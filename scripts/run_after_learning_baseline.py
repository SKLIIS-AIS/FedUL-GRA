import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import dill
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models
from experiments.trainer_private import TrainerPrivate


NUM_CLASSES = {
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100,
    'celeba': None,
}

IN_CHANNELS = {
    'mnist': 1,
    'cifar10': 3,
    'cifar100': 3,
    'celeba': 3,
}


def latest_file(root, pattern):
    paths = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f'No files matching {pattern} in {root}')
    return paths[0]


def read_last_json(path):
    last = None
    if not path or not path.exists():
        return None
    with path.open('r', encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def latest_source_log(artifact_dir):
    logs = sorted(artifact_dir.glob('*.log'), key=lambda path: path.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def load_model_state(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, dict):
        if 'model_state_dict' in payload:
            return payload['model_state_dict']
        if 'net_info' in payload and isinstance(payload['net_info'], dict):
            net_info = payload['net_info']
            if 'best_model' in net_info:
                return net_info['best_model']
        if all(torch.is_tensor(value) for value in payload.values()):
            return payload
    raise ValueError(f'Could not find a model state dict in {path}')


def infer_num_classes_from_state_dict(state_dict, dataset, override=None):
    if override is not None:
        return int(override)

    classifier_weight = state_dict.get('classifier.weight')
    if torch.is_tensor(classifier_weight) and classifier_weight.ndim == 2:
        return int(classifier_weight.shape[0])

    for key, value in reversed(list(state_dict.items())):
        if key.endswith('.weight') and torch.is_tensor(value) and value.ndim == 2:
            return int(value.shape[0])

    fallback = NUM_CLASSES[dataset]
    if fallback is None:
        raise ValueError(
            'Could not infer num_classes for {} from the saved model. '
            'Please pass --num_classes explicitly.'.format(dataset)
        )
    return int(fallback)


def make_model(model_name, dataset, device, num_classes):
    model = models.__dict__[model_name](
        num_classes=num_classes,
        in_channels=IN_CHANNELS[dataset],
    )
    return model.to(device)


def evaluate(model, dataloader, device, num_classes):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = (target % num_classes).to(device)
            logits = model(data)
            loss_sum += F.cross_entropy(logits, target.long(), reduction='sum').item()
            pred = logits.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total


def evaluate_unlearn_success(model, dataloader, device, num_classes):
    model.eval()
    loss_sum = 0.0
    success = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = (target % num_classes).to(device)
            logits = model(data)
            loss_sum += F.cross_entropy(logits, target.long(), reduction='sum').item()
            pred = logits.max(1, keepdim=True)[1]
            success += pred.ne(target.view_as(pred)).sum().item()
            total += data.size(0)
    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, success / total


def collect_conv_class_scores(model, dataloader, target_class, num_classes, device, max_batches):
    conv_modules = {
        name: module for name, module in model.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    }
    if not conv_modules:
        raise ValueError('Class-disc needs at least one Conv2d layer.')

    target_sums = {}
    other_sums = {}
    target_counts = {}
    other_counts = {}
    batch_features = {}
    hooks = []

    for name, module in conv_modules.items():
        target_sums[name] = torch.zeros(module.out_channels)
        other_sums[name] = torch.zeros(module.out_channels)
        target_counts[name] = 0
        other_counts[name] = 0

        def hook(_module, _inputs, output, layer_name=name):
            feature = F.relu(output.detach()).mean(dim=(2, 3)).cpu()
            batch_features[layer_name] = feature

        hooks.append(module.register_forward_hook(hook))

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch_features.clear()
            data = data.to(device)
            target = (target % num_classes).cpu()
            model(data)
            target_mask = target.eq(target_class)
            other_mask = ~target_mask
            for name, feature in batch_features.items():
                if target_mask.any():
                    target_sums[name].add_(feature[target_mask].sum(dim=0))
                    target_counts[name] += int(target_mask.sum().item())
                if other_mask.any():
                    other_sums[name].add_(feature[other_mask].sum(dim=0))
                    other_counts[name] += int(other_mask.sum().item())

    for hook in hooks:
        hook.remove()

    scores = {}
    for name in conv_modules:
        target_mean = target_sums[name] / max(target_counts[name], 1)
        other_mean = other_sums[name] / max(other_counts[name], 1)
        score = target_mean / (other_mean + 1e-6)
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        scores[name] = score
    return scores


def apply_class_disc_pruning(model, scores, sparsity):
    all_scores = torch.cat([score.flatten() for score in scores.values()])
    num_prune = max(1, int(math.ceil(float(sparsity) * all_scores.numel())))
    num_prune = min(num_prune, all_scores.numel())
    threshold = torch.topk(all_scores, num_prune).values.min()

    pruned = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if name not in scores:
                continue
            mask = scores[name] >= threshold
            if not mask.any():
                continue
            module.weight[mask.to(module.weight.device)] = 0
            if module.bias is not None:
                module.bias[mask.to(module.bias.device)] = 0
            pruned += int(mask.sum().item())
    return pruned


def finetune_without_class(model, dataloader, target_class, num_classes, device, epochs, lr, optim_name):
    if epochs <= 0:
        return
    model.train()
    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0005)

    for _epoch in range(epochs):
        for data, target in dataloader:
            data = data.to(device)
            target = (target % num_classes).to(device)
            keep = target.ne(target_class)
            if not keep.any():
                continue
            optimizer.zero_grad()
            logits = model(data[keep])
            loss = F.cross_entropy(logits, target[keep].long())
            loss.backward()
            optimizer.step()


def write_log(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        json.dump(payload, handle)
        handle.write('\n')


def artifact_dir(args):
    return Path(args.source_root) / args.model_name / args.dataset


def output_log_path(args, stem):
    output_dir = Path(args.output_root) / args.model_name / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / stem


def load_common_artifacts(args, device):
    root = artifact_dir(args)
    dataloader_path = latest_file(root, f'FedUL_dataloader_s{args.seed}_*.pkl')
    model_path = None
    model_candidates = sorted(root.glob(f'FedUL_model_s{args.seed}_*.pkl'), key=lambda path: path.stat().st_mtime, reverse=True)
    if model_candidates:
        model_path = model_candidates[0]
    else:
        model_path = latest_file(root, f'Final_s{args.seed}_*.pkl')

    with dataloader_path.open('rb') as handle:
        dataloaders = dill.load(handle)

    model_state = load_model_state(model_path)
    num_classes = infer_num_classes_from_state_dict(
        model_state,
        args.dataset,
        override=args.num_classes,
    )
    model = make_model(args.model_name, args.dataset, device, num_classes)
    model.load_state_dict(model_state, strict=False)

    source_log = latest_source_log(root)
    source_last = read_last_json(source_log) or {}
    return root, dataloader_path, model_path, model, dataloaders, source_last, num_classes


def run_class_disc(args):
    requested_device = args.device
    if requested_device == 'auto':
        requested_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Use --device cpu for a CPU run.')
    device = torch.device(requested_device)
    start = time.time()
    _root, dataloader_path, model_path, model, dataloaders, source_last, num_classes = load_common_artifacts(args, device)

    train_ldr = dataloaders['train_ldr']
    val_ldr = dataloaders['val_ldr']
    ul_ldr = dataloaders['ul_ldr']

    _pre_val_loss, pre_val_acc = evaluate(model, val_ldr, device, num_classes)
    _pre_ul_loss, pre_ul_success = evaluate_unlearn_success(model, ul_ldr, device, num_classes)
    pre_ul_acc = 1 - pre_ul_success

    scores = collect_conv_class_scores(
        model,
        train_ldr,
        target_class=args.ul_class_id,
        num_classes=num_classes,
        device=device,
        max_batches=args.class_disc_batches,
    )
    pruned_filters = apply_class_disc_pruning(model, scores, args.class_disc_sparsity)
    finetune_without_class(
        model,
        train_ldr,
        target_class=args.ul_class_id,
        num_classes=num_classes,
        device=device,
        epochs=args.class_disc_finetune_epochs,
        lr=args.lr,
        optim_name=args.optim,
    )

    _post_val_loss, post_val_acc = evaluate(model, val_ldr, device, num_classes)
    _post_ul_loss, post_ul_success = evaluate_unlearn_success(model, ul_ldr, device, num_classes)

    elapsed = time.time() - start
    total_time = elapsed + float(source_last.get('time', 0.0))
    log_path = output_log_path(
        args,
        f'Class-disc_classes_s{args.seed}_{args.num_users}_{args.batch_size}_{args.lr}_{args.iid}.log',
    )
    write_log(log_path, {
        'epoch': args.epochs - 1,
        'method': 'Class-disc',
        'scenario': 'classes',
        'train_acc': None,
        'test_acc': round(pre_val_acc, 4),
        'UL set acc': round(pre_ul_acc, 4),
        'UL val acc': round(post_val_acc, 4),
        'UL effect': round(post_ul_success, 4),
        'time': round(total_time, 2),
        'learn_time': round(float(source_last.get('time', 0.0)), 2),
        'unlearntime': round(elapsed, 2),
        'pruned_filters': pruned_filters,
        'sparsity': args.class_disc_sparsity,
        'num_classes': num_classes,
        'source_model': str(model_path),
        'source_dataloader': str(dataloader_path),
    })
    print(f'[Class-disc] wrote {log_path}')


def main():
    parser = argparse.ArgumentParser(description='Run after-learning baselines from saved FedAU artifacts.')
    parser.add_argument('--method', choices=['Class-disc'], required=True)
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100', 'celeba'], required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--scenario', choices=['classes'], required=True)
    parser.add_argument('--source_root', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--local_ep', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_users', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--iid', type=int, default=1)
    parser.add_argument('--optim', default='sgd')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='cuda')
    parser.add_argument('--gpu', default='0', help='legacy GPU label retained for compatibility')
    parser.add_argument('--dp', action='store_true', default=False)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--ul_class_id', type=int, default=9)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--class_disc_sparsity', type=float, default=0.05)
    parser.add_argument('--class_disc_batches', type=int, default=20)
    parser.add_argument('--class_disc_finetune_epochs', type=int, default=10)
    args = parser.parse_args()

    if args.method == 'Class-disc':
        if args.scenario != 'classes':
            raise ValueError('Class-disc is only supported for class unlearning.')
        run_class_disc(args)


if __name__ == '__main__':
    main()
