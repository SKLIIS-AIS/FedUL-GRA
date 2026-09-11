import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


METHOD_VARIANTS = {
    'FedAU': 'original',
    'FedUL': 'independent',
    'FedUL-MG-PRO-CNE': 'independent',
    'Retraining': 'original',
    'Amnesiac': 'original',
    'Class-disc': 'original',
}

METHOD_SCENARIO_MODES = {
    'FedAU': {
        'samples': 'ul_samples_backdoor',
        'classes': 'ul_class',
    },
    'FedUL': {
        'samples': 'ul_samples_backdoor',
        'classes': 'ul_class',
    },
    'FedUL-MG-PRO-CNE': {
        'samples': 'ul_samples_backdoor',
        'classes': 'ul_class',
    },
    'Retraining': {
        'samples': 'retrain_samples',
        'classes': 'retrain_class',
    },
    'Amnesiac': {
        'samples': 'amnesiac_ul_samples',
        'classes': 'amnesiac_ul_class',
    },
    'Class-disc': {
        'classes': 'amnesiac_ul_class',
    },
}

SCENARIO_MODES = METHOD_SCENARIO_MODES['FedAU']

DATASET_MODELS = {
    'mnist': 'lenet',
    'cifar10': 'alexnet',
    'cifar100': 'resnet18',
    'celeba': 'alexnet',
}

INDEPENDENT_DATASETS = {'mnist', 'cifar10', 'celeba'}


def parse_list(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def resolve_repo_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def experiment_proportion(args, scenario):
    if scenario == 'classes':
        return 1.0
    return args.proportion


def latest_completed_log(log_dir, seed, epochs):
    if not log_dir.exists():
        return None
    patterns = [f'*_s{seed}_*.log', f'*s{seed}_*.log']
    paths = []
    for pattern in patterns:
        paths.extend(log_dir.glob(pattern))
    seen = set()
    unique_paths = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    for path in sorted(unique_paths, key=lambda p: p.stat().st_mtime, reverse=True):
        last = None
        with path.open('r', encoding='utf-8', errors='ignore') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
        if last and last.get('epoch') == epochs - 1:
            return path
    return None


def build_command(args, method, dataset, scenario, seed):
    model_name = DATASET_MODELS[dataset]
    ul_mode = METHOD_SCENARIO_MODES[method][scenario]
    proportion = experiment_proportion(args, scenario)
    log_folder = str(Path(args.output_root) / method / scenario)
    cmd = [
        sys.executable,
        str(REPO_ROOT / 'main_zgx.py'),
        '--model_variant', METHOD_VARIANTS[method],
        '--num_users', str(args.num_users),
        '--dataset', dataset,
        '--model_name', model_name,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--proportion', str(proportion),
        '--num_ul_users', str(args.num_ul_users),
        '--ul_mode', ul_mode,
        '--local_ep', str(args.local_ep),
        '--lr', str(args.lr),
        '--iid', str(args.iid),
        '--optim', args.optim,
        '--ul_class_id', str(args.ul_class_id),
        '--log_folder_name', log_folder,
        '--data_root', args.data_root,
        '--seed', str(seed),
        '--gpu', args.gpu,
        '--device', args.device,
        '--celeba_attr', args.celeba_attr,
        '--celeba_target_mode', args.celeba_target_mode,
    ]
    if METHOD_VARIANTS[method] == 'independent':
        cmd.extend([
            '--recovery_epochs', str(args.recovery_epochs),
            '--recovery_freeze_aux', str(args.recovery_freeze_aux),
            '--recovery_objective', args.recovery_objective,
            '--recovery_accept_ul_drop', str(args.recovery_accept_ul_drop),
        ])
        if args.recovery_lr is not None:
            cmd.extend(['--recovery_lr', str(args.recovery_lr)])
        if args.disable_cne:
            cmd.append('--disable_cne')
        if args.disable_gra:
            cmd.append('--disable_gra')
    return cmd


def build_source_command(args, method, dataset, scenario, seed):
    model_name = DATASET_MODELS[dataset]
    ul_mode = METHOD_SCENARIO_MODES[method][scenario]
    proportion = experiment_proportion(args, scenario)
    log_folder = str(Path(args.output_root) / '_baseline_sources' / method / scenario)
    return [
        sys.executable,
        str(REPO_ROOT / 'main_zgx.py'),
        '--model_variant', 'original',
        '--num_users', str(args.num_users),
        '--dataset', dataset,
        '--model_name', model_name,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--proportion', str(proportion),
        '--num_ul_users', str(args.num_ul_users),
        '--ul_mode', ul_mode,
        '--local_ep', str(args.local_ep),
        '--lr', str(args.lr),
        '--iid', str(args.iid),
        '--optim', args.optim,
        '--ul_class_id', str(args.ul_class_id),
        '--log_folder_name', log_folder,
        '--data_root', args.data_root,
        '--seed', str(seed),
        '--gpu', args.gpu,
        '--device', args.device,
        '--celeba_attr', args.celeba_attr,
        '--celeba_target_mode', args.celeba_target_mode,
    ]


def build_after_learning_command(args, method, dataset, scenario, seed):
    model_name = DATASET_MODELS[dataset]
    source_root = str(Path(args.output_root) / '_baseline_sources' / method / scenario)
    output_root = str(Path(args.output_root) / method / scenario)
    cmd = [
        sys.executable,
        str(REPO_ROOT / 'scripts' / 'run_after_learning_baseline.py'),
        '--method', method,
        '--dataset', dataset,
        '--model_name', model_name,
        '--scenario', scenario,
        '--source_root', source_root,
        '--output_root', output_root,
        '--seed', str(seed),
        '--epochs', str(args.epochs),
        '--local_ep', str(args.local_ep),
        '--batch_size', str(args.batch_size),
        '--num_users', str(args.num_users),
        '--lr', str(args.lr),
        '--iid', str(args.iid),
        '--optim', args.optim,
        '--gpu', args.gpu,
        '--device', args.device,
        '--ul_class_id', str(args.ul_class_id),
        '--class_disc_sparsity', str(args.class_disc_sparsity),
        '--class_disc_batches', str(args.class_disc_batches),
        '--class_disc_finetune_epochs', str(args.class_disc_finetune_epochs),
    ]
    return cmd


def is_after_learning_method(method):
    return method in {'Class-disc'}


def supports_method_dataset(method, dataset):
    return METHOD_VARIANTS[method] != 'independent' or dataset in INDEPENDENT_DATASETS


def run_and_record(cmd, writer, manifest_file, method, dataset, scenario, seed, dry_run, stage):
    print(f'\n[run:{stage}]', ' '.join(cmd))
    if dry_run:
        return
    start = time.time()
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    seconds = time.time() - start
    writer.writerow({
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'method': method,
        'dataset': dataset,
        'scenario': scenario,
        'seed': seed,
        'stage': stage,
        'returncode': completed.returncode,
        'seconds': round(seconds, 2),
        'command': ' '.join(cmd),
    })
    manifest_file.flush()

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main():
    parser = argparse.ArgumentParser(description='Run FedAU/FedUL and main_zgx baselines.')
    parser.add_argument('--methods', default='FedAU,FedUL,Retraining,Amnesiac')
    parser.add_argument('--datasets', default='mnist')
    parser.add_argument('--scenarios', default='samples,classes')
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--local_ep', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--iid', type=int, default=1)
    parser.add_argument('--optim', default='sgd')
    parser.add_argument('--num_users', type=int, default=10)
    parser.add_argument('--num_ul_users', type=int, default=1)
    parser.add_argument('--ul_class_id', type=int, default=9)
    parser.add_argument(
        '--proportion',
        type=float,
        default=0.1,
        help='Sample-level request proportion; class-level experiments always use 1.0.',
    )
    parser.add_argument('--celeba_attr', default='Smiling')
    parser.add_argument('--celeba_target_mode', choices=['binary', 'attrs10'], default='attrs10')
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--output_root', default='results_compare')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='cuda')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--rerun_completed', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Run a tiny one-epoch sanity check.')
    parser.add_argument('--class_disc_sparsity', type=float, default=0.05)
    parser.add_argument('--class_disc_batches', type=int, default=20)
    parser.add_argument('--class_disc_finetune_epochs', type=int, default=10)
    parser.add_argument('--recovery_epochs', type=int, default=1)
    parser.add_argument('--recovery_lr', type=float, default=None)
    parser.add_argument('--recovery_freeze_aux', type=int, default=1)
    parser.add_argument('--recovery_objective', choices=['unlearn', 'learning'], default='unlearn')
    parser.add_argument('--recovery_accept_ul_drop', type=float, default=0.02)
    parser.add_argument('--disable_cne', action='store_true')
    parser.add_argument('--disable_gra', action='store_true')
    args = parser.parse_args()

    if args.smoke:
        args.methods = 'FedAU,FedUL,Retraining,Amnesiac'
        args.datasets = 'mnist'
        args.scenarios = 'samples'
        args.seeds = '0'
        args.epochs = 1
        args.local_ep = 1
        args.batch_size = 64
        args.num_users = 2
        args.proportion = 0.01
        args.output_root = 'results_smoke'
        args.class_disc_batches = 1
        args.class_disc_finetune_epochs = 1

    args.data_root = str(resolve_repo_path(args.data_root))
    args.output_root = str(resolve_repo_path(args.output_root))

    methods = parse_list(args.methods)
    datasets = parse_list(args.datasets)
    scenarios = parse_list(args.scenarios)
    seeds = [int(seed) for seed in parse_list(args.seeds)]

    for method in methods:
        if method not in METHOD_VARIANTS:
            raise ValueError(f'Unknown method: {method}')
    for dataset in datasets:
        if dataset not in DATASET_MODELS:
            raise ValueError(f'Unknown dataset: {dataset}')
    if 'celeba' in datasets:
        if args.celeba_target_mode == 'binary':
            if args.ul_class_id == 9:
                args.ul_class_id = 1
            elif args.ul_class_id not in (0, 1):
                raise ValueError('CelebA binary experiments use --ul_class_id 0 or 1.')
        elif not 0 <= args.ul_class_id < 10:
            raise ValueError('CelebA attrs10 experiments use --ul_class_id in [0, 9].')
    for scenario in scenarios:
        if scenario not in SCENARIO_MODES:
            raise ValueError(f'Unknown scenario: {scenario}')

    manifest_path = Path(args.output_root) / 'manifest.csv'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    first_manifest = not manifest_path.exists()

    with manifest_path.open('a', newline='', encoding='utf-8') as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=['time', 'method', 'dataset', 'scenario', 'seed', 'stage', 'returncode', 'seconds', 'command'],
        )
        if first_manifest:
            writer.writeheader()

        for dataset in datasets:
            model_name = DATASET_MODELS[dataset]
            for scenario in scenarios:
                for method in methods:
                    if not supports_method_dataset(method, dataset):
                        print(
                            f'[skip] {method} is not implemented for {dataset}; '
                            'the independent model currently supports LeNet/AlexNet datasets.'
                        )
                        continue
                    if scenario not in METHOD_SCENARIO_MODES[method]:
                        print(f'[skip] {method} does not apply to {scenario} unlearning.')
                        continue
                    for seed in seeds:
                        log_dir = Path(args.output_root) / method / scenario / model_name / dataset
                        done_log = latest_completed_log(log_dir, seed, args.epochs)
                        if done_log and not args.rerun_completed:
                            print(f'[skip] {method} {dataset} {scenario} seed={seed}: {done_log}')
                            continue

                        if is_after_learning_method(method):
                            source_log_dir = (
                                Path(args.output_root) / '_baseline_sources' / method / scenario / model_name / dataset
                            )
                            source_done_log = latest_completed_log(source_log_dir, seed, args.epochs)
                            if source_done_log and not args.rerun_completed:
                                print(f'[skip:source] {method} {dataset} {scenario} seed={seed}: {source_done_log}')
                            else:
                                source_cmd = build_source_command(args, method, dataset, scenario, seed)
                                run_and_record(
                                    source_cmd, writer, manifest_file,
                                    method, dataset, scenario, seed, args.dry_run, 'source',
                                )
                            baseline_cmd = build_after_learning_command(args, method, dataset, scenario, seed)
                            run_and_record(
                                baseline_cmd, writer, manifest_file,
                                method, dataset, scenario, seed, args.dry_run, 'baseline',
                            )
                        else:
                            cmd = build_command(args, method, dataset, scenario, seed)
                            run_and_record(
                                cmd, writer, manifest_file,
                                method, dataset, scenario, seed, args.dry_run, 'main',
                            )


if __name__ == '__main__':
    main()
