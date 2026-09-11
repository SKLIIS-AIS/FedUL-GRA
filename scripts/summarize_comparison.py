import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


SCENARIO_LABELS = {
    'samples': 'Samples',
    'classes': 'Classes',
}

MODEL_LABELS = {
    ('mnist', 'lenet'): 'MNIST LeNet',
    ('cifar10', 'alexnet'): 'CIFAR10 AlexNet',
    ('cifar100', 'resnet18'): 'CIFAR100 ResNet18',
    ('celeba', 'alexnet'): 'CelebA AlexNet',
}

METHOD_ORDER = {
    'Retraining': 0,
    'Amnesiac': 1,
    'Class-disc': 2,
    'FedAU': 3,
    'FedUL': 4,
    'FedUL-MG-PRO-CNE': 5,
}


def read_last_json(path):
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
    return last


def method_from_path(path, root):
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


def scenario_from_path(path, root):
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    return rel.parts[1] if len(rel.parts) > 1 else None


def infer_dataset_model(path):
    parts = path.parts
    for dataset in ('mnist', 'cifar10', 'cifar100', 'celeba'):
        if dataset in parts:
            idx = parts.index(dataset)
            model = parts[idx - 1] if idx > 0 else ''
            return dataset, model
    return '', ''


def seed_from_name(path):
    for token in path.stem.split('_'):
        if token.startswith('s') and token[1:].isdigit():
            return int(token[1:])
    return None


def mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def fmt_percent(mean, std):
    if mean is None:
        return '-'
    return f'{mean * 100:.2f} +/- {std * 100:.2f}'


def fmt_latex_percent(mean, std):
    if mean is None:
        return '-'
    return f'{mean * 100:.2f} $\\pm$ {std * 100:.2f}'


def collect_logs(paths, root, include_external=False):
    rows = []
    for path in paths:
        root_path = root
        method = method_from_path(path, root_path)
        scenario = scenario_from_path(path, root_path)
        if include_external and (method not in METHOD_ORDER):
            name = path.parent.parent.parent.name
            if name == 'original_full':
                method = 'FedAU'
                scenario = 'samples'
            elif name == 'independent_full':
                method = 'FedUL-MG-PRO-CNE'
                scenario = 'samples'
        if method not in METHOD_ORDER or scenario not in SCENARIO_LABELS:
            continue

        dataset, model = infer_dataset_model(path)
        payload = read_last_json(path)
        if not payload:
            continue
        ul_effect = payload.get('UL effect')
        ul_acc_after = None if ul_effect is None else 1 - ul_effect
        rows.append({
            'method': method,
            'scenario': scenario,
            'dataset': dataset,
            'model': model,
            'seed': seed_from_name(path),
            'log': str(path),
            'epoch': payload.get('epoch'),
            'fedavg_rm_acc': payload.get('test_acc'),
            'ul_acc_after': ul_acc_after,
            'ul_success_after': ul_effect,
            'rm_acc_after': payload.get('UL val acc'),
            'ul_acc_before': payload.get('UL set acc'),
            'time': payload.get('time'),
            'learn_time': payload.get('learn_time'),
            'unlearn_time': payload.get('unlearntime'),
            'recovery': payload.get('recovery'),
            'recovery_applied': payload.get('recovery_applied'),
            'recovery_blend': payload.get('recovery_blend'),
            'recovery_retain_gate_mean': payload.get('recovery_retain_gate_mean'),
            'recovery_ul_gate_mean': payload.get('recovery_ul_gate_mean'),
            'recovery_time': payload.get('recovery_time'),
            'pre_recovery_rm_acc': payload.get('pre_recovery_UL val acc'),
            'candidate_recovery_rm_acc': payload.get('candidate_recovery_UL val acc'),
            'selected_recovery_rm_acc': payload.get('selected_recovery_UL val acc'),
            'pre_recovery_ul_effect': payload.get('pre_recovery_UL effect'),
            'candidate_recovery_ul_effect': payload.get('candidate_recovery_UL effect'),
            'selected_recovery_ul_effect': payload.get('selected_recovery_UL effect'),
            'recovery_skip_reason': payload.get('recovery_skip_reason'),
            'mtime': path.stat().st_mtime,
        })
    return rows


def keep_latest_per_seed(rows):
    latest = {}
    unique_without_seed = []
    for row in rows:
        seed = row.get('seed')
        if seed is None:
            unique_without_seed.append(row)
            continue
        key = (row['dataset'], row['model'], row['scenario'], row['method'], seed)
        current = latest.get(key)
        if current is None or row['mtime'] > current['mtime']:
            latest[key] = row
    return list(latest.values()) + unique_without_seed


def write_csv(path, rows):
    fieldnames = [
        'dataset', 'model', 'scenario', 'method', 'num_runs', 'fedavg_rm_acc',
        'ul_acc_after', 'ul_success_after', 'rm_acc_after', 'ul_acc_before',
        'recovery_attempted', 'recovery_applied', 'recovery_blend',
        'recovery_retain_gate_mean', 'recovery_ul_gate_mean',
        'pre_recovery_rm_acc', 'candidate_recovery_rm_acc',
        'selected_recovery_rm_acc', 'pre_recovery_ul_effect',
        'candidate_recovery_ul_effect', 'selected_recovery_ul_effect',
        'recovery_time', 'time', 'logs'
    ]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_markdown(path, rows):
    lines = [
        '| Dataset | UL Method | Method | Runs | FedAvg Rm-Acc | Ul/Rm-Acc | Recovery | Blend | Gate R/U | Pre/Cand/Sel Rm-Acc | Time(s) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} {row['model']} | {SCENARIO_LABELS[row['scenario']]} | "
            f"{row['method']} | {row['num_runs']} | {row['fedavg_rm_acc']} | "
            f"{row['ul_acc_after']} / {row['rm_acc_after']} | "
            f"{row['recovery_applied']} | {row['recovery_blend']} | "
            f"{row['recovery_retain_gate_mean']} / {row['recovery_ul_gate_mean']} | "
            f"{row['pre_recovery_rm_acc']} / {row['candidate_recovery_rm_acc']} / "
            f"{row['selected_recovery_rm_acc']} | {row['time']} |"
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_latex(path, rows):
    lines = [
        r'\begin{tabular}{lllcccc}',
        r'\toprule',
        r'Dataset & UL Method & Method & Runs & FedAvg Rm-Acc & Ul Acc / Rm Acc & Time(s) \\',
        r'\midrule',
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} {row['model']} & {SCENARIO_LABELS[row['scenario']]} & "
            f"{row['method']} & {row['num_runs']} & {row['fedavg_rm_acc_latex']} & "
            f"{row['ul_acc_after_latex']} / {row['rm_acc_after_latex']} & {row['time_latex']} \\\\"
        )
    lines.extend([r'\bottomrule', r'\end{tabular}', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Summarize FedAU/FedUL JSON logs.')
    parser.add_argument('--root', default='results_compare')
    parser.add_argument('--output', default='results_compare')
    parser.add_argument('--include_external_logs', action='store_true')
    parser.add_argument('--external_roots', default='../original_full,../independent_full')
    args = parser.parse_args()

    root = Path(args.root)
    paths = list(root.rglob('*.log')) if root.exists() else []
    if args.include_external_logs:
        for external in [Path(item.strip()) for item in args.external_roots.split(',') if item.strip()]:
            if external.exists():
                paths.extend(external.rglob('*.log'))

    raw_rows = keep_latest_per_seed(collect_logs(paths, root, include_external=args.include_external_logs))
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[(row['dataset'], row['model'], row['scenario'], row['method'])].append(row)

    summary = []
    def group_key(item):
        dataset, model, scenario, method = item
        return dataset, model, list(SCENARIO_LABELS).index(scenario), METHOD_ORDER.get(method, 999)

    for (dataset, model, scenario, method), items in sorted(grouped.items(), key=lambda kv: group_key(kv[0])):
        fedavg_mean, fedavg_std = mean_std([item['fedavg_rm_acc'] for item in items if item['fedavg_rm_acc'] is not None])
        ul_mean, ul_std = mean_std([item['ul_acc_after'] for item in items if item['ul_acc_after'] is not None])
        rm_mean, rm_std = mean_std([item['rm_acc_after'] for item in items if item['rm_acc_after'] is not None])
        time_mean, time_std = mean_std([item['time'] for item in items if item['time'] is not None])
        recovery_items = [item for item in items if item.get('recovery')]
        recovery_attempted = len(recovery_items)
        recovery_applied = sum(1 for item in recovery_items if item.get('recovery_applied'))
        recovery_blend_mean, recovery_blend_std = mean_std([item['recovery_blend'] for item in items if item['recovery_blend'] is not None])
        recovery_retain_gate_mean, recovery_retain_gate_std = mean_std([item['recovery_retain_gate_mean'] for item in items if item.get('recovery_retain_gate_mean') is not None])
        recovery_ul_gate_mean, recovery_ul_gate_std = mean_std([item['recovery_ul_gate_mean'] for item in items if item.get('recovery_ul_gate_mean') is not None])
        recovery_time_mean, recovery_time_std = mean_std([item['recovery_time'] for item in items if item['recovery_time'] is not None])
        pre_recovery_rm_mean, pre_recovery_rm_std = mean_std([item['pre_recovery_rm_acc'] for item in items if item['pre_recovery_rm_acc'] is not None])
        candidate_recovery_rm_mean, candidate_recovery_rm_std = mean_std([item['candidate_recovery_rm_acc'] for item in items if item['candidate_recovery_rm_acc'] is not None])
        selected_recovery_rm_mean, selected_recovery_rm_std = mean_std([item['selected_recovery_rm_acc'] for item in items if item['selected_recovery_rm_acc'] is not None])
        pre_recovery_ul_mean, pre_recovery_ul_std = mean_std([item['pre_recovery_ul_effect'] for item in items if item['pre_recovery_ul_effect'] is not None])
        candidate_recovery_ul_mean, candidate_recovery_ul_std = mean_std([item['candidate_recovery_ul_effect'] for item in items if item['candidate_recovery_ul_effect'] is not None])
        selected_recovery_ul_mean, selected_recovery_ul_std = mean_std([item['selected_recovery_ul_effect'] for item in items if item['selected_recovery_ul_effect'] is not None])
        summary.append({
            'dataset': dataset.upper() if dataset.startswith('cifar') else dataset.upper(),
            'model': model,
            'scenario': scenario,
            'method': method,
            'num_runs': len(items),
            'fedavg_rm_acc': fmt_percent(fedavg_mean, fedavg_std),
            'ul_acc_after': fmt_percent(ul_mean, ul_std),
            'ul_success_after': fmt_percent(*mean_std([item['ul_success_after'] for item in items if item['ul_success_after'] is not None])),
            'rm_acc_after': fmt_percent(rm_mean, rm_std),
            'ul_acc_before': fmt_percent(*mean_std([item['ul_acc_before'] for item in items if item['ul_acc_before'] is not None])),
            'recovery_attempted': '-' if recovery_attempted == 0 else f'{recovery_attempted}/{len(items)}',
            'recovery_applied': '-' if recovery_attempted == 0 else f'{recovery_applied}/{recovery_attempted}',
            'recovery_blend': '-' if recovery_blend_mean is None else f'{recovery_blend_mean:.2f} +/- {recovery_blend_std:.2f}',
            'recovery_retain_gate_mean': '-' if recovery_retain_gate_mean is None else f'{recovery_retain_gate_mean:.2f} +/- {recovery_retain_gate_std:.2f}',
            'recovery_ul_gate_mean': '-' if recovery_ul_gate_mean is None else f'{recovery_ul_gate_mean:.2f} +/- {recovery_ul_gate_std:.2f}',
            'recovery_time': '-' if recovery_time_mean is None else f'{recovery_time_mean:.2f} +/- {recovery_time_std:.2f}',
            'pre_recovery_rm_acc': fmt_percent(pre_recovery_rm_mean, pre_recovery_rm_std),
            'candidate_recovery_rm_acc': fmt_percent(candidate_recovery_rm_mean, candidate_recovery_rm_std),
            'selected_recovery_rm_acc': fmt_percent(selected_recovery_rm_mean, selected_recovery_rm_std),
            'pre_recovery_ul_effect': fmt_percent(pre_recovery_ul_mean, pre_recovery_ul_std),
            'candidate_recovery_ul_effect': fmt_percent(candidate_recovery_ul_mean, candidate_recovery_ul_std),
            'selected_recovery_ul_effect': fmt_percent(selected_recovery_ul_mean, selected_recovery_ul_std),
            'time': '-' if time_mean is None else f'{time_mean:.2f} +/- {time_std:.2f}',
            'fedavg_rm_acc_latex': fmt_latex_percent(fedavg_mean, fedavg_std),
            'ul_acc_after_latex': fmt_latex_percent(ul_mean, ul_std),
            'rm_acc_after_latex': fmt_latex_percent(rm_mean, rm_std),
            'time_latex': '-' if time_mean is None else f'{time_mean:.2f} $\\pm$ {time_std:.2f}',
            'logs': '; '.join(item['log'] for item in items),
        })

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'comparison_summary.csv', summary)
    write_markdown(output / 'comparison_summary.md', summary)
    write_latex(output / 'comparison_summary.tex', summary)

    print(f'Found {len(raw_rows)} log runs.')
    print(f'Wrote {output / "comparison_summary.csv"}')
    print(f'Wrote {output / "comparison_summary.md"}')
    print(f'Wrote {output / "comparison_summary.tex"}')


if __name__ == '__main__':
    main()
