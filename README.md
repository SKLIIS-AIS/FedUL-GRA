# FedUL-GRA

This paper has been accepted by ICA3PP 2026, and the link to the original text is [https://anonymous.4open.science/r/FedHUA-6399](https://github.com/liuzixiao09/FedUL-GRA)

PyTorch research code for federated machine unlearning. The repository contains
the FedAU-compatible training path and the FedUL-GRA independent dual-branch
extension with calibrated negative evidence, gated retain adaptation, and
optional retained-set recovery.

## Supported Experiments

| Dataset | Base model | FedAU-compatible | FedUL-GRA independent |
|---|---|---:|---:|
| MNIST | LeNet | Yes | Yes |
| CIFAR-10 | AlexNet | Yes | Yes |
| CIFAR-100 | ResNet-18 | Yes | No |
| CelebA | AlexNet | Yes | Yes |

The maintained runner supports sample-level and class-level unlearning,
retraining, and Amnesiac update subtraction. The comparison launcher also
supports the after-learning Class-disc baseline for class unlearning.

Both IID and non-IID client partitions are supported. CelebA non-IID experiments
use the same Dirichlet `--beta` partition interface as the other datasets.

## Repository Layout

```text
.
|-- main_zgx.py                         # Main training and unlearning entry point
|-- dataset.py                          # Dataset wrappers and forget-set construction
|-- experiments/
|   |-- base.py                         # Experiment setup and device selection
|   `-- trainer_private.py              # Local train, unlearn, CNE, GRA, recovery, metrics
|-- models/                             # LeNet, AlexNet, ResNet and unlearning variants
|-- utils/
|   |-- args.py                         # Command-line arguments
|   |-- datasets.py                     # Data loading and preprocessing
|   `-- sampling.py                     # IID and non-IID client partitioning
|-- scripts/
|   |-- run_comparison_experiments.py   # Multi-method experiment launcher
|   |-- run_after_learning_baseline.py  # Class-disc baseline
|   `-- summarize_comparison.py         # CSV, Markdown and LaTeX summaries
|-- environment.yml
`-- requirements.txt
```

## Installation

The current code was checked against the following server environment:

- Ubuntu 22.04
- Python 3.12.3
- PyTorch 2.7.0+cu128
- torchvision 0.22.0+cu128
- NumPy 2.2.6
- NVIDIA RTX 4090 D
- NVIDIA driver 595.71.05

Create the complete Conda environment:

```bash
conda env create -f environment.yml
conda activate fedul-gra
```

For a clean manual installation:

```bash
conda create -n fedau python=3.12 -y
conda activate fedau
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

If PyTorch 2.7.0+cu128 and torchvision 0.22.0+cu128 are already installed, only
the remaining packages are needed:

```bash
python -m pip install -r requirements.txt
```

Verify the environment:

```bash
python - <<'PY'
import torch
import torchvision
import numpy
import scipy
import PIL
import dill

print('PyTorch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('NumPy:', numpy.__version__)
print('SciPy:', scipy.__version__)
print('Pillow:', PIL.__version__)
print('dill:', dill.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA runtime:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
PY
```

The CUDA-enabled PyTorch wheel includes its required CUDA runtime. A separate
`nvcc` installation is not required for running these experiments.

The reference experiments use CUDA. A CPU execution path is available for
debugging with `--device cpu`, but it is much slower and should not be mixed with
CUDA runs when reporting comparable results.

## Data

The default data directory is `./data`. It can be changed with `--data_root`.

MNIST, CIFAR-10, and CIFAR-100 are downloaded automatically by torchvision when
the required files are missing.

CelebA must be downloaded separately and arranged in torchvision's standard
layout under the selected data root:

```text
data/
`-- celeba/
    |-- img_align_celeba/
    |-- identity_CelebA.txt
    |-- list_attr_celeba.txt
    |-- list_bbox_celeba.txt
    |-- list_eval_partition.txt
    `-- list_landmarks_align_celeba.txt
```

## Check Commands Before Training

Preview the generated experiment commands without starting PyTorch training:

```bash
python scripts/run_comparison_experiments.py --smoke --dry_run --device cpu
```

Display all main-runner options:

```bash
python main_zgx.py --help
```

## Single Experiments

Run commands from the repository root.

FedAU-compatible MNIST sample-level unlearning:

```bash
python main_zgx.py \
  --model_variant original \
  --dataset mnist \
  --model_name lenet \
  --ul_mode ul_samples_backdoor \
  --num_users 10 \
  --num_ul_users 1 \
  --proportion 0.1 \
  --epochs 200 \
  --local_ep 2 \
  --batch_size 128 \
  --seed 0 \
  --device cuda \
  --data_root ./data \
  --log_folder_name ./results/FedAU/samples
```

FedUL-GRA with the same experiment settings:

```bash
python main_zgx.py \
  --model_variant independent \
  --dataset mnist \
  --model_name lenet \
  --ul_mode ul_samples_backdoor \
  --num_users 10 \
  --num_ul_users 1 \
  --proportion 0.1 \
  --epochs 200 \
  --local_ep 2 \
  --batch_size 128 \
  --seed 0 \
  --device cuda \
  --data_root ./data \
  --log_folder_name ./results/FedUL/samples
```

For class-level unlearning, use `--ul_mode ul_class` and set
`--ul_class_id`. Class-level runs always use the full target class internally;
the comparison launcher therefore forces their request proportion to `1.0`.

For non-IID experiments, set `--iid 0` and choose a Dirichlet concentration:

```bash
python main_zgx.py \
  --model_variant independent \
  --dataset cifar10 \
  --model_name alexnet \
  --ul_mode ul_class \
  --ul_class_id 9 \
  --iid 0 \
  --beta 0.5 \
  --device cuda \
  --data_root ./data \
  --log_folder_name ./results/FedUL/classes_noniid
```

To select a specific GPU, set `CUDA_VISIBLE_DEVICES` before starting the command.
For example:

```bash
CUDA_VISIBLE_DEVICES=0 python main_zgx.py --device cuda [other arguments]
```

## Reproduce Method Comparisons

Run FedAU, FedUL-GRA, Retraining, and Amnesiac over several seeds:

```bash
python scripts/run_comparison_experiments.py \
  --methods Retraining,Amnesiac,FedAU,FedUL \
  --datasets mnist,cifar10 \
  --scenarios samples,classes \
  --seeds 0,1,2 \
  --epochs 200 \
  --local_ep 2 \
  --batch_size 128 \
  --device cuda \
  --data_root ./data \
  --output_root ./results_compare
```

Add `Class-disc` to `--methods` when running class-level comparisons. The
launcher first creates the required source artifacts and then invokes
`scripts/run_after_learning_baseline.py`.

Completed runs are skipped automatically unless `--rerun_completed` is passed.
Every launched command and return code is recorded in `manifest.csv`.

## Summarize Results

```bash
python scripts/summarize_comparison.py \
  --root ./results_compare \
  --output ./results_compare
```

The summarizer writes:

- `comparison_summary.csv`
- `comparison_summary.md`
- `comparison_summary.tex`

For repeated runs with the same method, dataset, scenario, and seed, the newest
completed log is used.

## Outputs and Safety

Experiment outputs are stored under:

```text
<log_folder_name>/<model_name>/<dataset>/
```

The runner saves JSON-lines logs, model payloads, update histories, and serialized
DataLoader bundles. These files can be large and are excluded by `.gitignore`.
PyTorch pickle and dill files must only be loaded from trusted sources.

## License Notice

This repository contains code derived from or inspired by the FedAU project.
Read [NOTICE.md](NOTICE.md) before redistribution. The license or redistribution
permission for inherited source files must be confirmed before publishing the
repository under a new `LICENSE`.
