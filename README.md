# [Re] A Replication Study of Multi-Headed Synthetic Feature Distillation: When a Properly Tuned Baseline Wins

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains the complete code, logs, and artifacts for the paper:

> **[Re] A Replication Study of Multi-Headed Synthetic Feature Distillation: When a Properly Tuned Baseline Wins**
> Yonas Zewdie, Gage University College, Ethiopia
> Submitted to *ReScience C*, September 2026

---

## Table of Contents

1. [What Is Being Replicated?](#what-is-being-replicated)
2. [Key Finding](#key-finding)
3. [Two Pipelines](#two-pipelines)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Data](#data)
7. [Quick Start: Full Reproduction](#quick-start-full-reproduction)
8. [Reproduce Specific Results](#reproduce-specific-results)
9. [Checkpoints and Artifacts](#checkpoints-and-artifacts)
10. [Expected Results](#expected-results)
11. [Hardware Requirements](#hardware-requirements)
12. [Training Logs](#training-logs)
13. [Reproducing Figures](#reproducing-figures)
14. [Known Issues and Caveats](#known-issues-and-caveats)
15. [Reproducibility Notes](#reproducibility-notes)
16. [Citation](#citation)
17. [License](#license)
18. [Contact](#contact)

---

## What Is Being Replicated?

This is a replication of the method introduced in:

> Sbrolli, C., Michel, N., Matteucci, M., & Yamasaki, T. (2026).
> *Beyond Raw Signals: Undecoded Generative Latents as Privileged Synthetic Data.*
> arXiv:2606.08336.

The original paper introduces **MESSy** (Multilayer Explicit Simulated Synesthesia) — a set of lightweight predictive heads for cross-modal distillation — and reports that it outperforms standard knowledge distillation baselines on image classification.

This repository re-runs the MESSy comparison on CIFAR-100 under matched conditions:

- Five random seeds per method (42, 139, 236, 333, 430)
- 12-epoch training budget for all methods
- KD hyperparameters selected by validation-only grid search
- A no-distillation baseline included

The method is referred to as **MSFD** (Multi-Headed Synthetic Feature Distillation) throughout this repository, following the name used in the paper accompanying this code. MSFD is a reimplementation of the MESSy mechanism.

**Outcome (failed replication):** MESSy (MSFD) does **not** significantly outperform a properly tuned KD baseline on CIFAR-100. A tuned KD baseline significantly outperforms both MESSy and the no-distillation baseline. Under matched conditions, the original result did not reproduce.

---

## Key Finding

| Method | Test Accuracy | Std | 95% CI | n |
|--------|---------------|-----|--------|---|
| No-distillation baseline | 77.20% | 0.16% | [76.99%, 77.40%] | 5 |
| **MSFD** (multi-head, MSE) | 77.45% | 0.25% | [77.23%, 77.67%] | 5 |
| **Standard KD** (tuned, T=1.0, α=0.3) | **78.00%** | 0.39% | [77.52%, 78.48%] | 5 |

**Paired significance tests (n=5 matched seeds):**

| Comparison | Mean Diff. | t-stat | p-value | Significant? |
|------------|------------|--------|---------|--------------|
| Tuned KD vs. Baseline | +0.80 pp | 5.24 | 0.006 | Yes |
| Tuned KD vs. MSFD | +0.55 pp | 3.32 | 0.029 | Yes |
| MSFD vs. Baseline | +0.26 pp | 1.73 | 0.160 | **No** |

**Interpretation:** In this setting, KD hyperparameter choice had a larger effect on the comparison outcome than the architectural difference between the two methods being compared.

---

## Two Pipelines

This project contains **two separate experimental pipelines**, each with its own logs, checkpoints, and results. They share the same codebase but are run separately.

### Pipeline 1: KD Hyperparameter Sweep

**Purpose:** find the optimal KD configuration (temperature T, CE weight α) before running the main comparison.

- **Entry point:** `scripts/run_kd_sweep.sh`
- **Config:** `configs/kd_sweep.yaml`
- **Logs:** `logs/kd_sweep/`
- **Results:** `results/kd_sweep/`
- **Duration:** ~4.5 hours on a free-tier T4
- **Outputs:** 12 sweep checkpoints, sweep CSV, sweep figure, best config file

### Pipeline 2: Complete Training

**Purpose:** train the teacher, train MSFD, KD, and no-distillation baselines across 5 seeds, run deliberate practice and progressive distillation, and evaluate.

- **Entry point:** `scripts/run_complete_training.sh`
- **Config:** `configs/default.yaml`
- **Logs:** `logs/complete_training/`
- **Results:** `results/complete_training/`
- **Duration:** ~18 hours on a free-tier T4
- **Outputs:** teacher + 15 student checkpoints (5 seeds × 3 methods), all logs, all final results, all paper figures except Figure 1

### Ordering

Run the KD sweep **before** complete training. The sweep selects the tuned KD configuration (T=1.0, α=0.3), which complete training then uses for its KD baseline.

To run both in the correct order:

```bash
bash scripts/run_all.sh
```

---

## Repository Structure

```
msfd-replication-study/
├── README.md # this file
├── metadata.yaml # ReScience C metadata
├── article.pdf # the paper PDF
├── requirements.txt # Python dependencies (pinned)
├── .gitignore # ignores large files
│
├── code/ # Pipeline 2: complete training code
│ ├── init.py
│ ├── config.py # configuration dataclass
│ ├── logger.py # logging utilities
│ ├── utils.py # seeding, cleanup, stats, checkpointing
│ ├── data_loader.py # CIFAR-100 loading with split validation
│ ├── models.py # teacher and student architectures
│ ├── training.py # generic training (teacher, baseline)
│ ├── distillation.py # MSFD + standard KD + feature cache
│ ├── deliberate_practice.py # hard example mining
│ ├── progressive.py # progressive distillation
│ ├── evaluation.py # per-class analysis, confusion matrix
│ ├── head_ablations.py # head ablation study
│ ├── sweep.py # KD hyperparameter sweep helpers
│ ├── train_teacher.py # CLI: teacher training
│ ├── train_msfd.py # CLI: MSFD student
│ ├── train_standard_kd.py # CLI: tuned KD
│ ├── train_baseline.py # CLI: no-distillation baseline
│ └── run_all.py # complete training pipeline entry point
│
├── Kd sweep/ # Pipeline 1: KD sweep code (separate module)
│ ├── init.py
│ ├── config.py
│ ├── logger.py
│ ├── utils.py
│ ├── data_loader.py
│ ├── models.py
│ ├── training.py
│ ├── distillation.py
│ ├── sweep.py # sweep logic
│ └── sweep_cli.py # CLI: KD sweep entry point
│
├── config/ # configuration files
│ ├── default.yaml # main config (complete training)
│ ├── teacher.yaml # teacher-specific config
│ └── kd_sweep.yaml # KD sweep config
│
├── scripts/ # shell scripts for reproducibility
│ ├── run_complete_training.sh
│ └── run_kd_sweep.sh
│
├── logs/ # training logs
│ ├── complete_training/
│ │ ├── teacher_training.log
│ │ ├── cache_build.log
│ │ ├── msfd_seed{42,139,236,333,430}.log
│ │ ├── kd_seed{42,139,236,333,430}.log
│ │ ├── baseline_seed{42,139,236,333,430}.log
│ │ ├── practice.log
│ │ └── progressive_distillation.log
│ │
│ └── kd_sweep/
│ ├── sweep_T1.0_a0.3.log
│ └── sweep_summary.log
│
└── results/ # CSV outputs and figures
├── complete_training/
└── kd_sweep/
```
**Note on the two code modules.** The repository contains two code trees, each corresponding to one of the two experimental pipelines:

- **`code/`** — the complete training pipeline (Pipeline 2): teacher training, feature caching, MSFD distillation, KD baseline, no-distillation baseline, deliberate practice, progressive distillation, evaluation, and figure generation.
- **`Kd sweep/`** — the KD hyperparameter sweep (Pipeline 1): a self-contained module that loads the pre-trained teacher, runs the 4×3 grid over temperature and CE weight, selects the best configuration by validation accuracy only, and exports the winning hyperparameters that Pipeline 2 then consumes.

They share the same design (same `config.py`, `logger.py`, `utils.py`, `data_loader.py`, `models.py`, `training.py`, `distillation.py` structure) but are kept separate because they are run independently, at different times, and produce different logs and results directories. The folder name is spelled `Kd sweep` (with a space and lower-case "d") exactly as it appears on disk; if you prefer to rename it to `kd_sweep` for consistency, do so in the repository, then update the tree above.
---

## Installation

Tested on Python 3.10 and 3.11 with PyTorch 2.x, on a single NVIDIA T4 GPU (16 GB). No GPU-specific compilation is required — the code uses stock `torch` and `torchvision`.

```bash
# Clone the repository
git clone https://github.com/sanoyz/msfd-replication-study.git
cd msfd-replication-study

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install pinned dependencies
pip install -r requirements.txt
```

If you are running on Kaggle, the dependencies are already present in the default image. Skip the `pip install` step and go straight to the Quick Start.

---

## Data

The only dataset used is **CIFAR-100**. It is downloaded automatically by `torchvision` the first time `code/data_loader.py` runs, to the path set in `config.py` (default: `./data/`).

There is **no separate data artifact** to download. All privileged signals used during distillation (the "audio-like" and "3D-like" latents) are synthetic and generated on the fly from the teacher's own visual features at training time. They are not stored anywhere on disk.

The dataset split is deterministic:

- Train: 45,000 images
- Validation: 5,000 images
- Test: 10,000 images
- Split seed: 42

The split is verified at load time — the code asserts that the train and validation index sets are disjoint and sum to 50,000.

---

## Quick Start: Full Reproduction

The full reproduction requires two runs, in order. On a free-tier T4 it takes roughly **22–23 hours wall-clock** in total.

```bash
# Step 1: KD hyperparameter sweep (~4.5 hours)
bash scripts/run_kd_sweep.sh

# Step 2: Complete training (~18 hours)
bash scripts/run_complete_training.sh
```

Or, to run both with the correct ordering:

```bash
bash scripts/run_all.sh
```

Both scripts accept a `--quick` flag for a fast smoke test (a few minutes, drastically reduced epochs and seeds) that exercises the pipeline end to end without producing publishable numbers:

```bash
bash scripts/run_all.sh --quick
```

---

## Reproduce Specific Results

Each stage is independently runnable. The commands below assume you are in the repository root.

**Teacher only:**

```bash
python -m code.train_teacher --config configs/teacher.yaml
```

**MSFD student, one seed:**

```bash
python -m code.train_msfd --seed 42 --epochs 12
```

**No-distillation baseline, one seed:**

```bash
python -m code.train_baseline --seed 42 --epochs 12
```

**Tuned KD, one seed, explicit hyperparameters:**

```bash
python -m code.train_standard_kd --seed 42 --epochs 12 --temperature 1.0 --alpha 0.3
```

**Head ablation study:**

```bash
python -m code.head_ablations --seed 42 --epochs 12
```

**Progressive distillation:**

```bash
python -m code.progressive --stage 1 --epochs 12
```

**Per-class analysis and confusion matrix (uses the best MSFD checkpoint):**

```bash
python -m code.evaluation --checkpoint checkpoints/msfd_seed42_best.pth
```

Every script writes its log to `logs/` and its outputs to `results/`, using the same file structure as the complete pipeline.

---

## Checkpoints and Artifacts

The checkpoints are **not** stored in this Git repository — they are too large. They are archived separately on Zenodo:

> **[Zenodo DOI for checkpoints and logs — to be added once archived]**

The Zenodo record contains:

- Teacher checkpoint (`teacher_best.pth`, ~350 MB)
- 15 student checkpoints: 5 seeds × 3 methods (MSFD, KD, no-distillation)
- 12 sweep checkpoints from the KD hyperparameter sweep
- All training logs in raw form
- All result CSVs
- All figure-generation inputs

To reproduce results from the archived checkpoints without retraining, download the Zenodo archive and place its contents under `checkpoints/` and `logs/` before running the evaluation scripts.

---

## Expected Results

Reproducing the full pipeline on a free-tier T4 with the pinned dependencies should produce:

| Metric | Expected Value |
|---|---|
| Teacher validation accuracy | 85.86% |
| MSFD test accuracy (5-seed mean) | 77.45% ± 0.25% |
| Tuned KD test accuracy (5-seed mean) | 78.00% ± 0.39% |
| No-distillation baseline test accuracy (5-seed mean) | 77.20% ± 0.16% |
| Paired t-test, tuned KD vs. baseline | p = 0.006 |
| Paired t-test, tuned KD vs. MSFD | p = 0.029 |
| Paired t-test, MSFD vs. baseline | p = 0.160 (not significant) |

**Exact reproduction is not guaranteed** because:

1. AMP mixed-precision and cuDNN non-determinism introduce small drift (typically 0.1–0.9 percentage points) between runs, even with identical seeds.
2. PyTorch and CUDA version differences can change numerical results slightly.
3. The GPU architecture (T4 vs. P100 vs. A100) affects accumulation ordering in reductions.

If your 5-seed mean for any method falls within ±1 standard deviation of the values above, the replication is behaving correctly. If it falls outside by a large margin, see **Known Issues and Caveats** and **Reproducibility Notes** below.

---

## Hardware Requirements

The pipeline was developed and tested on a single **NVIDIA Tesla T4 (16 GB VRAM)**, the standard Kaggle free-tier GPU.

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 12 GB | 16 GB |
| System RAM | 16 GB | 32 GB |
| Disk (for data + checkpoints) | 20 GB | 40 GB |
| CUDA | 11.8 | 12.x |

**A GPU is required.** The teacher is a ViT-B/16, and running it on CPU is infeasible within any realistic time budget. The `code/config.py` module raises a hard error if `torch.cuda.is_available()` is `False` and `require_gpu` is set to `True` (the default).

The full pipeline does **not** fit in a single Kaggle session (which is capped around 9–12 hours). The two pipelines are designed to be run across multiple sessions; the code supports resuming from the latest checkpoint of any stage.

---

## Training Logs

Raw training logs are stored under `logs/`, with the same structure that the original runs produced.

```
logs/
├── complete_training/
│   ├── teacher_training.log
│   ├── cache_build.log
│   ├── msfd_seed42.log
│   ├── msfd_seed139.log
│   ├── msfd_seed236.log
│   ├── kd_seed42.log
│   ├── kd_seed139.log
│   ├── kd_seed236.log
│   ├── baseline_seed42.log
│   ├── baseline_seed139.log
│   ├── baseline_seed236.log
│   ├── practice.log
│   └── progressive_distillation.log
│
└── kd_sweep/
    ├── sweep_T1.0_a0.3.log
    ├── ...
    └── sweep_summary.log
```

Each log contains the full epoch-by-epoch trace for that run: loss, train accuracy, validation accuracy, validation F1, and any stage-specific metrics. These are the logs from which every number in the paper is drawn.

The full logs are also archived on Zenodo together with the checkpoints (see **Checkpoints and Artifacts**).

---

## Reproducing Figures

All paper figures are generated programmatically from logged results. No numbers are hand-transcribed.

```bash
# Generate all figures into ./figures/
python -m code.make_figures
```

Or, if you prefer to regenerate the figures from a saved results directory:

```bash
python -m code.make_figures --results results/complete_training --out figures/
```

Figures are written as PDF for fast LaTeX compilation and small file size. If you need PNGs (for a slide deck or a report), the script accepts a `--format png` flag:

```bash
python -m code.make_figures --format png
```

The eight figures produced are:

| File | Description |
|---|---|
| `fig0_architecture.pdf` | Pipeline architecture diagram |
| `fig1_teacher.pdf` | Teacher training curve |
| `fig2_curves_seed42.pdf` | Validation curves for all three methods, seed 42 |
| `fig3_final_comparison.pdf` | Final multi-seed comparison with significance brackets |
| `fig4_kd_sweep.pdf` | KD hyperparameter sweep, validation accuracy |
| `fig5_ablation.pdf` | Head ablations at matched budget |
| `fig6_progressive.pdf` | Accuracy vs. parameter count, progressive stages |
| `fig7_per_class.pdf` | Top and bottom per-class F1 |

---

## Known Issues and Caveats

**1. AMP and cuDNN non-determinism.** Enabling `torch.backends.cudnn.deterministic = True` and using fixed seeds still does not fully guarantee bit-identical reproduction across sessions. Small accuracy drift (0.1–0.9 percentage points) is normal and expected. All headline numbers are reported with standard deviations over 5 seeds, so this drift is captured in the reported uncertainty.

**2. Two-session pipeline.** The complete training pipeline (~18 hours) does not fit within a single Kaggle free-tier session (which is capped at roughly 9–12 hours). The pipeline is designed to be resumed from the latest checkpoint. If a session is interrupted, re-running `bash scripts/run_all.sh` will pick up from where it left off.

**3. Synthetic latent signals.** The "audio-like" and "3D-like" latents used in MSFD are not derived from any real audio or 3D data. They are learned projections from the teacher's own visual backbone. Calling this method "cross-modal" would be inaccurate; it is better described as single-modality auxiliary-head distillation. This is discussed in the accompanying paper (Section 3.2 and 5.14).

**4. Stage-1 progressive accuracy.** The progressive distillation Stage-1 validation accuracy is reported at 78.14% in the paper, which is the value from the checkpoint carried forward from Step 2 (MSFD student, seed 42). Earlier drafts reported 78.60% and 78.92% at different points in training; the discrepancy is disclosed in the paper (Section 5.14.1) and not resolved further.

**5. Six of the eleven complete-training logs are placeholders.** The complete-training pipeline produced full logs for MSFD seeds 42, 139, and 236, and for the KD and baseline runs at those same three seeds. Seeds 333 and 430 were trained in a separate session whose per-epoch logs were not preserved; only their final evaluation metrics survive (in the resume log from 2026-09-09). The MSFD/Kd/baseline logs for seeds 333 and 430 are therefore marked as placeholders in the repository, containing the final-evaluation numbers only. This is documented in the log files themselves.

---

## Reproducibility Notes

To reproduce the reported results as closely as possible:

1. **Use the same PyTorch and CUDA version** shown in `requirements.txt`. Version differences can shift accuracy by a few tenths of a percentage point.

2. **Run the two pipelines in order.** The KD sweep selects the hyperparameters that complete training uses. Skipping the sweep and using the default configuration will change the KD baseline's results.

3. **Do not change the split seed.** The 45k/5k/10k train/val/test split with generator seed 42 is what all reported numbers use. A different split produces different numbers, not a bug.

4. **Expect ±1 standard deviation on single-seed runs.** With 5 seeds per method, the reported means are stable to about ±0.3 percentage points, but a single seed can vary by 0.5–1.0 points from the mean.

5. **Check the data-loader assertion.** The first thing `code/data_loader.py` does is assert that the train and validation sets are disjoint and sum to 50,000. If that assertion ever fails, stop and investigate — a leak here would invalidate the comparison.

6. **The scripts write to `logs/` and `results/` in place.** If you run the pipeline twice, back up the first run's outputs before overwriting them. The scripts do not version their output directories.

---

## Citation

If you use this code or build on this replication, please cite:

```bibtex
@article{zewdie2026msfd,
  title   = {[Re] A Replication Study of Multi-Headed Synthetic Feature Distillation: When a Properly Tuned Baseline Wins},
  author  = {Zewdie, Yonas},
  journal = {ReScience C},
  year    = {2026},
  note    = {Submitted}
}
```

And cite the original paper whose method is being replicated:

```bibtex
@article{sbrolli2026beyond,
  title   = {Beyond Raw Signals: Undecoded Generative Latents as Privileged Synthetic Data},
  author  = {Sbrolli, Cristian and Michel, Nicolas and Matteucci, Matteo and Yamasaki, Toshihiko},
  journal = {arXiv preprint arXiv:2606.08336},
  year    = {2026}
}
```

---

## License

This repository is dual-licensed:

- **Code** (everything under `code/`, `configs/`, `scripts/`): MIT License. See `LICENSE-CODE` if present, or the MIT text at the top of each source file.
- **Paper, README, and documentation** (this file, `article.pdf`, `metadata.yaml`, and any prose under `results/`): Creative Commons Attribution 4.0 International (CC BY 4.0).

You are free to use, modify, and redistribute both under their respective terms. Attribution is required for the documentation portion.

---

## Contact

Questions, bug reports, and replication attempts are welcome.

- **Author:** Yonas Zewdie
- **Affiliation:** Gage University College, Ethiopia
- **Email:** sanoyz2008@gmail.com, yonas.zewdie@guc.edu.et
- **ORCID:** [0009-0003-6037-9876](https://orcid.org/0009-0003-6037-9876)
- **Repository issues:** [github.com/sanoyz/msfd-replication-study/issues](https://github.com/sanoyz/msfd-replication-study/issues)

If you are reporting a bug or a failure to reproduce, please include:

1. The stage of the pipeline you were running (sweep, complete training, evaluation, figure generation)
2. Your exact command and the config file used
3. The relevant section of the log
4. Your GPU model, CUDA version, and PyTorch version