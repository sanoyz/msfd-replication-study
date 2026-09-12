# MSFD Replication: Multi-Headed Synthetic Feature Distillation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains the complete code, logs, and artifacts for the paper:

> **A Replication Study of Multi-Headed Synthetic Feature Distillation: When a Properly Tuned Baseline Wins**
> [Author Name(s)], [Affiliation(s)]
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

This is a **self-replication** of an earlier claim made during this project:

> **Original claim (retracted):** *"MSFD outperforms standard knowledge distillation by 1.96 percentage points on CIFAR-100."*

**Original protocol:**
- Single random seed
- 4-epoch ablation training budget (vs. 12 for the main model)
- KD baseline with arbitrary, untuned hyperparameters (T=4.0, α=0.3)
- No no-distillation baseline measured

**This replication re-runs the comparison under matched conditions:**
- Five random seeds per method (42, 139, 236, 333, 430)
- 12-epoch training budget for all methods
- KD hyperparameters selected by validation-only grid search
- No-distillation baseline included

**Outcome:** The original claim does **not** hold. A properly tuned KD baseline significantly outperforms MSFD, and MSFD does not significantly outperform the no-distillation baseline.

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