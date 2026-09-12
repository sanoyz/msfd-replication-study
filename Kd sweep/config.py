"""
Configuration dataclass for the MSFD (MESSy) KD sweep pipeline.
"""
import os
from dataclasses import dataclass, field
from typing import List

import torch


def find_input_dir():
    """Find the MSFD dataset/notebook in Kaggle input."""
    base_dir = '/kaggle/input'

    if not os.path.exists(base_dir):
        print("⚠️ /kaggle/input not found")
        return None

    dirs = os.listdir(base_dir)
    print(f"Available directories in /kaggle/input/: {dirs}")

    for d in dirs:
        if 'notebook' in d.lower():
            return os.path.join(base_dir, d)
    for d in dirs:
        if 'messy' in d.lower():
            return os.path.join(base_dir, d)

    return None


def find_checkpoint_dir(input_dir):
    """Find checkpoints directory in the input."""
    if input_dir is None:
        return None

    for root, dirs, files in os.walk(input_dir):
        if 'checkpoints' in dirs:
            return os.path.join(root, 'checkpoints')
        for f in files:
            if f.endswith('.pth'):
                return root

    if input_dir and os.path.exists(input_dir):
        for f in os.listdir(input_dir):
            if f.endswith('.pth'):
                return input_dir

    return None


INPUT_DIR = find_input_dir()
OUTPUT_DIR = '/kaggle/working/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_DIR = find_checkpoint_dir(INPUT_DIR)


@dataclass
class Config:
    """Configuration for KD sweep using saved artifacts."""

    # ==================== PATHS ====================
    input_dir: str = INPUT_DIR
    output_dir: str = OUTPUT_DIR
    checkpoint_dir: str = os.path.join(OUTPUT_DIR, 'checkpoints')
    data_dir: str = os.path.join(OUTPUT_DIR, 'data')

    # ==================== DATA ====================
    dataset: str = 'cifar100'
    num_classes: int = 100
    img_size: int = 224
    batch_size: int = 32
    accumulation_steps: int = 4
    num_workers: int = 2

    # ==================== MODEL ====================
    student_dim: int = 256
    latent_dim: int = 512
    teacher_visual_dim: int = 768
    teacher_audio_dim: int = 512
    teacher_3d_dim: int = 1024

    # ==================== OPTIMIZATION ====================
    student_lr: float = 3e-4
    teacher_lr: float = 1e-4
    practice_lr: float = 1e-5
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    use_amp: bool = True
    warmup_epochs: int = 2

    # ==================== DISTILLATION ====================
    ce_weight: float = 0.3
    distill_weight: float = 0.7
    temperature: float = 4.0
    kd_alpha: float = 0.3
    mixup_alpha: float = 0.2

    # ==================== STATISTICAL ====================
    num_seeds: int = 5
    seed: int = 42
    confidence_level: float = 0.95

    # ==================== SWEEP ====================
    sweep_temperatures: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0])
    sweep_alphas: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    sweep_epochs: int = 6
    student_epochs: int = 12

    # ==================== EXPERIMENT FLAGS ====================
    run_baseline: bool = True
    run_kd_sweep: bool = True
    run_best_kd: bool = True
    run_messy: bool = True
    run_significance_tests: bool = True

    # ==================== DEVICE ====================
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = Config()

os.makedirs(config.checkpoint_dir, exist_ok=True)
os.makedirs(config.data_dir, exist_ok=True)