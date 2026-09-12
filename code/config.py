"""
Configuration dataclass for the MSFD replication pipeline.
"""
import os
from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class Config:
    """Memory-optimized configuration for Kaggle T4 (16GB VRAM)."""

    # ==================== MODE ====================
    QUICK_TEST: bool = False
    RUN_ALL_EXPERIMENTS: bool = True

    # Per-experiment toggles
    run_deliberate_practice: bool = True
    run_standard_kd: bool = True
    run_messy_kd: bool = True
    run_head_ablations: bool = True
    run_ensemble: bool = True
    run_progressive_distillation: bool = True
    run_per_class_analysis: bool = True
    run_baseline: bool = True

    # ==================== DATA ====================
    dataset: str = 'cifar100'
    num_classes: int = 100
    img_size: int = 224
    batch_size: int = 32
    accumulation_steps: int = 4
    num_workers: int = 2
    strong_augmentation: bool = False

    # ==================== EPOCHS ====================
    teacher_epochs: int = 10
    student_epochs: int = 12
    practice_rounds: int = 1
    ablation_epochs: int = 4
    ensemble_models: int = 2
    ensemble_epochs: int = 8
    progressive_stage2_epochs: int = 8
    progressive_stage3_epochs: int = 6
    progressive_stages: int = 3

    # ==================== OPTIMIZATION ====================
    teacher_lr: float = 1e-4
    student_lr: float = 3e-4
    lr: float = 1e-4
    practice_lr: float = 1e-5
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    use_amp: bool = True
    warmup_epochs: int = 2
    early_stopping_patience: int = 5
    early_stopping_delta: float = 0.001

    # ==================== MIXUP ====================
    mixup_alpha: float = 0.2

    # ==================== MODEL DIMENSIONS ====================
    teacher_visual_dim: int = 768
    teacher_audio_dim: int = 512
    teacher_3d_dim: int = 1024
    latent_dim: int = 512
    student_dim: int = 256

    # ==================== DISTILLATION ====================
    distill_weight: float = 0.7
    ce_weight: float = 0.3
    temperature: float = 4.0

    # ==================== DELIBERATE PRACTICE ====================
    hard_example_threshold: float = 0.6
    augmentations_per_sample: int = 2

    # ==================== PATHS ====================
    output_dir: str = '/kaggle/working/'
    checkpoint_dir: str = '/kaggle/working/checkpoints'
    data_dir: str = '/kaggle/working/data'

    # ==================== DEVICE ====================
    device: torch.device = field(
        default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    seed: int = 42

    # ==================== PERFORMANCE / KAGGLE-FIT ====================
    require_gpu: bool = True
    freeze_teacher_backbone_except_last_n_blocks: int = 4
    use_teacher_feature_cache: bool = True
    cache_batch_size: int = 64
    resume_from_latest: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    # ==================== STATISTICAL RIGOR ====================
    num_seeds: int = 3
    seed_stride: int = 97
    confidence_level: float = 0.95

    # ==================== ABLATIONS ====================
    match_ablation_budget: bool = True
    run_multiseed_ablations: bool = False

    def __post_init__(self):
        """Create output directories on instantiation."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)