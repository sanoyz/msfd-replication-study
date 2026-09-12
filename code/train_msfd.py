"""
Entry point: train the MSFD student for a single seed.

Usage:
    python code/train_msfd.py --seed 42 --config configs/default.yaml
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from code.config import Config
from code.logger import logger
from code.data_loader import build_dataloaders
from code.models import MultimodalTeacher, LightweightStudent
from code.distillation import train_student_messy, TeacherFeatureCache
from code.utils import set_seed, safe_load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()

    config = Config()
    set_seed(args.seed)

    train_loader, val_loader, _, cache_loader, _ = build_dataloaders(config)

    # Load teacher
    teacher = MultimodalTeacher(
        num_classes=config.num_classes,
        visual_dim=config.teacher_visual_dim,
        audio_dim=config.teacher_audio_dim,
        three_d_dim=config.teacher_3d_dim,
        latent_dim=config.latent_dim
    ).to(config.device)
    teacher_path = f"{config.checkpoint_dir}/teacher_best.pth"
    checkpoint = safe_load_checkpoint(teacher_path, config.device)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher.eval()

    # Build cache
    teacher_cache = TeacherFeatureCache()
    if config.use_teacher_feature_cache and cache_loader is not None:
        teacher_cache.build(teacher, cache_loader, config, with_logits=True)

    # Train MSFD student
    student = LightweightStudent(
        num_classes=config.num_classes,
        student_dim=config.student_dim,
        teacher_dim=config.latent_dim,
        active_heads=['visual', 'audio', '3d']
    ).to(config.device)

    student = train_student_messy(
        student, teacher, train_loader, val_loader, config,
        epochs=config.student_epochs, use_mixup=True,
        run_name=f"messy_seed{args.seed}",
        teacher_cache=teacher_cache
    )

    logger.log(f"\nMSFD training complete for seed {args.seed}.")


if __name__ == "__main__":
    main()