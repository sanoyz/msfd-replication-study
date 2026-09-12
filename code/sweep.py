"""
Entry point: train the teacher model.

Usage:
    python code/train_teacher.py --config configs/teacher.yaml
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from code.config import Config
from code.logger import logger
from code.data_loader import build_dataloaders
from code.models import MultimodalTeacher
from code.training import train_teacher
from code.utils import set_seed, safe_load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/teacher.yaml')
    args = parser.parse_args()

    config = Config()
    set_seed(config.seed)

    train_loader, val_loader, _, _, _ = build_dataloaders(config)

    teacher = MultimodalTeacher(
        num_classes=config.num_classes,
        visual_dim=config.teacher_visual_dim,
        audio_dim=config.teacher_audio_dim,
        three_d_dim=config.teacher_3d_dim,
        latent_dim=config.latent_dim
    ).to(config.device)

    teacher_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    logger.log(f"Teacher parameters: {teacher_params:,}")

    best_path = f"{config.checkpoint_dir}/teacher_best.pth"
    if os.path.exists(best_path) and not config.QUICK_TEST:
        logger.log("Loading existing teacher checkpoint...")
        checkpoint = safe_load_checkpoint(best_path, config.device)
        teacher.load_state_dict(checkpoint['model_state_dict'])
        logger.log(f"Loaded teacher with accuracy: {checkpoint['best_acc']:.4f}")
    else:
        teacher = train_teacher(teacher, train_loader, val_loader,
                                 config, config.teacher_epochs)

    logger.log("\nTeacher training complete.")


if __name__ == "__main__":
    main()