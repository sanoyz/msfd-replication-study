"""
Entry point: load saved teacher, build loaders, run the KD sweep.
This is the file you run to reproduce the sweep end-to-end.
"""
import json
import os
import sys

import torch
import torch.nn as nn

from .config import config, CHECKPOINT_DIR, INPUT_DIR, OUTPUT_DIR
from .logger import SimpleLogger
from .utils import safe_load_checkpoint
from .models import MultimodalTeacher
from .data_loader import build_loaders
from .sweep import run_kd_sweep_and_validate


def find_teacher_checkpoint():
    """Locate a teacher checkpoint either in the input dir or the output dir."""
    if CHECKPOINT_DIR and os.path.exists(CHECKPOINT_DIR):
        for f in os.listdir(CHECKPOINT_DIR):
            if 'teacher' in f and f.endswith('.pth'):
                return os.path.join(CHECKPOINT_DIR, f)
    if INPUT_DIR and os.path.exists(INPUT_DIR):
        for root, dirs, files in os.walk(INPUT_DIR):
            for f in files:
                if 'teacher' in f and f.endswith('.pth'):
                    return os.path.join(root, f)
    return None


def main():
    logger = SimpleLogger(OUTPUT_DIR)

    logger.log("\n" + "=" * 80)
    logger.log("LOADING SAVED ARTIFACTS")
    logger.log("=" * 80)

    teacher_checkpoint_path = find_teacher_checkpoint()
    if teacher_checkpoint_path:
        logger.log(f"✓ Teacher checkpoint found: {teacher_checkpoint_path}")
    else:
        logger.log("⚠️ Teacher checkpoint not found! Will train from scratch.")

    teacher = MultimodalTeacher(
        num_classes=config.num_classes,
        visual_dim=config.teacher_visual_dim,
        audio_dim=config.teacher_audio_dim,
        three_d_dim=config.teacher_3d_dim,
        latent_dim=config.latent_dim
    ).to(config.device)

    teacher_loaded = False
    if teacher_checkpoint_path and os.path.exists(teacher_checkpoint_path):
        try:
            checkpoint = safe_load_checkpoint(teacher_checkpoint_path, config.device)
            if 'model_state_dict' in checkpoint:
                teacher.load_state_dict(checkpoint['model_state_dict'])
            else:
                teacher.load_state_dict(checkpoint)
            teacher.eval()
            acc = checkpoint.get('best_acc', 0.0)
            logger.log(f"✓ Teacher loaded successfully (best_acc: {acc:.4f})")
            teacher_loaded = True
        except Exception as e:
            logger.log(f"⚠️ Error loading teacher: {e}")

    if not teacher_loaded:
        logger.log("⚠️ Teacher not loaded. Please ensure the checkpoint is available.")

    logger.log("\n" + "=" * 80)
    logger.log("LOADING CIFAR-100")
    logger.log("=" * 80)
    train_loader, val_loader, test_loader, _classes = build_loaders(INPUT_DIR)

    logger.log("\n" + "=" * 80)
    logger.log("STARTING KD SWEEP PIPELINE")
    logger.log("=" * 80)
    logger.log(f"Device: {config.device}")
    if torch.cuda.is_available():
        logger.log(f"GPU: {torch.cuda.get_device_properties(0).name}")
        logger.log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.log(f"Number of seeds: {config.num_seeds}")
    logger.log(f"Sweep temperatures: {config.sweep_temperatures}")
    logger.log(f"Sweep alphas: {config.sweep_alphas}")
    logger.log("=" * 80)

    try:
        results = run_kd_sweep_and_validate(
            teacher, teacher_loaded, train_loader, val_loader, test_loader, logger
        )
        with open(f'{OUTPUT_DIR}/kd_sweep_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.log(f"\n✓ Results saved to {OUTPUT_DIR}/kd_sweep_results.json")
        logger.save_results(results)
    except Exception as e:
        logger.log(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    logger.log("\n" + "=" * 80)
    logger.log("✅ KD SWEEP COMPLETE")
    logger.log("=" * 80)


if __name__ == '__main__':
    main()