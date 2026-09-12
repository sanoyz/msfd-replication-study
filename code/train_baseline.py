"""
Entry point: train the no-distillation baseline for a single seed.

Usage:
    python code/train_baseline.py --seed 42 --config configs/default.yaml
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from code.config import Config
from code.logger import logger
from code.data_loader import build_dataloaders
from code.models import LightweightStudent
from code.training import train_baseline
from code.utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()

    config = Config()
    set_seed(args.seed)

    train_loader, val_loader, _, _, _ = build_dataloaders(config)

    student = LightweightStudent(
        num_classes=config.num_classes,
        student_dim=config.student_dim,
        teacher_dim=config.latent_dim,
        active_heads=[]
    ).to(config.device)

    student = train_baseline(
        student, train_loader, val_loader, config,
        epochs=config.student_epochs,
        run_name=f"baseline_seed{args.seed}"
    )

    logger.log(f"\nBaseline training complete for seed {args.seed}.")


if __name__ == "__main__":
    main()