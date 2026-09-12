"""
Evaluation utilities: per-class analysis, confusion matrices, and ensemble.
"""
import os
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from code.config import Config
from code.logger import logger


def analyze_per_class(model: nn.Module, test_loader: DataLoader,
                       class_names: List[str], config: Config
                       ) -> Dict[str, Any]:
    """Per-class performance analysis."""
    logger.log("\n" + "=" * 80)
    logger.log("PER-CLASS ANALYSIS")
    logger.log("=" * 80)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Analyzing"):
            images = images.to(config.device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(
        all_labels, all_preds, target_names=class_names,
        output_dict=True, zero_division=0
    )

    class_acc = {}
    for name in class_names:
        if name in report:
            class_acc[name] = report[name]['f1-score']

    best_classes = sorted(class_acc.items(), key=lambda x: x[1], reverse=True)[:10]
    worst_classes = sorted(class_acc.items(), key=lambda x: x[1])[:10]

    logger.log("\nTop 10 Best Performing Classes:")
    for name, score in best_classes:
        logger.log(f"  [+] {name}: {score:.3f}")
    logger.log("\nTop 10 Worst Performing Classes:")
    for name, score in worst_classes:
        logger.log(f"  [-] {name}: {score:.3f}")

    # Full per-class CSV
    rows = []
    for name in class_names:
        if name in report:
            r = report[name]
            rows.append({'class': name, 'precision': r['precision'],
                          'recall': r['recall'], 'f1_score': r['f1-score'],
                          'support': r['support']})
    per_class_df = pd.DataFrame(rows).sort_values('f1_score').reset_index(drop=True)
    per_class_df.to_csv(logger.per_class_file, index=False)
    logger.log(f"[OK] Full per-class report saved to {logger.per_class_file}")

    return {
        'best': best_classes, 'worst': worst_classes,
        'report': report, 'class_acc': class_acc,
        'per_class_csv': logger.per_class_file,
        'all_labels': all_labels, 'all_preds': all_preds,
    }


def plot_confusion_matrix(model: nn.Module, test_loader: DataLoader,
                           class_names: List[str], config: Config,
                           worst_class_names: Optional[List[str]] = None,
                           all_labels: Optional[List[int]] = None,
                           all_preds: Optional[List[int]] = None) -> None:
    """Plot and save full + worst-class confusion matrices."""
    logger.log("\n" + "=" * 80)
    logger.log("CONFUSION MATRIX")
    logger.log("=" * 80)

    if all_labels is None or all_preds is None:
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Generating confusion matrix"):
                images = images.to(config.device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    plt.figure(figsize=(22, 20))
    sns.heatmap(cm, cmap='Blues', annot=False,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - MSFD Student (all 100 classes)')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(rotation=90, fontsize=5); plt.yticks(rotation=0, fontsize=5)
    plt.tight_layout()
    full_path = f'{config.output_dir}/confusion_matrix_full.png'
    plt.savefig(full_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.log(f"[OK] Full confusion matrix saved to {full_path}")

    if worst_class_names:
        name_to_idx = {name: i for i, name in enumerate(class_names)}
        worst_idx = [name_to_idx[n] for n in worst_class_names if n in name_to_idx]
        if worst_idx:
            sub_cm = cm[np.ix_(worst_idx, worst_idx)]
            sub_names = [class_names[i] for i in worst_idx]
            plt.figure(figsize=(max(8, len(worst_idx) * 0.9),
                                max(6, len(worst_idx) * 0.8)))
            sns.heatmap(sub_cm, annot=True, fmt='d', cmap='Reds',
                        xticklabels=sub_names, yticklabels=sub_names,
                        cbar_kws={'label': 'Count'})
            plt.title(f'Confusion Matrix - Worst {len(worst_idx)} Classes')
            plt.xlabel('Predicted'); plt.ylabel('True')
            plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
            plt.tight_layout()
            worst_path = f'{config.output_dir}/confusion_matrix_worst.png'
            plt.savefig(worst_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.log(f"[OK] Worst-classes confusion matrix saved to {worst_path}")


class EnsembleStudent:
    """Ensemble of MSFD students with different seeds."""

    def __init__(self, num_models: int = 3,
                 active_heads: Optional[List[str]] = None,
                 config: Config = None):
        self.num_models = num_models
        self.active_heads = active_heads or ['visual', 'audio', '3d']
        self.config = config
        self.models = []
        self.seeds = [42 + i * 100 for i in range(num_models)]

    def train_all(self, teacher: nn.Module, train_loader, val_loader,
                   config: Config, epochs: int = 8, teacher_cache=None) -> None:
        from code.distillation import train_student_messy
        from code.utils import set_seed, cleanup_memory

        for i, seed in enumerate(self.seeds):
            logger.log(f"\n{'='*60}")
            logger.log(f"Ensemble Model {i+1}/{self.num_models} (Seed {seed})")
            logger.log(f"{'='*60}")
            set_seed(seed)
            model = LightweightStudent(
                num_classes=config.num_classes,
                student_dim=config.student_dim,
                teacher_dim=config.latent_dim,
                active_heads=self.active_heads
            ).to(config.device)
            model = train_student_messy(
                model, teacher, train_loader, val_loader, config,
                epochs=epochs, use_mixup=True,
                run_name=f"ensemble_{i}", teacher_cache=teacher_cache
            )
            self.models.append(model)
            cleanup_memory()
        set_seed(config.seed)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                predictions.append(F.softmax(model(x), dim=1))
        return torch.stack(predictions).mean(dim=0)

    def evaluate(self, test_loader) -> float:
        all_preds, all_labels = [], []
        for images, labels in tqdm(test_loader, desc="Ensemble evaluating"):
            images = images.to(self.config.device)
            preds = torch.argmax(self.predict(images), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
        return accuracy_score(all_labels, all_preds)