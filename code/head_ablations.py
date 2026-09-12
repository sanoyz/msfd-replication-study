"""
Head ablation study: train student with different subsets of prediction heads.
"""
from typing import Dict, Optional
import torch
import torch.nn as nn

from code.config import Config
from code.logger import logger
from code.models import LightweightStudent
from code.distillation import train_student_messy
from code.training import evaluate
from code.utils import get_seed_list, log_seeded_summary, set_seed, cleanup_memory


def run_head_ablations(teacher: nn.Module, train_loader, val_loader,
                        test_loader, config: Config,
                        teacher_cache=None) -> Dict[str, Dict]:
    """Run head ablation experiments."""
    logger.log("\n" + "=" * 80)
    logger.log("HEAD ABLATIONS")
    logger.log("=" * 80)

    epochs = (config.student_epochs if config.match_ablation_budget
              else config.ablation_epochs)
    logger.log(f"[ABLATION BUDGET] Using {epochs} epochs per configuration.")

    head_configs = {
        'visual_only': ['visual'],
        'audio_only': ['audio'],
        '3d_only': ['3d'],
        'visual_audio': ['visual', 'audio'],
        'visual_3d': ['visual', '3d'],
        'full_messy': ['visual', 'audio', '3d']
    }

    results = {}
    seeds = get_seed_list(config) if config.run_multiseed_ablations else [config.seed]

    for name, heads in head_configs.items():
        logger.log(f"\n{'='*60}\nAblation: {name}\n{'='*60}")
        seed_results = []
        for seed in seeds:
            set_seed(seed)
            student = LightweightStudent(
                num_classes=config.num_classes,
                student_dim=config.student_dim,
                teacher_dim=config.latent_dim,
                active_heads=heads
            ).to(config.device)
            student = train_student_messy(
                student, teacher, train_loader, val_loader, config,
                epochs=epochs, use_mixup=True,
                run_name=f"ablation_{name}_seed{seed}",
                teacher_cache=teacher_cache
            )
            _, test_acc, test_f1, _, _ = evaluate(
                student, test_loader, nn.CrossEntropyLoss(), config
            )
            params = sum(p.numel() for p in student.parameters() if p.requires_grad)
            seed_results.append({'seed': seed, 'test_acc': test_acc,
                                  'test_f1': test_f1})
            del student
            cleanup_memory()

        set_seed(config.seed)
        stats_out = log_seeded_summary(f"ablation_{name}", seed_results,
                                        metric='test_acc')
        mean_f1 = float(sum(r['test_f1'] for r in seed_results) / len(seed_results))
        results[name] = {
            'accuracy': stats_out['mean'], 'std': stats_out['std'],
            'ci_low': stats_out['ci_low'], 'ci_high': stats_out['ci_high'],
            'n_seeds': stats_out['n'], 'f1': mean_f1, 'params': params,
            'per_seed': seed_results,
        }
        logger.log_ablation(name, stats_out['mean'], mean_f1, params)

    return results