#!/usr/bin/env bash
#
# KD hyperparameter sweep (Pipeline 2):
#   grid search over T in {1,2,4,8} x alpha in {0.3,0.5,0.7}
#
# Expected runtime: ~4.5 hours on a free-tier T4 GPU.
#
set -euo pipefail

python -c "
import sys; sys.path.insert(0, '.')
import torch
from code.config import Config
from code.logger import logger
from code.data_loader import build_dataloaders
from code.models import MultimodalTeacher
from code.distillation import TeacherFeatureCache
from code.sweep import run_kd_sweep
from code.utils import set_seed, safe_load_checkpoint

config = Config()
set_seed(config.seed)
train_loader, val_loader, _, cache_loader, _ = build_dataloaders(config)

teacher = MultimodalTeacher(
    num_classes=config.num_classes,
    visual_dim=config.teacher_visual_dim,
    audio_dim=config.teacher_audio_dim,
    three_d_dim=config.teacher_3d_dim,
    latent_dim=config.latent_dim
).to(config.device)
checkpoint = safe_load_checkpoint(f'{config.checkpoint_dir}/teacher_best.pth', config.device)
teacher.load_state_dict(checkpoint['model_state_dict'])
teacher.eval()

teacher_cache = TeacherFeatureCache()
if config.use_teacher_feature_cache and cache_loader is not None:
    teacher_cache.build(teacher, cache_loader, config, with_logits=True)

results = run_kd_sweep(teacher, train_loader, val_loader, config, teacher_cache)
import json
with open(f'{config.output_dir}/kd_sweep_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('Best config:', results['best_config'])
"