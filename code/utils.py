"""
Shared utilities: seeding, memory cleanup, checkpoint loading,
statistical aggregation, and significance testing.
"""
import os
import gc
import math
import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from code.config import Config
from code.logger import logger

try:
    from scipy import stats as scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def safe_load_checkpoint(path: str, map_location):
    """Robust checkpoint loading across PyTorch versions/sessions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class IndexedDataset(Dataset):
    """Wraps a dataset to also yield each sample's position (0..N-1)."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        return x, y, i


def make_loader(dataset: Dataset, config: Config, shuffle: bool,
                indexed: bool = False, drop_last: bool = False,
                batch_size: Optional[int] = None) -> DataLoader:
    """Central place for DataLoader kwargs."""
    ds = IndexedDataset(dataset) if indexed else dataset
    kwargs = dict(
        batch_size=batch_size if batch_size is not None else config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    if config.num_workers > 0:
        kwargs['persistent_workers'] = config.persistent_workers
        kwargs['prefetch_factor'] = config.prefetch_factor
    return DataLoader(ds, **kwargs)


def save_resume_checkpoint(path: str, model, optimizer, scaler,
                           epoch: int, best_acc: float) -> None:
    """Save 'latest' state every epoch so a killed session can resume."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_acc': best_acc,
    }, path)


def try_load_resume_checkpoint(path: str, model, optimizer, scaler, config: Config):
    """Returns (start_epoch, best_acc) -- (0, 0.0) if nothing to resume from."""
    if not config.resume_from_latest or not os.path.exists(path):
        return 0, 0.0
    ckpt = safe_load_checkpoint(path, config.device)
    model.load_state_dict(ckpt['model_state_dict'])
    try:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    except Exception as e:
        logger.log(f"[RESUME] Could not restore optimizer/scaler state ({e}); "
                   f"resuming with fresh optimizer state.")
    start_epoch = ckpt['epoch'] + 1
    best_acc = ckpt.get('best_acc', 0.0)
    logger.log(f"[RESUME] Resuming from epoch {start_epoch} (best_acc so far: {best_acc:.4f})")
    return start_epoch, best_acc


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 5, delta: float = 0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_acc: float) -> bool:
        if self.best_score is None:
            self.best_score = val_acc
            return False
        if val_acc > self.best_score + self.delta:
            self.best_score = val_acc
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def get_seed_list(config: Config) -> List[int]:
    """Seeds for multi-seed runs."""
    return [config.seed + i * config.seed_stride for i in range(config.num_seeds)]


def mean_std_ci(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """Mean, sample std (ddof=1), and a t-distribution confidence interval."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return {'mean': mean, 'std': 0.0, 'ci_low': mean, 'ci_high': mean, 'n': n}
    std = float(arr.std(ddof=1))
    sem = std / math.sqrt(n)
    if _SCIPY_AVAILABLE:
        t_crit = float(scipy_stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    else:
        _t95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}
        t_crit = _t95.get(n, 1.96)
    margin = t_crit * sem
    return {'mean': mean, 'std': std, 'ci_low': mean - margin,
            'ci_high': mean + margin, 'n': n}


def paired_significance_test(name_a: str, vals_a: List[float],
                              name_b: str, vals_b: List[float]) -> Dict[str, Any]:
    """Paired t-test + Wilcoxon signed-rank between two same-seed-matched sets."""
    result = {'comparison': f'{name_a} vs {name_b}', 'n': len(vals_a)}
    if len(vals_a) != len(vals_b) or len(vals_a) < 2:
        result['note'] = 'Not enough matched seeds for a significance test.'
        return result
    if not _SCIPY_AVAILABLE:
        result['note'] = 'scipy not available -- install scipy for significance tests.'
        return result
    t_stat, t_p = scipy_stats.ttest_rel(vals_a, vals_b)
    try:
        w_stat, w_p = scipy_stats.wilcoxon(vals_a, vals_b)
    except ValueError:
        w_stat, w_p = float('nan'), float('nan')
    result.update({
        'mean_diff': float(np.mean(np.array(vals_a) - np.array(vals_b))),
        'paired_t_stat': float(t_stat), 'paired_t_p': float(t_p),
        'wilcoxon_stat': float(w_stat), 'wilcoxon_p': float(w_p),
    })
    return result


def log_seeded_summary(name: str, seed_results: List[Dict[str, float]],
                       metric: str = 'test_acc') -> Dict[str, float]:
    """Logs a per-seed table plus mean/std/CI for one configuration."""
    vals = [r[metric] for r in seed_results]
    from code.logger import logger
    stats_out = mean_std_ci(vals, confidence=0.95)
    logger.log(f"\n[STATS] {name} -- {metric} across {len(vals)} seed(s):")
    for r in seed_results:
        logger.log(f"    seed {r['seed']}: {r[metric]:.4f}")
    logger.log(f"    mean={stats_out['mean']:.4f}  std={stats_out['std']:.4f}  "
               f"95% CI=[{stats_out['ci_low']:.4f}, {stats_out['ci_high']:.4f}]  "
               f"(n={stats_out['n']})")
    return stats_out