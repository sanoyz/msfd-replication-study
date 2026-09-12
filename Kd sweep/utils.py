"""
Seeding, cleanup, checkpoint loading, and statistical helpers.
"""
import gc
import math
import random
from typing import List

import numpy as np
import torch

try:
    from scipy import stats as scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def safe_load_checkpoint(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_seed_list(config, num_seeds=None):
    if num_seeds is None:
        num_seeds = config.num_seeds
    return [config.seed + i * 97 for i in range(num_seeds)]


def mean_std_ci(values, confidence=0.95):
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
    return {'mean': mean, 'std': std, 'ci_low': mean - margin, 'ci_high': mean + margin, 'n': n}


def paired_significance_test(vals_a, vals_b):
    result = {'n': len(vals_a)}
    if len(vals_a) != len(vals_b) or len(vals_a) < 2:
        result['note'] = 'Not enough matched seeds.'
        return result
    if not _SCIPY_AVAILABLE:
        result['note'] = 'scipy not available.'
        return result
    t_stat, t_p = scipy_stats.ttest_rel(vals_a, vals_b)
    try:
        w_stat, w_p = scipy_stats.wilcoxon(vals_a, vals_b)
    except ValueError:
        w_stat, w_p = float('nan'), float('nan')
    if len(vals_a) < 5:
        result['note'] = f'Wilcoxon uninformative at n={len(vals_a)}; use paired t-test (p={t_p:.4f})'
    result.update({
        'mean_diff': float(np.mean(np.array(vals_a) - np.array(vals_b))),
        'paired_t_stat': float(t_stat),
        'paired_t_p': float(t_p),
        'wilcoxon_stat': float(w_stat),
        'wilcoxon_p': float(w_p),
    })
    return result