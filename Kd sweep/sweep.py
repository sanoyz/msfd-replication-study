"""
Main KD sweep + multi-seed validation pipeline.
"""
import numpy as np
import torch
import torch.nn as nn

from .config import config
from .logger import SimpleLogger
from .utils import get_seed_list, mean_std_ci, paired_significance_test, set_seed, cleanup_memory, safe_load_checkpoint
from .models import LightweightStudent, MultimodalTeacher
from .data_loader import evaluate
from .distillation import train_standard_kd_with_overrides
from .training import train_baseline_with_seed, train_messy_with_seed


def run_kd_sweep_and_validate(teacher, teacher_loaded, train_loader, val_loader, test_loader, logger: SimpleLogger):
    """Run complete KD sweep and validation using saved artifacts."""

    if not teacher_loaded:
        logger.log("⚠️ Teacher not loaded. Cannot proceed with KD sweep.")
        logger.log("Please ensure teacher checkpoint is available in the input.")
        return {}

    results = {}
    seed_list = get_seed_list(config, num_seeds=config.num_seeds)
    logger.log(f"\n[SEEDS] Using {len(seed_list)} seeds: {seed_list}")

    # ========== Step 1: KD Sweep ==========
    if config.run_kd_sweep:
        logger.log("\n" + "=" * 80)
        logger.log("STEP 1: KD HYPERPARAMETER SWEEP")
        logger.log("=" * 80)
        logger.log("  Model selection: VALIDATION accuracy only")
        logger.log("  Sweep seeds: [42] (coarse)")
        logger.log("=" * 80)

        best_val_acc = 0.0
        best_config = {'temperature': 4.0, 'alpha': 0.3}
        sweep_results = []

        total_configs = len(config.sweep_temperatures) * len(config.sweep_alphas)
        config_count = 0

        for temp in config.sweep_temperatures:
            for alpha in config.sweep_alphas:
                config_count += 1
                logger.log(f"\n[{config_count}/{total_configs}] Testing KD: T={temp}, α={alpha}")

                seed_results = []
                sweep_seed = 42
                set_seed(sweep_seed)

                student = LightweightStudent(
                    num_classes=config.num_classes,
                    student_dim=config.student_dim,
                    teacher_dim=config.latent_dim,
                    active_heads=[]
                ).to(config.device)

                run_name = f"kd_sweep_T{temp}_a{alpha}"
                student = train_standard_kd_with_overrides(
                    student, teacher, train_loader, val_loader, config,
                    epochs=config.sweep_epochs, run_name=run_name, logger=logger,
                    temperature_override=temp, alpha_override=alpha
                )

                val_loss, val_acc, val_f1, _, _ = evaluate(student, val_loader, nn.CrossEntropyLoss(), config)
                seed_results.append(val_acc)
                logger.log(f"    Val Acc={val_acc:.4f}")

                mean_val_acc = np.mean(seed_results)
                logger.log(f"  Result: T={temp}, α={alpha} → {mean_val_acc*100:.2f}%")

                sweep_results.append({
                    'temperature': temp,
                    'alpha': alpha,
                    'mean_val_accuracy': mean_val_acc
                })

                if mean_val_acc > best_val_acc:
                    best_val_acc = mean_val_acc
                    best_config = {'temperature': temp, 'alpha': alpha}
                    logger.log(f"  ✓ New best (validation): {best_val_acc*100:.2f}%")

                cleanup_memory()

        logger.log(f"\n{'='*80}")
        logger.log("SWEEP COMPLETE")
        logger.log(f"Best config by validation: T={best_config['temperature']}, α={best_config['alpha']}")
        logger.log(f"  Best validation accuracy: {best_val_acc*100:.2f}%")
        logger.log(f"{'='*80}")

        results['sweep_results'] = sweep_results
        results['best_kd_config'] = best_config
        results['best_kd_val_accuracy'] = best_val_acc
    else:
        best_config = {'temperature': 2.0, 'alpha': 0.5}
        logger.log("\n[SKIP] KD Sweep disabled, using default config: T=2.0, α=0.5")
        results['best_kd_config'] = best_config

    # ========== Step 2: Baseline ==========
    if config.run_baseline:
        logger.log("\n" + "=" * 80)
        logger.log("STEP 2: BASELINE (NO DISTILLATION)")
        logger.log("=" * 80)

        baseline_results = []
        for seed in seed_list:
            set_seed(seed)
            logger.log(f"\nSeed {seed}:")

            student = LightweightStudent(
                num_classes=config.num_classes,
                student_dim=config.student_dim,
                teacher_dim=config.latent_dim,
                active_heads=[]
            ).to(config.device)

            run_name = f"baseline_seed{seed}"
            student = train_baseline_with_seed(
                student, train_loader, val_loader, config,
                epochs=config.student_epochs, run_name=run_name, logger=logger
            )

            test_loss, test_acc, test_f1, _, _ = evaluate(student, test_loader, nn.CrossEntropyLoss(), config)
            baseline_results.append({'seed': seed, 'test_acc': test_acc, 'test_f1': test_f1})
            logger.log(f"  Seed {seed}: Test Acc={test_acc:.4f}")

            cleanup_memory()

        baseline_stats = mean_std_ci([r['test_acc'] for r in baseline_results])
        results['baseline'] = {
            'per_seed': baseline_results,
            'mean': baseline_stats['mean'],
            'std': baseline_stats['std'],
            'ci_low': baseline_stats['ci_low'],
            'ci_high': baseline_stats['ci_high'],
            'n': baseline_stats['n']
        }
        logger.log(f"\nBaseline: {baseline_stats['mean']*100:.2f}% ± {baseline_stats['std']*100:.2f}")
        logger.log(f"  95% CI: [{baseline_stats['ci_low']*100:.2f}%, {baseline_stats['ci_high']*100:.2f}%]")

    # ========== Step 3: Best KD ==========
    if config.run_best_kd:
        logger.log("\n" + "=" * 80)
        logger.log("STEP 3: BEST KD (FULL MULTI-SEED)")
        logger.log("=" * 80)
        logger.log(f"  Config: T={best_config['temperature']}, α={best_config['alpha']}")
        logger.log(f"  Seeds: {seed_list}")
        logger.log("=" * 80)

        kd_results = []
        for seed in seed_list:
            set_seed(seed)
            logger.log(f"\nSeed {seed}:")

            student = LightweightStudent(
                num_classes=config.num_classes,
                student_dim=config.student_dim,
                teacher_dim=config.latent_dim,
                active_heads=[]
            ).to(config.device)

            run_name = f"kd_winner_seed{seed}"
            student = train_standard_kd_with_overrides(
                student, teacher, train_loader, val_loader, config,
                epochs=config.student_epochs, run_name=run_name, logger=logger,
                temperature_override=best_config['temperature'],
                alpha_override=best_config['alpha']
            )

            test_loss, test_acc, test_f1, _, _ = evaluate(student, test_loader, nn.CrossEntropyLoss(), config)
            kd_results.append({'seed': seed, 'test_acc': test_acc, 'test_f1': test_f1})
            logger.log(f"  Seed {seed}: Test Acc={test_acc:.4f}")

            cleanup_memory()

        kd_stats = mean_std_ci([r['test_acc'] for r in kd_results])
        results['best_kd'] = {
            'per_seed': kd_results,
            'config': best_config,
            'mean': kd_stats['mean'],
            'std': kd_stats['std'],
            'ci_low': kd_stats['ci_low'],
            'ci_high': kd_stats['ci_high'],
            'n': kd_stats['n']
        }
        logger.log(f"\nBest KD: {kd_stats['mean']*100:.2f}% ± {kd_stats['std']*100:.2f}")
        logger.log(f"  95% CI: [{kd_stats['ci_low']*100:.2f}%, {kd_stats['ci_high']*100:.2f}%]")

    # ========== Step 4: MSFD ==========
    if config.run_messy:
        logger.log("\n" + "=" * 80)
        logger.log("STEP 4: MSFD (FULL MULTI-SEED)")
        logger.log("=" * 80)
        logger.log(f"  Seeds: {seed_list}")
        logger.log("=" * 80)

        messy_results = []
        for seed in seed_list:
            set_seed(seed)
            logger.log(f"\nSeed {seed}:")

            student = LightweightStudent(
                num_classes=config.num_classes,
                student_dim=config.student_dim,
                teacher_dim=config.latent_dim,
                active_heads=['visual', 'audio', '3d']
            ).to(config.device)

            run_name = f"MSFD_seed{seed}"
            student = train_messy_with_seed(
                student, teacher, train_loader, val_loader, config,
                epochs=config.student_epochs, run_name=run_name, logger=logger
            )

            test_loss, test_acc, test_f1, _, _ = evaluate(student, test_loader, nn.CrossEntropyLoss(), config)
            messy_results.append({'seed': seed, 'test_acc': test_acc, 'test_f1': test_f1})
            logger.log(f"  Seed {seed}: Test Acc={test_acc:.4f}")

            cleanup_memory()

        messy_stats = mean_std_ci([r['test_acc'] for r in messy_results])
        results['messy'] = {
            'per_seed': messy_results,
            'mean': messy_stats['mean'],
            'std': messy_stats['std'],
            'ci_low': messy_stats['ci_low'],
            'ci_high': messy_stats['ci_high'],
            'n': messy_stats['n']
        }
        logger.log(f"\nMSFD: {messy_stats['mean']*100:.2f}% ± {messy_stats['std']*100:.2f}")
        logger.log(f"  95% CI: [{messy_stats['ci_low']*100:.2f}%, {messy_stats['ci_high']*100:.2f}%]")

    # ========== Step 5: Significance Tests ==========
    if config.run_significance_tests and 'messy' in results and 'best_kd' in results and 'baseline' in results:
        logger.log("\n" + "=" * 80)
        logger.log("STEP 5: STATISTICAL SIGNIFICANCE TESTS")
        logger.log("=" * 80)

        messy_accs = [r['test_acc'] for r in results['messy']['per_seed']]
        kd_accs = [r['test_acc'] for r in results['best_kd']['per_seed']]
        baseline_accs = [r['test_acc'] for r in results['baseline']['per_seed']]

        sig_messy_kd = paired_significance_test(messy_accs, kd_accs)
        logger.log(f"\nMSFD vs Best KD:")
        logger.log(f"  Mean diff: {sig_messy_kd['mean_diff']*100:.2f}%")
        logger.log(f"  p-value (t-test): {sig_messy_kd.get('paired_t_p', 'N/A'):.4f}")
        logger.log("  ✅ Statistically significant" if sig_messy_kd.get('paired_t_p', 1.0) < 0.05 else "  ❌ Not statistically significant")
        results['significance_messy_vs_kd'] = sig_messy_kd

        sig_messy_baseline = paired_significance_test(messy_accs, baseline_accs)
        logger.log(f"\nMSFD vs Baseline:")
        logger.log(f"  Mean diff: {sig_messy_baseline['mean_diff']*100:.2f}%")
        logger.log(f"  p-value (t-test): {sig_messy_baseline.get('paired_t_p', 'N/A'):.4f}")
        logger.log("  ✅ Statistically significant" if sig_messy_baseline.get('paired_t_p', 1.0) < 0.05 else "  ❌ Not statistically significant")
        results['significance_messy_vs_baseline'] = sig_messy_baseline

        sig_kd_baseline = paired_significance_test(kd_accs, baseline_accs)
        logger.log(f"\nBest KD vs Baseline:")
        logger.log(f"  Mean diff: {sig_kd_baseline['mean_diff']*100:.2f}%")
        logger.log(f"  p-value (t-test): {sig_kd_baseline.get('paired_t_p', 'N/A'):.4f}")
        logger.log("  ✅ Statistically significant" if sig_kd_baseline.get('paired_t_p', 1.0) < 0.05 else "  ❌ Not statistically significant")
        results['significance_kd_vs_baseline'] = sig_kd_baseline

    # ========== Step 6: Final Summary ==========
    logger.log("\n" + "=" * 80)
    logger.log("FINAL SUMMARY")
    logger.log("=" * 80)
    logger.log(f"  Best KD Config: T={best_config['temperature']}, α={best_config['alpha']}")
    if 'baseline' in results:
        logger.log(f"  Baseline (No Distillation): {results['baseline']['mean']*100:.2f}% ± {results['baseline']['std']*100:.2f}")
    if 'best_kd' in results:
        logger.log(f"  Best KD: {results['best_kd']['mean']*100:.2f}% ± {results['best_kd']['std']*100:.2f}")
    if 'messy' in results:
        logger.log(f"  MSFD: {results['messy']['mean']*100:.2f}% ± {results['messy']['std']*100:.2f}")

    return results