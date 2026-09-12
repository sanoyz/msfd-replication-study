"""
Main pipeline: run all experiments and produce the paper's results.

This is the "complete training" pipeline (Pipeline 1).

Usage:
    python code/run_all.py --config configs/default.yaml
"""
import argparse
import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from code.config import Config
from code.logger import logger
from code.data_loader import build_dataloaders
from code.models import MultimodalTeacher, LightweightStudent
from code.training import train_teacher, train_baseline, evaluate
from code.distillation import (
    train_student_messy, train_standard_kd, train_messy_kd,
    TeacherFeatureCache, run_seeded_students
)
from code.deliberate_practice import deliberate_practice
from code.progressive import progressive_distillation
from code.evaluation import analyze_per_class, plot_confusion_matrix, EnsembleStudent
from code.head_ablations import run_head_ablations
from code.utils import (
    set_seed, cleanup_memory, safe_load_checkpoint, get_seed_list,
    log_seeded_summary, paired_significance_test
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()

    config = Config()
    set_seed(config.seed)

    # ---------- Step 1: Data ----------
    train_loader, val_loader, test_loader, cache_loader, CIFAR100_CLASSES = \
        build_dataloaders(config)

    # ---------- Step 2: Teacher ----------
    logger.log("\n" + "=" * 60)
    logger.log("STEP 1: TRAIN TEACHER")
    logger.log("=" * 60)

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
        checkpoint = safe_load_checkpoint(best_path, config.device)
        teacher.load_state_dict(checkpoint['model_state_dict'])
        logger.log(f"Loaded teacher with accuracy: {checkpoint['best_acc']:.4f}")
    else:
        teacher = train_teacher(teacher, train_loader, val_loader,
                                 config, config.teacher_epochs)

    # ---------- Step 3: Build teacher feature cache ----------
    teacher_cache = TeacherFeatureCache()
    if config.use_teacher_feature_cache and cache_loader is not None:
        logger.log("\n" + "=" * 60)
        logger.log("BUILDING TEACHER FEATURE CACHE")
        logger.log("=" * 60)
        teacher_cache.build(teacher, cache_loader, config, with_logits=True)

    # ---------- Step 4: MSFD students (multi-seed) ----------
    logger.log("\n" + "=" * 60)
    logger.log("STEP 2: TRAIN MSFD STUDENTS (multi-seed)")
    logger.log("=" * 60)

    results_dict = {}
    seed_list = get_seed_list(config)
    logger.log(f"[SEEDS] Using {len(seed_list)} seeds: {seed_list}")

    messy_seed_results, messy_seed_models = run_seeded_students(
        kind='messy', config=config, train_loader=train_loader,
        val_loader=val_loader, test_loader=test_loader,
        seeds=seed_list, epochs=config.student_epochs,
        teacher=teacher, teacher_cache=teacher_cache
    )
    messy_stats = log_seeded_summary("MSFD student (full, pre-practice)",
                                      messy_seed_results, metric='test_acc')
    results_dict['messy_student_seed_results'] = messy_seed_results
    results_dict['messy_student_mean_acc'] = messy_stats['mean']
    results_dict['messy_student_std_acc'] = messy_stats['std']
    results_dict['messy_student_ci95'] = [messy_stats['ci_low'],
                                            messy_stats['ci_high']]

    # Carry primary seed forward
    primary_idx = (seed_list.index(config.seed)
                    if config.seed in seed_list else 0)
    student = messy_seed_models[primary_idx]
    student_params = sum(p.numel() for p in student.parameters()
                          if p.requires_grad)
    logger.log(f"Student parameters: {student_params:,}")
    logger.log(f"[PRIMARY] Seed {seed_list[primary_idx]}'s student carried forward.")

    # ---------- Step 5: Deliberate Practice ----------
    if (config.practice_rounds > 0
            and config.run_deliberate_practice
            and not config.QUICK_TEST):
        logger.log("\n" + "=" * 60)
        logger.log("STEP 3: DELIBERATE PRACTICE")
        logger.log("=" * 60)
        for round_num in range(config.practice_rounds):
            logger.log(f"\n--- Round {round_num + 1}/{config.practice_rounds} ---")
            student, teacher = deliberate_practice(
                student, teacher, train_loader, val_loader, config,
                teacher_cache=teacher_cache
            )
            torch.save({
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'round': round_num
            }, f"{config.checkpoint_dir}/practice_round_{round_num}.pth")
            cleanup_memory()

    # ---------- Step 6: Experiments ----------
    if config.RUN_ALL_EXPERIMENTS and not config.QUICK_TEST:
        logger.log("\n" + "=" * 60)
        logger.log("STEP 4: RUNNING EXPERIMENTS")
        logger.log("=" * 60)

        # 6.1 No-distillation baseline
        baseline_seed_results = None
        if config.run_baseline:
            logger.log("\n--- 4.0 No-Distillation Baseline ---")
            baseline_seed_results, baseline_models = run_seeded_students(
                kind='baseline', config=config,
                train_loader=train_loader, val_loader=val_loader,
                test_loader=test_loader, seeds=seed_list,
                epochs=config.student_epochs
            )
            baseline_stats = log_seeded_summary(
                "No-distillation baseline", baseline_seed_results,
                metric='test_acc'
            )
            results_dict['baseline_seed_results'] = baseline_seed_results
            results_dict['baseline_mean_acc'] = baseline_stats['mean']
            results_dict['baseline_std_acc'] = baseline_stats['std']
            results_dict['baseline_ci95'] = [baseline_stats['ci_low'],
                                              baseline_stats['ci_high']]
            del baseline_models
            cleanup_memory()

        # 6.2 Standard KD (tuned)
        kd_seed_results = None
        if config.run_standard_kd:
            logger.log("\n--- 4.1 Standard KD Baseline ---")
            kd_seed_results, kd_models = run_seeded_students(
                kind='standard_kd', config=config,
                train_loader=train_loader, val_loader=val_loader,
                test_loader=test_loader, seeds=seed_list,
                epochs=config.student_epochs,
                teacher=teacher, teacher_cache=teacher_cache
            )
            kd_stats = log_seeded_summary("Standard KD", kd_seed_results,
                                            metric='test_acc')
            results_dict['standard_kd_seed_results'] = kd_seed_results
            results_dict['standard_kd_mean_acc'] = kd_stats['mean']
            results_dict['standard_kd_std_acc'] = kd_stats['std']
            results_dict['standard_kd_ci95'] = [kd_stats['ci_low'],
                                                  kd_stats['ci_high']]
            del kd_models
            cleanup_memory()

        # 6.3 MSFD + KD
        messy_kd_seed_results = None
        if config.run_messy_kd:
            logger.log("\n--- 4.2 MSFD + KD with Temperature ---")
            messy_kd_seed_results, messy_kd_models = run_seeded_students(
                kind='messy_kd', config=config,
                train_loader=train_loader, val_loader=val_loader,
                test_loader=test_loader, seeds=seed_list,
                epochs=config.student_epochs,
                teacher=teacher, teacher_cache=teacher_cache
            )
            messy_kd_stats = log_seeded_summary("MSFD+KD", messy_kd_seed_results,
                                                  metric='test_acc')
            results_dict['messy_kd_seed_results'] = messy_kd_seed_results
            results_dict['messy_kd_mean_acc'] = messy_kd_stats['mean']
            results_dict['messy_kd_std_acc'] = messy_kd_stats['std']
            results_dict['messy_kd_ci95'] = [messy_kd_stats['ci_low'],
                                                messy_kd_stats['ci_high']]
            del messy_kd_models
            cleanup_memory()

        # 6.4 Head ablations
        if config.run_head_ablations:
            logger.log("\n--- 4.3 Head Ablations ---")
            ablation_results = run_head_ablations(
                teacher, train_loader, val_loader, test_loader,
                config, teacher_cache=teacher_cache
            )
            for name, result in ablation_results.items():
                results_dict[f'ablation_{name}_accuracy'] = result['accuracy']
                results_dict[f'ablation_{name}_std'] = result['std']
                results_dict[f'ablation_{name}_ci95'] = [result['ci_low'],
                                                          result['ci_high']]
                results_dict[f'ablation_{name}_f1'] = result['f1']
            cleanup_memory()

        # 6.5 Ensemble
        if config.run_ensemble:
            logger.log("\n--- 4.4 Ensemble of Students ---")
            ensemble = EnsembleStudent(
                num_models=len(messy_seed_models),
                active_heads=['visual', 'audio', '3d'], config=config
            )
            ensemble.models = messy_seed_models
            ensemble_acc = ensemble.evaluate(test_loader)
            results_dict['ensemble_accuracy'] = ensemble_acc
            results_dict['ensemble_n_models'] = len(messy_seed_models)
            logger.log(f"[OK] Ensemble Test Accuracy: {ensemble_acc:.4f}")
            cleanup_memory()

        # 6.6 Significance tests
        logger.log("\n--- 4.4b Significance Tests ---")
        sig_results = {}
        messy_accs = [r['test_acc'] for r in messy_seed_results]
        if kd_seed_results is not None:
            kd_accs = [r['test_acc'] for r in kd_seed_results]
            sig = paired_significance_test("MSFD student", messy_accs,
                                             "Standard KD", kd_accs)
            sig_results['messy_vs_standard_kd'] = sig
            logger.log(f"[SIG] MSFD vs Standard KD: {sig}")
        if baseline_seed_results is not None:
            base_accs = [r['test_acc'] for r in baseline_seed_results]
            sig = paired_significance_test("MSFD student", messy_accs,
                                             "No-distillation baseline", base_accs)
            sig_results['messy_vs_baseline'] = sig
            logger.log(f"[SIG] MSFD vs Baseline: {sig}")
            if kd_seed_results is not None:
                sig = paired_significance_test("Standard KD", kd_accs,
                                                 "No-distillation baseline", base_accs)
                sig_results['standard_kd_vs_baseline'] = sig
                logger.log(f"[SIG] KD vs Baseline: {sig}")
        results_dict['significance_tests'] = sig_results

        # 6.7 Progressive distillation
        if config.run_progressive_distillation:
            logger.log("\n--- 4.5 Progressive Distillation ---")
            prog_results, st1, st2, st3 = progressive_distillation(
                teacher, train_loader, val_loader, config,
                pretrained_student1=student, teacher_cache=teacher_cache
            )
            for key, value in prog_results.items():
                results_dict[f'progressive_{key}'] = value
            del st1, st2, st3
            cleanup_memory()

        # 6.8 Per-class analysis + confusion matrix
        if config.run_per_class_analysis:
            logger.log("\n--- 4.6 Per-Class Analysis ---")
            per_class_results = analyze_per_class(
                student, test_loader, CIFAR100_CLASSES, config
            )
            results_dict['per_class_analysis_complete'] = True
            logger.log("\n--- 4.7 Confusion Matrix ---")
            plot_confusion_matrix(
                student, test_loader, CIFAR100_CLASSES, config,
                worst_class_names=[n for n, _ in per_class_results['worst']],
                all_labels=per_class_results['all_labels'],
                all_preds=per_class_results['all_preds']
            )

    # ---------- Step 7: Final evaluation ----------
    logger.log("\n" + "=" * 60)
    logger.log("STEP 5: FINAL EVALUATION")
    logger.log("=" * 60)

    _, val_acc, val_f1, _, _ = evaluate(student, val_loader,
                                          nn.CrossEntropyLoss(), config)
    _, test_acc, test_f1, _, _ = evaluate(student, test_loader,
                                            nn.CrossEntropyLoss(), config)

    results_dict['final_val_accuracy'] = val_acc
    results_dict['final_val_f1'] = val_f1
    results_dict['final_test_accuracy'] = test_acc
    results_dict['final_test_f1'] = test_f1

    logger.log(f"Final Validation - Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
    logger.log(f"Final Test - Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

    # ---------- Step 8: Save ----------
    logger.save_results(results_dict)
    torch.save({
        'teacher_state_dict': teacher.state_dict(),
        'student_state_dict': student.state_dict(),
        'config': config.__dict__,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'results': results_dict
    }, f'{config.output_dir}/final_models.pth')

    logger.log(f"\nAll outputs saved to: {config.output_dir}")


if __name__ == "__main__":
    main()