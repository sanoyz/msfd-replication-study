"""
Progressive distillation: Teacher -> Student -> smaller Student -> tiniest Student.
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from code.config import Config
from code.logger import logger
from code.models import LightweightStudent
from code.distillation import train_student_messy
from code.training import evaluate


def progressive_distillation(teacher: nn.Module,
                              train_loader, val_loader, config: Config,
                              pretrained_student1: Optional[nn.Module] = None,
                              teacher_cache=None
                              ) -> Tuple[Dict[str, float], nn.Module,
                                          nn.Module, nn.Module]:
    """
    Progressive distillation: Teacher -> Student -> Student 2 -> Student 3.

    Stage 1 can reuse a pretrained student (if supplied); stages 2 and 3
    use each previous stage as the "teacher" for the next, smaller student.
    """
    logger.log("\n" + "=" * 80)
    logger.log("PROGRESSIVE DISTILLATION")
    logger.log("=" * 80)

    results = {}

    # Stage 1
    if pretrained_student1 is not None:
        logger.log("\n--- Stage 1: reusing already-trained student ---")
        student_1 = pretrained_student1
    else:
        logger.log("\n--- Stage 1: Teacher -> Student ---")
        student_1 = LightweightStudent(
            num_classes=config.num_classes,
            student_dim=config.student_dim,
            teacher_dim=config.latent_dim,
            active_heads=['visual', 'audio', '3d']
        ).to(config.device)
        student_1 = train_student_messy(
            student_1, teacher, train_loader, val_loader, config,
            epochs=config.student_epochs, use_mixup=True,
            run_name="progressive_stage1", teacher_cache=teacher_cache
        )

    _, acc_1, f1_1, _, _ = evaluate(
        student_1, val_loader, nn.CrossEntropyLoss(), config
    )
    results['stage_1_accuracy'] = acc_1
    results['stage_1_f1'] = f1_1
    logger.log(f"[OK] Stage 1 Accuracy: {acc_1:.4f}")

    # Stage 2 (half width)
    logger.log("\n--- Stage 2: Student -> Student 2 ---")
    student_2 = LightweightStudent(
        num_classes=config.num_classes,
        student_dim=config.student_dim // 2,
        teacher_dim=config.latent_dim,
        active_heads=['visual', 'audio', '3d']
    ).to(config.device)
    student_2 = train_student_messy(
        student_2, student_1, train_loader, val_loader, config,
        epochs=config.progressive_stage2_epochs, use_mixup=True,
        run_name="progressive_stage2"
    )
    _, acc_2, f1_2, _, _ = evaluate(
        student_2, val_loader, nn.CrossEntropyLoss(), config
    )
    results['stage_2_accuracy'] = acc_2
    results['stage_2_f1'] = f1_2
    logger.log(f"[OK] Stage 2 Accuracy: {acc_2:.4f}")

    # Stage 3 (quarter width)
    logger.log("\n--- Stage 3: Student 2 -> Student 3 ---")
    student_3 = LightweightStudent(
        num_classes=config.num_classes,
        student_dim=config.student_dim // 4,
        teacher_dim=config.latent_dim,
        active_heads=['visual', 'audio', '3d']
    ).to(config.device)
    student_3 = train_student_messy(
        student_3, student_2, train_loader, val_loader, config,
        epochs=config.progressive_stage3_epochs, use_mixup=True,
        run_name="progressive_stage3"
    )
    _, acc_3, f1_3, _, _ = evaluate(
        student_3, val_loader, nn.CrossEntropyLoss(), config
    )
    results['stage_3_accuracy'] = acc_3
    results['stage_3_f1'] = f1_3
    logger.log(f"[OK] Stage 3 Accuracy: {acc_3:.4f}")

    return results, student_1, student_2, student_3