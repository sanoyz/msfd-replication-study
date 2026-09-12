"""
Deliberate practice: hard example mining + targeted augmentation + retraining.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T

from code.config import Config
from code.logger import logger
from code.data_loader import IMAGENET_MEAN, IMAGENET_STD
from code.distillation import train_student_fast, train_student_messy


def identify_hard_examples(student: nn.Module, dataloader,
                            threshold: float = 0.6,
                            config: Config = None
                            ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Identify hard examples using classification confidence."""
    student.eval()
    hard_images, hard_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Identifying hard examples", leave=False):
            images, labels = batch[0], batch[1]
            images, labels = images.to(config.device), labels.to(config.device)
            logits = student(images)
            probs = F.softmax(logits, dim=1)
            max_prob, pred = torch.max(probs, dim=1)

            is_hard = (max_prob < threshold) | (pred != labels)

            if is_hard.any():
                hard_images.append(images[is_hard].cpu())
                hard_labels.append(labels[is_hard].cpu())

    if hard_images:
        return torch.cat(hard_images), torch.cat(hard_labels)
    return None, None


def deliberate_practice(student: nn.Module, teacher: nn.Module,
                         train_loader, val_loader, config: Config,
                         teacher_cache=None
                         ) -> Tuple[nn.Module, nn.Module]:
    """Improved deliberate practice using classification-based hard mining."""
    logger.log("\n" + "=" * 80)
    logger.log("IMPROVED DELIBERATE PRACTICE")
    logger.log("=" * 80)

    hard_images, hard_labels = identify_hard_examples(
        student, train_loader,
        threshold=config.hard_example_threshold, config=config
    )

    if hard_images is None or len(hard_images) < 50:
        logger.log(f"[WARN] Only {len(hard_images) if hard_images is not None else 0} "
                   f"hard examples found. Skipping.")
        return student, teacher

    logger.log(f"Found {len(hard_images)} hard examples")

    # Stronger augmentation pipeline for deliberate practice
    practice_transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224, padding=8, padding_mode='reflect'),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    augmented_dataset = []
    logger.log(f"Generating {config.augmentations_per_sample} augmentations per sample...")

    for i in range(len(hard_images)):
        img = hard_images[i]
        label = hard_labels[i]
        img = img * std_t + mean_t
        img = torch.clamp(img, 0, 1)
        img = T.ToPILImage()(img)
        for _ in range(config.augmentations_per_sample):
            aug_img = practice_transform(img)
            augmented_dataset.append((aug_img, label))

    augmented_loader = DataLoader(
        augmented_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True
    )
    logger.log(f"Created {len(augmented_dataset)} augmented samples")

    student = train_student_fast(
        student, teacher, augmented_loader, val_loader, config,
        epochs=2, run_name="practice_fast"
    )

    logger.log("Fine-tuning on full dataset...")
    student = train_student_messy(
        student, teacher, train_loader, val_loader, config,
        epochs=3, use_mixup=True, lr=config.practice_lr,
        run_name="practice_finetune", teacher_cache=teacher_cache
    )

    return student, teacher