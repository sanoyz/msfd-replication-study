"""
MixUp helpers and standard KD distillation training.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .logger import SimpleLogger
from .utils import safe_load_checkpoint
from .data_loader import evaluate


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_accuracy(preds, y_a, y_b, lam):
    correct_a = (preds == y_a).float()
    correct_b = (preds == y_b).float()
    return (lam * correct_a + (1 - lam) * correct_b).mean().item()


def train_standard_kd_with_overrides(
    student, teacher, train_loader, val_loader, config,
    epochs, run_name, logger: SimpleLogger,
    temperature_override=None, alpha_override=None
):
    temperature = temperature_override if temperature_override is not None else config.temperature
    alpha = alpha_override if alpha_override is not None else config.kd_alpha

    logger.log(f"  Training KD '{run_name}': T={temperature}, α={alpha} ({epochs} epochs)")

    optimizer = torch.optim.AdamW(student.parameters(), lr=config.student_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=config.use_amp)

    teacher.eval()
    best_acc = 0.0

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"KD {run_name} Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            images, labels, idx = batch
            images, labels = images.to(config.device), labels.to(config.device)

            mixed_x, y_a, y_b, lam = mixup_data(images, labels, alpha=config.mixup_alpha)

            with autocast(enabled=config.use_amp):
                student_logits = student(mixed_x)
                with torch.no_grad():
                    teacher_logits = teacher(mixed_x)
                ce_loss = mixup_criterion(criterion, student_logits, y_a, y_b, lam)
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction='batchmean'
                ) * (temperature ** 2)
                loss = alpha * ce_loss + (1 - alpha) * kd_loss
                loss = loss / config.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % config.accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * config.accumulation_steps
            pbar.set_postfix({'loss': loss.item() * config.accumulation_steps})

        scheduler.step()

        val_loss, val_acc, val_f1, _, _ = evaluate(student, val_loader, criterion, config)
        logger.log(f"    Epoch {epoch+1}: Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'best_acc': best_acc,
                'temperature': temperature,
                'alpha': alpha
            }, f"{config.checkpoint_dir}/{run_name}_best.pth")

        torch.save({
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'best_acc': best_acc
        }, f"{config.checkpoint_dir}/{run_name}_latest.pth")

    if os.path.exists(f"{config.checkpoint_dir}/{run_name}_best.pth"):
        ckpt = safe_load_checkpoint(f"{config.checkpoint_dir}/{run_name}_best.pth", config.device)
        student.load_state_dict(ckpt['model_state_dict'])
        logger.log(f"    Best val acc: {ckpt['best_acc']:.4f}")

    return student