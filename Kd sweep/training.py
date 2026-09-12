"""
Generic training loops: baseline (no distillation) and MSFD student
with privileged-feature distillation.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .logger import SimpleLogger
from .utils import safe_load_checkpoint
from .data_loader import evaluate
from .distillation import mixup_data, mixup_criterion, mixup_accuracy


def train_baseline_with_seed(student, train_loader, val_loader, config, epochs, run_name, logger: SimpleLogger):
    logger.log(f"  Training Baseline '{run_name}' ({epochs} epochs)")

    optimizer = torch.optim.AdamW(student.parameters(), lr=config.student_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=config.use_amp)

    best_acc = 0.0

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Baseline {run_name} Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            images, labels, idx = batch
            images, labels = images.to(config.device), labels.to(config.device)

            mixed_x, y_a, y_b, lam = mixup_data(images, labels, alpha=config.mixup_alpha)

            with autocast(enabled=config.use_amp):
                logits = student(mixed_x)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                loss = loss / config.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % config.accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * config.accumulation_steps
            preds = torch.argmax(logits, dim=1)
            acc = mixup_accuracy(preds, y_a, y_b, lam)
            pbar.set_postfix({'loss': loss.item() * config.accumulation_steps, 'acc': acc})

        scheduler.step()

        val_loss, val_acc, val_f1, _, _ = evaluate(student, val_loader, criterion, config)
        logger.log(f"    Epoch {epoch+1}: Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'best_acc': best_acc
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


def train_messy_with_seed(student, teacher, train_loader, val_loader, config, epochs, run_name, logger: SimpleLogger):
    logger.log(f"  Training MSFD '{run_name}' ({epochs} epochs)")

    optimizer = torch.optim.AdamW(student.parameters(), lr=config.student_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    scaler = GradScaler(enabled=config.use_amp)

    teacher.eval()
    best_acc = 0.0

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        running_acc = 0.0

        pbar = tqdm(train_loader, desc=f"MSFD {run_name} Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            images, labels, idx = batch
            images, labels = images.to(config.device), labels.to(config.device)

            mixed_x, y_a, y_b, lam = mixup_data(images, labels, alpha=config.mixup_alpha)

            with autocast(enabled=config.use_amp):
                student_logits, student_preds = student(mixed_x, return_predictions=True)
                with torch.no_grad():
                    teacher_features = teacher.get_privileged_features(mixed_x)
                ce_loss = mixup_criterion(ce_criterion, student_logits, y_a, y_b, lam)

                distill_losses = []
                for head_name in student.active_heads:
                    pred_key = f'pred_{head_name}'
                    if pred_key in student_preds and head_name in teacher_features:
                        d_loss = mse_criterion(student_preds[pred_key], teacher_features[head_name])
                        distill_losses.append(d_loss)

                distill_loss = sum(distill_losses) / len(distill_losses) if distill_losses else torch.tensor(0.0, device=config.device)
                loss = config.ce_weight * ce_loss + config.distill_weight * distill_loss
                loss = loss / config.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % config.accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * config.accumulation_steps
            preds = torch.argmax(student_logits, dim=1)
            acc = mixup_accuracy(preds, y_a, y_b, lam)
            running_acc += acc
            pbar.set_postfix({'loss': loss.item() * config.accumulation_steps, 'acc': acc})

        scheduler.step()

        val_loss, val_acc, val_f1, _, _ = evaluate(student, val_loader, ce_criterion, config)
        logger.log(f"    Epoch {epoch+1}: Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'best_acc': best_acc
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