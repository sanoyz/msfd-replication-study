"""
Generic training utilities: optimizer/scheduler creation, MixUp,
evaluation, and the no-distillation baseline.
"""
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from code.config import Config
from code.logger import logger
from code.utils import (
    EarlyStopping, save_resume_checkpoint, try_load_resume_checkpoint,
    cleanup_memory
)


# ---------- MIXUP ----------

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Apply MixUp augmentation."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion: nn.Module, pred: torch.Tensor,
                    y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_accuracy(preds: torch.Tensor, y_a: torch.Tensor,
                   y_b: torch.Tensor, lam: float) -> float:
    """Lambda-weighted accuracy during MixUp steps."""
    correct_a = (preds == y_a).float()
    correct_b = (preds == y_b).float()
    return (lam * correct_a + (1 - lam) * correct_b).mean().item()


# ---------- EVALUATION ----------

def evaluate(model: nn.Module, loader, criterion: nn.Module, config: Config):
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(config.device), labels.to(config.device)
            with autocast(enabled=config.use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, acc, f1, all_preds, all_labels


# ---------- OPTIMIZER ----------

def create_optimizer_and_scheduler(model: nn.Module, config: Config,
                                    epochs: int, lr: Optional[float] = None):
    """Create optimizer with warmup + cosine annealing scheduler."""
    effective_lr = lr if lr is not None else config.lr
    optimizer = AdamW(model.parameters(), lr=effective_lr,
                      weight_decay=config.weight_decay)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1,
        total_iters=min(config.warmup_epochs, epochs)
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(epochs - config.warmup_epochs, 1)
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_epochs]
    )
    return optimizer, scheduler


# ---------- TRAIN TEACHER ----------

def train_teacher(teacher: nn.Module, train_loader, val_loader,
                  config: Config, epochs: int) -> nn.Module:
    """Train teacher model with MixUp and gradient accumulation."""
    from code.models import freeze_vit_backbone_except_last_n

    logger.log(f"\nTraining Teacher ({epochs} epochs)")
    freeze_vit_backbone_except_last_n(
        teacher, config.freeze_teacher_backbone_except_last_n_blocks
    )

    optimizer, scheduler = create_optimizer_and_scheduler(
        teacher, config, epochs, lr=config.teacher_lr
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=config.use_amp)
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)

    resume_path = f"{config.checkpoint_dir}/teacher_latest.pth"
    start_epoch, best_acc = try_load_resume_checkpoint(
        resume_path, teacher, optimizer, scaler, config
    )
    if best_acc > 0:
        early_stopping.best_score = best_acc

    for epoch in range(start_epoch, epochs):
        teacher.train()
        running_loss, running_acc = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Teacher Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()

        for step, (images, labels, _idx) in enumerate(pbar):
            images, labels = images.to(config.device), labels.to(config.device)
            mixed_x, y_a, y_b, lam = mixup_data(images, labels, alpha=config.mixup_alpha)

            with autocast(enabled=config.use_amp):
                logits = teacher(mixed_x)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                loss = loss / config.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % config.accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * config.accumulation_steps
            preds = torch.argmax(logits, dim=1)
            acc = mixup_accuracy(preds, y_a, y_b, lam)
            running_acc += acc
            pbar.set_postfix({'loss': loss.item() * config.accumulation_steps, 'acc': acc})

        scheduler.step()
        val_loss, val_acc, val_f1, _, _ = evaluate(teacher, val_loader, criterion, config)

        logger.log(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, "
                   f"Train Acc={running_acc/len(train_loader):.4f}, "
                   f"Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
        logger.log_metrics('teacher', epoch, {
            'train_loss': running_loss / len(train_loader),
            'train_acc': running_acc / len(train_loader),
            'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': teacher.state_dict(),
                'best_acc': best_acc
            }, f"{config.checkpoint_dir}/teacher_best.pth")
            logger.log(f"[OK] New best teacher: {best_acc:.4f}")

        save_resume_checkpoint(resume_path, teacher, optimizer, scaler, epoch, best_acc)

        if early_stopping(val_acc):
            logger.log(f"[WARN] Early stopping at epoch {epoch+1}")
            break
        cleanup_memory()

    best_path = f"{config.checkpoint_dir}/teacher_best.pth"
    if os.path.exists(best_path):
        from code.utils import safe_load_checkpoint
        checkpoint = safe_load_checkpoint(best_path, config.device)
        teacher.load_state_dict(checkpoint['model_state_dict'])
        logger.log(f"[OK] Teacher best accuracy: {checkpoint['best_acc']:.4f}")
    return teacher


# ---------- TRAIN BASELINE ----------

def train_baseline(student: nn.Module, train_loader, val_loader,
                   config: Config, epochs: int,
                   run_name: str = "baseline") -> nn.Module:
    """No-distillation baseline: cross-entropy + MixUp only."""
    import os
    from code.utils import safe_load_checkpoint

    logger.log(f"\nTraining No-Distillation Baseline '{run_name}' ({epochs} epochs)")

    optimizer, scheduler = create_optimizer_and_scheduler(
        student, config, epochs, lr=config.student_lr
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=config.use_amp)
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)

    resume_path = f"{config.checkpoint_dir}/{run_name}_latest.pth"
    best_path = f"{config.checkpoint_dir}/{run_name}_best.pth"
    start_epoch, best_acc = try_load_resume_checkpoint(
        resume_path, student, optimizer, scaler, config
    )
    if best_acc > 0:
        early_stopping.best_score = best_acc

    for epoch in range(start_epoch, epochs):
        student.train()
        running_loss, running_acc = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Baseline Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            images, labels = batch[0], batch[1]
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
            running_acc += acc
            pbar.set_postfix({'loss': loss.item() * config.accumulation_steps, 'acc': acc})

        scheduler.step()
        val_loss, val_acc, val_f1, _, _ = evaluate(student, val_loader, criterion, config)
        logger.log(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, "
                   f"Train Acc={running_acc/len(train_loader):.4f}, "
                   f"Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'epoch': epoch, 'model_state_dict': student.state_dict(),
                        'best_acc': best_acc}, best_path)
            logger.log(f"[OK] New best baseline ({run_name}): {best_acc:.4f}")

        save_resume_checkpoint(resume_path, student, optimizer, scaler, epoch, best_acc)

        if early_stopping(val_acc):
            logger.log(f"[WARN] Early stopping at epoch {epoch+1}")
            break
        cleanup_memory()

    if os.path.exists(best_path):
        checkpoint = safe_load_checkpoint(best_path, config.device)
        student.load_state_dict(checkpoint['model_state_dict'])
        logger.log(f"[OK] Baseline ({run_name}) best accuracy: {checkpoint['best_acc']:.4f}")
    return student