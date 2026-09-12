"""
Distillation methods: MSFD (multi-headed synthetic feature distillation)
and standard KD baseline.
"""
import os
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from code.config import Config
from code.logger import logger
from code.utils import (
    EarlyStopping, save_resume_checkpoint, try_load_resume_checkpoint,
    safe_load_checkpoint, cleanup_memory
)
from code.training import (
    mixup_data, mixup_criterion, mixup_accuracy, evaluate
)


# ---------- TEACHER FEATURE CACHE ----------

class TeacherFeatureCache:
    """Precompute teacher features and logits once per training image."""

    def __init__(self):
        self.features: Dict[str, torch.Tensor] = {}
        self.logits: Optional[torch.Tensor] = None
        self.ready = False

    @torch.no_grad()
    def build(self, model: nn.Module, loader, config: Config,
              with_logits: bool = False) -> None:
        from collections import defaultdict
        model.eval()
        feat_chunks = defaultdict(list)
        logit_chunks = []

        for images, _labels in tqdm(loader, desc="Building teacher feature cache"):
            images = images.to(config.device)
            with autocast(enabled=config.use_amp):
                feats = model.get_privileged_features(images)
                if with_logits:
                    logits = model(images)
            for k, v in feats.items():
                feat_chunks[k].append(v.float().cpu())
            if with_logits:
                logit_chunks.append(logits.float().cpu())

        for k, chunks in feat_chunks.items():
            self.features[k] = torch.cat(chunks, dim=0)
        if with_logits:
            self.logits = torch.cat(logit_chunks, dim=0)
        self.ready = True
        n = next(iter(self.features.values())).shape[0]
        logger.log(f"[CACHE] Built teacher feature cache for {n} samples "
                   f"(keys: {list(self.features.keys())}, logits: {with_logits})")

    def lookup_features(self, idx_a: torch.Tensor, idx_b: torch.Tensor,
                        lam: float, device) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in self.features.items():
            fa, fb = v[idx_a], v[idx_b]
            out[k] = (lam * fa + (1 - lam) * fb).to(device)
        return out

    def lookup_logits(self, idx_a: torch.Tensor, idx_b: torch.Tensor,
                      lam: float, device) -> torch.Tensor:
        la, lb = self.logits[idx_a], self.logits[idx_b]
        return (lam * la + (1 - lam) * lb).to(device)


# ---------- MSFD STUDENT ----------

def train_student_messy(student: nn.Module, teacher: nn.Module,
                        train_loader, val_loader, config: Config,
                        epochs: int, use_mixup: bool = True,
                        lr: Optional[float] = None,
                        run_name: str = "student_messy",
                        teacher_cache: Optional[TeacherFeatureCache] = None
                        ) -> nn.Module:
    """Train student with MSFD distillation."""
    logger.log(f"\nTraining MSFD Student '{run_name}' ({epochs} epochs)")

    effective_lr = lr if lr is not None else config.student_lr
    optimizer = AdamW(student.parameters(), lr=effective_lr,
                      weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    scaler = GradScaler(enabled=config.use_amp)
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)

    resume_path = f"{config.checkpoint_dir}/{run_name}_latest.pth"
    best_path = f"{config.checkpoint_dir}/{run_name}_best.pth"
    start_epoch, best_acc = try_load_resume_checkpoint(
        resume_path, student, optimizer, scaler, config
    )
    if best_acc > 0:
        early_stopping.best_score = best_acc

    teacher.eval()

    for epoch in range(start_epoch, epochs):
        student.train()
        running_loss, running_acc, running_distill = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"MSFD Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            if len(batch) == 3:
                images, labels, idx = batch
            else:
                images, labels = batch
                idx = None
            images, labels = images.to(config.device), labels.to(config.device)

            if use_mixup:
                perm = torch.randperm(images.size(0))
                lam = float(np.random.beta(config.mixup_alpha, config.mixup_alpha))
                mixed_x = lam * images + (1 - lam) * images[perm]
                y_a, y_b = labels, labels[perm]
            else:
                mixed_x, y_a, y_b, lam = images, labels, labels, 1.0
                perm = torch.arange(images.size(0))

            with autocast(enabled=config.use_amp):
                student_logits, student_preds = student(mixed_x, return_predictions=True)

                if (teacher_cache is not None and teacher_cache.ready
                        and idx is not None):
                    teacher_features = teacher_cache.lookup_features(
                        idx, idx[perm], lam, config.device
                    )
                else:
                    with torch.no_grad():
                        teacher_features = teacher.get_privileged_features(mixed_x)

                ce_loss = mixup_criterion(ce_criterion, student_logits, y_a, y_b, lam)

                distill_losses = []
                for head_name in student.active_heads:
                    pred_key = f'pred_{head_name}'
                    if pred_key in student_preds and head_name in teacher_features:
                        d_loss = mse_criterion(student_preds[pred_key],
                                                teacher_features[head_name])
                        distill_losses.append(d_loss)

                distill_loss = (sum(distill_losses) / len(distill_losses)
                                if distill_losses
                                else torch.tensor(0.0, device=config.device))
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
            running_distill += (distill_loss.item()
                                if isinstance(distill_loss, torch.Tensor)
                                else distill_loss)
            preds = torch.argmax(student_logits, dim=1)
            acc = mixup_accuracy(preds, y_a, y_b, lam)
            running_acc += acc
            pbar.set_postfix({'loss': loss.item() * config.accumulation_steps, 'acc': acc})

        scheduler.step()
        val_loss, val_acc, val_f1, _, _ = evaluate(student, val_loader, ce_criterion, config)

        logger.log(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, "
                   f"Train Acc={running_acc/len(train_loader):.4f}, "
                   f"Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
        logger.log_metrics(run_name, epoch, {
            'train_loss': running_loss / len(train_loader),
            'train_acc': running_acc / len(train_loader),
            'distill_loss': running_distill / len(train_loader),
            'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'best_acc': best_acc
            }, best_path)
            logger.log(f"[OK] New best student ({run_name}): {best_acc:.4f}")

        save_resume_checkpoint(resume_path, student, optimizer, scaler, epoch, best_acc)

        if early_stopping(val_acc):
            logger.log(f"[WARN] Early stopping at epoch {epoch+1}")
            break
        cleanup_memory()

    if os.path.exists(best_path):
        checkpoint = safe_load_checkpoint(best_path, config.device)
        student.load_state_dict(checkpoint['model_state_dict'])
        logger.log(f"[OK] Student ({run_name}) best accuracy: {checkpoint['best_acc']:.4f}")
    return student


def train_student_fast(student: nn.Module, teacher: nn.Module,
                       train_loader, val_loader, config: Config,
                       epochs: int = 2, run_name: str = "practice_fast",
                       teacher_cache: Optional[TeacherFeatureCache] = None
                       ) -> nn.Module:
    """Fast training for deliberate practice (lower LR)."""
    return train_student_messy(
        student, teacher, train_loader, val_loader,
        config, epochs=epochs, use_mixup=True, lr=config.practice_lr,
        run_name=run_name, teacher_cache=teacher_cache
    )


# ---------- STANDARD KD ----------

def train_standard_kd(student: nn.Module, teacher: nn.Module,
                       train_loader, val_loader, config: Config,
                       epochs: int = 12, run_name: str = "student_kd",
                       teacher_cache: Optional[TeacherFeatureCache] = None,
                       temperature_override: Optional[float] = None,
                       alpha_override: Optional[float] = None
                       ) -> nn.Module:
    """Standard Knowledge Distillation (Hinton et al., 2015)."""
    logger.log("\n" + "=" * 80)
    logger.log("STANDARD KNOWLEDGE DISTILLATION")
    logger.log("=" * 80)

    temperature = (temperature_override if temperature_override is not None
                   else config.temperature)
    alpha = (alpha_override if alpha_override is not None else 0.3)

    optimizer = AdamW(student.parameters(), lr=config.student_lr,
                      weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=config.use_amp)

    resume_path = f"{config.checkpoint_dir}/{run_name}_latest.pth"
    best_path = f"{config.checkpoint_dir}/{run_name}_best.pth"
    start_epoch, best_acc = try_load_resume_checkpoint(
        resume_path, student, optimizer, scaler, config
    )

    teacher.eval()
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    if best_acc > 0:
        early_stopping.best_score = best_acc

    for epoch in range(start_epoch, epochs):
        student.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"KD Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            if len(batch) == 3:
                images, labels, idx = batch
            else:
                images, labels = batch
                idx = None
            images, labels = images.to(config.device), labels.to(config.device)

            perm = torch.randperm(images.size(0))
            lam = float(np.random.beta(config.mixup_alpha, config.mixup_alpha))
            mixed_x = lam * images + (1 - lam) * images[perm]
            y_a, y_b = labels, labels[perm]

            with autocast(enabled=config.use_amp):
                student_logits = student(mixed_x)

                if (teacher_cache is not None and teacher_cache.ready
                        and teacher_cache.logits is not None and idx is not None):
                    teacher_logits = teacher_cache.lookup_logits(
                        idx, idx[perm], lam, config.device
                    )
                else:
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
        logger.log(f"Epoch {epoch+1}: Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'best_acc': best_acc
            }, best_path)

        save_resume_checkpoint(resume_path, student, optimizer, scaler, epoch, best_acc)

        if early_stopping(val_acc):
            logger.log(f"[WARN] Early stopping at epoch {epoch+1}")
            break
        cleanup_memory()

    if os.path.exists(best_path):
        checkpoint = safe_load_checkpoint(best_path, config.device)
        student.load_state_dict(checkpoint['model_state_dict'])
        logger.log(f"[OK] Standard KD best accuracy: {checkpoint['best_acc']:.4f}")
    return student


# ---------- MSFD + KD ----------

def train_messy_kd(student: nn.Module, teacher: nn.Module,
                    train_loader, val_loader, config: Config,
                    epochs: int = 12, run_name: str = "student_messy_kd",
                    teacher_cache: Optional[TeacherFeatureCache] = None
                    ) -> nn.Module:
    """MSFD + Standard KD with temperature."""
    logger.log("\n" + "=" * 80)
    logger.log("MSFD + KD WITH TEMPERATURE")
    logger.log("=" * 80)

    optimizer = AdamW(student.parameters(), lr=config.student_lr,
                      weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    scaler = GradScaler(enabled=config.use_amp)

    temperature = config.temperature
    alpha, beta = 0.3, 0.7

    resume_path = f"{config.checkpoint_dir}/{run_name}_latest.pth"
    best_path = f"{config.checkpoint_dir}/{run_name}_best.pth"
    start_epoch, best_acc = try_load_resume_checkpoint(
        resume_path, student, optimizer, scaler, config
    )

    teacher.eval()
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    if best_acc > 0:
        early_stopping.best_score = best_acc

    for epoch in range(start_epoch, epochs):
        student.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"MSFD+KD Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            if len(batch) == 3:
                images, labels, idx = batch
            else:
                images, labels = batch
                idx = None
            images, labels = images.to(config.device), labels.to(config.device)

            perm = torch.randperm(images.size(0))
            lam = float(np.random.beta(config.mixup_alpha, config.mixup_alpha))
            mixed_x = lam * images + (1 - lam) * images[perm]
            y_a, y_b = labels, labels[perm]

            with autocast(enabled=config.use_amp):
                student_logits, student_preds = student(mixed_x, return_predictions=True)

                have_cache = (teacher_cache is not None
                              and teacher_cache.ready and idx is not None)
                if have_cache and teacher_cache.logits is not None:
                    teacher_logits = teacher_cache.lookup_logits(
                        idx, idx[perm], lam, config.device
                    )
                else:
                    with torch.no_grad():
                        teacher_logits = teacher(mixed_x)
                if have_cache:
                    teacher_features = teacher_cache.lookup_features(
                        idx, idx[perm], lam, config.device
                    )
                else:
                    with torch.no_grad():
                        teacher_features = teacher.get_privileged_features(mixed_x)

                ce_loss = mixup_criterion(ce_criterion, student_logits, y_a, y_b, lam)
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction='batchmean'
                ) * (temperature ** 2)

                distill_losses = []
                for head_name in student.active_heads:
                    pred_key = f'pred_{head_name}'
                    if pred_key in student_preds and head_name in teacher_features:
                        distill_losses.append(
                            mse_criterion(student_preds[pred_key],
                                          teacher_features[head_name])
                        )

                distill_loss = (sum(distill_losses) / len(distill_losses)
                                if distill_losses
                                else torch.tensor(0.0, device=config.device))
                loss = alpha * (ce_loss + kd_loss) + beta * distill_loss
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
        val_loss, val_acc, val_f1, _, _ = evaluate(student, val_loader, ce_criterion, config)
        logger.log(f"Epoch {epoch+1}: Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch, 'model_state_dict': student.state_dict(),
                'best_acc': best_acc
            }, best_path)

        save_resume_checkpoint(resume_path, student, optimizer, scaler, epoch, best_acc)

        if early_stopping(val_acc):
            logger.log(f"[WARN] Early stopping at epoch {epoch+1}")
            break
        cleanup_memory()

    if os.path.exists(best_path):
        checkpoint = safe_load_checkpoint(best_path, config.device)
        student.load_state_dict(checkpoint['model_state_dict'])
        logger.log(f"[OK] MSFD+KD best accuracy: {checkpoint['best_acc']:.4f}")
    return student


# ---------- SEEDED RUNNER ----------

def run_seeded_students(kind: str, config: Config,
                         train_loader, val_loader, test_loader,
                         seeds: List[int], epochs: int,
                         teacher: Optional[nn.Module] = None,
                         teacher_cache: Optional[TeacherFeatureCache] = None
                         ) -> tuple:
    """Train one fresh student per seed for a given experiment kind."""
    from code.models import LightweightStudent
    from code.utils import set_seed
    from code.training import train_baseline

    assert kind in ('messy', 'standard_kd', 'messy_kd', 'baseline')
    if kind in ('standard_kd', 'messy_kd', 'messy') and teacher is None:
        raise ValueError(f"kind='{kind}' requires a teacher model.")

    results, models = [], []
    for seed in seeds:
        set_seed(seed)
        active_heads = ([] if kind in ('standard_kd', 'baseline')
                        else ['visual', 'audio', '3d'])
        student = LightweightStudent(
            num_classes=config.num_classes, student_dim=config.student_dim,
            teacher_dim=config.latent_dim, active_heads=active_heads
        ).to(config.device)

        run_name = f"{kind}_seed{seed}"
        if kind == 'messy':
            student = train_student_messy(
                student, teacher, train_loader, val_loader, config,
                epochs=epochs, use_mixup=True, run_name=run_name,
                teacher_cache=teacher_cache
            )
        elif kind == 'standard_kd':
            student = train_standard_kd(
                student, teacher, train_loader, val_loader, config,
                epochs=epochs, run_name=run_name, teacher_cache=teacher_cache
            )
        elif kind == 'messy_kd':
            student = train_messy_kd(
                student, teacher, train_loader, val_loader, config,
                epochs=epochs, run_name=run_name, teacher_cache=teacher_cache
            )
        elif kind == 'baseline':
            student = train_baseline(
                student, train_loader, val_loader, config,
                epochs=epochs, run_name=run_name
            )

        test_loss, test_acc, test_f1, _, _ = evaluate(
            student, test_loader, nn.CrossEntropyLoss(), config
        )
        results.append({'seed': seed, 'test_acc': test_acc, 'test_f1': test_f1})
        models.append(student)
        cleanup_memory()

    set_seed(config.seed)
    return results, models