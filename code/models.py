"""
Teacher and student model definitions.
"""
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from torchvision.models import mobilenet_v3_small, vit_b_16
from torchvision.models import MobileNet_V3_Small_Weights, ViT_B_16_Weights

from code.config import Config
from code.logger import logger


# ---------- TEACHER COMPONENTS ----------

class AudioLatentGenerator(nn.Module):
    def __init__(self, input_dim: int = 512, latent_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, latent_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ThreeDLatentGenerator(nn.Module):
    def __init__(self, input_dim: int = 512, latent_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, latent_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalTeacher(nn.Module):
    """ViT-based teacher with synthetic audio and 3D heads."""

    def __init__(self, num_classes: int = 100, visual_dim: int = 768,
                 audio_dim: int = 512, three_d_dim: int = 1024,
                 latent_dim: int = 512):
        super().__init__()

        try:
            self.visual_backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        except Exception:
            self.visual_backbone = vit_b_16(weights=None)

        try:
            self.visual_backbone.heads = nn.Identity()
        except AttributeError:
            try:
                self.visual_backbone.head = nn.Identity()
            except AttributeError:
                pass

        self.visual_proj = nn.Linear(visual_dim, latent_dim)

        self.audio_generator = AudioLatentGenerator(input_dim=latent_dim, latent_dim=audio_dim)
        self.audio_proj = nn.Linear(audio_dim, latent_dim)

        self.three_d_generator = ThreeDLatentGenerator(input_dim=latent_dim, latent_dim=three_d_dim)
        self.three_d_proj = nn.Linear(three_d_dim, latent_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=8, batch_first=True, dropout=0.1
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        visual_feats = self.visual_backbone(x)
        visual_feats = self.visual_proj(visual_feats)

        audio_latents = self.audio_generator(visual_feats)
        audio_feats = self.audio_proj(audio_latents)

        three_d_latents = self.three_d_generator(visual_feats)
        three_d_feats = self.three_d_proj(three_d_latents)

        multimodal = torch.stack([visual_feats, audio_feats, three_d_feats], dim=1)
        fused, _ = self.cross_attn(
            query=visual_feats.unsqueeze(1), key=multimodal, value=multimodal
        )
        fused = fused.squeeze(1)

        logits = self.classifier(fused)

        if return_features:
            return logits, {
                'visual': visual_feats,
                'audio': audio_feats,
                'three_d': three_d_feats
            }
        return logits

    def get_privileged_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features without gradients for distillation."""
        with torch.no_grad():
            visual_feats = self.visual_backbone(x)
            visual_feats = self.visual_proj(visual_feats)

            audio_latents = self.audio_generator(visual_feats)
            audio_feats = self.audio_proj(audio_latents)

            three_d_latents = self.three_d_generator(visual_feats)
            three_d_feats = self.three_d_proj(three_d_latents)

            return {
                'visual': visual_feats.detach(),
                'audio': audio_feats.detach(),
                'three_d': three_d_feats.detach()
            }


# ---------- STUDENT ----------

class LightweightStudent(nn.Module):
    """MobileNetV3 student with MSFD prediction heads."""

    def __init__(self, num_classes: int = 100, student_dim: int = 256,
                 teacher_dim: int = 512, active_heads: Optional[List[str]] = None):
        super().__init__()

        try:
            self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        except Exception:
            self.backbone = mobilenet_v3_small(weights=None)

        self.backbone.classifier = nn.Identity()

        self.student_proj = nn.Sequential(
            nn.Linear(576, student_dim),
            nn.ReLU(inplace=True)
        )

        self.active_heads = active_heads or ['visual', 'audio', '3d']
        self.heads = nn.ModuleDict()

        for head_name in self.active_heads:
            self.heads[head_name] = nn.Sequential(
                nn.Linear(student_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, teacher_dim)
            )

        self.classifier = nn.Linear(student_dim, num_classes)

    def forward(self, x: torch.Tensor, return_predictions: bool = False):
        features = self.backbone(x)
        student_feats = self.student_proj(features)

        predictions = {}
        for head_name, head in self.heads.items():
            predictions[f'pred_{head_name}'] = head(student_feats)

        logits = self.classifier(student_feats)

        if return_predictions:
            return logits, predictions
        return logits

    def get_privileged_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Allows a trained student to act as 'teacher' for progressive distillation."""
        with torch.no_grad():
            features = self.backbone(x)
            student_feats = self.student_proj(features)
            out = {}
            for head_name, head in self.heads.items():
                out[head_name] = head(student_feats).detach()
            return out


# ---------- TEACHER FREEZING ----------

def freeze_vit_backbone_except_last_n(teacher: MultimodalTeacher, n: int) -> None:
    """Freeze all ViT encoder blocks except the last n."""
    if n <= 0:
        return
    vit = teacher.visual_backbone
    for p in vit.parameters():
        p.requires_grad = False
    blocks = list(vit.encoder.layers)
    for block in blocks[-n:]:
        for p in block.parameters():
            p.requires_grad = True
    if hasattr(vit.encoder, 'ln'):
        for p in vit.encoder.ln.parameters():
            p.requires_grad = True
    trainable = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    total = sum(p.numel() for p in teacher.parameters())
    logger.log(f"[FREEZE] ViT backbone: last {n} of {len(blocks)} blocks trainable "
               f"({trainable:,}/{total:,} params trainable, {trainable/total*100:.1f}%)")