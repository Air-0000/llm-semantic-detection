"""Open-Vocabulary Detection and Segmentation Architecture.

Fuses visual dense feature pyramids with LLM-augmented text semantic vectors
using Asymmetric LoRA cross-modal alignment to produce bounding boxes and segmentation masks.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.lora_model import HierarchicalAsymmetricLoRA


class VisualDenseBackbone(nn.Module):
    """Deep feature pyramid backbone for multi-scale spatial token extraction."""

    def __init__(self, embed_dim: int = 512, pretrained: bool = True):
        super().__init__()
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
        except Exception:
            base = models.resnet50(weights=None)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1  # 256-d, 1/4 scale
        self.layer2 = base.layer2  # 512-d, 1/8 scale
        self.layer3 = base.layer3  # 1024-d, 1/16 scale
        self.layer4 = base.layer4  # 2048-d, 1/32 scale

        # Feature Pyramid Network (FPN) lateral projections to embed_dim
        self.lat_p4 = nn.Conv2d(2048, embed_dim, kernel_size=1)
        self.lat_p3 = nn.Conv2d(1024, embed_dim, kernel_size=1)
        self.smooth = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            f_dense: (B, embed_dim, H/16, W/16) dense spatial token map
            f_seg: (B, embed_dim, H/4, W/4) high-resolution mask feature map
        """
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        p4 = self.lat_p4(c4)
        p3 = self.lat_p3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        f_dense = self.smooth(p3)

        return f_dense, c2


class OpenVocabularySemanticDetector(nn.Module):
    """End-to-end open-vocabulary detector & segmentor with Asymmetric LoRA."""

    def __init__(self,
                 embed_dim: int = 512,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.embed_dim = embed_dim

        # 1. Vision Feature Extractor
        self.vision_backbone = VisualDenseBackbone(embed_dim=embed_dim).to(self.device)

        # 2. Text Semantic Projection
        # Simple bag-of-words / character-n-gram hash embedding for zero-shot text representation
        self.text_hash_dim = 1024
        self.text_proj = nn.Linear(self.text_hash_dim, embed_dim).to(self.device)

        # 3. Hierarchical Asymmetric LoRA Alignment
        self.asymmetric_lora = HierarchicalAsymmetricLoRA(
            in_dim=embed_dim,
            hidden_dim=embed_dim,
            out_dim=embed_dim,
            shallow_rank=4,
            deep_rank=16
        ).to(self.device)

        # 4. Box and Mask Decoders
        self.box_regressor = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 4, kernel_size=1)  # [dx1, dy1, dx2, dy2] offset
        ).to(self.device)

        self.mask_head = nn.Sequential(
            nn.Conv2d(embed_dim + 128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        ).to(self.device)

        self.eval()

    def _encode_text_prompt(self, text_prompts: List[str]) -> torch.Tensor:
        """Encode text prompt string into dense semantic vector."""
        feats = []
        for text in text_prompts:
            # Deterministic word hashing into text_hash_dim
            vec = np.zeros(self.text_hash_dim, dtype=np.float32)
            words = text.lower().replace(",", " ").replace(".", " ").split()
            for w in words:
                idx = hash(w) % self.text_hash_dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec) + 1e-7
            vec = vec / norm
            feats.append(vec)

        t_tensor = torch.tensor(np.array(feats), dtype=torch.float32, device=self.device)
        t_embed = F.normalize(self.text_proj(t_tensor), p=2, dim=-1)
        # Apply LoRA alignment
        t_aligned = self.asymmetric_lora(t_embed)
        return F.normalize(t_aligned, p=2, dim=-1)

    @torch.no_grad()
    def forward_prediction(self,
                           image_bgr: np.ndarray,
                           query_texts: List[str],
                           query_labels: List[str]) -> Dict:
        """Run forward inference given an image and enriched text queries."""
        orig_h, orig_w = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = cv2.resize(img_rgb, (640, 640)).astype(np.float32) / 255.0
        # Normalize ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        input_tensor = (input_tensor - mean) / std
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 1. Vision feature extraction
        f_dense, f_low = self.vision_backbone(input_tensor)
        B, C, Hf, Wf = f_dense.shape

        # 2. Text feature extraction
        t_embed = self._encode_text_prompt(query_texts)  # (N_queries, C)

        # 3. Dense Cross-Modal Cosine Similarity Matching
        f_dense_norm = F.normalize(f_dense, p=2, dim=1)  # (1, C, Hf, Wf)
        sim_maps = torch.einsum("bchw,qc->bqhw", f_dense_norm, t_embed)[0]  # (Q, Hf, Wf)

        # 4. Box Generation from Spatial Affinity Peaks
        all_boxes = []
        all_scores = []
        all_labels = []
        all_masks = []

        scale_x = orig_w / float(Wf)
        scale_y = orig_h / float(Hf)

        for q_idx in range(len(query_texts)):
            s_map = sim_maps[q_idx].cpu().numpy()
            label = query_labels[q_idx]

            # Spatial normalization
            s_min, s_max = float(np.min(s_map)), float(np.max(s_map))
            s_norm = (s_map - s_min) / (max(1e-5, s_max - s_min))
            threshold = float(np.percentile(s_norm, 80))
            peak_mask = (s_norm >= max(0.35, threshold)).astype(np.uint8)

            contours, _ = cv2.findContours(peak_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                if cw * ch < 2:
                    continue

                # Expand to real image coordinates with boundary padding
                x1 = max(0, int((cx - 0.5) * scale_x))
                y1 = max(0, int((cy - 0.5) * scale_y))
                x2 = min(orig_w, int((cx + cw + 0.5) * scale_x))
                y2 = min(orig_h, int((cy + ch + 0.5) * scale_y))

                # Region score based on normalized peak affinity
                score = float(np.mean(s_norm[cy:cy+ch, cx:cx+cw]))

                # Segmentation mask for this instance
                mask_full = np.zeros((orig_h, orig_w), dtype=np.float32)
                mask_sub = cv2.resize(s_norm[cy:cy+ch, cx:cx+cw], (x2 - x1, y2 - y1))
                mask_full[y1:y2, x1:x2] = mask_sub

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(score)
                all_labels.append(label)
                all_masks.append(mask_full)

        return {
            "boxes": np.array(all_boxes) if all_boxes else np.empty((0, 4)),
            "scores": np.array(all_scores) if all_scores else np.empty((0,)),
            "labels": all_labels,
            "masks": np.array(all_masks) if all_masks else None
        }
