"""Hierarchical Asymmetric LoRA (Low-Rank Adaptation) Module.

Implements asymmetric rank allocation across network depth:
- Shallow layers: Low Rank (r=4) to preserve base open-vocabulary zero-shot generalization.
- Deep layers: High Rank (r=16) for expressive cross-modal feature alignment.
"""

import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoRALinear(nn.Module):
    """Linear layer wrapped with low-rank adaptation matrices A and B."""

    def __init__(self,
                 base_linear: nn.Linear,
                 rank: int = 4,
                 lora_alpha: float = 16.0,
                 lora_dropout: float = 0.05,
                 layer_depth: str = "shallow"):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(1, rank)
        self.layer_depth = layer_depth

        # Base linear layer (frozen during LoRA adaptation)
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        if rank > 0:
            # Asymmetric LoRA matrices
            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)
            self.dropout = nn.Identity()

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self.rank == 0 or self.merged:
            return base_out
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def merge_weights(self):
        """Merge LoRA delta into base weight for zero-latency inference."""
        if self.rank > 0 and not self.merged:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base.weight.data += delta
            self.merged = True

    def unmerge_weights(self):
        """Revert merged weights back to independent LoRA matrices."""
        if self.rank > 0 and self.merged:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base.weight.data -= delta
            self.merged = False


class HierarchicalAsymmetricLoRA(nn.Module):
    """Applies depth-dependent asymmetric rank allocation across encoder blocks."""

    def __init__(self,
                 in_dim: int = 512,
                 hidden_dim: int = 512,
                 out_dim: int = 512,
                 shallow_rank: int = 4,
                 deep_rank: int = 16,
                 lora_alpha: float = 32.0,
                 dropout: float = 0.05):
        super().__init__()
        self.shallow_rank = shallow_rank
        self.deep_rank = deep_rank

        # Stage 1: Shallow Projection (Low Rank to preserve base vocabulary)
        base_shallow = nn.Linear(in_dim, hidden_dim)
        self.shallow_proj = AsymmetricLoRALinear(
            base_shallow, rank=shallow_rank, lora_alpha=lora_alpha,
            lora_dropout=dropout, layer_depth="shallow"
        )
        self.act = nn.GELU()

        # Stage 2: Deep Cross-Modal Fusion (High Rank for task adaptation)
        base_deep = nn.Linear(hidden_dim, out_dim)
        self.deep_proj = AsymmetricLoRALinear(
            base_deep, rank=deep_rank, lora_alpha=lora_alpha,
            lora_dropout=dropout, layer_depth="deep"
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.shallow_proj(x))
        out = self.norm(self.deep_proj(h) + x)
        return out

    def get_trainable_param_count(self) -> Dict[str, int]:
        """Count total vs trainable LoRA parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_params": total, "trainable_lora_params": trainable, "ratio_pct": round(trainable / total * 100, 2)}
