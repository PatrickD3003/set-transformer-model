import torch   
import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import os
from modules_modified import ISAB, SAB, PMA, OrdinalHead
import pandas as pd
import openpyxl
from typing import Mapping, Iterable, Tuple, List, Union, Optional


# ----------------------------------------------- Classifier Model with XY-----------------------------------------------

# --- set transformer model ---
class SetTransformerClassifierXY(nn.Module):
    expects_tuple_input = True
    def __init__(self, vocab_size, dim_in=64, dim_hidden=128, num_heads=4, num_inds=16, num_classes=8, type_vec_dim=10):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim_in)  # hold embedding
        input_dim = dim_in + 1 + type_vec_dim + 2  # +1 for difficulty, +2 for (x, y)

        self.encoder = nn.Sequential(
            ISAB(input_dim, dim_hidden, num_heads, num_inds, ln=True),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=True),
        )

        self.decoder = nn.Sequential(
            PMA(dim_hidden, num_heads, 1, ln=True),
            nn.Flatten(start_dim=1),
            nn.Linear(dim_hidden, num_classes)
        )

    def forward(self, inputs):
        hold_idx, difficulty, type_tensor, xy_tensor = inputs  # shapes: (B, N), (B, N), (B, N, T), (B, N, 2)
        x_embed = self.embedding(hold_idx)              # (B, N, dim_in)
        difficulty = difficulty.unsqueeze(-1)           # (B, N, 1)
        x = torch.cat([x_embed, difficulty, type_tensor, xy_tensor], dim=-1)  # (B, N, D+1+T+2)
        x_enc = self.encoder(x)
        return self.decoder(x_enc)

# revised set transformer, embedding for type + XY
class SetTransformerClassifierXYAdditive(nn.Module):
    expects_tuple_input = True
    def __init__(self, vocab_size, feat_dim=64, dim_hidden=128, num_heads=4, num_inds=16, num_classes=8, type_vec_dim=10):
        super().__init__()
        # (1) hold ID → embedding
        self.hold_emb   = nn.Embedding(vocab_size, feat_dim)          # (B,N,feat_dim)
        # (2) difficulty (scalar) → linear projection to feat_dim
        self.diff_proj  = nn.Linear(1, feat_dim)                      # (B,N,1) → (B,N,feat_dim)
        # (3) type (multi-hot length T) → embedding matrix, then matmul
        self.type_emb   = nn.Parameter(torch.randn(type_vec_dim, feat_dim))  # (T,feat_dim)
        # (4) XY (2-dim) → MLP to feat_dim
        self.xy_mlp     = nn.Sequential(
                nn.Linear(2, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim)
            )
        # Learnable scalar weights for each feature
        self.w_hold = nn.Parameter(torch.tensor(1.0))
        self.w_diff = nn.Parameter(torch.tensor(1.0))
        self.w_type = nn.Parameter(torch.tensor(1.0))
        self.w_xy   = nn.Parameter(torch.tensor(1.0))

        self.encoder = nn.Sequential(
            ISAB(feat_dim, dim_hidden, num_heads, num_inds, ln=True),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=True),
        )
        self.decoder = nn.Sequential(
            PMA(dim_hidden, num_heads, 1, ln=True),
            nn.Flatten(start_dim=1),
            nn.Linear(dim_hidden, num_classes)
        )

    def forward(self, inputs):
        hold_idx, difficulty, type_tensor, xy_tensor = inputs
        # shapes: (B,N)  (B,N)  (B,N,T)  (B,N,2)
        # hold ID
        h  = self.hold_emb(hold_idx)                       # (B,N,feat_dim)
        # hold difficulty
        d  = difficulty.unsqueeze(-1)                     # (B,N,1)
        d  = self.diff_proj(d)                            # (B,N,feat_dim)
        # hold types
        t  = type_tensor @ self.type_emb                  # (B,N,feat_dim)
        # XY position
        xy = self.xy_mlp(xy_tensor)                       # (B,N,feat_dim)
        # element-wise addition
        # TODO: 重みを入れてみる 学習可能な重みを掛け算して
        # x = h + d + t + xy                                # (B,N,feat_dim)
        # Weighted sum
        x = self.w_hold * h + self.w_diff * d + self.w_type * t + self.w_xy * xy
        x_enc = self.encoder(x)
        return self.decoder(x_enc)
    
# --- deepset model ---
class DeepSetClassifierXY(nn.Module):
    expects_tuple_input = True
    def __init__(self, vocab_size, dim_in=64, dim_hidden=128, num_classes=8, type_vec_dim=10):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim_in)
        input_dim = dim_in + 1 + type_vec_dim + 2  # hold_emb + difficulty + type_vec + (x, y)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_classes)
        )

    def forward(self, inputs):
        hold_idx, difficulty, type_tensor, xy_tensor = inputs  # (B, N), (B, N), (B, N, T), (B, N, 2)

        x_embed = self.embedding(hold_idx)             # (B, N, dim_in)
        difficulty = difficulty.unsqueeze(-1)          # (B, N, 1)

        x = torch.cat([x_embed, difficulty, type_tensor, xy_tensor], dim=-1)  # (B, N, total_input_dim)
        x = self.encoder(x)                            # (B, N, hidden_dim)
        x = x.mean(dim=1)                              # (B, hidden_dim)
        return self.decoder(x)                         # (B, num_classes)
    
    
class DeepSetClassifierXYAdditive(nn.Module):
    expects_tuple_input = True
    def __init__(self, vocab_size, feat_dim=64, dim_hidden=128, num_classes=8, type_vec_dim=10):
        super().__init__()
        # (1) hold ID → embedding
        self.hold_emb   = nn.Embedding(vocab_size, feat_dim)           # (B,N,feat_dim)
        # (2) difficulty (scalar) → linear projection to feat_dim
        self.diff_proj  = nn.Linear(1, feat_dim)                       # (B,N,1) → (B,N,feat_dim)
        # (3) type (multi-hot length T) → embedding matrix, then matmul
        self.type_emb   = nn.Parameter(torch.randn(type_vec_dim, feat_dim))  # (T,feat_dim)
        # (4) XY (2-dim) → MLP to feat_dim
        self.xy_mlp     = nn.Sequential(
            nn.Linear(2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        # Learnable scalar weights for each feature
        self.w_hold = nn.Parameter(torch.tensor(1.0))
        self.w_diff = nn.Parameter(torch.tensor(1.0))
        self.w_type = nn.Parameter(torch.tensor(1.0))
        self.w_xy   = nn.Parameter(torch.tensor(1.0))

        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_classes)
        )

    def forward(self, inputs):
        hold_idx, difficulty, type_tensor, xy_tensor = inputs  # (B, N), (B, N), (B, N, T), (B, N, 2)

        # Embed hold indices
        h = self.hold_emb(hold_idx)                    # (B, N, feat_dim)
        # Project difficulty
        d = self.diff_proj(difficulty.unsqueeze(-1))   # (B, N, feat_dim)
        # Embed type (multi-hot)
        t = type_tensor @ self.type_emb                # (B, N, feat_dim)
        # Process XY
        xy = self.xy_mlp(xy_tensor)                    # (B, N, feat_dim)

        # Weighted additive fusion
        x = self.w_hold * h + self.w_diff * d + self.w_type * t + self.w_xy * xy  # (B, N, feat_dim)

        # Encode + aggregate
        x = self.encoder(x)       # (B, N, hidden_dim)
        x = x.mean(dim=1)         # (B, hidden_dim)
        return self.decoder(x)    # (B, num_classes)


# -----------------------------------------------Classifier Model Original-----------------------------------------------

# --- set transformer model ---
class SetTransformerClassifier(nn.Module):
    expects_tuple_input = False
    def __init__(self, vocab_size, dim_in=64, dim_hidden=128, num_heads=4, num_inds=16, num_classes=8):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, dim_in)
        self.encoder = nn.Sequential(
            ISAB(dim_in, dim_hidden, num_heads, num_inds, ln=True),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=True),
        )
        self.decoder = nn.Sequential(
            PMA(dim_hidden, num_heads, 1, ln=True),
            nn.Flatten(start_dim=1),
            nn.Linear(dim_hidden, num_classes)
        )

    def forward(self, x):
        # x: (B, N, dim_in)
        x = self.embedding(x)
        x_enc = self.encoder(x)
        return self.decoder(x_enc)
    
# --- deepset model ---
class DeepSetClassifier(nn.Module):
    expects_tuple_input = False
    def __init__(self, vocab_size, dim_in=64, dim_hidden=128, num_classes=8):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim_in)  # Embed hold indices

        self.encoder = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_classes)
        )

    def forward(self, hold_idx):  # hold_idx: (B, N)
        x = self.embedding(hold_idx)   # (B, N, dim_in)
        x = self.encoder(x)            # (B, N, hidden)
        x = x.mean(dim=1)              # (B, hidden)
        out = self.decoder(x)          # (B, num_classes)
        return out


# -----------------------------------------------Ordinal Models -------------------------------------------------

class SetTransformerOrdinalXY(nn.Module):
    expects_tuple_input = True
    def __init__(self, vocab_size, dim_in=64, dim_hidden=128, num_heads=4, num_inds=16, num_classes=8, type_vec_dim=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_in)
        input_dim = dim_in + 1 + type_vec_dim + 2
        self.encoder = nn.Sequential(
            ISAB(input_dim, dim_hidden, num_heads, num_inds, ln=True),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=True),
        )
        self.pool = nn.Sequential(
            PMA(dim_hidden, num_heads, 1, ln=True),
            nn.Flatten(start_dim=1),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )
        self.ordinal_head = OrdinalHead(dim_hidden, num_classes)

    def forward(self, inputs):
        hold_idx, difficulty, type_tensor, xy_tensor = inputs
        x_embed = self.embedding(hold_idx)
        difficulty = difficulty.unsqueeze(-1)
        x = torch.cat([x_embed, difficulty, type_tensor, xy_tensor], dim=-1)
        x_enc = self.encoder(x)
        features = self.pool(x_enc)
        return self.ordinal_head(features)


class SetTransformerOrdinalXYAdditive(nn.Module):
    expects_tuple_input = True
    def __init__(self, vocab_size, feat_dim=64, dim_hidden=128, num_heads=4, num_inds=16, num_classes=8, type_vec_dim=10):
        super().__init__()
        self.hold_emb = nn.Embedding(vocab_size, feat_dim)
        self.diff_proj = nn.Linear(1, feat_dim)
        self.type_emb = nn.Parameter(torch.randn(type_vec_dim, feat_dim))
        self.xy_mlp = nn.Sequential(
            nn.Linear(2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        self.w_hold = nn.Parameter(torch.tensor(1.0))
        self.w_diff = nn.Parameter(torch.tensor(1.0))
        self.w_type = nn.Parameter(torch.tensor(1.0))
        self.w_xy = nn.Parameter(torch.tensor(1.0))
        self.encoder = nn.Sequential(
            ISAB(feat_dim, dim_hidden, num_heads, num_inds, ln=True),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=True),
        )
        self.pool = nn.Sequential(
            PMA(dim_hidden, num_heads, 1, ln=True),
            nn.Flatten(start_dim=1),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )
        self.ordinal_head = OrdinalHead(dim_hidden, num_classes)

    def forward(self, inputs):
        hold_idx, difficulty, type_tensor, xy_tensor = inputs
        h = self.hold_emb(hold_idx)
        d = self.diff_proj(difficulty.unsqueeze(-1))
        t = type_tensor @ self.type_emb
        xy = self.xy_mlp(xy_tensor)
        x = self.w_hold * h + self.w_diff * d + self.w_type * t + self.w_xy * xy
        x_enc = self.encoder(x)
        features = self.pool(x_enc)
        return self.ordinal_head(features)


class SetTransformerOrdinal(nn.Module):
    expects_tuple_input = False
    def __init__(self, vocab_size, dim_in=64, dim_hidden=128, num_heads=4, num_inds=16, num_classes=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_in)
        self.encoder = nn.Sequential(
            ISAB(dim_in, dim_hidden, num_heads, num_inds, ln=True),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=True),
        )
        self.pool = nn.Sequential(
            PMA(dim_hidden, num_heads, 1, ln=True),
            nn.Flatten(start_dim=1),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )
        self.ordinal_head = OrdinalHead(dim_hidden, num_classes)

    def forward(self, hold_idx):
        x = self.embedding(hold_idx)
        x_enc = self.encoder(x)
        features = self.pool(x_enc)
        return self.ordinal_head(features)


class DeepSetOrdinalXY(nn.Module):
    expects_tuple_input = True
    def __init__(self, vocab_size, dim_in=64, dim_hidden=128, num_classes=8, type_vec_dim=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_in)
        input_dim = dim_in + 1 + type_vec_dim + 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.proj = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )
        self.ordinal_head = OrdinalHead(dim_hidden, num_classes)

    def forward(self, inputs):
        hold_idx, difficulty, type_tensor, xy_tensor = inputs
        x_embed = self.embedding(hold_idx)
        difficulty = difficulty.unsqueeze(-1)
        x = torch.cat([x_embed, difficulty, type_tensor, xy_tensor], dim=-1)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        features = self.proj(pooled)
        return self.ordinal_head(features)


class DeepSetOrdinalXYAdditive(nn.Module):
    expects_tuple_input = True
    def __init__(self, vocab_size, feat_dim=64, dim_hidden=128, num_classes=8, type_vec_dim=10):
        super().__init__()
        self.hold_emb = nn.Embedding(vocab_size, feat_dim)
        self.diff_proj = nn.Linear(1, feat_dim)
        self.type_emb = nn.Parameter(torch.randn(type_vec_dim, feat_dim))
        self.xy_mlp = nn.Sequential(
            nn.Linear(2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        self.w_hold = nn.Parameter(torch.tensor(1.0))
        self.w_diff = nn.Parameter(torch.tensor(1.0))
        self.w_type = nn.Parameter(torch.tensor(1.0))
        self.w_xy = nn.Parameter(torch.tensor(1.0))
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.proj = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )
        self.ordinal_head = OrdinalHead(dim_hidden, num_classes)

    def forward(self, inputs):
        hold_idx, difficulty, type_tensor, xy_tensor = inputs
        h = self.hold_emb(hold_idx)
        d = self.diff_proj(difficulty.unsqueeze(-1))
        t = type_tensor @ self.type_emb
        xy = self.xy_mlp(xy_tensor)
        x = self.w_hold * h + self.w_diff * d + self.w_type * t + self.w_xy * xy
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        features = self.proj(pooled)
        return self.ordinal_head(features)


class DeepSetOrdinal(nn.Module):
    expects_tuple_input = False
    def __init__(self, vocab_size, dim_in=64, dim_hidden=128, num_classes=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_in)
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.proj = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )
        self.ordinal_head = OrdinalHead(dim_hidden, num_classes)

    def forward(self, hold_idx):
        x = self.embedding(hold_idx)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        features = self.proj(pooled)
        return self.ordinal_head(features)


# -----------------------------------------------ensemble model-----------------------------------------------
TensorLike = Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]

# =========================
# Base + Helpers
# =========================
class BaseEnsemble(nn.Module):
    """
    Base class for classification ensembles that combine member *probabilities*.
    Subclasses should implement `_combine(probs: Tensor [M,B,C], weights: Tensor [M]) -> Tensor [B,C]`.
    - Members can declare `expects_tuple_input=True` to receive tuple inputs; otherwise they get inputs[0].
    - Members may return a tensor (logits) or a tuple where the first tensor-like is treated as logits.
    """

    def __init__(
        self,
        models: Union[Mapping[str, nn.Module], Iterable[Tuple[str, nn.Module]]],
        weights: Optional[Union[Mapping[str, float], Iterable[float]]] = None,
        freeze_members: bool = True,
    ):
        super().__init__()

        if isinstance(models, dict):
            items = list(models.items())
        else:
            items = list(models)
            if not all(isinstance(it, (tuple, list)) and len(it) == 2 for it in items):
                raise ValueError("models must be a mapping or iterable of (name, module) pairs.")
        if not items:
            raise ValueError("At least one base model is required for an ensemble.")

        self._model_names = [name for name, _ in items]
        self.models = nn.ModuleDict((name, module) for name, module in items)

        w = self._resolve_weights(weights, self._model_names)
        self.register_buffer("_weights", w)  # shape [M], sums to 1

        if freeze_members:
            for m in self.models.values():
                for p in m.parameters():
                    p.requires_grad_(False)

        # Hints for your pipeline
        self.is_ensemble = True
        self.is_ordinal = False

    @staticmethod
    def _resolve_weights(weights, names):
        if weights is None:
            vals = torch.ones(len(names), dtype=torch.float32)
        elif isinstance(weights, dict):
            vals = torch.tensor([float(weights[n]) for n in names], dtype=torch.float32)
        else:
            vals = torch.tensor(list(weights), dtype=torch.float32)
            if vals.numel() != len(names):
                raise ValueError("weights length must match number of models.")
        s = vals.sum()
        if s <= 0:
            raise ValueError("Ensemble weights must sum to a positive value.")
        return vals / s

    @staticmethod
    def _extract_logits(outputs: TensorLike) -> torch.Tensor:
        """Pick the first tensor from outputs if it's a tuple; otherwise return outputs."""
        if isinstance(outputs, (tuple, list)):
            for x in outputs:
                if isinstance(x, torch.Tensor):
                    return x
            raise ValueError("Model returned a tuple/list without any tensor outputs.")
        if not isinstance(outputs, torch.Tensor):
            raise ValueError(f"Unsupported model output type: {type(outputs)}")
        return outputs

    @staticmethod
    def _prepare_inputs(module: nn.Module, raw_inputs: TensorLike) -> TensorLike:
        expects_tuple = getattr(module, "expects_tuple_input", False)
        if expects_tuple:
            if not isinstance(raw_inputs, (tuple, list)):
                raise ValueError(
                    f"Model {module.__class__.__name__} expects tuple inputs but received {type(raw_inputs)}"
                )
            return raw_inputs
        # Non-tuple model gets the first element (hold_idx)
        if isinstance(raw_inputs, (tuple, list)):
            if not raw_inputs:
                raise ValueError("Empty inputs given to ensemble.")
            return raw_inputs[0]
        return raw_inputs

    def _member_probs(self, inputs: TensorLike) -> torch.Tensor:
        """
        Run each member, collect *probabilities* stack of shape [M, B, C].
        """
        tuple_inputs = tuple(inputs) if isinstance(inputs, list) else inputs
        probs_list = []
        for module in self.models.values():
            prepared = self._prepare_inputs(module, tuple_inputs)
            outputs = module(prepared)
            logits = self._extract_logits(outputs)
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)
        # [M, B, C]
        return torch.stack(probs_list, dim=0)

    # --- abstract-ish: combine member probs into [B, C]
    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, inputs: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            probs:  [B, C]
            logits: [B, C] (log of probs for convenience)
        """
        member_probs = self._member_probs(inputs)    # [M, B, C]
        combined = self._combine(member_probs, self._weights)  # [B, C]
        # ensure normalization & numeric safety
        combined = combined.clamp_min(1e-12)
        combined = combined / combined.sum(dim=-1, keepdim=True)
        logits = torch.log(combined)
        return combined, logits

    def predict(self, inputs: TensorLike) -> torch.Tensor:
        probs, _ = self.forward(inputs)
        return probs.argmax(dim=-1)


# =========================
# 1) Soft Voting (Arithmetic Mean)
# =========================
class SoftVotingEnsemble(BaseEnsemble):
    """Weighted arithmetic mean of member probabilities (classic soft voting)."""
    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # member_probs: [M,B,C], weights: [M]
        w = weights.view(-1, 1, 1)
        return (member_probs * w).sum(dim=0)  # [B,C]


# =========================
# 2) Geometric Mean
# =========================
class GeometricMeanEnsemble(BaseEnsemble):
    """
    Weighted geometric mean of probabilities:
    exp( sum_i w_i * log(p_i) ), then renormalize.
    Tends to penalize classes that any member rates near zero.
    """
    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        logp = torch.log(member_probs.clamp_min(eps))  # [M,B,C]
        w = weights.view(-1, 1, 1)                     # [M,1,1]
        g = torch.exp((w * logp).sum(dim=0))           # [B,C]
        return g


# =========================
# 3) Median Ensemble
# =========================
class MedianEnsemble(BaseEnsemble):
    """
    Element-wise median of probabilities across members.
    NOTE: Ignores weights (robust estimator).
    """
    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # weights are ignored by design for median
        return member_probs.median(dim=0).values  # [B,C]


# =========================
# 4) Trimmed Mean Ensemble
# =========================
class TrimmedMeanEnsemble(BaseEnsemble):
    """
    Per-class trimmed mean across members.
    - trim_frac: fraction to drop from both ends (0 <= trim_frac < 0.5).
    Ignores weights (robust estimator).
    """

    def __init__(self, models, weights=None, freeze_members=True, trim_frac: float = 0.2):
        super().__init__(models, weights=weights, freeze_members=freeze_members)
        if not (0.0 <= trim_frac < 0.5):
            raise ValueError("trim_frac must be in [0, 0.5).")
        self.trim_frac = float(trim_frac)

    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        M = member_probs.shape[0]
        k = int(M * self.trim_frac)
        if k == 0:
            # falls back to simple mean (unweighted) if nothing to trim
            return member_probs.mean(dim=0)
        # sort along member axis for each (B,C)
        sorted_probs, _ = torch.sort(member_probs, dim=0)  # [M,B,C]
        trimmed = sorted_probs[k: M - k]                   # [M-2k,B,C]
        return trimmed.mean(dim=0)                         # [B,C]


# =========================
# 5) Stacking Ensemble (meta-learner)
# =========================
class StackingEnsemble(BaseEnsemble):
    """
    Stacking: feed member outputs to a meta-learner.
    - meta_model: nn.Module mapping features -> logits [B,C]
    - feature_source: 'logits' or 'probs' (what to feed)
    - combine: 'concat' (default) to concatenate member features along last dim,
               or 'mean' to average features before meta-model.

    Typical usage:
        meta = nn.Linear(M*C, C)   # simple logistic regression layer
        ens  = StackingEnsemble(models, meta_model=meta, feature_source='logits', combine='concat')

    Train the `meta_model` on a held-out validation set using the frozen base models.
    """

    def __init__(
        self,
        models,
        weights=None,
        freeze_members=True,
        meta_model: Optional[nn.Module] = None,
        feature_source: str = "logits",   # 'logits' or 'probs'
        combine: str = "concat",          # 'concat' or 'mean'
    ):
        super().__init__(models, weights=weights, freeze_members=freeze_members)
        if feature_source not in {"logits", "probs"}:
            raise ValueError("feature_source must be 'logits' or 'probs'.")
        if combine not in {"concat", "mean"}:
            raise ValueError("combine must be 'concat' or 'mean'.")
        if meta_model is None:
            raise ValueError("StackingEnsemble requires a meta_model.")
        self.meta_model = meta_model
        self.feature_source = feature_source
        self.combine = combine

    def _member_features(self, inputs: TensorLike) -> torch.Tensor:
        """
        Collect either logits or probs per member.
        Returns [M,B,C]
        """
        tuple_inputs = tuple(inputs) if isinstance(inputs, list) else inputs
        feats = []
        for module in self.models.values():
            prepared = self._prepare_inputs(module, tuple_inputs)
            out = module(prepared)
            logits = self._extract_logits(out)
            if self.feature_source == "probs":
                feats.append(F.softmax(logits, dim=-1))
            else:
                feats.append(logits)
        return torch.stack(feats, dim=0)  # [M,B,C]

    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Not used; forward is overridden to use meta_model.
        Implemented to satisfy BaseEnsemble API; returns arithmetic mean.
        """
        w = weights.view(-1, 1, 1)
        return (member_probs * w).sum(dim=0)

    def forward(self, inputs: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
        member_feats = self._member_features(inputs)  # [M,B,C]
        M, B, C = member_feats.shape

        if self.combine == "mean":
            feat = member_feats.mean(dim=0)           # [B,C]
        else:
            feat = member_feats.permute(1, 0, 2).reshape(B, M * C)  # [B, M*C]

        logits = self.meta_model(feat)                # [B,C]
        probs = F.softmax(logits, dim=-1)
        return probs, logits