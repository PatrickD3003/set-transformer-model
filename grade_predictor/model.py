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


# Classifier Model -----------------------------------------------
# with XY -----------------------------------------------

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


# Classifier Model Original-----------------------------------------------

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


class EnsembleClassifier(nn.Module):
    """
    Soft-voting ensemble that averages member model probabilities.
    Each member declares whether it expects tuple inputs via an `expects_tuple_input` attribute.
    """

    def __init__(self, models, weights=None, freeze_members=True):
        """
        Args:
            models: Mapping[str, nn.Module] or iterable of (name, nn.Module).
            weights: Optional mapping or iterable of weights aligned with `models`.
            freeze_members: If True, disable gradient updates on member parameters.
        """
        super().__init__()

        if isinstance(models, dict):
            items = list(models.items())
        else:
            items = list(models)
            if not all(isinstance(item, (tuple, list)) and len(item) == 2 for item in items):
                raise ValueError("models must be a mapping or iterable of (name, module) pairs.")

        if not items:
            raise ValueError("At least one base model is required for EnsembleClassifier.")

        self._model_names = [name for name, _ in items]
        self.models = nn.ModuleDict((name, module) for name, module in items)

        weight_values = self._resolve_weights(weights, self._model_names)
        self.register_buffer("_weights", weight_values)

        if freeze_members:
            for module in self.models.values():
                for param in module.parameters():
                    param.requires_grad_(False)

        self.is_ensemble = True
        self.is_ordinal = False

    @staticmethod
    def _resolve_weights(weights, names):
        if weights is None:
            values = torch.ones(len(names), dtype=torch.float32)
        elif isinstance(weights, dict):
            try:
                values = torch.tensor([float(weights[name]) for name in names], dtype=torch.float32)
            except KeyError as exc:
                missing = exc.args[0]
                raise KeyError(f"Missing weight for ensemble member: {missing}") from exc
        else:
            values = torch.tensor(list(weights), dtype=torch.float32)
            if values.numel() != len(names):
                raise ValueError("weights iterable length must match number of models.")

        total = values.sum()
        if total <= 0:
            raise ValueError("Ensemble weights must sum to a positive value.")
        return values / total

    def _prepare_inputs(self, module, raw_inputs):
        expects_tuple = getattr(module, "expects_tuple_input", False)
        if expects_tuple:
            if not isinstance(raw_inputs, (tuple, list)):
                raise ValueError(
                    f"Model {module.__class__.__name__} expects tuple inputs but received {type(raw_inputs)}"
                )
            return raw_inputs

        if isinstance(raw_inputs, (tuple, list)):
            if not raw_inputs:
                raise ValueError("Received empty inputs when at least the hold indices tensor is required.")
            return raw_inputs[0]
        return raw_inputs

    def forward(self, inputs):
        tuple_inputs = tuple(inputs) if isinstance(inputs, list) else inputs

        weighted_prob = None
        for idx, module in enumerate(self.models.values()):
            prepared = self._prepare_inputs(module, tuple_inputs)
            outputs = module(prepared)

            if isinstance(outputs, tuple):
                logits = outputs[1] if len(outputs) > 1 else outputs[0]
            else:
                logits = outputs

            probs = F.softmax(logits, dim=-1)
            weight = self._weights[idx]
            contrib = probs * weight
            weighted_prob = contrib if weighted_prob is None else weighted_prob + contrib

        avg_prob = weighted_prob
        avg_logits = torch.log(avg_prob.clamp_min(1e-12))
        return avg_prob, avg_logits

    def predict(self, inputs):
        probs, _ = self.forward(inputs)
        return probs.argmax(dim=-1)


# Ordinal Models -------------------------------------------------

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
