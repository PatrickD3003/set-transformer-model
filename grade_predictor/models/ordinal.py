import torch   
import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import os
from ..modules_modified import ISAB, SAB, PMA, OrdinalHead
import pandas as pd
import openpyxl
import warnings
from typing import Mapping, Iterable, Tuple, List, Union, Optional

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
