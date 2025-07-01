import torch    
import numpy as np
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import os
from modules_modified import ISAB, SAB, PMA
import pandas as pd
import openpyxl


# Classifier Model -----------------------------------------------

# --- set transformer model ---
class SetTransformerClassifier(nn.Module):
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

        # 全部embedding
        # positional encoding
        # concatではなく、足し算する
        # entity embedding
        x = torch.cat([x_embed, difficulty, type_tensor, xy_tensor], dim=-1)  # (B, N, D+1+T+2)
        x_enc = self.encoder(x)
        return self.decoder(x_enc)

# revised settransformer, embedding for type + XY
class SetTransformerAdditive(nn.Module):
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

        h  = self.hold_emb(hold_idx)                       # (B,N,feat_dim)

        d  = difficulty.unsqueeze(-1)                     # (B,N,1)
        d  = self.diff_proj(d)                            # (B,N,feat_dim)

        t  = type_tensor @ self.type_emb                  # (B,N,feat_dim)

        xy = self.xy_mlp(xy_tensor)                       # (B,N,feat_dim)

        # 🔑 element-wise addition
        x = h + d + t + xy                                # (B,N,feat_dim)

        x_enc = self.encoder(x)
        return self.decoder(x_enc)
    
# --- deepset model ---
class DeepSetClassifier(nn.Module):
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


