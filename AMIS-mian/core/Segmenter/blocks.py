"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
from einops import rearrange
from pathlib import Path

import torch.nn.functional as F

from timm.models.layers import DropPath


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None
        self.v = None
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    @property
    def unwrapped(self):
        return self
    
    def get_attn(self):
        return self.attn
    
    def forward(self, x, general_token=None):
        B, N, C = x.shape  # [B, 1025, 768]
        if general_token is not None:
            # 拼接 x 和 general_token
            general_token = general_token.expand(B, -1, -1)
            general_token = self.proj(general_token)
            general_token = self.norm(general_token)
            kv = torch.cat((x, general_token), dim=1)  # [B, N+1, C]
            kv = self.qkv(kv).reshape(B, kv.shape[1], 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
            q, k, v = kv[0], kv[1], kv[2]
        else:
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.heads, C // self.heads)
                .permute(2, 0, 3, 1, 4)  # qkv (3, B, heads, N, head_dim)
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )
            
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.attn = attn
        self.v = v

        if general_token is not None:
            x = (attn @ v).transpose(1, 2).reshape(B, N+1, C)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def get_attn(self):
        return self.attn.get_attn()

    def forward(self, x, general_token=None):
        y = self.attn(self.norm1(x), general_token)
        if general_token is not None and y.shape[1] != x.shape[1]:
            y = y[:, :x.shape[1], :]
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
