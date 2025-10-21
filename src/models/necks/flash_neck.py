from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL.features import features
from jaxtyping import Float

from src.primitives.batch import PYRAMID, WINDOW, TIME, BatchDict
from src.primitives.model import BaseNeck, BlockMixin
from src.utils.windows_operations import pyramid2windows, windows2pyramid, pyramid2windows2, windows2pyramid2


class FlashBlock(nn.Module):
    """

    """

    def __init__(
            self,
            in_channel: int,
            num_heads: int,
            dropout: float = 0.05,
    ):
        super(FlashBlock, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channel,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(in_channel)
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channel, in_channel, bias=True),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(in_channel)

    def forward(
            self,
            features_pw: WINDOW,
            features_fw: WINDOW,
    ) -> WINDOW:
        attn_output, attn_weight = self.attn(features_fw, features_pw, features_pw)
        features_fw = features_fw + attn_output
        features_fw = self.norm1(features_fw)
        features_fw = features_fw + self.mlp(features_fw)
        features_fw = self.norm2(features_fw)
        return features_fw


class FlashNeck(BaseNeck):
    input_frames: int = 4
    output_frames: int = 1

    def __init__(
            self,
            in_channels: Tuple[int, ...],
            hidden_channel: int = 512,
            num_heads: int = 4,
            depth: int = 1,
            dropout: float = 0.05,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.depth = depth
        self.dropout = dropout
        self.input_channel = 16 * in_channels[0] + 4 * in_channels[1] + in_channels[2]
        self.hidden_channel = hidden_channel

        self.input_mlp = nn.Sequential(
            nn.Linear(self.input_channel, self.hidden_channel),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_channel, self.input_channel),
        )

        self.layers = nn.ModuleList([
            FlashBlock(self.hidden_channel, self.num_heads, self.dropout)
            for d in range(depth)
        ])

        self.tpe_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_channel, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channel, self.hidden_channel, bias=False)
        )


    def forward(self, batch: BatchDict) -> BatchDict:
        features = batch['intermediate']['features_p']
        past_clip_ids = batch['past_clip_ids'].float()
        future_clip_ids = batch['future_clip_ids'].float()
        B, TP = past_clip_ids.size()
        _, TF = future_clip_ids.size()

        # B, T, hC
        past_tpe = self.tpe_mlp(past_clip_ids.unsqueeze(-1))
        future_tpe = self.tpe_mlp(future_clip_ids.unsqueeze(-1))

        features_w, params = pyramid2windows2(features)
        features_fw_base = torch.expand_copy(features_w[:, -1:], size=[-1, TF, -1])
        features_pw = self.input_mlp(features_w)
        features_fw = torch.expand_copy(features_pw[:, -1:], size=[-1, TF, -1])
        features_pw = features_pw.detach()

        n_windows = features_pw.size(0) // B
        features_pw = features_pw + torch.repeat_interleave(past_tpe, n_windows, dim=0)
        features_fw = features_fw + torch.repeat_interleave(future_tpe, n_windows, dim=0)

        for layer in self.layers:
            features_fw = layer(features_pw, features_fw)

        features_fw = features_fw_base + self.output_mlp(features_fw)
        features_f = windows2pyramid2((features_fw, params))
        batch['intermediate']['features_f'] = features_f
        return batch
