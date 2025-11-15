from typing import Tuple, Optional, Literal

import torch
import torch.nn.functional as F
from torch import nn

from src.models.layers.pafpn.network_blocks import BaseConv
from src.primitives.batch import PYRAMID, TIME, BatchDict
from src.primitives.model import BaseNeck


class TemporalDifferenceNeck(BaseNeck):
    input_frames: int = 4
    output_frames: int = 1

    def __init__(
            self,
            in_channels: Tuple[int, ...],
            type_difference: Literal["default", "ratio", "gru", "gru2"] = "default",
    ):
        super().__init__()
        self.type_difference = type_difference

        self.convs = nn.ModuleList([
            BaseConv(f, f, ksize=1, stride=1)
            for f in in_channels
        ])
        if self.type_difference == "gru":
            self.grus = nn.ModuleList([
                nn.GRU(f, f, batch_first=True)
                for f in in_channels
            ])
        if self.type_difference == "gru2":
            self.grus = nn.ModuleList([
                nn.GRU(f, f, batch_first=True)
                for f in in_channels
            ])
            self.convs_2 = nn.ModuleList([
                BaseConv(f, f, ksize=1, stride=1)
                for f in in_channels
            ])

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_yolo)

    def forward(self, batch: BatchDict) -> BatchDict:
        features = batch['intermediate']['features_p']
        B, TP = batch['past_clip_ids'].size()
        _, TF = batch['future_clip_ids'].size()

        # use_buffer = 'buffer' in batch and not self.training
        use_buffer = False

        if self.type_difference == "default":
            outputs = []
            for i, conv in enumerate(self.convs):
                features_conv = conv(features[i].flatten(0, 1)).unflatten(0, (B, TP)).flip(1)
                features_conv = features_conv[:, :1] - features_conv[:, 1:]
                features_conv = torch.sum(features_conv, dim=1, keepdim=True)
                outputs.append(features[i][:, -1:]+features_conv)
        elif self.type_difference == "ratio":
            ratio = F.softmax(batch['past_clip_ids'][:, :-1].float(), dim=-1)[..., None, None, None]
            outputs = []
            for i, conv in enumerate(self.convs):
                features_conv = conv(features[i].flatten(0, 1)).unflatten(0, (B, TP)).flip(1)
                features_conv = features_conv[:, :1] - features_conv[:, 1:]
                features_conv = torch.sum(features_conv*ratio, dim=1, keepdim=True)
                outputs.append(features[i][:, -1:]+features_conv)
        elif self.type_difference == "gru":
            hidden = batch['buffer']['gru'] if 'gru' in batch['buffer'] and use_buffer else [None]*len(features)
            outputs = []
            new_hidden = []
            for i, (conv, gru, h) in enumerate(zip(self.convs, self.grus, hidden)):
                features_conv = conv(features[i].flatten(0, 1)).unflatten(0, (B, TP))
                _, _, C, H, W = features_conv.shape
                features_conv = features_conv.permute(0, 3, 4, 1, 2).contiguous().flatten(0, 2)
                if use_buffer:
                    features_conv = features_conv[:, -1:]
                out, new_h = gru(features_conv, h)
                out = out.unflatten(0, (B, H, W)).permute(0, 3, 4, 1, 2).contiguous()[:, -1:]
                outputs.append(features[i][:, -1:]+out)
                new_hidden.append(new_h)
            if use_buffer:
                batch['buffer']['gru'] = tuple(new_hidden)
        elif self.type_difference == "gru2":
            hidden = batch['buffer']['gru'] if 'gru' in batch['buffer'] and use_buffer else [None] * len(features)
            outputs = []
            new_hidden = []
            for i, (conv, conv2, gru, h) in enumerate(zip(self.convs, self.convs_2, self.grus, hidden)):
                features_conv = conv(features[i].flatten(0, 1)).unflatten(0, (B, TP))
                _, _, C, H, W = features_conv.shape
                features_conv = features_conv.permute(0, 3, 4, 1, 2).contiguous().flatten(0, 2)
                if use_buffer:
                    features_conv = features_conv[:, -1:]
                out, new_h = gru(features_conv, h)
                out = out.unflatten(0, (B, H, W)).permute(0, 3, 4, 1, 2).contiguous()[:, -1]
                outputs.append(conv2(features[i][:, -1]-out).unsqueeze(1))
                new_hidden.append(new_h)
            if use_buffer:
                batch['buffer']['gru'] = tuple(new_hidden)
        else:
            raise ValueError(f"Unknown type_difference: {self.type_difference}")

        batch['intermediate']['features_f'] = tuple(outputs)
        return batch
