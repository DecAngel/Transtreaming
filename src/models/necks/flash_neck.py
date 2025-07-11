from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

from src.primitives.batch import PYRAMID, TIME, BatchDict
from src.primitives.model import BaseNeck, BlockMixin
from src.utils.windows_operations import pyramid2windows, windows2pyramid


class FlashNeck(BaseNeck):
    input_frames: int = 4
    output_frames: int = 1

    def __init__(
            self,
            in_channels: Tuple[int, ...],
            num_heads: int = 4,
            window_size: Tuple[int, int] = (8, 8),
            depth: int = 1,
            hidden_channel: Optional[int] = None,
            dropout: float = 0.05,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.depth = depth
        self.hidden_channel = hidden_channel or in_channels[0]
        self.dropout = dropout

        flash_attn_func()
        nn.MultiheadAttention()

    def test(self, func: {__le__, __ge__}):
        pass


class RelativePositionalEncoding3D(nn.Module):
    def __init__(
            self,
            window_size: Tuple[int, int],
            num_heads: int,
            rpe_coef: float = 16,
    ) -> None:
        super(RelativePositionalEncoding3D, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.rpe_coef = rpe_coef
        T = 50
        H, W = window_size
        rct_coef = 8

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # get relative_coords_table
        relative_coords_t = torch.arange(0, 2*T-1, dtype=torch.float32)  # 2*T-1
        relative_coords_h = torch.arange(-H+1, H, dtype=torch.float32)  # 2*H-1
        relative_coords_w = torch.arange(-W+1, W, dtype=torch.float32)  # 2*W-1

        relative_coords_table = torch.stack(
            torch.meshgrid([
                relative_coords_t,
                relative_coords_h,
                relative_coords_w])
        ).permute(1, 2, 3, 0).contiguous()  # 2*T-1, 2*H-1, 2*W-1, 3
        relative_coords_table[:, :, :, 0] /= 2*T-1
        relative_coords_table[:, :, :, 1] /= H-1
        relative_coords_table[:, :, :, 2] /= W-1

        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) * rct_coef + 1.0
        ) / np.log2(rct_coef)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # get pair-wise relative position index for each token inside the window
        coords_f = torch.stack(torch.meshgrid([torch.arange(T), torch.arange(H), torch.arange(W)]))  # 3, TF, H, W
        coords_p = torch.stack(torch.meshgrid([-torch.arange(T), torch.arange(-H+1, 1), torch.arange(-W+1, 1)]))  # 3, TP, H, W
        coords_flatten_f = torch.flatten(coords_f, 1)  # 3, TFHW
        coords_flatten_p = torch.flatten(coords_p, 1)  # 3, TPHW
        relative_coords = coords_flatten_f[:, :, None] - coords_flatten_p[:, None, :]  # 3, TFHW, TPHW
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # TFHW, TPHW, 3
        relative_coords[:, :, 0] *= (2 * H - 1)*(2 * W - 1)
        relative_coords[:, :, 1] *= 2 * W - 1
        relative_position_index = relative_coords.sum(-1).view(T, H*W, T, H*W)  # TF, HW, TP, HW
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.cached_rpt = None

    def forward(self, ptc: TIME, ftc: TIME) -> Float[torch.Tensor, 'batch head TFdHdW TPdHdW']:
        ptc = (-ptc).long()
        ftc = ftc.long()
        B, TP = ptc.size()
        _, TF = ftc.size()
        H, W = self.window_size

        if self.training:
            self.cached_rpt = None
            rpt = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        elif self.cached_rpt is None:
            rpt = self.cached_rpt = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        else:
            rpt = self.cached_rpt

        res = []
        for p, f in zip(ptc, ftc):  # permute batch
            rpi = self.relative_position_index[f]
            rpi = rpi[:, :, p]
            rpi = rpi.flatten()   # TFHWTPHW

            rpb = rpt[rpi].view(TF*H*W, TP*H*W, self.num_heads)
            rpb = self.rpe_coef * torch.sigmoid(rpb)
            res.append(rpb)
        return torch.stack(res, dim=0).permute(0, 3, 1, 2).contiguous()


class TATWindowAttention(nn.Module):
    def __init__(self, in_channel: int, hidden_channel: int, window_size: Tuple[int, int], num_heads: int):
        super().__init__()
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.window_size = window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.register_buffer('logit_max', torch.log(torch.tensor(1. / 0.01)), persistent=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
            self,
            query: Float[torch.Tensor, 'batch n length_q channels'],
            key: Float[torch.Tensor, 'batch n length_v channels'],
            value: Float[torch.Tensor, 'batch n length_v channels'],
            position: Float[torch.Tensor, 'batch head length_q length_v']
    ) -> Tuple[
        Float[torch.Tensor, 'batch n length_q channels_v'],
        Float[torch.Tensor, 'batch n length_q length_v'],
    ]:
        B, n, Lq, C = query.size()
        _, _, Lv, _ = key.size()

        query = query.reshape(B, n, Lq, self.num_heads, C // self.num_heads).transpose(2, 3)
        key = key.reshape(B, n, Lv, self.num_heads, C // self.num_heads).transpose(2, 3)
        value = value.reshape(B, n, Lv, self.num_heads, C // self.num_heads).transpose(2, 3)

        # qkv attention
        attn = query @ key.transpose(-2, -1)    # B n heads Lq Lv
        logit_scale = torch.clamp(self.logit_scale, max=self.logit_max).exp()
        attn = attn * logit_scale

        # relative positional encoding
        attn = attn + position[:, None]

        # mask and softmax
        attn = self.softmax(attn)

        # res
        x = attn @ value    # B n heads Lq C/h
        x = x.transpose(2, 3).flatten(3, 4)

        # proj
        # x = self.proj(x)

        return x, attn


class TATLayer(BlockMixin, nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            hidden_channel: int,
            window_size: Tuple[int, int],
            num_heads: int,
            shift: bool,
            dropout: float = 0.05,
    ):
        super(TATLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift = shift
        self.dropout = dropout

        # self.learnable_query = nn.Parameter(torch.rand(1, 1, self.window_size[0]*self.window_size[1], self.hidden_channel), requires_grad=True)

        self.fc_in = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, self.hidden_channel, bias=True),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channel, self.hidden_channel, bias=True),
                nn.Dropout(self.dropout),
            )
            for c in self.in_channels
        ])
        self.attn = TATWindowAttention(hidden_channel, hidden_channel, window_size, num_heads)
        self.fc_out = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_channel, self.hidden_channel, bias=True),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channel, c, bias=True),
                nn.Dropout(self.dropout),
            )
            for c in self.in_channels
        ])
        self.norm1s = nn.ModuleList([
            nn.LayerNorm(c)
            for c in self.in_channels
        ])
        self.norm2s = nn.ModuleList([
            nn.LayerNorm(c)
            for c in self.in_channels
        ])
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, c, bias=True),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(c, c, bias=True),
                nn.Dropout(self.dropout),
            )
            for c in self.in_channels
        ])

    def forward(
            self,
            features_p: PYRAMID,
            features_f: PYRAMID,
            position: Float[torch.Tensor, 'batch head length_q length_v']
    ) -> PYRAMID:
        B, TP0, _, _, _ = features_p[0].size()
        B, TF, _, _, _ = features_f[0].size()

        x = features_f

        with self.record_time('mlp_in'):
            features_p_in = tuple(fc_in(f) for f, fc_in in zip(features_p, self.fc_in))
            features_f_in = tuple(fc_in(f) for f, fc_in in zip(features_f, self.fc_in))

        with self.record_time('p2w'):
            query, q_p = pyramid2windows(features_f_in, self.window_size, self.shift)
            # _, N, L, C = query.size()
            # query = self.learnable_query.expand((B, N, L, C))
            key, k_p = pyramid2windows(tuple(f[:, :-1] for f in features_p_in), self.window_size, self.shift)
            value, v_p = pyramid2windows(tuple(f[:, :-1] - f[:, -1:] for f in features_p_in), self.window_size, self.shift)
            # value, v_p = pyramid2windows(tuple(f[:, :-1] for f in features_p_in), self.window_size, self.shift)

        with self.record_time('attn'):
            attn = self.attn(query, key, value, position)[0]

        with self.record_time('w2p'):
            features_attn = windows2pyramid(attn, q_p, self.window_size, self.shift)

        with self.record_time('mlp_out'):
            features_attn = tuple(fc_out(f) for f, fc_out in zip(features_attn, self.fc_out))

            x = tuple(f + norm(fa) for f, fa, norm in zip(x, features_attn, self.norm1s))
            x = tuple(f + norm(mlp(f)) for f, mlp, norm in zip(x, self.mlps, self.norm2s))
        return x


class TATLayer2(BlockMixin, nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            hidden_channel: int,
            window_size: Tuple[int, int],
            num_heads: int,
            shift: bool,
            dropout: float = 0.05,
    ):
        super(TATLayer2, self).__init__()
        self.in_channels = in_channels
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift = shift
        self.dropout = dropout

        # self.learnable_query = nn.Parameter(torch.rand(1, 1, self.window_size[0]*self.window_size[1], self.hidden_channel), requires_grad=True)

        self.fc_in = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, self.hidden_channel, bias=True),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channel, self.hidden_channel, bias=True),
                nn.Dropout(self.dropout),
            )
            for c in self.in_channels
        ])
        self.attn = TATWindowAttention(hidden_channel, hidden_channel, window_size, num_heads)
        self.fc_out = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_channel, self.hidden_channel, bias=True),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channel, c, bias=True),
                nn.Dropout(self.dropout),
            )
            for c in self.in_channels
        ])
        self.norm1s = nn.ModuleList([
            nn.LayerNorm(c)
            for c in self.in_channels
        ])
        self.norm2s = nn.ModuleList([
            nn.LayerNorm(c)
            for c in self.in_channels
        ])
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, c, bias=True),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(c, c, bias=True),
                nn.Dropout(self.dropout),
            )
            for c in self.in_channels
        ])

    def forward(
            self,
            features: PYRAMID,
            position: Float[torch.Tensor, 'batch head length_q length_v'],
            TP0: int,
    ) -> PYRAMID:

        x = tuple(f[:, TP0:] for f in features)

        with self.record_time('mlp_in'):
            features_p_in = tuple(fc_in(f) for f, fc_in in zip(features_p, self.fc_in))
            features_f_in = tuple(fc_in(f) for f, fc_in in zip(features_f, self.fc_in))

        with self.record_time('p2w'):
            query, q_p = pyramid2windows(features_f_in, self.window_size, self.shift)
            # _, N, L, C = query.size()
            # query = self.learnable_query.expand((B, N, L, C))
            key, k_p = pyramid2windows(tuple(f[:, :-1] for f in features_p_in), self.window_size, self.shift)
            value, v_p = pyramid2windows(tuple(f[:, :-1] - f[:, -1:] for f in features_p_in), self.window_size, self.shift)
            # value, v_p = pyramid2windows(tuple(f[:, :-1] for f in features_p_in), self.window_size, self.shift)

        with self.record_time('attn'):
            attn = self.attn(query, key, value, position)[0]

        with self.record_time('w2p'):
            features_attn = windows2pyramid(attn, q_p, self.window_size, self.shift)

        with self.record_time('mlp_out'):
            features_attn = tuple(fc_out(f) for f, fc_out in zip(features_attn, self.fc_out))

            x = tuple(f + norm(fa) for f, fa, norm in zip(x, features_attn, self.norm1s))
            x = tuple(f + norm(mlp(f)) for f, mlp, norm in zip(x, self.mlps, self.norm2s))
        return x


class TATNeck(BaseNeck):
    input_frames: int = 4
    output_frames: int = 1

    def __init__(
            self,
            in_channels: Tuple[int, ...],
            num_heads: int = 4,
            window_size: Tuple[int, int] = (8, 8),
            depth: int = 1,
            hidden_channel: Optional[int] = None,
            dropout: float = 0.05,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.depth = depth
        self.hidden_channel = hidden_channel or in_channels[0]
        self.dropout = dropout

        self.rpe = RelativePositionalEncoding3D(window_size=self.window_size, num_heads=self.num_heads)
        self.layers = nn.ModuleList([
            TATLayer(self.in_channels, self.hidden_channel, window_size, num_heads, d % 2 != 0, self.dropout)
            for d in range(depth)
        ])

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_yolo)

    def forward(self, batch: BatchDict) -> BatchDict:
        features = batch['intermediate']['features_p']
        past_clip_ids = batch['past_clip_ids'].float()
        future_clip_ids = batch['future_clip_ids'].float()
        B, TP = past_clip_ids.size()
        _, TF = future_clip_ids.size()

        position = self.rpe(past_clip_ids[:, :-1], future_clip_ids)

        features_p = tuple(f.permute(0, 1, 3, 4, 2) for f in features)                    # B T H W C
        features_f = tuple(f[:, -1:].expand(-1, TF, -1, -1, -1) for f in features_p)            # B T H W C

        for layer in self.layers:
            features_f = layer(features_p, features_f, position)

        batch['intermediate']['features_f'] = tuple(f.permute(0, 1, 4, 2, 3) for f in features_f)

        return batch
