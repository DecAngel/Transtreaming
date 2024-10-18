import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from src.models.layers.hybrid_encoder import TransformerEncoder, TransformerEncoderLayer, ConvNormLayer, CSPRepLayer
from src.primitives.batch import PYRAMID, TIME
from src.primitives.model import BaseNeck


class HybridEncoderNeck(BaseNeck):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='v2',
                 diff_stride=2,
                 diff_radius=4):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.diff_stride = diff_stride
        self.diff_radius = diff_radius

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if version == 'v1':
                proj = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim))
            elif version == 'v2':
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                raise AttributeError()

            self.input_proj.append(proj)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self.diff_mlp = nn.Sequential(
            nn.Linear((2*self.diff_radius+1)**2, 1, bias=False),
        )
        self.diff_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def get_diff(
            self,
            feature1: Float[torch.Tensor, '*batch_time channels height width'],
            feature2: Float[torch.Tensor, '*batch_time channels height width'],
    ):
        B, T, C, H, W = feature1.shape
        f1 = feature1.flatten(0, 2).unsqueeze(0)    # 1, B*T*C
        f2 = feature2.flatten(0, 2).unsqueeze(1)    # B*T*C, 1

        # 1, B*T*C, 2*radius+1, 2*radius+1
        diff = F.conv2d(f1, f2, bias=None, stride=self.diff_stride, padding=self.diff_radius * self.diff_stride, groups=B*T*C)

        return diff.reshape(B, T, C, 2*self.diff_radius+1, 2*self.diff_radius+1)

    def forward(
            self,
            features: PYRAMID,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
    ) -> PYRAMID:
        B, TP = past_clip_ids.shape
        _, TF = future_clip_ids.shape

        base_features = [f[:, -1] for f in features]
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(base_features)]

        # B, TP, C, H, W
        proj_time_features = [self.input_proj[-1](feat) for i, feat in enumerate(features[-1].unbind(1)) if i != TP-1]
        proj_time_features = torch.stack(proj_time_features+[proj_feats[-1]], dim=1)

        # B, TP-1
        diff_clip_ids = past_clip_ids[..., 1:] - past_clip_ids[..., :-1]
        # B, TP-1, C, 2R+1, 2R+1
        diff_features = self.get_diff(proj_time_features[:, 1:], proj_time_features[:, :-1])

        # B, TP-1, C, (2R+1)^2
        diff_features = F.softmax(diff_features.flatten(3, 4), dim=-1)
        # B, TP-1, C
        diff_features = self.diff_mlp(diff_features).squeeze(3)
        diff_features = diff_features / diff_clip_ids.unsqueeze(-1)
        diff_features, _ = self.diff_rnn(diff_features)
        # B, 1, C
        diff_features = diff_features[:, -1:]

        # CA
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory: torch.Tensor = self.encoder[i](src_flatten, extra=diff_features, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            # shape
            _, _, H1, W1 = feat_heigh.shape
            _, _, H2, W2 = feat_low.shape
            upsample_feat = F.interpolate(feat_heigh, scale_factor=(H2/H1, W2/W1), mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return tuple(outs)
