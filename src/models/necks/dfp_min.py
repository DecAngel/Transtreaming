from typing import Tuple, Optional

from torch import nn

from ..layers.network_blocks import BaseConv
from ..types import PYRAMID, BaseNeck, TIME


class DFPMIN(BaseNeck):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            **kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            BaseConv(f, f, ksize=1, stride=1)
            for f in in_channels
        ])

    def forward(
            self,
            features: PYRAMID,
            past_time_constant: Optional[TIME] = None,
            future_time_constant: Optional[TIME] = None,
    ) -> PYRAMID:
        B, T, _, _, _ = features[0].size()

        outputs = []
        for i, conv in enumerate(self.convs):
            features_conv = conv(features[i].flatten(0, 1)).unflatten(0, (B, T)).flip(1)
            features_conv = [features_conv[:, :1].expand(-1, T-1, -1, -1, -1) - features_conv[:, 1:]]
            outputs.append(features[i][:, -1:]+features_conv)

        return tuple(outputs)
