from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..layers.darknet import CSPDarknet
from ..layers.network_blocks import DWConv, BaseConv, CSPLayer
from ..types import IMAGE, PYRAMID, BaseBackbone


class YOLOXPAFPNBackbone(BaseBackbone):
    """This module extracts the FPN feature of a single image,
    similar to StreamYOLO's DFPPAFPN without the last DFP concat step.

    """
    def __init__(
            self,
            base_depth: int = 3,
            base_channel: int = 64,
            depthwise: bool = False,
            act='silu',
            **kwargs
    ):
        # select FPN features
        super().__init__()
        self.out_channels: Tuple[int, ...] = tuple(i*base_channel for i in (4, 8, 16))
        self.feature_names = ('dark3', 'dark4', 'dark5')

        # build network (hardcoded)
        self.backbone = CSPDarknet(
            base_depth=base_depth,
            base_channel=base_channel,
            out_features=self.feature_names,
            depthwise=depthwise, act=act
        )
        Conv = DWConv if depthwise else BaseConv

        self.down_conv_5_4 = BaseConv(self.out_channels[2], self.out_channels[1], 1, 1, act=act)
        self.down_csp_4_4 = CSPLayer(
            2*self.out_channels[1],
            self.out_channels[1],
            base_depth,
            False,
            depthwise=depthwise,
            act=act,
        )

        self.down_conv_4_3 = BaseConv(self.out_channels[1], self.out_channels[0], 1, 1, act=act)
        self.down_csp_3_3 = CSPLayer(
            2*self.out_channels[0],
            self.out_channels[0],
            base_depth,
            False,
            depthwise=depthwise,
            act=act,
        )

        self.up_conv_3_3 = Conv(self.out_channels[0], self.out_channels[0], 3, 2, act=act)
        self.up_csp_3_4 = CSPLayer(
            2*self.out_channels[0],
            self.out_channels[1],
            base_depth,
            False,
            depthwise=depthwise,
            act=act,
        )

        self.up_conv_4_4 = Conv(self.out_channels[1], self.out_channels[1], 3, 2, act=act)
        self.up_csp_4_5 = CSPLayer(
            2*self.out_channels[1],
            self.out_channels[2],
            base_depth,
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, image: IMAGE) -> PYRAMID:
        """ Extract the FPN feature (p3, p4, p5) of an image tensor of (b, t, 3, h, w)

        """
        B, T, C, H, W = image.size()
        image = image.flatten(0, 1)

        feature = self.backbone(image)
        f3, f4, f5 = list(feature[f_name] for f_name in self.feature_names)

        # 5 -> 4 -> 3 -> 4 -> 5
        x = self.down_conv_5_4(f5)
        m4 = x
        x = F.interpolate(x, size=f4.shape[2:], mode='nearest')
        x = torch.cat([x, f4], dim=1)
        x = self.down_csp_4_4(x)

        x = self.down_conv_4_3(x)
        m3 = x
        x = F.interpolate(x, size=f3.shape[2:], mode='nearest')
        x = torch.cat([x, f3], dim=1)
        x = self.down_csp_3_3(x)
        p3 = x

        x = self.up_conv_3_3(x)
        x = torch.cat([x, m3], dim=1)
        x = self.up_csp_3_4(x)
        p4 = x

        x = self.up_conv_4_4(x)
        x = torch.cat([x, m4], dim=1)
        x = self.up_csp_4_5(x)
        p5 = x

        return p3.unflatten(0, (B, T)), p4.unflatten(0, (B, T)), p5.unflatten(0, (B, T))
