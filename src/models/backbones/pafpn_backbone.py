from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.layers.darknet import CSPDarknet
from src.models.layers.network_blocks import DWConv, BaseConv, CSPLayer
from src.primitives.batch import IMAGE, PYRAMID, BatchDict
from src.primitives.model import BaseBackbone


class PAFPNBackbone(BaseBackbone):
    """This module extracts the FPN feature of a single image,
    similar to StreamYOLO's DFPPAFPN without the last DFP concat step.

    """
    state_dict_replace = [
        ('lateral_conv0', 'down_conv_5_4'),
        ('C3_p4', 'down_csp_4_4'),
        ('reduce_conv1', 'down_conv_4_3'),
        ('C3_p3', 'down_csp_3_3'),
        ('bu_conv2', 'up_conv_3_3'),
        ('C3_n3', 'up_csp_3_4'),
        ('bu_conv1', 'up_conv_4_4'),
        ('C3_n4', 'up_csp_4_5'),
    ]

    def __init__(
            self,
            base_depth: int = 3,
            base_channel: int = 64,
            depthwise: bool = False,
            act='silu',
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

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_yolo)

    def forward(self, batch: BatchDict) -> BatchDict:
        """ Extract the FPN feature (p3, p4, p5) of an image tensor of (b, t, 3, h, w)

        """
        image = batch['image']['image']
        B, T, C, H, W = image.size()
        image = image.flatten(0, 1)

        with self.record_time('PAFPN_darknet'):
            feature = self.backbone(image)
        f3, f4, f5 = list(feature[f_name] for f_name in self.feature_names)

        with self.record_time('PAFPN_fpn'):
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

        batch['intermediate']['features_p'] = p3.unflatten(0, (B, T)), p4.unflatten(0, (B, T)), p5.unflatten(0, (B, T))

        return batch
