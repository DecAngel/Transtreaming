import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from src.models.layers.presnet import ResNet_cfg, donwload_url, ConvNormLayer, BasicBlock, BottleNeck, Blocks
from src.primitives.batch import IMAGE, PYRAMID
from src.utils.detr_utils import FrozenBatchNorm2d
from src.primitives.model import BaseBackbone


class PResNetBackbone(BaseBackbone):

    def __init__(
            self,
            depth,
            variant='d',
            num_stages=4,
            return_idx=[0, 1, 2, 3],
            act='relu',
            freeze_at=-1,
            freeze_norm=True,
            pretrained=False):
        super().__init__()

        block_nums = ResNet_cfg[depth]
        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(OrderedDict([
            (name, ConvNormLayer(cin, cout, k, s, act=act)) for cin, cout, k, s, name in conv_def
        ]))

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(
                Blocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act, variant=variant)
            )
            ch_in = _out_channels[i]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            if isinstance(pretrained, bool) or 'http' in pretrained:
                state = torch.hub.load_state_dict_from_url(donwload_url[depth], map_location='cpu')
            else:
                state = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(state)
            print(f'Load PResNet{depth} state_dict')

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, image: IMAGE) -> PYRAMID:
        B, T, C, H, W = image.shape
        x = image.flatten(0, 1) / 255.
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x.unflatten(0, (B, T)))
        return tuple(outs)
