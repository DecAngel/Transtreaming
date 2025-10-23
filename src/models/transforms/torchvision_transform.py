""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import importlib.metadata
from collections import defaultdict

from torch import Tensor

if importlib.metadata.version('torchvision') == '0.15.2':
    import torchvision

    torchvision.disable_beta_transforms_warning()

    from torchvision.datapoints import BoundingBox as BoundingBoxes
    from torchvision.datapoints import BoundingBoxFormat, Mask, Image, Video
    from torchvision.transforms.v2 import SanitizeBoundingBox as SanitizeBoundingBoxes

    _boxes_keys = ['format', 'spatial_size']

elif '0.17' > importlib.metadata.version('torchvision') >= '0.16':
    import torchvision

    torchvision.disable_beta_transforms_warning()

    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)

    _boxes_keys = ['format', 'canvas_size']

elif importlib.metadata.version('torchvision') >= '0.17':
    import torchvision
    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)

    _boxes_keys = ['format', 'canvas_size']

else:
    raise RuntimeError('Please make sure torchvision version >= 0.15.2')


from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torchvision
import PIL.Image
torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from src.primitives.batch import BatchDict
from src.primitives.model import BaseTransform


"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

def convert_to_tv_tensor(tensor: Tensor, key: str, box_format='xyxy', spatial_size=None) -> Tensor:
    """
    Args:
        tensor (Tensor): input tensor
        key (str): transform to key

    Return:
        Dict[str, TV_Tensor]
    """
    assert key in ('boxes', 'masks',), "Only support 'boxes' and 'masks'"

    if key == 'boxes':
        box_format = getattr(BoundingBoxFormat, box_format.upper())
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
        return BoundingBoxes(tensor, **_kwargs)

    if key == 'masks':
        return Mask(tensor)


class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return self._get_params(flat_inputs)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5,
                 max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40,
                 p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )

    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)


class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )

    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)


""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


class Compose(T.Compose):
    def __init__(self, ops: list[T.Transform], policy=None) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                transforms.append(op)
        else:
            transforms = [EmptyTransform(), ]

        super().__init__(transforms=transforms)

        if policy is None:
            policy = {'name': 'default'}

        self.policy = policy
        self.global_samples = 0

    def forward(self, *inputs: Any) -> Any:
        return self.get_forward(self.policy['name'])(*inputs)

    def get_forward(self, name):
        forwards = {
            'default': self.default_forward,
            'stop_epoch': self.stop_epoch_forward,
            'stop_sample': self.stop_sample_forward,
        }
        return forwards[name]

    def default_forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def stop_epoch_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]

        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']
        policy_epoch = self.policy['epoch']

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and cur_epoch >= policy_epoch:
                pass
            else:
                sample = transform(sample)

        return sample

    def stop_sample_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        policy_ops = self.policy['ops']
        policy_sample = self.policy['sample']

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and self.global_samples >= policy_sample:
                pass
            else:
                sample = transform(sample)

        self.global_samples += 1

        return sample


class TVTransform(BaseTransform):
    def __init__(
            self,
            train_ops: list[T.Transform],
            eval_ops: list[T.Transform],
            train_policy: dict = None,
    ) -> None:
        super().__init__()
        self.train_transform = Compose(train_ops, train_policy)
        self.eval_transform = Compose(eval_ops)

    def preprocess(self, batch: BatchDict) -> BatchDict:
        if self.training:
            transform = self.train_transform
        else:
            transform = self.eval_transform
        targets: dict = {
            'video': Video(batch['image']['image'])
        }
        if 'bbox' in batch:
            h, w = batch['meta']['current_size'][0].tolist()
            b, t, s = batch['bbox']['coordinate'].shape[:3]
            # num_objs = torch.count_nonzero(torch.count_nonzero(batch['bbox']['coordinate'], dim=3), dim=2)
            targets['boxes'] = BoundingBoxes(
                batch['bbox']['coordinate'].flatten(0, 2),
                format=BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            )
            label_b = torch.arange(0, b, device=batch['bbox']['coordinate'].device)[:, None, None].tile([1, t, s])
            label_t = torch.arange(0, t, device=batch['bbox']['coordinate'].device)[None, :, None].tile([b, 1, s])
            label = batch['bbox']['label']

            targets['labels'] = torch.stack([label, label_b, label_t], dim=-1).flatten(0, 2)

        results = transform(targets)

        video: Video = results['video']
        h, w = video.shape[-2:]
        batch['image']['image'] = video.data.float()
        batch['meta']['current_size'][..., 0] = h
        batch['meta']['current_size'][..., 1] = w

        if 'bbox' in batch:
            bbox: BoundingBoxes = results['boxes']
            label, label_b, label_t = results['labels'].unbind(-1)
            container = defaultdict(list)
            for b, l, lb, lt in zip(bbox, label, label_b, label_t):
                container[(lb.item(), lt.item())].append((b, l))

            coord_container = torch.zeros_like(batch['bbox']['coordinate'])
            label_container = torch.zeros_like(batch['bbox']['label'])
            for (lb, lt), v in container.items():
                b = torch.stack([i[0] for i in v], dim=0)
                l = torch.stack([i[1] for i in v], dim=0)
                num = b.shape[0]
                coord_container[lb, lt, :num] = b
                label_container[lb, lt, :num] = l

            batch['bbox']['coordinate'] = coord_container
            batch['bbox']['label'] = label_container

        return batch

    def postprocess(self, batch: BatchDict) -> BatchDict:
        return batch
