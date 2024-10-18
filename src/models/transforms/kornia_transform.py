from typing import Optional, Tuple, List

import kornia.augmentation as ka
from kornia.augmentation import random_generator as rg
import torch

from src.primitives.batch import BatchDict
from src.primitives.model import BaseTransform


def listify(x):
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    else:
        return [x]


class KorniaTransform(BaseTransform):
    def __init__(
            self,
            train_aug: Optional[List[ka.AugmentationBase2D]] = None,
            train_size: Optional[List[List[int]]] = None,
            eval_size: Optional[List[int]] = None,
            original_size: bool = False,
    ):
        super().__init__()
        self.train_aug_transform = ka.AugmentationSequential(
            ka.VideoSequential(*train_aug),
            data_keys=['image', 'bbox_xyxy'],
            same_on_batch=False,
            keepdim=False,
        ) if train_aug is not None else None
        self.train_resize_transform = ka.AugmentationSequential(
            ka.VideoSequential(
                *[ka.Resize((h, w)) for h, w in train_size],
                random_apply=1,
            ),
            data_keys=['image', 'bbox_xyxy'],
            same_on_batch=True,
            keepdim=False,
        ) if train_size is not None else None
        self.eval_resize_transform = ka.AugmentationSequential(
            ka.VideoSequential(ka.Resize(tuple(eval_size))),
            data_keys=['image', 'bbox_xyxy'],
            same_on_batch=True,
            keepdim=False,
        ) if eval_size is not None else None

        if original_size:
            self.original_size = (100, 100)
            self.original_resizer = ka.Resize(self.original_size)   # will change according to batch original_size
            self.original_resize_transform = ka.AugmentationSequential(
                ka.VideoSequential(self.original_resizer),
                data_keys=['image', 'bbox_xyxy'],
                same_on_batch=True,
                keepdim=False,
            )
        else:
            self.original_size = None
            self.original_resizer = None
            self.original_resize_transform = None

        self.register_buffer('c255', tensor=torch.tensor(255.0, dtype=torch.float32), persistent=False)

    def reset_original_resize(self, new_size: Tuple[int, int]) -> None:
        if new_size != self.original_size:
            self.original_size = new_size
            self.original_resizer._param_generator = rg.ResizeGenerator(resize_to=new_size, side=self.original_resizer.flags['side'])
            self.original_resizer.flags['size'] = new_size

    def train_transform(self, *args, data_keys: List[str]):
        args = listify(self.train_aug_transform(*args, data_keys=data_keys)) if self.train_aug_transform is not None else args
        args = listify(self.train_resize_transform(*args, data_keys=data_keys)) if self.train_resize_transform is not None else args
        return args

    def eval_transform(self, *args, data_keys: List[str]):
        args = listify(self.eval_resize_transform(*args, data_keys=data_keys)) if self.eval_resize_transform is not None else args
        return args

    def original_transform(self, *args, data_keys: List[str]):
        args = listify(self.original_resize_transform(*args, data_keys=data_keys)) if self.eval_resize_transform is not None else args
        return args

    def transform(self, batch: BatchDict, transform_fn) -> BatchDict:
        inputs = [batch['image']['image'] / self.c255]
        data_keys = ['image']
        for key in ('bbox', 'bbox_pred'):
            if key in batch:
                coordinates = batch[key]['coordinate']
                b, t, o, c = coordinates.shape
                coordinates = coordinates[..., [0, 1, 2, 1, 2, 3, 0, 3]].reshape(b, t, o, 4, 2)
                inputs.append(coordinates)
                data_keys.append('bbox')

        max_time = max([i.size(1) for i in inputs])
        difference_time = [max_time - i.size(1) for i in inputs]
        for i, d in enumerate(difference_time):
            if d > 0:
                inputs[i] = torch.cat([inputs[i], inputs[i][:, [0] * d]], dim=1)
        outputs = list(transform_fn(*inputs, data_keys=data_keys))
        for i, d in enumerate(difference_time):
            if d > 0:
                outputs[i] = outputs[i][:, :-d]

        batch['image']['image'] = torch.round(outputs.pop(0) * self.c255)
        batch['meta']['current_size'][:] = torch.tensor(batch['image']['image'].size()[-2:], dtype=torch.long, device=self.c255.device)
        for key in ('bbox', 'bbox_pred'):
            if key in batch:
                xy_min, xy_max = torch.aminmax(outputs.pop(0), dim=-2)
                coordinates = torch.cat([xy_min, xy_max], dim=-1)
                batch[key]['coordinate'] = coordinates

                # filter bounding box
                """
                B, T, O, _ = coordinates.shape
                for b in range(B):
                    for t in range(T):
                        mask = coordinates[b, t, :, 0]
                """

        return batch

    def preprocess(self, batch: BatchDict) -> BatchDict:
        if self.training:
            if self.train_aug_transform is not None or self.train_aug_transform is not None:
                return self.transform(batch, self.train_transform)
            else:
                return batch
        else:
            if self.eval_resize_transform is not None:
                return self.transform(batch, self.eval_transform)
            else:
                return batch

    """
    def postprocess(self, batch: BatchDict) -> BatchDict:
        if self.original_resize_transform is not None:
            self.reset_original_resize(tuple(batch['meta']['original_size'][0].cpu().tolist()))
            return self.transform(batch, self.original_transform)
    """
