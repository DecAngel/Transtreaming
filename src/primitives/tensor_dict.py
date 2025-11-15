import random

import torch
from tensordict import tensorclass, NonTensorData, TensorClass
from torch import Tensor


class BBoxDict(TensorClass):
    coordinates: Tensor
    labels: Tensor
    probabilities: Tensor


class SourceDict(TensorClass):
    video: Tensor
    bboxes: list[list[BBoxDict]]


if __name__ == '__main__':
    B, T, C, H, W = 4, 5, 3, 10, 20
    video = torch.randn(B, T, C, H, W)
    bboxes = []
    for b in range(B):
        bboxes.append([])
        for t in range(T):
            num_objs = random.randint(10, 100)
            bboxes[-1].append(BBoxDict(
                coordinates=torch.rand(num_objs, 4),
                labels=torch.randint(0, 10, size=(num_objs,)),
                probabilities=torch.rand(num_objs),
            ))
    source = SourceDict(
        video=video,
        bboxes=bboxes,
        batch_size=(B, T)
    )

    print(source[0, 0])
