import torch

from src.primitives.batch import BatchDict
from src.primitives.model import BaseTransform

from torchvision.tv_tensors import Video, BoundingBoxes, Image, BoundingBoxFormat
import torchvision.transforms.v2 as T


# TODO: finish
class TVTransform(BaseTransform):
    def __init__(self, transform):
        super().__init__()

    def preprocess(self, batch: BatchDict) -> BatchDict:
        targets = {
            'image': Video(batch['image']['image']),
            '': BoundingBoxes(batch['bbox']['coordinate'], batch['meta']['current_size']),
        }
        BoundingBoxFormat.CXCYWH

    def postprocess(self, batch: BatchDict) -> BatchDict:
        return batch
