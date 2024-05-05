import functools
import itertools
from typing import Optional, Generic, TypeVar, Iterable, Iterator, Sequence, Tuple, Union, List, TypedDict, Dict, Any

import torch
from jaxtyping import Int, Float, UInt8, Int32, Float32
import kornia.augmentation as ka
import bisect

import torch
from torch.utils.data import Dataset


class MetaDict(TypedDict):
    """Meta tensor component of a batch.

    **image_id**: the image id in COCO

    **seq_id**: the id of the sequence

    **frame_id**: the id of the frame in <seq_id> sequence
    """
    image_id: Int32[torch.Tensor, '*batch']
    seq_id: Int32[torch.Tensor, '*batch']
    frame_id: Int32[torch.Tensor, '*batch']


class ImageDict(TypedDict):
    """Image tensor component of a batch.

    **image**: a BGR image with CHW shape in 0-255

    **original_size**: Original HW size of image
    """
    image: UInt8[torch.Tensor, '*batch channels=3 height width']
    original_size: Int32[torch.Tensor, '*batch hw=2']


class BBoxDict(TypedDict):
    """BBox tensor component of a batch.

    **coordinate**: the xyxy array of bbox coordinates

    **label**: the category of the bbox

    **probability**: the probability of the bbox

    **current_size**: Current HW size of the bbox

    **original_size**: Original HW size of the bbox
    """
    coordinate: Float32[torch.Tensor, '*batch objects xyxy=4']
    label: Int32[torch.Tensor, '*batch objects']
    probability: Float32[torch.Tensor, '*batch objects']
    current_size: Int32[torch.Tensor, '*batch hw=2']
    original_size: Int32[torch.Tensor, '*batch hw=2']


class BatchDict(TypedDict, total=False):
    meta: MetaDict
    image: ImageDict
    image_clip_ids: Int32[torch.Tensor, '*batch time']
    bbox: BBoxDict
    bbox_clip_ids: Int32[torch.Tensor, '*batch time']
    bbox_pred: BBoxDict
    bbox_pred_clip_ids: Int32[torch.Tensor, '*batch time']


class DataPipe(Dataset):
    """A multi-dimensional map data loading component with pickleable state.

    Caution: invoke __len__ directly
    """
    def __init__(self):
        super().__init__()
        self._sources: List['DataPipe'] = []

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_sources')
        return d

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)

    @property
    def source(self):
        return self._sources[0]

    @property
    def sources(self):
        return self._sources

    def __post_init__(self) -> None: raise NotImplementedError()
    def __getitem__(self, items: Tuple[int, ...]) -> Dict[str, Any]: raise NotImplementedError()
    def __len__(self) -> Tuple[Union[int, List[int]], ...]: raise NotImplementedError()


class DataPipe2Dataset(DataPipe):
    def __init__(self):
        super().__init__()

    def __post_init__(self) -> None:
        self._length = self.source.__len__()
        self._length_acc = [l if isinstance(l, int) else list(itertools.accumulate(l)) for l in self._length]

    def __getitem__(self, item: int) -> Dict[str, Any]:
        indices = []
        for ad in reversed(self._length_acc):
            if isinstance(ad, int):
                item, residue = divmod(item, ad)
                indices.insert(0, residue)
            else:
                res = bisect.bisect(ad, item)
                indices.insert(0, item - (ad[res-1] if res > 0 else 0))
                item = res
        return self.source[tuple(indices)]

    def __len__(self) -> int:
        size = 1
        flag = True
        for ad in reversed(self._length_acc):
            if isinstance(ad, int):
                size *= ad
            elif flag:
                flag = False
                size *= ad[-1]
        return size
