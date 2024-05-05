import functools
import itertools
from typing import TypedDict, Dict, Any, Tuple, NotRequired, Union

import torch
from jaxtyping import UInt8, Int32, Float32, Float, Int


META = Int[torch.Tensor, '*batch']
IMAGE_RAW = UInt8[torch.Tensor, '*batch_time channels_rgb=3 height width']
IMAGE = Float[torch.Tensor, '*batch_time channels_rgb=3 height width']
SIZE = Int[torch.Tensor, '*batch_time hw=2']
PYRAMID = Tuple[Float[torch.Tensor, '*batch_time channels height width'], ...]
COORDINATE = Float[torch.Tensor, '*batch_time max_objs coords_xyxy=4']
PROBABILITY = Float[torch.Tensor, '*batch_time max_objs']
LABEL = Int[torch.Tensor, '*batch_time max_objs']
TIME = Float[torch.Tensor, '*batch time']
SCALAR = Float[torch.Tensor, '']


class MetaDict(TypedDict):
    """Meta tensor component of a batch.

    **image_id**: the image id in COCO

    **seq_id**: the id of the sequence

    **frame_id**: the id of the frame in <seq_id> sequence
    """
    image_id: META
    seq_id: META
    frame_id: META


class ImageDict(TypedDict):
    """Image tensor component of a batch.

    **image**: a BGR image with CHW shape in 0-255 or 0-1

    **original_size**: Original HW size of image
    """
    image: Union[IMAGE, IMAGE_RAW]
    original_size: SIZE


class BBoxDict(TypedDict):
    """BBox tensor component of a batch.

    **coordinate**: the xyxy array of bbox coordinates

    **label**: the category of the bbox

    **probability**: the probability of the bbox

    **current_size**: Current HW size of the bbox

    **original_size**: Original HW size of the bbox
    """
    coordinate: COORDINATE
    label: LABEL
    probability: PROBABILITY
    current_size: SIZE
    original_size: SIZE


class BatchDict(TypedDict, total=False):
    meta: MetaDict
    image: ImageDict
    image_clip_ids: TIME
    bbox: BBoxDict
    bbox_clip_ids: TIME
    bbox_pred: BBoxDict
    bbox_pred_clip_ids: TIME


class BufferDict(TypedDict, total=False):
    buffer: PYRAMID
    buffer_clip_ids: TIME


class LossDict(TypedDict, total=False):
    loss: SCALAR


class MetricDict(TypedDict, total=False):
    mAP: SCALAR
    sAP: SCALAR
