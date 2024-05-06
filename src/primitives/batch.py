from typing import TypedDict, Dict, Any, Tuple, NotRequired, Union, List

import torch
from jaxtyping import UInt8, Int32, Float32, Float, Int


META = Int[torch.Tensor, '*batch']
IMAGE_RAW = UInt8[torch.Tensor, '*batch_time channels_rgb=3 height width']
IMAGE = Float[torch.Tensor, '*batch_time channels_rgb=3 height width']
PYRAMID = Tuple[Float[torch.Tensor, '*batch_time channels height width'], ...]
COORDINATE = Float[torch.Tensor, '*batch_time max_objs coords_xyxy=4']
PROBABILITY = Float[torch.Tensor, '*batch_time max_objs']
LABEL = Int[torch.Tensor, '*batch_time max_objs']
TIME = Float[torch.Tensor, '*batch time']
SIZE = Int[torch.Tensor, 'hw=2']
SCALAR = Float[torch.Tensor, '']


class MetaDict(TypedDict):
    """Meta tensor component of a batch.

    **image_id**: the image id in COCO

    **seq_id**: the id of the sequence

    **frame_id**: the id of the frame in <seq_id> sequence

    **current_size**: the size of current image and bbox

    **original_size**: the size of original image and bbox
    """
    image_id: META
    seq_id: META
    frame_id: META
    current_size: SIZE
    original_size: SIZE


class ImageDict(TypedDict):
    """Image tensor component of a batch.

    **image**: a BGR image with CHW shape in 0-255 or 0-1
    """
    image: Union[IMAGE, IMAGE_RAW]


class BBoxDict(TypedDict):
    """BBox tensor component of a batch.

    **coordinate**: the xyxy array of bbox coordinates

    **label**: the category of the bbox

    **probability**: the probability of the bbox
    """
    coordinate: COORDINATE
    label: LABEL
    probability: PROBABILITY


class BatchDict(TypedDict, total=False):
    meta: MetaDict
    image: ImageDict
    bbox: BBoxDict
    bbox_pred: BBoxDict
    image_clip_ids: TIME
    bbox_clip_ids: TIME


class BufferDict(TypedDict, total=False):
    buffer_list: List[PYRAMID]
    buffer_clip_id_list: List[int]


class LossDict(TypedDict, total=False):
    loss: SCALAR


class MetricDict(TypedDict, total=False):
    mAP: SCALAR
    sAP: SCALAR
