import functools
from typing import TypedDict, Tuple, Union, List

import torch
from jaxtyping import UInt8, Float, Int, Shaped


META = Int[torch.Tensor, '*batch']
IMAGE_RAW = UInt8[torch.Tensor, '*batch_time channels_rgb=3 height width']
IMAGE = Float[torch.Tensor, '*batch_time channels_rgb=3 height width']
FEATURE = Float[torch.Tensor, '*batch_time channels height width']
PYRAMID = Tuple[FEATURE, ...]
WINDOW = Tuple[Float[torch.Tensor, '*batch_windows length channels'], List[Tuple[int, ...]]]
COORDINATE = Float[torch.Tensor, '*batch_time max_objs coords_xyxy=4']
PROBABILITY = Float[torch.Tensor, '*batch_time max_objs']
LABEL = Int[torch.Tensor, '*batch_time max_objs']
TIME = Shaped[torch.Tensor, '*batch time']
SIZE = Int[torch.Tensor, '*batch hw=2']
SCALAR = Float[torch.Tensor, '']


def is_pyramid(obj) -> bool:
    return isinstance(obj, tuple) and functools.reduce(
            bool.__and__,
            [isinstance(o, torch.Tensor) and o.ndim == 5 for o in obj]
    )


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

    **track**: the track id of the bbox
    """
    coordinate: COORDINATE
    label: LABEL
    probability: PROBABILITY


class LossDict(TypedDict, total=False):
    """Loss tensor component of a batch.

    **loss**: the computed loss of the batch

    Can also include additional losses
    """
    loss: SCALAR


class IntermediateDict(TypedDict, total=False):
    """Intermediate tensor component of a batch.

    **features_p**: the features of the past

    **features_f**: the features of the future

    **features_flow**: the features of intermediate flows

    Can also include additional features
    """
    features_p: PYRAMID | WINDOW
    features_f: PYRAMID | WINDOW
    features_flow: FEATURE


class MetricDict(TypedDict, total=False):
    """Metric tensor component of a batch.

    **mAP**: the computed mean Average Precision

    **sAP**: the computed streaming mAP

    Can also include additional metrics
    """
    mAP: SCALAR
    sAP: SCALAR


class BufferDict(TypedDict, total=False):
    """Buffer tensor component of a batch.

    **features_p**: past features

    **features_flow**: the features of intermediate flows

    """
    features_p: PYRAMID
    features_flow: FEATURE
    gru: PYRAMID


class BatchDict(TypedDict, total=False):
    meta: MetaDict
    image: ImageDict
    bbox: BBoxDict
    bbox_pred: BBoxDict
    image_clip_ids: TIME
    bbox_clip_ids: TIME
    past_clip_ids: TIME
    future_clip_ids: TIME
    loss: LossDict | SCALAR
    intermediate: IntermediateDict
    metric: MetricDict
    buffer: BufferDict
