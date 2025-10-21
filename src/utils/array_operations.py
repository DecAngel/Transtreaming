from typing import Any, TypeVar

import torch
import numpy as np


ArrayType = TypeVar('ArrayType', torch.Tensor, np.ndarray)


def slice_along(
        arr: ArrayType, axis: int, start: int, end: int, step: int = 1
) -> ArrayType:
    """Slice the array along a specific dimension.

    :param arr: the array to be sliced, `np.ndarray` or `torch.Tensor`
    :param axis: the dimension to slice
    :param start: the start of slice
    :param end: the end of slice
    :param step: the step of slice
    :return: the result array
    """
    return arr[(slice(None), ) * (axis % arr.ndim) + (slice(start, end, step),)]


def clip_or_pad_along(
        arr: ArrayType, axis: int, fixed_length: int, pad_value: Any = 0
) -> ArrayType:
    """Clip or pad the array along a specific dimension to a length of `fixed_length`. Pad with `pad_value`.

    :param arr: the array to be clipped or padded, `np.ndarray` or `torch.Tensor`
    :param axis: the index of the dimension
    :param fixed_length: the desired length of the dimension
    :param pad_value: the value to pad
    :return: the result array
    """
    if arr.shape[axis] > fixed_length:
        return slice_along(arr, axis, 0, fixed_length)
    elif arr.shape[axis] < fixed_length:
        if isinstance(arr, np.ndarray):
            pad_widths = [(0, 0)] * arr.ndim
            pad_widths[axis] = (0, fixed_length - arr.shape[axis])
            return np.pad(arr, pad_widths, constant_values=pad_value)
        else:
            pad_widths = [0] * (2 * arr.ndim)
            pad_widths[-2*(axis % arr.ndim)-1] = fixed_length - arr.shape[axis]
            res = torch.nn.functional.pad(arr, pad_widths, value=pad_value)
            return res
    else:
        return arr


def remove_pad_along(
        arr: ArrayType, axis: int, pad_value: Any = 0
) -> ArrayType:
    """Remove the pad along a specific dimension.

    :param arr: the array to be clipped, `np.ndarray` or `torch.Tensor`
    :param axis: the index of the dimension
    :param pad_value: the pad value to remove
    :return: the result array
    """
    data_length = 0
    if isinstance(arr, np.ndarray):
        for a in np.split(arr, arr.shape[axis],axis=axis):
            if np.isnan(pad_value) and np.all(np.isnan(a)):
                break
            elif not np.isnan(pad_value) and np.all(a == pad_value):
                break
            else:
                data_length += 1
        return slice_along(arr, axis, 0, max(data_length, 1))
    else:
        for a in torch.split(arr, 1, dim=axis):
            if np.isnan(pad_value) and torch.all(torch.isnan(a)):
                break
            elif not np.isnan(pad_value) and torch.all(a == pad_value):
                break
            else:
                data_length += 1
        return slice_along(arr, axis, 0, max(data_length, 1))


def xyxy2cxcywh(coordinates: ArrayType) -> ArrayType:
    cxcy = (coordinates[..., :2] + coordinates[..., 2:]) / 2
    wh = (coordinates[..., 2:] - coordinates[..., :2])
    return np.concatenate([cxcy, wh], axis=-1) if isinstance(coordinates, np.ndarray) else torch.cat([cxcy, wh], dim=-1)


def xyxy2xywh(coordinates: ArrayType) -> ArrayType:
    xy = coordinates[..., :2]
    wh = coordinates[..., 2:] - coordinates[..., :2]
    return np.concatenate([xy, wh], axis=-1) if isinstance(coordinates, np.ndarray) else torch.cat([xy, wh], dim=-1)


def cxcywh2xyxy(coordinates: ArrayType) -> ArrayType:
    wh2 = coordinates[..., 2:] / 2
    x1y1 = coordinates[..., :2] - wh2
    x2y2 = coordinates[..., :2] + wh2
    return np.concatenate([x1y1, x2y2], axis=-1) if isinstance(coordinates, np.ndarray) else torch.cat([x1y1, x2y2], dim=-1)


def xywh2xyxy(coordinates: ArrayType) -> ArrayType:
    x1y1 = coordinates[..., :2]
    x2y2 = coordinates[..., :2] + coordinates[..., 2:]
    return np.concatenate([x1y1, x2y2], axis=-1) if isinstance(coordinates, np.ndarray) else torch.cat([x1y1, x2y2], dim=-1)


if __name__ == '__main__':
    array = torch.randn(100, 4)
    print(torch.all(torch.abs(cxcywh2xyxy(xyxy2cxcywh(array)) - array) < 1e-5))
    print(torch.all(torch.abs(xywh2xyxy(xyxy2xywh(array)) - array) < 1e-5))
