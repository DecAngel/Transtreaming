import collections
from typing import Dict, List, Union, Callable, Tuple

import torch
from torch.utils.data import default_collate, default_convert

from src.primitives.batch import PYRAMID
# from torch._six import string_classes
# from torch.utils.data._utils.collate import np_str_obj_array_pattern

from src.utils.array_operations import ArrayType, slice_along

CollectionType = Union[ArrayType, Tuple[ArrayType], List[ArrayType], Dict[str, ArrayType]]


class ApplyCollection:
    def __init__(self, fn: Callable):
        self.fn = fn

    def __call__(self, collection: CollectionType) -> CollectionType:
        elem_type = type(collection)
        if isinstance(collection, torch.Tensor):
            return self.fn(collection)
        elif isinstance(collection, collections.abc.Mapping):
            try:
                return elem_type({key: self(collection[key]) for key in collection})
            except TypeError:
                # The mapping type may not support `__init__(iterable)`.
                return {key: self(collection[key]) for key in collection}
        elif isinstance(collection, tuple) and hasattr(collection, '_fields'):  # namedtuple
            return elem_type(*(self(d) for d in collection))
        elif isinstance(collection, tuple):
            return [self(d) for d in collection]  # Backwards compatibility.
        elif isinstance(collection, collections.abc.Sequence) and not isinstance(collection, str):
            try:
                return elem_type([self(d) for d in collection])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [self(d) for d in collection]
        else:
            return collection


def get_one_element(collection: CollectionType) -> ArrayType:
    if isinstance(collection, dict):
        return get_one_element(list(collection.values())[0])
    elif isinstance(collection, (list, tuple)):
        return get_one_element(collection[0])
    else:
        return collection


ndarray2tensor = ApplyCollection(torch.from_numpy)
tensor2ndarray = ApplyCollection(lambda t: t.detach().cpu().numpy())
collate = default_collate


def reverse_collate(collection: CollectionType) -> List[CollectionType]:
    batch_size = get_one_element(collection).shape[0]
    return [ApplyCollection(lambda t: t[i])(collection) for i in range(batch_size)]


def select_collate(collection: CollectionType, batch_id: int) -> CollectionType:
    return ApplyCollection(lambda t: t[batch_id])(collection)


def to_device(collection: CollectionType, device: torch.device) -> CollectionType:
    return ApplyCollection(lambda t: t.to(device))(collection)


def concat_pyramids(pyramids: List[PYRAMID], dim: int = 1) -> PYRAMID:
    return tuple(
        torch.cat([
            p[i]
            for p in pyramids
        ], dim=dim)
        for i in range(len(pyramids[0]))
    )


def slice_pyramid(pyramid: PYRAMID, start: int, end: int, step: int = 1, dim: int = 1) -> PYRAMID:
    return tuple(
        slice_along(p, dim, start, end, step)
        for p in pyramid
    )
