import functools
from typing import List, Tuple, Dict, Any, Protocol, Optional, Union, Callable

import lightning as L
import torch
from torch.utils.data import Dataset, default_collate, DataLoader

from src.primitives.batch import MetaDict, ImageDict, BBoxDict, BatchDict
from src.utils.pylogger import RankedLogger
from src.primitives.model import BlockMixin

log = RankedLogger(__name__, rank_zero_only=False)


class BaseDataSource(BlockMixin, Dataset):
    """Base data source for video detection datasets"""
    def __init__(
            self,
            image_clip_ids: List[int],
            bbox_clip_ids: List[int],
            image_clip_fn: Optional[Callable[[BlockMixin], List[int]]] = None,
            bbox_clip_fn: Optional[Callable[[BlockMixin], List[int]]] = None,
    ):
        super().__init__()
        self.image_clip_ids = image_clip_ids
        self.bbox_clip_ids = bbox_clip_ids
        self.image_clip_fn = image_clip_fn
        self.bbox_clip_fn = bbox_clip_fn
        indices = image_clip_ids + bbox_clip_ids + [0]
        self.margin_left = -min(indices)
        self.margin_right = max(indices)
        self.margin = self.margin_left + self.margin_right

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)

    def __post_init__(self): raise NotImplementedError()

    def get_meta(self, seq_id: int, frame_id: int) -> MetaDict: raise NotImplementedError()

    def get_image(self, seq_id: int, frame_id: int) -> ImageDict: raise NotImplementedError()

    def get_bbox(self, seq_id: int, frame_id: int) -> BBoxDict: raise NotImplementedError()

    def get_length(self) -> List[int]: raise NotImplementedError()

    @functools.cached_property
    def _length(self) -> List[int]: return self.get_length()

    def __len__(self) -> int:
        return sum([l - self.margin for l in self._length])

    def __getitem__(self, item: int) -> BatchDict:
        for seq_id, seq_len in enumerate(self._length):
            if item < seq_len - self.margin:
                frame_id = item + self.margin_left

                ici = self.image_clip_fn(self) if self.image_clip_fn else self.image_clip_ids
                bci = self.bbox_clip_fn(self) if self.bbox_clip_fn else self.bbox_clip_ids

                batch: BatchDict = {
                    'meta': self.get_meta(seq_id, frame_id),
                    'image': default_collate([self.get_image(seq_id, frame_id+c) for c in ici]),
                    'bbox': default_collate([self.get_bbox(seq_id, frame_id+c) for c in bci]),
                    'image_clip_ids': torch.tensor(ici),
                    'bbox_clip_ids': torch.tensor(bci),
                }
                return batch
            else:
                item -= seq_len - self.margin
        else:
            raise IndexError(f'item {item} is out of bounds')


class BaseDataSpace(Protocol):
    """Manages the sharing of interprocess dictionaries, with shared tensors but non-shared non-tensors.

    """
    def __contains__(self, name: str) -> bool: ...
    def __getitem__(self, name: str) -> Dict[str, Any]: ...
    def __setitem__(self, name: str, d: Dict[str, Any]): ...


class BaseDataModule(L.LightningDataModule):
    TRAIN_NAMESPACE = 'dm_train'
    VAL_NAMESPACE = 'dm_val'
    TEST_NAMESPACE = 'dm_test'

    def __init__(
            self,
            train_data_source: Optional[BaseDataSource] = None,
            val_data_source: Optional[BaseDataSource] = None,
            test_data_source: Optional[BaseDataSource] = None,
            data_space: Optional[BaseDataSpace] = None,
            data_space_train: bool = False,
            data_space_val: bool = False,
            data_space_test: bool = False,
            batch_size: int = 4,
            num_workers: int = 0,
            loky: bool = False,
    ):
        super().__init__()
        self.train_init = False
        self.val_init = False
        self.test_init = False

        # post init inside __init__ if not using data_space
        if (data_space is None or not data_space_train) and train_data_source is not None:
            train_data_source.__post_init__()
            self.train_init = True
        if (data_space is None or not data_space_val) and val_data_source is not None:
            val_data_source.__post_init__()
            self.val_init = True
        if (data_space is None or not data_space_test) and test_data_source is not None:
            test_data_source.__post_init__()
            self.test_init = True

        self.train_data_source = train_data_source
        self.val_data_source = val_data_source
        self.test_data_source = test_data_source
        self.data_space = data_space
        self.data_space_train = data_space_train
        self.data_space_val = data_space_val
        self.data_space_test = data_space_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loky = loky

    def prepare_data(self):
        if self.data_space is not None:
            if self.train_data_source is not None and self.data_space_train and self.TRAIN_NAMESPACE not in self.data_space:
                self.train_data_source.__post_init__()
                self.train_init = True
                self.data_space[self.TRAIN_NAMESPACE] = self.train_data_source.__getstate__()
            if self.val_data_source is not None and self.data_space_val and self.VAL_NAMESPACE not in self.data_space:
                self.val_data_source.__post_init__()
                self.val_init = True
                self.data_space[self.VAL_NAMESPACE] = self.val_data_source.__getstate__()
            if self.test_data_source is not None and self.data_space_test and self.TEST_NAMESPACE not in self.data_space:
                self.test_data_source.__post_init__()
                self.test_init = True
                self.data_space[self.TEST_NAMESPACE] = self.test_data_source.__getstate__()

    def setup(self, stage: str) -> None:
        if self.data_space is not None:
            # post init
            if stage == 'fit':
                if self.data_space_train and self.train_init is False:
                    self.train_data_source.__setstate__(self.data_space[self.TRAIN_NAMESPACE])
                    self.train_init = True
                if self.data_space_val and self.val_init is False:
                    self.val_data_source.__setstate__(self.data_space[self.VAL_NAMESPACE])
                    self.val_init = True
            elif stage == 'validate':
                if self.data_space_val and self.val_init is False:
                    self.val_data_source.__setstate__(self.data_space[self.VAL_NAMESPACE])
                    self.val_init = True
            elif stage == 'test':
                if self.data_space_test and self.test_init is False:
                    self.test_data_source.__setstate__(self.data_space[self.TEST_NAMESPACE])
                    self.test_init = True

    def worker_init_fn(self, worker_id: int) -> None:
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')
        # seed = uuid.uuid4().int % 2 ** 32
        # L.seed_everything(seed)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data_source,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            worker_init_fn=self.worker_init_fn if self.num_workers != 0 else None,
            persistent_workers=self.num_workers != 0,
            multiprocessing_context='fork' if self.loky else None,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data_source,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            worker_init_fn=self.worker_init_fn if self.num_workers != 0 else None,
            persistent_workers=self.num_workers != 0,
            multiprocessing_context='fork' if self.loky else None,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data_source,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            worker_init_fn=self.worker_init_fn if self.num_workers != 0 else None,
            persistent_workers=self.num_workers != 0,
            multiprocessing_context='fork' if self.loky else None,
        )
