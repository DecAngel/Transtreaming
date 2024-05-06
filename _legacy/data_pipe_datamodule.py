from typing import Tuple, Optional, Dict

import lightning as L
from torch.utils.data import DataLoader

from src.datamodules.components.data_spaces.nsm_space import TensorDictSpace
from src._legacy.data_pipes.base import DataPipe, DataPipe2Dataset


class DataPipeDataModule(L.LightningDataModule):
    NAMESPACE = 'data'
    TRAIN_NAME = 'train'
    VAL_NAME = 'val'
    TEST_NAME = 'test'

    def __init__(
            self,
            data_pipes: Dict[str, Tuple[DataPipe, ...]],
            space: Optional[TensorDictSpace] = None,
            batch_size: int = 4,
            num_workers: int = 0,
    ):
        super().__init__()
        self.data_pipes = data_pipes
        self.data_pipes.update({
            self.TRAIN_NAME + '_dataset': (DataPipe2Dataset(), self.TRAIN_NAME),
            self.VAL_NAME + '_dataset': (DataPipe2Dataset(), self.VAL_NAME),
            self.TEST_NAME + '_dataset': (DataPipe2Dataset(), self.TEST_NAME),
        })
        self.space = space
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.space is None:
            # if None, init inside __init__
            self.link()
            for k, v in self.data_pipes.items():
                v[0].__post_init__()

    def link(self):
        for k, v in self.data_pipes.items():
            v[0].sources.clear()
            for name in v[1:]:
                v[0].sources.append(self.data_pipes[name][0])

    def prepare_data(self):
        if self.space is not None:
            self.link()
            if self.NAMESPACE in self.space:
                existing_dict = self.space[self.NAMESPACE]
            else:
                existing_dict = {}
            flag = False
            for k, v in self.data_pipes.items():
                if k not in existing_dict:
                    v[0].__post_init__()
                    existing_dict[k] = v[0].__getstate__()
                    flag = True

            if flag:
                self.space[self.NAMESPACE] = existing_dict

    def setup(self, stage: str) -> None:
        if self.space is not None:
            self.link()
            existing_dict = self.space[self.NAMESPACE]
            for k, v in self.data_pipes.items():
                v[0].__setstate__(existing_dict[k])

    def train_dataloader(self):
        return DataLoader(
            self.data_pipes[self.TRAIN_NAME+'_dataset'][0],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.num_workers != 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_pipes[self.VAL_NAME + '_dataset'][0],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.num_workers != 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_pipes[self.TEST_NAME + '_dataset'][0],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.num_workers != 0,
        )
