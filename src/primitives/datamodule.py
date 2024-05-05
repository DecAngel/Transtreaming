from typing import Tuple, Union, List, Dict, Any, Protocol, Optional
from torch.utils.data import Dataset, DataLoader

import lightning as L


class BaseDataPipe(Dataset):
    """A multi-dimensional map data loading component with pickleable state.

    Caution: invoke __len__ directly
    """
    def __init__(self):
        super().__init__()
        self._sources: List['BaseDataPipe'] = []

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


class BaseDataSpace(Protocol):
    """Manages the sharing of interprocess dictionaries, with shared tensors but non-shared non-tensors.

    """
    def __contains__(self, name: str) -> bool: ...
    def __getitem__(self, name: str) -> Dict[str, Any]: ...
    def __setitem__(self, name: str, d: Dict[str, Any]): ...


class BaseDataModule(L.LightningDataModule):
    NAMESPACE = 'data'
    TRAIN_NAME = 'train'
    VAL_NAME = 'val'
    TEST_NAME = 'test'

    def __init__(
            self,
            data_pipes: Dict[str, Tuple[BaseDataPipe, ...]],
            space: Optional[BaseDataSpace] = None,
            batch_size: int = 4,
            num_workers: int = 0,
    ):
        super().__init__()
        self.data_pipes = data_pipes
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
