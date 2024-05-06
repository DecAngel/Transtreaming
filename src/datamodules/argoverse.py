from pathlib import Path
from typing import List, Optional, Sequence

from src.primitives.datamodule import BaseDataModule, BaseDataSpace
from src.datamodules.data_sources.argoverse_source import ArgoverseDataSource


class ArgoverseDataModule(BaseDataModule):
    def __init__(
            self,
            img_dir: str,
            ann_dir: str,
            image_clip_ids: List[int],
            bbox_clip_ids: List[int],
            size: Sequence[int] = (600, 960),
            max_objs: int = 100,
            data_space: Optional[BaseDataSpace] = None,
            data_space_train: bool = False,
            data_space_val: bool = False,
            data_space_test: bool = False,
            batch_size: int = 4,
            num_workers: int = 0,
    ):
        super().__init__(
            ArgoverseDataSource(
                img_dir, str(Path(ann_dir) / 'train.json'), image_clip_ids, [0] + bbox_clip_ids,
                data_space_train, size, max_objs,
            ),
            ArgoverseDataSource(
                img_dir, str(Path(ann_dir) / 'val.json'), image_clip_ids, bbox_clip_ids,
                data_space_val, size, max_objs,
            ),
            ArgoverseDataSource(
                img_dir, str(Path(ann_dir) / 'val.json'), image_clip_ids, bbox_clip_ids,
                data_space_test, size, max_objs,
            ),
            data_space,
            data_space_train,
            data_space_val,
            data_space_test,
            batch_size,
            num_workers
        )
