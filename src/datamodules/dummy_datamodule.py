from src.primitives.datamodule import BaseDataModule
from src.datamodules.data_sources.dummy_source import DummyDataSource


class DummyDataModule(BaseDataModule):
    def __init__(self):
        super().__init__(
            DummyDataSource(),
            DummyDataSource(),
            DummyDataSource(),
        )
