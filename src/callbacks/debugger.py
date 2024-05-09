from lightning import Callback
import lightning as L

from src.primitives.batch import BatchDict, LossDict
from src.primitives.model import BaseModel
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class Debugger(Callback):
    def __init__(self):
        super().__init__()
        self.batches = []
        self._state = None

    def change(self):
        log.info(f'{self.state} epoch image_ids: {self.batches}')
        self.batches.clear()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if self._state is None:
            self._state = state
        elif self._state != state:
            self.change()
            self._state = state

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: BaseModel) -> None:
        self.state = 'train'

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: BaseModel) -> None:
        self.state = 'val'

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: BaseModel) -> None:
        self.state = 'test'

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: BaseModel) -> None:
        self.state = None

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: BaseModel) -> None:
        self.state = None

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: BaseModel) -> None:
        self.state = None

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: BaseModel, outputs: LossDict, batch: BatchDict, batch_idx: int
    ) -> None:
        self.batches.extend(batch["meta"]["image_id"].cpu().tolist())

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: BaseModel,
        outputs: BatchDict,
        batch: BatchDict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.batches.extend(batch["meta"]["image_id"].cpu().tolist())

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: BaseModel,
        outputs: BatchDict,
        batch: BatchDict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.batches.extend(batch["meta"]["image_id"].cpu().tolist())
