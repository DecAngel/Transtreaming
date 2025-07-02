import sys
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import convert_inf, _update_n, Tqdm
from lightning.pytorch.utilities.types import STEP_OUTPUT


class NewTqdmProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The same as a train bar
        return Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )

    def on_train_start(self, *_: Any) -> None:
        pass

    def on_train_epoch_start(self, trainer: "L.Trainer", *_: Any) -> None:
        self.train_progress_bar = self.init_train_tqdm()
        self.train_progress_bar.reset(convert_inf(self.total_train_batches))
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"{'Train':>10s} Epoch {trainer.current_epoch}")

    def on_train_batch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            _update_n(self.train_progress_bar, n)
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
        if n == self.train_progress_bar.total:
            # close train bar on last batch, let val bar be the only bar.
            self.train_progress_bar.close()

    def on_validation_batch_start(
            self,
            trainer: "L.Trainer",
            pl_module: "L.LightningModule",
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        self.val_progress_bar.reset(convert_inf(self.total_val_batches_current_dataloader))
        self.val_progress_bar.initial = 0
        desc = self.sanity_check_description if trainer.sanity_checking else self.validation_description
        self.val_progress_bar.set_description(f"{desc:>10s} Epoch {trainer.current_epoch}")

    def on_test_batch_start(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        self.test_progress_bar.reset(convert_inf(self.total_test_batches_current_dataloader))
        self.test_progress_bar.initial = 0
        self.test_progress_bar.set_description(f"{self.test_description:>10s} Epoch {trainer.current_epoch}")

    def on_predict_batch_start(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        self.predict_progress_bar.reset(convert_inf(self.total_predict_batches_current_dataloader))
        self.predict_progress_bar.initial = 0
        self.predict_progress_bar.set_description(f"{self.predict_description:>10s} Epoch {trainer.current_epoch}")
