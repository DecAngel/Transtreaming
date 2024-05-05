import contextlib
from pathlib import Path
from typing import Optional, Tuple, Union, Dict

import lightning as L
import torch

import torch.nn as nn
from torchmetrics import Metric

from src.primitives.batch import (
    BatchDict, MetricDict, LossDict, IMAGE, PYRAMID, TIME,
    COORDINATE, LABEL, PROBABILITY, SCALAR, BBoxDict,
    BufferDict,
)
from src.utils.pylogger import RankedLogger


logger = RankedLogger(__name__, rank_zero_only=True)


class BaseBackbone(nn.Module):
    def forward(self, image: IMAGE) -> PYRAMID:
        raise NotImplementedError()


class BaseNeck(nn.Module):
    def forward(
            self,
            features: PYRAMID,
            past_clip_ids: Optional[TIME] = None,
            future_clip_ids: Optional[TIME] = None,
    ) -> PYRAMID:
        raise NotImplementedError()


class BaseHead(nn.Module):
    def forward(
            self,
            features: PYRAMID,
            gt_coordinates: Optional[COORDINATE] = None,
            gt_labels: Optional[LABEL] = None,
            shape: Optional[Tuple[int, int]] = (600, 960),
    ) -> Union[BBoxDict, LossDict]:
        raise NotImplementedError()



class BaseMetric(Metric):
    def update(self, batch: BatchDict, pred: BatchDict) -> None: raise NotImplementedError()

    def compute(self) -> LossDict: raise NotImplementedError()


class BaseTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.flag = True

    def forward(self, batch: BatchDict) -> BatchDict:
        if self.flag:
            self.flag = False
            return self.preprocess(batch)
        else:
            self.flag = True
            return self.postprocess(batch)

    def preprocess(self, batch: BatchDict) -> BatchDict:
        return batch

    def postprocess(self, batch: BatchDict) -> BatchDict:
        return batch


class Model(L.LightningModule):
    def __init__(
            self,
            backbone: BaseBackbone,
            neck: BaseNeck,
            head: BaseHead,
            transform: Optional[BaseTransform] = None,
            metric: Optional[BaseMetric] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.transform = transform
        self.metric = metric

    def pth_adapter(self, state_dict: Dict) -> Dict:
        raise NotImplementedError()

    def load_from_pth(self, file_path: Union[str, Path]) -> None:
        state_dict = torch.load(str(file_path), map_location='cpu')
        with contextlib.suppress(NotImplementedError):
            state_dict = self.pth_adapter(state_dict)

        misshaped_keys = []
        ssd = self.state_dict()
        for k in list(state_dict.keys()):
            if k in ssd and ssd[k].shape != state_dict[k].shape:
                misshaped_keys.append(k)
                del state_dict[k]

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        missing_keys = list(filter(lambda key: key not in misshaped_keys, missing_keys))

        if len(missing_keys) > 0:
            logger.warning(f'Missing keys in ckpt: {missing_keys}')
        if len(unexpected_keys) > 0:
            logger.warning(f'Unexpected keys in ckpt: {unexpected_keys}')
        if len(misshaped_keys) > 0:
            logger.warning(f'Misshaped keys in ckpt: {misshaped_keys}')
        logger.info(f'pth file {file_path} loaded!')

    def load_from_ckpt(self, file_path: Union[str, Path], strict: bool = True):
        self.load_state_dict(
            torch.load(
                str(file_path),
                map_location='cpu'
            )['state_dict'],
            strict=strict,
        )
        logger.info(f'ckpt file {file_path} loaded!')

    @property
    def fraction_epoch(self):
        return self.trainer.global_step / self.trainer.estimated_stepping_batches


    def _forward_impl(
            self,
            image: IMAGE,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
            buffer: Optional[BufferDict] = None,
            bbox_coordinate: Optional[COORDINATE] = None,
            bbox_label: Optional[LABEL] = None,
    ) -> :

    def forward_impl(
            self,
            batch: BatchDict,
            buffer: Optional[Dict],
    ) -> Tuple[BatchDict, LossDict, Optional[Dict]]:
        raise NotImplementedError()

    def forward(self, batch: BatchDict) -> Tuple[BatchDict, LossDict]

    def inference(
            self,
            batch: BatchDict,
            buffer: Optional[Dict],
    ) -> Tuple[BatchDict, LossDict, Optional[Dict]]:
        with torch.inference_mode():
            pred, metric, buf = self.forward_impl(batch, buffer)
            return self.inference_impl(batch, buffer)

    def forward(
            self,
            batch: BatchTDict,
    ) -> Union[BatchTDict, LossDict]:
        with torch.inference_mode():
            batch = self.transform.preprocess_tensor(batch) if self.transform is not None else batch
        with contextlib.nullcontext() if self.training else torch.inference_mode():
            return self.forward_impl(batch)

    def training_step(self, batch: BatchTDict, *args, **kwargs) -> LossDict:
        output: LossDict = self(batch)
        self.log_dict(output, on_step=True, prog_bar=True)
        return output

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        if self.metric is not None:
            self.metric.reset()

    def validation_step(self, batch: BatchTDict, *args, **kwargs) -> BatchTDict:
        output: BatchTDict = self(batch)
        if not self.trainer.sanity_checking and self.metric is not None:
            self.metric.update(batch, output)
        return output

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        if not self.trainer.sanity_checking and self.metric is not None:
            self.log_dict(self.metric.compute(), on_epoch=True, prog_bar=True)
        return None

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        return self.on_validation_epoch_start()

    def test_step(self, batch: BatchTDict, *args, **kwargs) -> BatchTDict:
        return self.validation_step(batch)

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        return self.on_validation_epoch_end()

    @staticmethod
    def freeze_module(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    @staticmethod
    def unfreeze_module(module: nn.Module):
        for param in module.parameters():
            param.requires_grad = True
