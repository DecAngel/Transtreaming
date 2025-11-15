import contextlib
import json
from pathlib import Path
from typing import Optional, Tuple, Union, Literal, List, ClassVar, Dict

import lightning as L
import torch

import torch.nn as nn
from torchmetrics import Metric

from src.primitives.batch import (
    BatchDict, MetricDict, LossDict
)
from src.utils.collection_operations import concat_pyramids, slice_pyramid
from src.utils.inspection import inspect
from src.utils.pylogger import RankedLogger
from src.utils.time_recorder import TimeRecorder

log = RankedLogger(__name__, rank_zero_only=True)


class BlockMixin:
    _trainer: ClassVar[Optional[L.Trainer]] = None
    _time_recorder: ClassVar[Optional[TimeRecorder]] = TimeRecorder()
    _time_recorder_enable: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def freeze(self):
        if isinstance(self, nn.Module):
            for p in self.parameters():
                p.requires_grad = False

    def unfreeze(self):
        if isinstance(self, nn.Module):
            for p in self.parameters():
                p.requires_grad = True

    @property
    def fraction_epoch(self):
        return self._trainer.global_step / self._trainer.estimated_stepping_batches

    @property
    def total_epoch(self):
        return self._trainer.max_epochs

    @contextlib.contextmanager
    def record_time(self, tag: str):
        if self._time_recorder_enable:
            self._time_recorder.record_start(tag)
            try:
                yield None
            finally:
                torch.cuda.synchronize()
                self._time_recorder.record_end(tag)
        else:
            yield None

    def print_record_time(self):
        self._time_recorder.print()


class BaseBackbone(BlockMixin, nn.Module):
    state_dict_location: List[str] = ['model']
    state_dict_replace: List[Tuple[str, str]] = []
    state_dict_remap: Dict[str, List[int]] = {}

    def forward(self, batch: BatchDict) -> BatchDict:
        raise NotImplementedError()


class BaseNeck(BlockMixin, nn.Module):
    state_dict_replace: List[Tuple[str, str]] = []
    state_dict_remap: Dict[str, List[int]] = {}
    input_frames: int = 2
    output_frames: int = 2

    def forward(self, batch: BatchDict) -> BatchDict:
        raise NotImplementedError()


class BaseHead(BlockMixin, nn.Module):
    state_dict_replace: List[Tuple[str, str]] = []
    state_dict_remap: Dict[str, List[int]] = {}
    require_prev_frame: bool = True

    def forward(self, batch: BatchDict) -> BatchDict:
        raise NotImplementedError()


class BaseMetric(BlockMixin, Metric):
    def update(self, output: BatchDict, **kwargs) -> None: raise NotImplementedError()

    def compute(self) -> MetricDict: raise NotImplementedError()


class BaseTransform(BlockMixin, nn.Module):
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


class BaseOptim(BlockMixin):
    def configure_optimizers(
            self,
            backbone: BaseBackbone,
            neck: BaseNeck,
            head: BaseHead,
    ): raise NotImplementedError()


class BaseModel(BlockMixin, L.LightningModule):
    def __init__(
            self,
            backbone: BaseBackbone,
            neck: BaseNeck,
            head: BaseHead,
            transform: Optional[BaseTransform] = None,
            metric: Optional[BaseMetric] = None,
            optim: Optional[BaseOptim] = None,
            torch_compile: Optional[Literal['default', 'reduce-overhead', 'max-autotune']] = None,
            record_interval: int = 0,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.transform = transform
        self.metric = metric
        self.optim = optim
        self.record_interval = record_interval
        self.record_count = 0
        if torch_compile is not None:
            self.forward = torch.compile(
                self.forward, fullgraph=False, dynamic=True, mode=torch_compile
            )
            self.inference = torch.compile(
                self.inference, fullgraph=False, dynamic=False, mode=torch_compile
            )

    def setup(self, stage: str) -> None:
        BlockMixin._time_recorder_enable = (self.record_interval != 0)
        BlockMixin._trainer = self.trainer

    @property
    def example_input_array(self):
        b = 2
        tp = self.neck.input_frames
        tf = self.neck.output_frames
        return {
            'meta': {
                'seq_id': torch.randint(0, 10, size=(b, ), dtype=torch.long),
                'frame_id': torch.randint(0, 900, size=(b, ), dtype=torch.long),
                'image_id': torch.randint(0, 10000, size=(b, ), dtype=torch.long),
                'current_size': torch.tensor([[600, 960]]*b, dtype=torch.long),
                'original_size': torch.tensor([[1200, 1920]]*b, dtype=torch.long),
            },
            'image': {
                'image': torch.randint(0, 255, size=(b, tp, 3, 600, 960), dtype=torch.float32),
            },
            'image_clip_ids': torch.stack([torch.arange(-tp+1, 1, dtype=torch.long)]*b, dim=0),
            'bbox': {
                'coordinate': torch.randint(0, 100, size=(b, tf, 100, 4), dtype=torch.float32),
                'label': torch.randint(0, 10, size=(b, tf, 100,), dtype=torch.long),
                'probability': torch.ones(b, tf, 100, dtype=torch.float32),
            },
            'bbox_clip_ids': torch.stack([torch.arange(1, tf+1, dtype=torch.long)]*b, dim=0),
        },

    def configure_optimizers(self):
        if self.optim is not None:
            return self.optim.configure_optimizers(self.backbone, self.neck, self.head)
        else:
            return None

    def load_from_pth(self, file_path: Union[str, Path]) -> None:
        state_dict = torch.load(str(file_path), map_location='cpu', weights_only=False)
        for key in self.backbone.state_dict_location:
            if key in state_dict:
                state_dict = state_dict[key]
            else:
                break

        # replace
        replacements = self.backbone.state_dict_replace + self.neck.state_dict_replace + self.head.state_dict_replace
        new_ckpt = {}
        for k, v in state_dict.items():
            for r1, r2 in replacements:
                k = k.replace(r1, r2)
            new_ckpt[k] = v
        state_dict = new_ckpt

        # remap
        remaps = self.backbone.state_dict_remap | self.neck.state_dict_remap | self.head.state_dict_remap
        for k in state_dict.keys():
            if k in remaps:
                state_dict[k] = state_dict[k][remaps[k]]

        misshaped_keys = []
        ssd = self.state_dict()
        for k in list(state_dict.keys()):
            if k in ssd and ssd[k].shape != state_dict[k].shape:
                misshaped_keys.append(k)
                del state_dict[k]

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        missing_keys = list(filter(lambda key: key not in misshaped_keys, missing_keys))

        if len(missing_keys) > 0:
            log.warning(f'Missing keys in ckpt: {missing_keys}')
        if len(unexpected_keys) > 0:
            log.warning(f'Unexpected keys in ckpt: {unexpected_keys}')
        if len(misshaped_keys) > 0:
            log.warning(f'Misshaped keys in ckpt: {misshaped_keys}')
        log.info(f'pth file {file_path} loaded!')

    def load_from_ckpt(self, file_path: Union[str, Path], strict: bool = True):
        self.load_state_dict(
            torch.load(
                str(file_path),
                map_location='cpu',
                weights_only=False,
            )['state_dict'],
            strict=strict,
        )
        log.info(f'ckpt file {file_path} loaded!')

    def forward(self, batch: BatchDict) -> BatchDict:
        """
        Image:                     image
        Buffer:     buffer
        BBox:                                       pred
        Clip_id:    |---image_clip_ids---|  |---bbox_clip_ids---|
                    |---past_clip_ids---|  |---future_clip_ids---|
        """
        # Define past and future clip_ids
        batch['past_clip_ids'] = batch['image_clip_ids']
        batch['future_clip_ids'] = batch['bbox_clip_ids'][:, (int(not self.head.require_prev_frame) if self.training else 0):]
        batch['loss'] = {}
        batch['intermediate'] = {}
        batch['metric'] = {}
        if 'buffer' not in batch:
            batch['buffer'] = {}

        # Preprocess
        # log.info(f'origin: {inspect(batch)}')
        # log.info(f'origin_bbox: {batch["bbox"]["coordinate"][0]}')
        with self.record_time('preprocess'):
            with torch.inference_mode():
                batch = self.transform.preprocess(batch) if self.transform is not None else batch
        # log.info(f'preprocess: {inspect(batch)}')
        # log.info(f'preprocess_bbox: {batch["bbox"]["coordinate"][0]}')
        # Forward
        with self.record_time('backbone'):
            batch: BatchDict = self.backbone(batch)
        with self.record_time('neck'):
            batch = self.neck(batch)
        with self.record_time('head'):
            batch = self.head(batch)

        # Postprocess
        with self.record_time('postprocess'):
            batch = self.transform.postprocess(batch) if self.transform is not None else batch

        self.record_count += 1
        if self.record_interval != 0 and self.record_count % self.record_interval == 0:
            self.print_record_time()
            self._time_recorder.restart()
        return batch

    def inference_backbone_neck(self, batch: BatchDict) -> BatchDict:
        """Get forecasted features and buffer features"""
        batch['intermediate'] = {}
        past_clip_ids = batch['past_clip_ids']
        future_clip_ids = batch['future_clip_ids']
        B, TP = past_clip_ids.size()
        _, TF = future_clip_ids.size()

        # Backbone
        with self.record_time('backbone'):
            batch = self.backbone(batch)

        # Concatenate
        if 'buffer' in batch:
            # append buffer to features_p and past_time_constant
            batch['intermediate']['features_p'] = concat_pyramids(
                [batch['buffer']['features_p'], batch['intermediate']['features_p']],
                dim=1
            )
        else:
            batch['buffer'] = {}

        # pad
        TB = batch['intermediate']['features_p'][0].size(1)
        if TB < TP:
            batch['intermediate']['features_p'] = concat_pyramids(
                [slice_pyramid(batch['intermediate']['features_p'], dim=1, start=0, end=1)] * (TP - TB) + [batch['intermediate']['features_p']],
                dim=1
            )
        elif TB > TP:
            batch['intermediate']['features_p'] = slice_pyramid(batch['intermediate']['features_p'], dim=1, start=TB - TP, end=TB)

        batch['buffer']['features_p'] = batch['intermediate']['features_p']
        # Neck
        with self.record_time('neck'):
            batch = self.neck(batch)

        return batch

    def inference_head(self, batch: BatchDict) -> BatchDict:
        """Get detection"""
        with self.record_time('head'):
            return self.head(batch)

    def inference(self, batch: BatchDict) -> BatchDict:
        with torch.inference_mode():
            batch = self.inference_head(self.inference_backbone_neck(batch))
            self.record_count += 1
            if self.record_interval != 0 and self.record_count % self.record_interval == 0:
                self.print_record_time()
                self._time_recorder.restart()
            return batch

    def training_step(self, batch: BatchDict, *args, **kwargs) -> LossDict:
        batch = self(batch)
        self.log('loss', batch['loss']['loss'], on_step=True, prog_bar=True)
        self.log_dict({k: v for k, v in batch['loss'].items() if k != 'loss' and 'loss' in k}, on_step=True)
        batch['loss'] = batch['loss']['loss']
        return batch

    def on_validation_epoch_start(self) -> None:
        if self.metric is not None:
            self.metric.reset()

    def validation_step(self, batch: BatchDict, *args, **kwargs) -> BatchDict:
        batch = self(batch)
        if not self._trainer.sanity_checking and self.metric is not None:
            self.metric.update(batch)
        return batch

    def on_validation_epoch_end(self) -> None:
        if not self._trainer.sanity_checking and self.metric is not None:
            d = {f'val_{k}': v.cpu().item() for k, v in self.metric.compute().items()}
            log.info(f'Val Metric Dict: {json.dumps(d, indent=2)}')
            self.log('val_mAP', d.pop('val_mAP'), on_epoch=True, prog_bar=True)
            self.log_dict(d, on_epoch=True)
        return None

    def on_test_epoch_start(self) -> None:
        if self.metric is not None:
            self.metric.reset()

    def test_step(self, batch: BatchDict, *args, **kwargs) -> BatchDict:
        batch = self(batch)
        if not self._trainer.sanity_checking and self.metric is not None:
            self.metric.update(batch)
        return batch

    def on_test_epoch_end(self) -> None:
        if not self._trainer.sanity_checking and self.metric is not None:
            d = {f'test_{k}': v.cpu().item() for k, v in self.metric.compute().items()}
            log.info(f'Test Metric Dict: {json.dumps(d, indent=2)}')
            self.log('test_mAP', d.pop('test_mAP'), on_epoch=True, prog_bar=True)
            self.log_dict(d, on_epoch=True)
        return None
