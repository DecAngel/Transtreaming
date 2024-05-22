import contextlib
import json
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Literal, Mapping, Any, List

import lightning as L
import torch

import torch.nn as nn
from torchmetrics import Metric

from src.primitives.batch import (
    BatchDict, MetricDict, LossDict, IMAGE, PYRAMID, TIME,
    COORDINATE, LABEL, PROBABILITY, SCALAR, BBoxDict,
    BufferDict, SIZE
)
from src.utils.pylogger import RankedLogger
from src.utils.array_operations import slice_along
from src.utils.time_recorder import TimeRecorder

log = RankedLogger(__name__, rank_zero_only=True)


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


class BlockMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer: Optional[L.Trainer] = None

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
        return self.trainer.global_step / self.trainer.estimated_stepping_batches

    @property
    def total_epoch(self):
        return self.trainer.max_epochs


class BaseBackbone(BlockMixin, nn.Module):
    state_dict_replace: List[Tuple[str, str]] = []

    def forward(self, image: IMAGE) -> PYRAMID:
        raise NotImplementedError()


class BaseNeck(BlockMixin, nn.Module):
    state_dict_replace: List[Tuple[str, str]] = []
    input_frames: int = 2
    output_frames: int = 2

    def forward(
            self,
            features: PYRAMID,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
    ) -> PYRAMID:
        raise NotImplementedError()


class BaseHead(BlockMixin, nn.Module):
    state_dict_replace: List[Tuple[str, str]] = []
    require_prev_frame: bool = True

    def forward(
            self,
            features: PYRAMID,
            gt_coordinate: Optional[COORDINATE] = None,
            gt_label: Optional[LABEL] = None,
            shape: Optional[SIZE] = None,
    ) -> Union[BBoxDict, LossDict]:
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


class BaseModel(L.LightningModule):
    def __init__(
            self,
            backbone: BaseBackbone,
            neck: BaseNeck,
            head: BaseHead,
            transform: Optional[BaseTransform] = None,
            metric: Optional[BaseMetric] = None,
            optim: Optional[BaseOptim] = None,
            torch_compile: Optional[Literal['default', 'reduce-overhead', 'max-autotune']] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.transform = transform
        self.metric = metric
        self.optim = optim
        if torch_compile is not None:
            self._forward_impl = torch.compile(
                self.forward_impl, fullgraph=False, dynamic=True, mode=torch_compile
            )

    def setup(self, stage: str) -> None:
        for b in filter(None, [self.backbone, self.neck, self.head, self.transform, self.metric, self.optim]):
            b.trainer = self.trainer

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
        state_dict = torch.load(str(file_path), map_location='cpu')['model']

        # replace
        replacements = self.backbone.state_dict_replace + self.neck.state_dict_replace + self.head.state_dict_replace
        new_ckpt = {}
        for k, v in state_dict.items():
            for r1, r2 in replacements:
                k = k.replace(r1, r2)
            new_ckpt[k] = v
        state_dict = new_ckpt

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
                map_location='cpu'
            )['state_dict'],
            strict=strict,
        )
        log.info(f'ckpt file {file_path} loaded!')

    def forward_impl(
            self,
            image: IMAGE,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
            gt_coordinate: Optional[COORDINATE] = None,
            gt_label: Optional[LABEL] = None,
            shape: Optional[SIZE] = None,
    ) -> Union[BBoxDict, LossDict]:
        """
        Image:                     image
        Buffer:     buffer
        BBox:                                       pred
        Clip_id:    |---past_clip_ids---|  |---future_clip_ids---|
        """
        pci = past_clip_ids.float()
        fci = future_clip_ids[:, (int(self.head.require_prev_frame) if self.training else 0):].float()
        _, TP = pci.size()
        _, TF = fci.size()

        # inference
        features_p = self.backbone(image)
        features_f = self.neck(features_p, pci, fci)
        loss_pred = self.head(features_f, gt_coordinate, gt_label, shape)
        return loss_pred

    def inference_backbone_neck(
            self,
            image: IMAGE,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
            buffer: Optional[PYRAMID] = None,
    ) -> Tuple[PYRAMID, PYRAMID]:
        """Get forecasted features and buffer features"""
        B, TI, C, H, W = image.size()
        pci = past_clip_ids.float()
        fci = future_clip_ids[:, (int(self.head.require_prev_frame) if self.training else 0):].float()
        _, TP = pci.size()
        _, TF = fci.size()
        assert B == 1

        # Backbone
        features_p = self.backbone(image)

        # Concatenate
        if buffer is not None:
            # append buffer to features_p and past_time_constant
            features_p = concat_pyramids([buffer, features_p], dim=1)
        # pad
        TB = features_p[0].size(1)
        if TB < TP:
            features_p = concat_pyramids(
                [slice_pyramid(features_p, dim=1, start=0, end=1)] * (TP - TB) + [features_p], dim=1
            )
        elif TB > TP:
            features_p = slice_pyramid(features_p, dim=1, start=TB - TP, end=TB)

        # Neck
        features_f = self.neck(features_p, pci, fci)

        return features_f, features_p

    def inference_head(
            self,
            features_f: PYRAMID,
            shape: Optional[SIZE] = None
    ) -> BBoxDict:
        """Get detection"""
        return self.head(features_f, None, None, shape)

    def forward(self, batch: BatchDict) -> Union[BatchDict, LossDict]:
        with torch.inference_mode():
            batch = self.transform.preprocess(batch) if self.transform is not None else batch
        if self.training:
            loss_dict = self.forward_impl(
                image=batch['image']['image'],
                past_clip_ids=batch['image_clip_ids'],
                future_clip_ids=batch['bbox_clip_ids'],
                gt_coordinate=batch['bbox']['coordinate'],
                gt_label=batch['bbox']['label'],
                shape=batch['meta']['current_size'],
            )
            return loss_dict
        else:
            with torch.inference_mode():
                bbox = self.forward_impl(
                    image=batch['image']['image'],
                    past_clip_ids=batch['image_clip_ids'],
                    future_clip_ids=batch['bbox_clip_ids'],
                    shape=batch['meta']['current_size'],
                )
                batch['bbox_pred'] = bbox
                batch = self.transform.postprocess(batch) if self.transform is not None else batch
            return batch

    def inference(
            self,
            image: IMAGE,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
            buffer: Optional[PYRAMID] = None,
    ) -> Tuple[BBoxDict, PYRAMID]:
        with torch.inference_mode():
            shape = torch.tensor([image.shape[-2:]], dtype=torch.long, device=self.device)
            feature_f, features_p = self.inference_backbone_neck(
                image, past_clip_ids, future_clip_ids, buffer
            )
            return self.inference_head(feature_f, shape), features_p

    def training_step(self, batch: BatchDict, *args, **kwargs) -> LossDict:
        output: LossDict = self(batch)
        self.log('loss', output['loss'], on_step=True, prog_bar=True)
        self.log_dict({k: v for k, v in output.items() if k != 'loss'}, on_step=True)
        return output

    def on_validation_epoch_start(self) -> None:
        if self.metric is not None:
            self.metric.reset()

    def validation_step(self, batch: BatchDict, *args, **kwargs) -> BatchDict:
        output: BatchDict = self(batch)
        if not self.trainer.sanity_checking and self.metric is not None:
            self.metric.update(output)
        return output

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking and self.metric is not None:
            d = {f'val_{k}': v.cpu().item() for k, v in self.metric.compute().items()}
            log.info(f'Val Metric Dict: {json.dumps(d, indent=2)}')
            self.log('val_mAP', d.pop('val_mAP'), on_epoch=True, prog_bar=True)
            self.log_dict(d, on_epoch=True)
        return None

    def on_test_epoch_start(self) -> None:
        if self.metric is not None:
            self.metric.reset()

    def test_step(self, batch: BatchDict, *args, **kwargs) -> BatchDict:
        output: BatchDict = self(batch)
        if not self.trainer.sanity_checking and self.metric is not None:
            self.metric.update(output)
        return output

    def on_test_epoch_end(self) -> None:
        if not self.trainer.sanity_checking and self.metric is not None:
            d = {f'test_{k}': v.cpu().item() for k, v in self.metric.compute().items()}
            log.info(f'Test Metric Dict: {json.dumps(d, indent=2)}')
            self.log('test_mAP', d.pop('test_mAP'), on_epoch=True, prog_bar=True)
            self.log_dict(d, on_epoch=True)
        return None
