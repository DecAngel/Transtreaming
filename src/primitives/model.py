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


logger = RankedLogger(__name__, rank_zero_only=True)


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
            past_clip_ids: Optional[TIME] = None,
            future_clip_ids: Optional[TIME] = None,
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
            self._forward_impl = torch.compile(self._forward_impl, fullgraph=True, dynamic=True, mode=torch_compile)

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
                'current_size': torch.tensor([[100, 200]]*b, dtype=torch.long),
                'original_size': torch.tensor([[200, 400]]*b, dtype=torch.long),
            },
            'image': {
                'image': torch.rand(b, tp, 3, 100, 200),
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

    def _forward_impl(
            self,
            image: IMAGE,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
            buffer: Optional[BufferDict] = None,
            buffer_delay: Optional[int] = None,
            gt_coordinate: Optional[COORDINATE] = None,
            gt_label: Optional[LABEL] = None,
            shape: Optional[SIZE] = None,
    ) -> Tuple[Union[BBoxDict, LossDict], BufferDict]:
        if buffer is not None:
            # requires buffer, batch_size must be 1
            assert image.size(0) == 1
            buffer_list = buffer['buffer_list'] if 'buffer_list' in buffer else []
            buffer_clip_id_list = buffer['buffer_clip_id_list'] if 'buffer_clip_id_list' in buffer else []
            buffer_clip_id_list = [b - buffer_delay for b in buffer_clip_id_list]

            new_buffer_list = []
            new_buffer_clip_id_list = []
            for i, p in enumerate(past_clip_ids[0].cpu().tolist()):
                if p in buffer_clip_id_list:
                    new_buffer_list.append(buffer_list[buffer_clip_id_list.index(p)])
                else:
                    new_buffer_list.append(self.backbone(image[:, i:i+1]))
                new_buffer_clip_id_list.append(p)
            features_p = tuple([torch.cat([f[i] for f in new_buffer_list], dim=1) for i in range(len(new_buffer_list[0]))])
            new_buffer = {
                'buffer_list': new_buffer_list,
                'buffer_clip_id_list': new_buffer_clip_id_list
            }
        else:
            features_p = self.backbone(image)
            new_buffer = None

        pci = past_clip_ids[:, :-1].float()
        fci = future_clip_ids[:, (int(self.head.require_prev_frame) if self.training else 0):].float()
        features_f = self.neck(features_p, pci, fci)

        return self.head(features_f, gt_coordinate, gt_label, shape), new_buffer

    def forward(self, batch: BatchDict) -> Union[BatchDict, LossDict]:
        with torch.inference_mode():
            batch = self.transform.preprocess(batch) if self.transform is not None else batch
        if self.training:
            loss_dict, _ = self._forward_impl(
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
                bbox, _ = self._forward_impl(
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
            buffer: Optional[BufferDict] = None,
            buffer_delay: Optional[int] = None,
    ) -> Tuple[BBoxDict, BufferDict]:
        with torch.inference_mode():
            return self._forward_impl(image, past_clip_ids, future_clip_ids, buffer, buffer_delay)

    def training_step(self, batch: BatchDict, *args, **kwargs) -> LossDict:
        output: LossDict = self(batch)
        self.log_dict(output, on_step=True, prog_bar=True)
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
            self.log_dict(d, on_epoch=True, prog_bar=True)
            logger.info(f'{d}')
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
            self.log_dict(d, on_epoch=True, prog_bar=False)
            logger.info(f'{d}')
        return None
