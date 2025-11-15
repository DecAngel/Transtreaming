import functools

import torch

from src.models.layers.detr.rtdetrv2_decoder import RTDETRTransformerv2
from src.models.layers.detr.rtdetrv2_criterion import RTDETRCriterionv2
from src.models.layers.detr.rtdetr_postprocessor import RTDETRPostProcessor
from src.primitives.batch import PYRAMID, TIME, COORDINATE, LABEL, SIZE, BBoxDict, LossDict, BatchDict
from src.primitives.model import BaseHead
from src.utils.array_operations import xyxy2cxcywh, cxcywh2xyxy, remove_pad_along, clip_or_pad_along
from src.utils.inspection import inspect
from src.utils.pylogger import RankedLogger


logger = RankedLogger(__name__)


class DetrHead(BaseHead):
    state_dict_replace = [
        ('decoder.decoder', 'head.detr.xxxxx'),
        ('decoder', 'head.detr'),
        ('xxxxx', 'decoder'),
    ]

    def __init__(
            self,
            transformer: RTDETRTransformerv2,
            criterion: RTDETRCriterionv2,
            postprocessor: RTDETRPostProcessor,
            remap: dict[str, list[int]] | None = None,
            prev_frame: bool = False,
            relative: bool = False,
    ):
        super().__init__()
        self.detr = transformer
        self.criterion = criterion
        self.postprocessor = postprocessor
        if remap is not None:
            self.state_dict_remap = remap
        self.require_prev_frame = prev_frame
        self.relative = relative

    def forward(self, batch: BatchDict) -> BatchDict:
        features = batch['intermediate']['features_f']
        past_time = batch['past_clip_ids'].to(dtype=features[0].dtype)
        future_time = batch['future_clip_ids'].to(dtype=features[0].dtype)
        B, TP = past_time.size()
        _, TF = future_time.size()
        if self.relative and TP > 1:
            unit = torch.maximum(torch.abs(past_time[:, -2:-1]), torch.ones_like(past_time[:, -1:]))
            future_time = future_time / unit

        h, w = batch['meta']['current_size'][0]
        size = torch.tensor([w, h, w, h], device=features[0].device)

        # if type(self.detr) is RTDETRTransformerv2:
        #     # original DETR
        #     if self.training:
        #         gt_coordinate = xyxy2cxcywh(batch['bbox']['coordinate']) / size
        #         gt_label = batch['bbox']['label']
        #
        #         targets = []
        #         for c, l in zip(gt_coordinate, gt_label):
        #             c = c[0]
        #             l = l[0]
        #             mask = torch.any(torch.abs(c) > 1e-5, 1)
        #             targets.append(
        #                 {
        #                     'boxes': c[mask],
        #                     'labels': l[mask].long(),
        #                 }
        #             )
        #     else:
        #         targets = None
        #
        #     features = tuple(f[:, -1] for f in features)
        #     out = self.detr(features, targets)
        #     # logger.info(f'out_pred_boxes0: {inspect({i: out["pred_boxes"][0, ..., i] for i in range(4)})}')
        #     # if targets is not None:
        #     #     logger.info(f'targets_boxes0: {inspect({i: targets[0]["boxes"][..., i] for i in range(4)})}')
        #
        #     if self.training:
        #         loss_dict = self.criterion(out, targets)
        #         loss = functools.reduce(lambda x,y: x+y, list(loss_dict.values()))
        #         loss_dict['loss'] = loss
        #         batch['loss'] = loss_dict
        #     else:
        #         results = self.postprocessor(out, torch.tensor([[w, h]]*B, device=features[0].device))
        #         # logger.info(f'results: {inspect({i: results[0]["boxes"][..., i] for i in range(4)})}')
        #         labels = []
        #         boxes = []
        #         scores = []
        #         for r in results:
        #             l = r['labels']
        #             b = r['boxes']
        #             s = r['scores']
        #             b[..., [0, 2]] = b[..., [0, 2]].clamp(min=0, max=w)
        #             b[..., [1, 3]] = b[..., [1, 3]].clamp(min=0, max=h)
        #             l = clip_or_pad_along(l, 0, 100, 0)
        #             b = clip_or_pad_along(b, 0, 100, 0)
        #             s = clip_or_pad_along(s, 0, 100, 0)
        #             labels.append(l)
        #             boxes.append(b)
        #             scores.append(s)
        #         batch['bbox_pred'] = {
        #             'coordinate': torch.stack(boxes, dim=0).unsqueeze(1).repeat(1, TF, 1, 1),
        #             'label': torch.stack(labels, dim=0).unsqueeze(1).repeat(1, TF, 1),
        #             'probability': torch.stack(scores, dim=0).unsqueeze(1).repeat(1, TF, 1),
        #         }
        #     return batch
        # else:
        if self.training:
            gt_coordinate = xyxy2cxcywh(batch['bbox']['coordinate']) / size
            gt_label = batch['bbox']['label']

            targets = []
            for cs, ls in zip(gt_coordinate, gt_label):
                for i, (c, l) in enumerate(zip(cs, ls)):
                    if i == 0 and self.require_prev_frame is False:
                        continue
                    mask = torch.any(torch.abs(c) > 1e-5, 1)
                    targets.append(
                        {
                            'boxes': c[mask],
                            'labels': l[mask].long(),
                        }
                    )
        else:
            targets = None

        # targets: list<B*TF>[dict<boxes,labels>[nobj,4]]
        # logger.info(f'DetrHead: {inspect(locals())}')
        # logger.info(f'DetrHead targets:{inspect(targets)}')

        out = self.detr(features, targets, past_time=past_time, future_time=future_time)
        # logger.info(f'DetrHead out:{inspect(out)}')

        if self.training:
            loss_dict = self.criterion(out, targets, future_time=future_time)
            loss = functools.reduce(lambda x, y: x + y, list(loss_dict.values()))
            loss_dict['loss'] = loss
            batch['loss'] = loss_dict
        else:
            results = self.postprocessor(out, torch.tensor([[w, h]] * B * TF, device=features[0].device))
            # logger.info(f'results: {inspect(results)}')
            labels = []
            boxes = []
            scores = []
            for batch_index in range(B):
                labels.append([])
                boxes.append([])
                scores.append([])
                for fut_index in range(TF):
                    r = results[batch_index*TF+fut_index]
                    l = r['labels']
                    b = r['boxes']
                    s = r['scores']
                    b[..., [0, 2]] = b[..., [0, 2]].clamp(min=0, max=w)
                    b[..., [1, 3]] = b[..., [1, 3]].clamp(min=0, max=h)
                    l = clip_or_pad_along(l, 0, 100, 0)
                    b = clip_or_pad_along(b, 0, 100, 0)
                    s = clip_or_pad_along(s, 0, 100, 0)
                    labels[-1].append(l)
                    boxes[-1].append(b)
                    scores[-1].append(s)
            batch['bbox_pred'] = {
                'coordinate': torch.stack([torch.stack(b, dim=0) for b in boxes], dim=0),
                'label': torch.stack([torch.stack(l, dim=0) for l in labels], dim=0),
                'probability': torch.stack([torch.stack(p, dim=0) for p in scores], dim=0),
            }
        return batch
