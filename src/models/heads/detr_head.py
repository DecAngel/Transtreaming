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
    require_prev_frame = False

    def __init__(
            self,
            transformer: RTDETRTransformerv2,
            criterion: RTDETRCriterionv2,
            postprocessor: RTDETRPostProcessor,
    ):
        super().__init__()
        self.detr = transformer
        self.criterion = criterion
        self.postprocessor = postprocessor
        # self.detr = RTDETRTransformerv2(
        #     num_classes=8,
        #     hidden_dim=128,
        #     feat_channels=[256, 512, 1024],
        #     feat_strides=[8, 16, 32],
        #     num_layers=6,
        #     dim_feedforward=512,
        #     learn_query_content=True,
        # )
        # self.criterion = RTDETRCriterionv2(
        #     HungarianMatcher(
        #         {'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
        #         alpha=0.25,
        #         gamma=2.0,
        #     ),
        #     weight_dict={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2,},
        #     losses=['vfl', 'boxes', ],
        #     alpha=0.75,
        #     gamma=2.0,
        #     num_classes=8,
        # )
        # self.postprocessor = RTDETRPostProcessor(
        #     num_classes=8,
        #     use_focal_loss=True,
        # )


    def forward(self, batch: BatchDict) -> BatchDict:
        features = batch['intermediate']['features_p']
        past_clip_ids = batch['past_clip_ids'].float()
        future_clip_ids = batch['future_clip_ids'].float()
        B, TP = past_clip_ids.size()
        _, TF = future_clip_ids.size()

        h, w = batch['meta']['current_size'][0]
        # H, W = batch['meta']['original_size'][0]
        size = torch.tensor([w, h, w, h], device=features[0].device)

        if self.training:
            # logger.info(f'gt_coordinate: {inspect({i: batch["bbox"]["coordinate"][..., i] for i in range(4)})}')
            # logger.info(f"batch: {batch['bbox']['coordinate']}")
            # logger.info(f"size: {size}")
            gt_coordinate = xyxy2cxcywh(batch['bbox']['coordinate']) / size
            # logger.info(f'gt_coordinate: {inspect({i: gt_coordinate[..., i] for i in range(4)})}')
            gt_label = batch['bbox']['label']

            targets = []
            for c, l in zip(gt_coordinate, gt_label):
                c = c[-1]
                l = l[-1]
                mask = torch.any(torch.abs(c) > 1e-5, 1)
                targets.append(
                    {
                        'boxes': c[mask],
                        'labels': l[mask].long(),
                    }
                )
                # if c[mask].shape[0] == 100:
                #     logger.info(f'100: {c[mask]}')
            # logger.info(f'targets: {inspect(targets)}')
        else:
            targets = None

        features = tuple(f.flatten(1, 2) for f in features)
        out = self.detr(features, targets)
        # logger.info(f'out_pred_boxes0: {inspect({i: out["pred_boxes"][0, ..., i] for i in range(4)})}')
        # if targets is not None:
        #     logger.info(f'targets_boxes0: {inspect({i: targets[0]["boxes"][..., i] for i in range(4)})}')

        if self.training:
            loss_dict = self.criterion(out, targets)
            # logger.info(f'loss_dict: {inspect(loss_dict)}')
            loss = functools.reduce(lambda x,y: x+y, list(loss_dict.values()))
            loss_dict['loss'] = loss
            batch['loss'] = loss_dict
        else:
            results = self.postprocessor(out, torch.tensor([[w, h]]*B, device=features[0].device))
            # logger.info(f'results: {inspect({i: results[0]["boxes"][..., i] for i in range(4)})}')
            labels = []
            boxes = []
            scores = []
            for r in results:
                l = r['labels']
                b = r['boxes']
                s = r['scores']
                b[..., [0, 2]] = b[..., [0, 2]].clamp(min=0, max=w)
                b[..., [1, 3]] = b[..., [1, 3]].clamp(min=0, max=h)
                # argsort
                # indices = torch.argsort(s, descending=True)
                # l = l[indices]
                # b = b[indices]
                # s = s[indices]
                l = clip_or_pad_along(l, 0, 100, 0)
                b = clip_or_pad_along(b, 0, 100, 0)
                s = clip_or_pad_along(s, 0, 100, 0)
                labels.append(l)
                boxes.append(b)
                scores.append(s)
            batch['bbox_pred'] = {
                'coordinate': torch.stack(boxes, dim=0).unsqueeze(1).repeat(1, TF, 1, 1),
                'label': torch.stack(labels, dim=0).unsqueeze(1).repeat(1, TF, 1),
                'probability': torch.stack(scores, dim=0).unsqueeze(1).repeat(1, TF, 1),
            }
            # logger.info(f'coordinate: {inspect({i: batch["bbox_pred"]["coordinate"][..., i] for i in range(4)})}')
        return batch
