import platform
import contextlib
import io
import json
import tempfile
from typing import Optional, List

import numpy as np
import torch
from torchmetrics.utilities.data import dim_zero_cat
from pycocotools.coco import COCO

from src.primitives.model import BaseMetric
from src.primitives.batch import BatchDict, MetricDict
from src.utils.array_operations import xyxy2xywh
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


coco_eval_version = ""
try:
    if platform.system() == 'Linux':
        from .yolox_cocoeval import COCOeval_opt as COCOeval
        coco_eval_version = 'Using YOLOX COCOeval.'
    else:
        from pycocotools.cocoeval import COCOeval
        coco_eval_version = 'Using pycocotools COCOeval.'
except ImportError:
    from pycocotools.cocoeval import COCOeval
    coco_eval_version = 'Using pycocotools COCOeval.'
logger.info(coco_eval_version)


coco_eval_metric_names = [
    ('Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ', 'AP5095'),
    ('Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = ', 'AP50'),
    ('Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = ', 'AP75'),
    ('Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ', 'APsmall'),
    ('Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ', 'APmedium'),
    ('Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ', 'APlarge'),
    ('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = ', 'AR5095_1'),
    ('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = ', 'AR5095_10'),
    ('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ', 'AR5095_100'),
    ('Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ', 'ARsmall'),
    ('Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ', 'ARmedium'),
    ('Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ', 'ARlarge'),
]


class COCOEvalMetric(BaseMetric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(
            self,
            eval_coco: str,
            test_coco: str,
            future_time_constant: Optional[List[int]] = None,
    ):
        super().__init__(
            dist_sync_on_step=False,
            process_group=None,
            dist_sync_fn=None,
            compute_on_cpu=False,
        )
        self.eval_coco = eval_coco
        self.test_coco = test_coco
        self.future_time_constant = future_time_constant or [0]
        for i, t in enumerate(self.future_time_constant):
            self.add_state(f'image_id_list_{i}', default=[], dist_reduce_fx='cat')
            self.add_state(f'category_id_list_{i}', default=[], dist_reduce_fx='cat')
            self.add_state(f'bbox_list_{i}', default=[], dist_reduce_fx='cat')
            self.add_state(f'score_list_{i}', default=[], dist_reduce_fx='cat')

    def get_coco(self, path: str) -> COCO:
        with contextlib.redirect_stdout(io.StringIO()):
            return COCO(path)

    def update(
            self,
            output: BatchDict,
            **kwargs
    ) -> None:
        # target is ignored
        original_size = output['meta']['original_size'][0]     # (2)
        current_size = output['meta']['current_size'][0]       # (2)
        r = (original_size / current_size)[[1, 0, 1, 0]]    # (4)

        image_ids = output['meta']['image_id'][..., None] + output['bbox_clip_ids']
        category_ids = output['bbox_pred']['label']
        bboxes = xyxy2xywh(r*output['bbox_pred']['coordinate'])
        scores = output['bbox_pred']['probability']
        for i, (ii, c, b, s) in enumerate(zip(
                image_ids.unbind(1), category_ids.unbind(1), bboxes.unbind(1), scores.unbind(1)
        )):
            getattr(self, f'image_id_list_{i}').extend(ii.unbind(0))
            getattr(self, f'category_id_list_{i}').extend(c.unbind(0))
            getattr(self, f'bbox_list_{i}').extend(b.unbind(0))
            getattr(self, f'score_list_{i}').extend(s.unbind(0))

    def compute(self) -> MetricDict:
        # gt, outputs
        if self.trainer.validating:
            cocoGt = self.get_coco(self.eval_coco)
        elif self.trainer.testing:
            cocoGt = self.get_coco(self.test_coco)
        else:
            raise ValueError('Trainer not in validating or testing mode.')

        # construct outputs
        res = {}
        for i, t in enumerate(self.future_time_constant):
            outputs = []
            ii = dim_zero_cat(getattr(self, f'image_id_list_{i}')).cpu().numpy()
            c = dim_zero_cat(getattr(self, f'category_id_list_{i}')).cpu().numpy()
            b = dim_zero_cat(getattr(self, f'bbox_list_{i}')).cpu().numpy()
            p = dim_zero_cat(getattr(self, f'score_list_{i}')).cpu().numpy()
            n_obj = c.shape[0] // ii.shape[0]
            ii = np.repeat(ii, n_obj, axis=0)
            mask = p > 1e-5
            for _ii, _c, _b, _p in zip(ii[mask], c[mask], b[mask], p[mask]):
                outputs.append({
                    'image_id': _ii.item(),
                    'category_id': _c.item(),
                    'bbox': _b.tolist(),
                    'score': _p.item(),
                    'segmentation': [],
                })

            if len(outputs) == 0:
                return {'mAP': torch.tensor(0.0, dtype=torch.float32, device=self.device)}
            else:
                outputs = sorted(outputs, key=lambda x: x['image_id'])

            res_str = io.StringIO()
            with contextlib.redirect_stdout(res_str):
                # pred
                _, tmp = tempfile.mkstemp()
                json.dump(outputs, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)

                # eval
                cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
                cocoEval.evaluate()
                cocoEval.accumulate()

                cocoEval.summarize()

            for v, (k1, k2) in zip(cocoEval.stats, coco_eval_metric_names):
                res[f'T{t}_{k2}'] = torch.tensor(float(v), dtype=torch.float32, device=self.device)
            if i == 0:
                res['mAP'] = torch.tensor(float(cocoEval.stats[0]), dtype=torch.float32, device=self.device)
        return res
