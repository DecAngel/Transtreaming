from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional, Union, Tuple, Any, List

import cv2
import numpy as np
import torch
from lightning import Callback
import lightning as L
from matplotlib import pyplot as plt

from src.primitives.batch import BatchDict, LossDict, IMAGE_RAW, IMAGE, COORDINATE, LABEL, PROBABILITY, FEATURE, \
    MetaDict, META, PYRAMID, IntermediateDict
from src.primitives.model import BaseModel
from src.utils.pylogger import RankedLogger
from src.utils.visualization import draw_images, draw_bboxes, draw_features, draw_grid_clip_id

log = RankedLogger(__name__, rank_zero_only=False)


class VisualizeFeature(Callback):
    def __init__(
            self,
            tag: str,
            mode: Literal['show_opencv', 'show_plt', 'write_image', 'write_video'] = 'show_plt',
            size: Tuple[int, int] = (300, 480),
            visualization_dir: Union[Path, str, None] = None,
            visualize_train: bool = True,
            visualize_val: bool = True,
            visualize_test: bool = True,
            visualize_interval: int = 100,
    ):
        self.tag = tag
        self.mode = mode
        self.size = size
        self.visualization_dir = Path(visualization_dir) if visualization_dir is not None else None
        self.visualize_train = visualize_train
        self.visualize_val = visualize_val
        self.visualize_test = visualize_test
        self.visualize_interval = visualize_interval
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

        self.vis_counter = 0
        self.counter = defaultdict(lambda: 0)
        self.video_writer = None
        self.video_shape = None

    def write(self, image: np.ndarray, subtag: Optional[str] = None):
        identifier = self.tag if subtag is None else f'{self.tag}_{subtag}'
        self.counter[subtag] += 1
        count = self.counter[subtag]
        if self.mode == 'show_opencv':
            cv2.imshow(identifier, image)
            cv2.waitKey(1)
        elif self.mode == 'show_plt':
            plt.close(identifier)
            plt.figure(identifier, figsize=(10, 10), dpi=200)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        elif self.mode == 'write_image':
            cv2.imwrite(str(self.visualization_dir.joinpath(f'{identifier}_{count:05d}.jpg')), image)
        elif self.mode == 'write_video':
            if self.video_writer is None:
                self.video_writer = cv2.VideoWriter(
                    str(self.visualization_dir.joinpath(f'{identifier}.mp4')),
                    cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                    30,
                    (image.shape[1], image.shape[0])
                )
                self.video_shape = image.shape[:2]
            else:
                assert self.video_shape == image.shape[:2]
            self.video_writer.write(image)
        else:
            raise ValueError(f'Unsupported mode {self.mode}')

    def close(self):
        if self.mode == 'write_video' and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def visualize(
            self,
            past_clip_ids: List[int],
            future_clip_ids: List[int],
            images: Union[IMAGE_RAW, IMAGE],
            gt_coordinates: COORDINATE,
            gt_labels: LABEL,
            gt_size: Tuple[int, int] = (300, 480),
            pred_coordinates: Optional[COORDINATE] = None,
            pred_labels: Optional[LABEL] = None,
            pred_probabilities: Optional[PROBABILITY] = None,
            pred_size: Tuple[int, int] = (300, 480),
            features_p: Optional[PYRAMID] = None,
            features_f: Optional[PYRAMID] = None,
    ):
        TP, C, H, W = images.size()
        TF, O, _ = gt_coordinates.size()

        future_images = images[-1:].expand(TF, C, H, W)
        img = draw_images(images, self.size)
        empty = np.zeros_like(img[0])
        gt = draw_bboxes(gt_coordinates, gt_labels, images=future_images, current_size=gt_size, size=self.size)
        gt = img + gt
        pred = None if pred_coordinates is None else draw_bboxes(
            pred_coordinates, pred_labels, pred_probabilities, images=future_images, current_size=pred_size, size=self.size
        )
        pred = img + pred if pred is not None else None

        if features_p is not None and features_f is not None:
            f_all = [draw_features(torch.cat([p, f], dim=0), self.size) for p, f in zip(features_p, features_f)]
        elif features_p is not None and features_f is None:
            f_all = [draw_features(p) + [empty]*TF for p in features_p]
        elif features_p is None and features_f is not None:
            f_all = [[empty]*TP + draw_features(f) for f in features_f]
        else:
            f_all = [None]
        vis_images = list(filter(None, [gt, pred] + f_all))
        vis_image = draw_grid_clip_id(vis_images, past_clip_ids+future_clip_ids)
        self.write(vis_image)

    def on_train_batch_end(
        self, trainer: "L.Trainer", pl_module: BaseModel, outputs: IntermediateDict, batch: BatchDict, batch_idx: int
    ) -> None:
        self.vis_counter += 1
        if self.vis_counter % self.visualize_interval == 0 and self.visualize_train:
            start = int(pl_module.head.require_prev_frame)
            self.visualize(
                past_clip_ids=batch['image_clip_ids'][0].cpu().numpy().tolist(),
                future_clip_ids=batch['bbox_clip_ids'][0, start:].cpu().numpy().tolist(),
                images=batch['image']['image'][0],
                gt_coordinates=batch['bbox']['coordinate'][0, start:],
                gt_labels=batch['bbox']['label'][0, start:],
                gt_size=batch['meta']['current_size'][0].cpu().numpy().tolist(),
                features_p=tuple(f[0] for f in outputs['features_p']) if 'features_p' in outputs else None,
                features_f=tuple(f[0] for f in outputs['features_f']) if 'features_f' in outputs else None,
            )

    def on_validation_batch_end(
            self,
            trainer: "L.Trainer",
            pl_module: "L.LightningModule",
            outputs: BatchDict,
            batch: BatchDict,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.vis_counter += 1
        if self.vis_counter % self.visualize_interval == 0 and self.visualize_val:
            self.visualize(
                past_clip_ids=outputs['image_clip_ids'][0].cpu().numpy().tolist(),
                future_clip_ids=outputs['bbox_clip_ids'][0].cpu().numpy().tolist(),
                images=outputs['image']['image'][0],
                gt_coordinates=outputs['bbox']['coordinate'][0],
                gt_labels=outputs['bbox']['label'][0],
                gt_size=outputs['meta']['current_size'][0].cpu().numpy().tolist(),
                pred_coordinates=outputs['bbox_pred']['coordinate'][0] if 'bbox_pred' in batch else None,
                pred_labels=outputs['bbox_pred']['label'][0] if 'bbox_pred' in batch else None,
                pred_probabilities=outputs['bbox_pred']['probability'][0] if 'bbox_pred' in batch else None,
                pred_size=outputs['meta']['current_size'][0].cpu().numpy().tolist() if 'bbox_pred' in batch else None,
            )

    def on_test_batch_end(
            self,
            trainer: "L.Trainer",
            pl_module: "L.LightningModule",
            outputs: BatchDict,
            batch: BatchDict,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.vis_counter += 1
        if self.vis_counter % self.visualize_interval == 0 and self.visualize_test:
            self.visualize(
                past_clip_ids=outputs['image_clip_ids'][0].cpu().numpy().tolist(),
                future_clip_ids=outputs['bbox_clip_ids'][0].cpu().numpy().tolist(),
                images=outputs['image']['image'][0],
                gt_coordinates=outputs['bbox']['coordinate'][0],
                gt_labels=outputs['bbox']['label'][0],
                gt_size=outputs['meta']['current_size'][0].cpu().numpy().tolist(),
                pred_coordinates=outputs['bbox_pred']['coordinate'][0] if 'bbox_pred' in batch else None,
                pred_labels=outputs['bbox_pred']['label'][0] if 'bbox_pred' in batch else None,
                pred_probabilities=outputs['bbox_pred']['probability'][0] if 'bbox_pred' in batch else None,
                pred_size=outputs['meta']['current_size'][0].cpu().numpy().tolist() if 'bbox_pred' in batch else None,
            )

    def on_train_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.close()
        self.vis_counter = 0
        self.counter = defaultdict(lambda: 0)

    def on_validation_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.close()
        self.vis_counter = 0
        self.counter = defaultdict(lambda: 0)

    def on_test_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.close()
        self.vis_counter = 0
        self.counter = defaultdict(lambda: 0)
