import functools
import time

import cv2
import hydra
import numpy as np
import rootutils
import torch
import torch.multiprocessing as mp
from PIL import Image
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

import json
import platform
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch

from src.utils.visualization import draw_images, draw_bboxes, draw_features, draw_grid_clip_id
from src.primitives.model import BaseModel
from src.primitives.batch import BufferDict, BatchDict


def load_image(filepath: Path, resize_ratio: int):
    img = cv2.imread(str(filepath))
    img = img[::resize_ratio, ::resize_ratio]
    return img


def construct_batch(
        image: np.ndarray,
        index: int,
        device_id: int,
        buffer: Optional[BufferDict] = None
) -> BatchDict:
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    h, w, _ = image.shape
    image  = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0).unsqueeze(0)
    image = image.to(dtype=torch.half, device=device)

    batch = {
        'meta': {
            'image_id': torch.tensor([index], dtype=torch.long, device=device),
            'seq_id': torch.tensor([0], dtype=torch.long, device=device),
            'frame_id': torch.tensor([index], dtype=torch.long, device=device),
            'current_size': torch.tensor([[h, w]], dtype=torch.long, device=device),
            'original_size': torch.tensor([[h, w]], dtype=torch.long, device=device),
        },
        'image': {'image': image},
        'image_clip_ids': torch.tensor([[0]], dtype=torch.long, device=device),
        'past_clip_ids': torch.tensor([[-3, -2, -1, 0]], dtype=torch.long, device=device),
        'future_clip_ids': torch.tensor([[1]], dtype=torch.long, device=device),
        'intermediate': {},
    }
    if buffer is not None:
        batch['buffer'] = buffer

    return batch


@task_wrapper
def demo(cfg: DictConfig) -> None:

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: BaseModel = hydra.utils.instantiate(cfg.model)

    device_id = cfg.get('device_id', 0)
    demo_dir = cfg.get('demo_dir')
    resize_ratio = cfg.get('resize_ratio', 2)
    fps = cfg.get('fps', 30)

    # load ckpt or pth
    path = Path(cfg.ckpt_path).resolve()
    log.info(f"Loading model from {str(path)}")
    if path.suffix == '.pth':
        model.load_from_pth(str(path))
    elif path.suffix == '.ckpt':
        model.load_from_ckpt(str(path))
    else:
        raise ValueError(f"Unsupported file type {path.suffix}")

    model = model.eval().half().to(torch.device(f'cuda:{device_id}'))

    image_loader = functools.partial(load_image, resize_ratio=resize_ratio)
    batch_constructor = functools.partial(construct_batch, device_id=device_id)

    buffer = None

    # load demo images
    log.info('Loading demo images...')
    demo_images = [path for path in list(sorted(Path(demo_dir).iterdir()))]

    log.info('Inferring ...')

    communication_delays = []
    computation_delays = []

    current_time = 0
    current_pred = None
    next_pred = None
    visualizations = []
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for i in range(len(demo_images)):
                if i == 0:
                    # warm up
                    for _ in range(3):
                        image = image_loader(demo_images[i])
                        batch = batch_constructor(image, index=0, buffer=buffer)
                        batch = model.inference(batch)
                        buffer = batch['buffer']
                    buffer = None

                if i / fps >= current_time:
                    current_pred = next_pred
                    start_time = time.perf_counter()
                    image = image_loader(demo_images[i])
                    batch = batch_constructor(image, index=i, buffer=buffer)
                    load_time = time.perf_counter() - start_time
                    batch = model.inference(batch)
                    compute_time = time.perf_counter() - start_time - load_time
                    buffer = batch['buffer']

                    current_time += time.perf_counter() - start_time
                    current_time = max(current_time, i / fps)
                    communication_delays.append(load_time*1000)
                    computation_delays.append(compute_time*1000)
                    next_pred = cv2.resize(draw_bboxes(
                        batch['bbox_pred']['coordinate'] / 2,
                        batch['bbox_pred']['label'],
                        current_size=(300, 480),
                        size=(300, 480),
                        color_fg=(0, 0, 255),
                    )[0], dsize=(960, 600), interpolation=cv2.INTER_NEAREST)

                # current_pred = next_pred
                if current_pred is None:
                    visualizations.append(image_loader(demo_images[i]))
                else:
                    b, g, r = cv2.split(current_pred)
                    mask = cv2.bitwise_not(cv2.merge([g, g, g]))
                    image = cv2.bitwise_and(image_loader(demo_images[i]), mask)
                    visualizations.append(cv2.add(image, current_pred))

    # display
    """
    for v in visualizations[::2]:
        cv2.imshow('visualization', v)
        cv2.waitKey()

    cv2.waitKey()
    cv2.destroyAllWindows()
    """
    print('communication')
    for d in communication_delays:
        print(d)
    print('computation')
    for d in computation_delays:
        print(d)


@hydra.main(version_base="1.3", config_path="../configs", config_name="demo.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # run demo
    demo(cfg)


if __name__ == '__main__':
    main()
