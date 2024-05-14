from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
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
from typing import Optional, Tuple, Dict, Any

import torch

from src.primitives.model import BaseModel
from src.primitives.sap_strategy import BaseSAPStrategy, SAPServer, SAPClient


@task_wrapper
def sap(cfg: DictConfig) -> Dict[str, float]:
    if platform.system() != 'Linux':
        raise EnvironmentError('sAP evaluation is only supported on Linux!')

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: BaseModel = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating sap_strategy <{cfg.sap_strategy._target_}>")
    strategy: BaseSAPStrategy = hydra.utils.instantiate(cfg.sap_strategy)

    # load ckpt or pth
    path = Path(cfg.ckpt_path).resolve()
    log.info(f"Loading model from {str(path)}")
    if path.suffix == '.pth':
        model.load_from_pth(str(path))
    elif path.suffix == '.ckpt':
        model.load_from_ckpt(str(path))
    else:
        raise ValueError(f"Unsupported file type {path.suffix}")

    data_dir = cfg.get('data_dir')
    ann_file = cfg.get('ann_file')
    output_dir = cfg.get('output_dir')

    sap_factor = cfg.get('sap_factor', 1.0)
    dataset_resize_ratio = cfg.get('dataset_resize_ratio', 2)
    device_id = cfg.get('device_id', 0)
    visualize_plot = cfg.get('visualize_plot', False)
    visualize_print = cfg.get('visualize_print', False)

    model = model.eval().half().to(torch.device(f'cuda:{device_id}'))

    with SAPServer(
            data_dir=data_dir,
            ann_file=ann_file,
            output_dir=output_dir,
            sap_factor=sap_factor
    ) as server:
        client = SAPClient(server, dataset_resize_ratio, device_id)
        try:
            return strategy.infer_all(
                model,
                server,
                client,
                visualize_plot=visualize_plot,
                visualize_print=visualize_print,
                output_dir=output_dir,
            )
        except KeyboardInterrupt as e:
            log.warning('Ctrl+C detected. Shutting down sAP server & client.', exc_info=e)
            raise
        finally:
            client.close()


@hydra.main(version_base="1.3", config_path="../configs", config_name="sap.yaml")
def main(cfg: DictConfig) -> Dict[str, float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # run sap
    metric_dict = sap(cfg)

    log.info(f'sAP result: {json.dumps(metric_dict, indent=2)}')

    return metric_dict


if __name__ == "__main__":
    main()
