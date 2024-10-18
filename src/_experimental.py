import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import lightning as L
import rootutils
import torch
from hydra.core.hydra_config import HydraConfig
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

from src.primitives.datamodule import BaseDataModule
from src.primitives.model import BaseModel
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="_experimental.yaml")
def main(cfg: DictConfig):
    model = hydra.utils.instantiate(cfg.model)
    path = cfg.get("ckpt_path")
    if path is not None:
        path = Path(path).resolve()
        log.info(f"Loading model from {str(path)}")
        if path.suffix == '.pth':
            model.load_from_pth(str(path))
        elif path.suffix == '.ckpt':
            model.load_from_ckpt(str(path))
        else:
            raise ValueError(f"Unsupported file type {path.suffix}")
    else:
        log.info(f"Skip loading ckpt")
    print(model.state_dict().keys())


if __name__ == '__main__':
    main()
