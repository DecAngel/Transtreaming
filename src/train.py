from initialization import root

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.primitives.datamodule import BaseDataModule
from src.primitives.model import BaseModel
from src.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    log.info(f'Project root: {root}')

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    h = HydraConfig.get()
    if str(h.mode) == 'RunMode.MULTIRUN':
        single_gpu_id = h.job.num % cfg.get('num_gpus', 1)
        cfg.trainer.devices = [single_gpu_id]
        log.info(f'Using gpu id: {single_gpu_id}, delay for {single_gpu_id*10} seconds')
        time.sleep(single_gpu_id*10)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: BaseDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: BaseModel = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg.get("loggers"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "loggers": loggers,
        "trainer": trainer,
    }

    if loggers:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # load ckpt or pth
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

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)
        if trainer.interrupted:
            raise KeyboardInterrupt('Training interrupted!')

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        if trainer.interrupted:
            raise KeyboardInterrupt('Testing interrupted!')
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, object_dict = train(cfg)
    metric_dict = {k: v.cpu().item() for k, v in metric_dict.items()}

    # output result
    log.info(f'Metric Dict: {json.dumps(metric_dict, indent=2)}')
    return metric_dict['test_mAP'] if 'test_mAP' in metric_dict else metric_dict['val_mAP']


if __name__ == "__main__":
    main()
