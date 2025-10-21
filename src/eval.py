from initialization import root

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    log.info(f'Project root: {root}')

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # load ckpt or pth
    ckpt_path = cfg.get('ckpt_path')
    if ckpt_path:
        path = Path(cfg.ckpt_path).resolve()
        log.info(f"Loading model from {str(path)}")
        if path.suffix == '.pth':
            model.load_from_pth(str(path))
        elif path.suffix == '.ckpt':
            model.load_from_ckpt(str(path))
        else:
            raise ValueError(f"Unsupported file type {path.suffix}")
    else:
        log.warning("No checkpoint path specified!")

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> float:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    metric_dict, object_dict = evaluate(cfg)
    metric_dict = {k: v.cpu().item() for k, v in metric_dict.items()}

    # output result
    log.info(f'Metric Dict: {json.dumps(metric_dict, indent=2)}')
    return metric_dict['test_mAP'] if 'test_mAP' in metric_dict else metric_dict['val_mAP']


if __name__ == "__main__":
    main()
