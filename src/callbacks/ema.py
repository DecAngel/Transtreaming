import math
from typing import Dict, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from torch import nn

from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model: pl.LightningModule, decay=0.9999):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = model.__class__(**model.hparams)
        self.module.load_state_dict(model.state_dict())
        self.module.eval()
        self.decay = decay
        self.device = model.device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=self.device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class EMACallback(Callback):
    """
    Model Exponential Moving Average. Empirically it has been found that using the moving average
    of the trained parameters of a deep network is better than using its trained parameters directly.

    If `use_ema_weights`, then the ema parameters of the network is set after training end.
    """

    def __init__(self, decay=0.9998, use_ema_weights: bool = True):
        self.decay = decay
        self.ema = None
        self.use_ema_weights = use_ema_weights

    def on_fit_start(self, trainer, pl_module):
        "Initialize `ModelEmaV2` from timm to keep a copy of the moving average of the weights"
        self.ema = ModelEmaV2(pl_module, decay=self.decay)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # disable EMA
        self.ema = None

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_validation_start(trainer, pl_module)

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_validation_start(trainer, pl_module)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        "Update the stored parameters using a moving average"
        # Update currently maintained parameters.
        if self.ema is not None:
            self.ema.decay = self.decay * (1 - math.exp(-trainer.global_step / 2000))
            self.ema.update(pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        "do validation using the stored parameters"
        if self.ema is not None:
            # save original parameters before replacing with EMA version
            self.store(pl_module.parameters())

            # update the LightningModule with the EMA weights
            # ~ Copy EMA parameters to LightningModule
            self.copy_to(self.ema.module.parameters(), pl_module.parameters())

    def on_validation_epoch_end(self, trainer, pl_module):
        "Restore original parameters to resume training later"
        if self.ema is not None:
            self.restore(pl_module.parameters())

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self.on_validation_epoch_start(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self.on_validation_epoch_end(trainer, pl_module)

    def on_predict_epoch_start(self, trainer, pl_module) -> None:
        self.on_validation_epoch_start(trainer, pl_module)

    def on_predict_epoch_end(self, trainer, pl_module) -> None:
        self.on_validation_epoch_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        # update the LightningModule with the EMA weights
        if self.ema is not None and self.use_ema_weights:
            self.copy_to(self.ema.module.parameters(), pl_module.parameters())
            logger.info("End of training. Model weights replaced with the EMA version.")

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        if self.ema is not None:
            checkpoint['callbacks'][self.__class__.__name__] = {'state_dict_ema_train': pl_module.state_dict()}
            checkpoint['state_dict'] = self.ema.module.state_dict()

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        if self.ema is not None:
            train_params = checkpoint['callbacks'][self.__class__.__name__]['state_dict_ema_train']
            eval_params = checkpoint['state_dict']
            if trainer.training:
                pl_module.load_state_dict(train_params)
            else:
                pl_module.load_state_dict(eval_params)
            self.ema.module.load_state_dict(eval_params)

    """
    def state_dict(self) -> Dict[str, Any]:
        if self.ema is not None:
            return {"state_dict_ema": get_state_dict(self.ema, unwrap_model)}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.ema is not None:
            self.ema.module.load_state_dict(state_dict["state_dict_ema"])
    """

    def store(self, parameters):
        "Save the current parameters for restoring later."
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def copy_to(self, shadow_parameters, parameters):
        "Copy current parameters into given collection of parameters."
        for s_param, param in zip(shadow_parameters, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)
