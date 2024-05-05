from typing import Optional, Union, Tuple, Dict

import torch
from lightning import LightningModule
from torch import Tensor


class DummyModule(LightningModule):
    def __init__(self):
        super().__init__()

    def example_input_array(self):
        return {
            'image': {
                'image': torch.rand(2, 4, 3, 100, 200),
                ''
            }
        }

    def training_step(self, batch):
        pass

    def validation_step(self, )
