from lightning import Callback
from torch.optim import Optimizer


from src.primitives.model import BaseModel
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def plot_grad_flow(title, named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    name_dict = defaultdict(lambda: 0)
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            base_name = n.split('.')[0]
            name_dict[base_name] += 1
            layers.append(f'{base_name}.{name_dict[base_name]}')
            ave_grads.append(p.grad.abs().mean().cpu().numpy())
            max_grads.append(p.grad.abs().max().cpu().numpy())

    plt.figure(title)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.05, top=0.5)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()


class VisualizeGrad(Callback):
    def __init__(self, interval: int = 100, vis_backbone: bool = False, vis_neck: bool = True, vis_head: bool = False):
        super().__init__()
        self.interval = interval
        self.vis_backbone = vis_backbone
        self.vis_neck = vis_neck
        self.vis_head = vis_head
        self.counter = 0

    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: BaseModel, optimizer: Optimizer
    ) -> None:
        self.counter += 1
        if self.counter == self.interval:
            if self.vis_backbone:
                plot_grad_flow('backbone_grad', pl_module.backbone.named_parameters())
            if self.vis_neck:
                plot_grad_flow('neck_grad', pl_module.neck.named_parameters())
            if self.vis_head:
                plot_grad_flow('head_grad', pl_module.head.named_parameters())
            self.counter = 0
