import functools
import re
from collections import defaultdict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.primitives.model import BaseOptim, BaseBackbone, BaseNeck, BaseHead


class RegexOptim(BaseOptim):
    def __init__(
            self,
            partial_optimizer: functools.partial[Optimizer],
            partial_scheduler: functools.partial[LRScheduler],
            params: list[dict],
    ):
        super().__init__()
        self.partial_optimizer = partial_optimizer
        self.partial_scheduler = partial_scheduler
        self.regex = []
        self.params = []
        for d in params:
            self.regex.append(re.compile(d.pop('params')))
            self.params.append(d)

    def configure_optimizers(
            self,
            backbone: BaseBackbone,
            neck: BaseNeck,
            head: BaseHead,
    ):
        def named_parameter_generator(
                _backbone: BaseBackbone,
                _neck: BaseNeck,
                _head: BaseHead
        ):
            for n, p in _backbone.named_parameters():
                yield 'backbone.'+n, p
            for n, p in _neck.named_parameters():
                yield 'neck.'+n, p
            for n, p in _head.named_parameters():
                yield 'head.'+n, p

        param_groups = defaultdict(list)
        for name, param in named_parameter_generator(backbone, neck, head):
            if not param.requires_grad:
                continue

            for i, r in enumerate(self.regex):
                if re.search(r, name):
                    param_groups[i].append(param)
                    break
            else:
                param_groups[-1].append(param)

        optimizer = self.partial_optimizer(
            params=[
                {
                    'params': param_groups[i],
                    **self.params[i],
                } for i in range(len(self.regex))
            ] + [{'params': param_groups[-1]}],
        )
        scheduler = self.partial_scheduler(optimizer)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step',}]
