import random
from typing import List, Union

from src.primitives.model import BlockMixin


def sample(block: BlockMixin, steps: List[List[Union[int, float]]]) -> List[int]:
    """
    steps: start_percent, end_percent, weight, *clip_ids
    """
    percent = block.fraction_epoch / block.total_epoch
    available_steps = list(filter(lambda s: s[0] <= percent <= s[1], steps))
    return random.choices([s[3:] for s in available_steps], weights=[s[2] for s in available_steps], k=1)[0]
