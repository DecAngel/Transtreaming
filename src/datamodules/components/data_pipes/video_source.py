import itertools
from typing import List, Union, Tuple, Dict, Any

from torch.utils.data import default_collate

from .base import DataPipe


class VideoClipperDataPipe(DataPipe):
    def __init__(self, component_indices: Dict[int, List[int]], interval: int):
        """Make clip batches from video datapipes.

        Source DataPipe should have a length of *batch_seq, frames, components
        Output DataPipe has a length of *batch_seq, clipped_frames
        """
        super().__init__()
        self.component_indices = component_indices
        self.interval = interval
        indices = list(itertools.chain(*self.component_indices.values(), [0]))
        self.margin = min(indices), max(indices)

    def __post_init__(self) -> None:
        frames, components = self.source.__len__()[-2:]
        assert isinstance(frames, list) and isinstance(components, int)
        self._length = [f + self.margin[0] - self.margin[1] for f in frames]

    def __getitem__(self, items: Tuple[int, ...]) -> Dict[str, Any]:
        res = {}
        for i, indices in self.component_indices.items():
            components = []
            for j in indices:
                new_indice = items[:-1]+(items[-1]-self.margin[0]+j, i)
                components.append(self.source.__getitem__(new_indice))
            res.update(default_collate(components))
        return res

    def __len__(self):
        return *self.source.__len__()[:-2], self._length
