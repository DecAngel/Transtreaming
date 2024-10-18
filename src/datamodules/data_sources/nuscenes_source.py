import contextlib
import io
import itertools
from pathlib import Path
from typing import Sequence, List, Union

import torch
from PIL import Image
from nuscenes.nuscenes import NuScenes
from torchvision.transforms import PILToTensor

from src.primitives.datamodule import BaseDataSource

from src.utils import RankedLogger
from src.primitives.batch import MetaDict, ImageDict, BBoxDict

logger = RankedLogger(__name__, rank_zero_only=False)


# TODO:finish
class NuScenesDataSource(BaseDataSource):
    def __init__(
            self,
            data_root: str,
            image_clip_ids: Union[List[int], List[List[int]]],
            bbox_clip_ids: Union[List[int], List[List[int]]],
            cache: bool = False,
            size: Sequence[int] = (600, 960),
            max_objs: int = 100,
    ):
        super().__init__(image_clip_ids, bbox_clip_ids)

        n = NuScenes(version='v1.0-trainval', verbose=True, dataroot='/home/zhangxiang/datasets/nuscenes')
        self.img_dir = Path(img_dir)
        self.ann_file = Path(ann_file)
        self.cache = cache
        self.size = tuple(size)
        self.max_objs = max_objs
        self.size_tensor = torch.tensor(self.size, dtype=torch.float32)

    def __post_init__(self) -> None:
        logger.info(f'Building indices for Argoverse dataset at {self.img_dir} with {self.ann_file}...')

        # load coco json
        coco = self.coco

        # get class ids
        class_ids = sorted(coco.getCatIds())
        class_names = list(c['name'] for c in coco.loadCats(class_ids))

        # get sequence lengths
        seq_dirs = [Path(d) for d in coco.dataset['seq_dirs']]
        seq_lens = []
        first_img_id = None
        first_seq_id = None
        for i, image_id in enumerate(coco.getImgIds()):
            img = coco.loadImgs([image_id])[0]
            if i == 0:
                first_img_id = image_id
                first_seq_id = img['sid']
            assert i + first_img_id == image_id, 'img_id not contiguous'
            if img['sid'] == len(seq_lens) - 1 + first_seq_id:
                # continuous seq
                assert img['fid'] == seq_lens[-1], 'fid not contiguous'
                seq_lens[-1] += 1
            else:
                # new seq
                assert img['sid'] == len(seq_lens) + first_seq_id, 'sid not contiguous'
                assert img['fid'] == 0, 'fid not starting from 0'
                seq_lens.append(1)
        seq_start_img_ids = list(itertools.accumulate([first_img_id] + seq_lens[:-1]))
        seq_start_img_ids_0 = list(itertools.accumulate([0] + seq_lens[:-1]))

        # get image and annotations
        self.seq_lens = seq_lens
        self.seq_start_img_ids = seq_start_img_ids
        self.seq_start_img_ids_0 = seq_start_img_ids_0
        self.total_images = sum(seq_lens)
        self.image_paths = []
        self.image_ids = torch.arange(first_img_id, first_img_id + self.total_images, dtype=torch.int32)
        self.image_sizes = torch.zeros(self.total_images, 2, dtype=torch.int32)
        self.gt_coordinates = torch.zeros(self.total_images, self.max_objs, 4, dtype=torch.float32)
        self.gt_labels = torch.zeros(self.total_images, self.max_objs, dtype=torch.int32)
        if self.cache:
            self.image_images = torch.zeros(self.total_images, 3, *self.size, dtype=torch.uint8)
            self.image_bools = torch.zeros(self.total_images, dtype=torch.bool)

        for seq_len, seq_dir, seq_start_img_id in zip(seq_lens, seq_dirs, seq_start_img_ids):
            for frame_id in range(seq_len):
                image_id = frame_id + seq_start_img_id
                image_id_0 = image_id - first_img_id
                image_ann = coco.loadImgs([image_id])[0]
                self.image_sizes[image_id_0, 0] = image_ann['height']
                self.image_sizes[image_id_0, 1] = image_ann['width']
                self.image_paths.append(str(seq_dir / image_ann['name']))

                label_ids = coco.getAnnIds(imgIds=[image_id])
                label_ann = coco.loadAnns(label_ids)
                if len(label_ann) > 0:
                    bbox = torch.tensor([l['bbox'] for l in label_ann], dtype=torch.float32)
                    cls = torch.tensor([class_ids.index(l['category_id']) for l in label_ann], dtype=torch.int32)
                    bbox[:, 2:] += bbox[:, :2]

                    # clip bbox & filter small bbox
                    torch.clip_(bbox[:, [0, 2]], min=0, max=image_ann['width'])
                    torch.clip_(bbox[:, [1, 3]], min=0, max=image_ann['height'])
                    mask = torch.min(bbox[:, 2:] - bbox[:, :2], dim=1)[0] >= 2
                    if not torch.all(mask).item():
                        logger.warning(f'filtered {bbox[torch.logical_not(mask)]}!')

                    bbox = bbox[mask]
                    cls = cls[mask]
                    length = bbox.size(0)
                    self.gt_coordinates[image_id_0, :length] = bbox
                    self.gt_labels[image_id_0, :length] = cls

        logger.info(f'Successfully built indices for Argoverse dataset.')

    @property
    def coco(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return COCO(str(self.ann_file))

    def _get_image_id_0(self, seq_id: int, frame_id: int) -> int:
        return self.seq_start_img_ids_0[seq_id] + frame_id

    def _load_image(self, image_id_0: int):
        image_path = str(self.img_dir / self.image_paths[image_id_0])
        image = Image.open(image_path).resize(self.size[::-1])
        return PILToTensor()(image)[[2, 1, 0]]

    def get_meta(self, seq_id: int, frame_id: int) -> MetaDict:
        image_id_0 = self._get_image_id_0(seq_id, frame_id)
        return MetaDict(
            image_id=self.image_ids[image_id_0],
            seq_id=torch.tensor(seq_id, dtype=torch.int32),
            frame_id=torch.tensor(frame_id, dtype=torch.int32),
            current_size=torch.tensor(self.size, dtype=torch.int32),
            original_size=self.image_sizes[image_id_0]
        )

    def get_image(self, seq_id: int, frame_id: int) -> ImageDict:
        image_id_0 = self._get_image_id_0(seq_id, frame_id)
        if self.cache:
            if self.image_bools[image_id_0].item() is True:
                image = self.image_images[image_id_0]
            else:
                image = self._load_image(image_id_0)
                self.image_images[image_id_0] = image
                self.image_bools[image_id_0] = 1
        else:
            image = self._load_image(image_id_0)

        return ImageDict(
            image=image,
        )

    def get_bbox(self, seq_id: int, frame_id: int) -> BBoxDict:
        image_id_0 = self._get_image_id_0(seq_id, frame_id)
        ratio = (self.size_tensor / self.image_sizes[image_id_0])[[1, 0, 1, 0]]
        return BBoxDict(
            coordinate=self.gt_coordinates[image_id_0]*ratio,
            label=self.gt_labels[image_id_0],
            probability=torch.ones(self.max_objs, dtype=torch.float32),
        )

    def get_length(self) -> List[int]:
        return self.seq_lens
