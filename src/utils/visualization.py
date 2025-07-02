from typing import Tuple, List, Iterable, Union, Optional

from kornia.utils.draw import draw_rectangle
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from src.primitives.batch import IMAGE_RAW, IMAGE, COORDINATE, LABEL, PROBABILITY, FEATURE


def normalize_images(
        images: Union[IMAGE_RAW, IMAGE],
) -> IMAGE_RAW:
    img: torch.Tensor = images.detach().cpu().requires_grad_(False)
    batch_dims = img.ndim - 3
    if batch_dims < 0:
        raise ValueError(f'Image dimensions less than 3: {img.size()}')
    elif batch_dims == 0:
        img = img.unsqueeze(0)
    else:
        img = img.flatten(0, batch_dims - 1)
    if torch.is_floating_point(img):
        max_img = torch.amax(img)
        img = (img/(max_img+1e-5)*255).to(dtype=torch.uint8)
    return img


def normalize_bboxes(
        coordinates: COORDINATE,
        labels: LABEL,
        probabilities: Optional[PROBABILITY] = None,
) -> Tuple[COORDINATE, LABEL, PROBABILITY]:
    c = coordinates.detach().cpu().requires_grad_(False)
    l = labels.detach().cpu().requires_grad_(False)
    p = probabilities.detach().cpu().requires_grad_(False) if probabilities is not None else None
    assert c.ndim - 1 == l.ndim and (True if p is None else p.ndim == c.ndim - 1)
    batch_dims = c.ndim - 2
    if batch_dims < 0:
        raise ValueError(f'Coordinate dimensions less than 2: {c.size()}')
    elif batch_dims == 0:
        c = c.unsqueeze(0)
        l = l.unsqueeze(0)
        p = p.unsqueeze(0) if p is not None else None
    else:
        c = c.flatten(0, batch_dims - 1)
        l = l.flatten(0, batch_dims - 1)
        p = p.flatten(0, batch_dims - 1) if p is not None else None
    return c, l, p


def normalize_features(
        features: FEATURE,
        feature_enhance_gamma: float = 0.7,
) -> IMAGE_RAW:
    f = features.detach().cpu().requires_grad_(False)
    batch_dims = f.ndim - 3
    if batch_dims < 0:
        raise ValueError(f'Feature dimensions less than 3: {f.size()}')
    elif batch_dims == 0:
        f = f.unsqueeze(0)
    else:
        f = f.flatten(0, batch_dims - 1)
    B, C, H, W = f.size()
    A = f.permute(0, 2, 3, 1).flatten(0, 2).to(dtype=torch.float32)  # BHW, C
    U, S, V = torch.pca_lowrank(A)
    f = torch.matmul(A, V[:, :3])
    std_f, mean_f = torch.std_mean(f, dim=0, keepdim=True)
    f = (f - mean_f) / std_f
    f = torch.float_power(torch.abs(f), feature_enhance_gamma) * torch.sign(f)
    min_f, max_f = torch.aminmax(f, dim=0, keepdim=True)
    f = (f - min_f) / (max_f - min_f + 1e-5) * 255
    f = f.to(dtype=torch.uint8)
    f = f.unflatten(0, (B, H, W)).permute(0, 3, 1, 2)
    return f


def draw_images(
        images: Union[IMAGE_RAW, IMAGE],
        size: Tuple[int, int] = (600, 960)
) -> List[np.ndarray]:
    img = normalize_images(images)
    img = F.interpolate(img, size=size, mode='nearest')
    return list(img.permute(0, 2, 3, 1).numpy())


def draw_bboxes(
        coordinates: COORDINATE,
        labels: LABEL,
        probabilities: Optional[PROBABILITY] = None,
        images: Union[None, IMAGE_RAW, IMAGE] = None,
        current_size: Tuple[int, int] = (600, 960),
        size: Tuple[int, int] = (600, 960),
        color_fg: Tuple[int, int, int] = (0, 255, 0),
        color_bg: Tuple[int, int, int] = (0, 0, 0),
) -> List[np.ndarray]:
    coordinates, labels, probabilities = normalize_bboxes(coordinates, labels, probabilities)
    resize_ratio = (torch.tensor(size, dtype=torch.float32) / torch.tensor(current_size, dtype=torch.float32))[[1, 0, 1, 0]]
    coordinates *= resize_ratio

    if images is None:
        images = torch.zeros(coordinates.size(0), 3, *size, dtype=torch.uint8, device='cpu')
    elif torch.is_tensor(images):
        images = F.interpolate(normalize_images(images), size=size, mode='nearest')

    color_fg = torch.tensor(color_fg, dtype=torch.uint8)
    color_bg = torch.tensor(color_bg, dtype=torch.uint8)
    if probabilities is not None:
        colors = (probabilities.unsqueeze(-1)*color_fg + (1-probabilities.unsqueeze(-1))*color_bg).to(dtype=torch.uint8)
    else:
        colors = color_fg
    img = draw_rectangle(images, coordinates, colors)

    return list(img.permute(0, 2, 3, 1).numpy())


def draw_features(features: FEATURE, size: Tuple[int, int] = (600, 960)) -> List[np.ndarray]:
    f = normalize_features(features)
    f = F.interpolate(f, size=size, mode='nearest')
    return list(f.permute(0, 2, 3, 1).numpy())


def draw_grid_clip_id(image_list: List[List[np.ndarray]], clip_ids: List[int]):
    # constants
    pad_width = 2
    H, W, C = image_list[0][0].shape
    x, y = len(image_list[0]), len(image_list)

    res = []

    title = np.zeros((40, (W+2*pad_width)*x, 3), dtype=np.uint8)
    for i, c in enumerate(clip_ids):
        title = cv2.putText(
            title, f'Frame T{"+" if c >= 0 else "-"}{abs(c)}', ((H+2*pad_width)*i, 40), fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=2.5, color=(255, 255, 255), thickness=2
        )
    res.append(title)

    for il in image_list:
        res.append(np.concatenate(
            [np.pad(i, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), constant_values=(0, 0)) for i in il],
            axis=1,
        ))

    return np.concatenate(res, axis=0)
