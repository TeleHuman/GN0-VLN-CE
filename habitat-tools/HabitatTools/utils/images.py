from __future__ import annotations

import cv2
import numpy as np


def ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def build_history_mosaic(
    rgb_frames: list[np.ndarray],
    grid_size: int = 4,
    tile_w: int = 160,
    tile_h: int = 120,
) -> np.ndarray:
    num_tiles = grid_size * grid_size
    canvas_h = grid_size * tile_h
    canvas_w = grid_size * tile_w

    selected = list(reversed(rgb_frames[-num_tiles:]))
    tiles = []
    for frame in selected:
        frame = ensure_uint8_rgb(frame)
        tiles.append(cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA))

    while len(tiles) < num_tiles:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

    grid = np.array(tiles).reshape(grid_size, grid_size, tile_h, tile_w, 3)
    mosaic = grid.swapaxes(1, 2).reshape(canvas_h, canvas_w, 3)
    return mosaic
