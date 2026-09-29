"""Shared fixtures: a tiny synthetic dataset in the Miniscapes layout, so the training system can be tested on CPU."""
import os

import numpy as np
import pytest
import torch
from PIL import Image

from source.datasets.dataset_miniscapes import DatasetMiniscapes

SPLIT_SIZES = {'train': 8, 'val': 4, 'test': 2}
IMG_H, IMG_W = 128, 256
NUM_BANDS = 4


def _write_sample(root, split, index, rng):
    """Vertical bands: each band has its own class, color and depth, so the tasks are learnable."""
    band_width = IMG_W // NUM_BANDS
    semseg = np.zeros((IMG_H, IMG_W), np.uint8)
    rgb = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    depth = np.zeros((IMG_H, IMG_W), np.uint8)
    classes = rng.permutation(19)[:NUM_BANDS]
    for band, cls in enumerate(classes):
        sl = slice(band * band_width, (band + 1) * band_width)
        semseg[:, sl] = cls
        rgb[:, sl] = (cls * 13 % 256, cls * 29 % 256, cls * 47 % 256)
        depth[:, sl] = 20 + cls * 10
    semseg[:4] = 255                               # some void pixels
    depth[:, :8] = 0                               # some invalid depth pixels
    rgb = np.clip(rgb.astype(int) + rng.integers(-5, 6, rgb.shape), 0, 255).astype(np.uint8)
    for modality, array, ext in (('rgb', rgb, 'jpg'), ('semseg', semseg, 'png'), ('depth', depth, 'png')):
        folder = os.path.join(root, split, modality)
        os.makedirs(folder, exist_ok=True)
        Image.fromarray(array).save(os.path.join(folder, f'{index}.{ext}'))


@pytest.fixture()
def tiny_dataset(tmp_path, monkeypatch):
    root = tmp_path / 'dataset'
    rng = np.random.default_rng(0)
    for split, size in SPLIT_SIZES.items():
        for index in range(size):
            _write_sample(str(root), split, index, rng)
    monkeypatch.setattr(DatasetMiniscapes, '__len__', lambda self: SPLIT_SIZES[self.split])
    return str(root)


@pytest.fixture()
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
