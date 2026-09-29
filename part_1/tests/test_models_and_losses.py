from argparse import Namespace

import numpy as np
import pytest
import torch

from source.losses import MaskedDepthRegressionLoss
from source.models import ModelDeepLabV3Plus, ModelDeepLabV3PlusMultiTask


@pytest.mark.parametrize('encoder', ['resnet18', 'resnet34'])
def test_joint_model_output_shapes(encoder):
    cfg = Namespace(model_encoder_name=encoder, pretrained=False)
    model = ModelDeepLabV3Plus(cfg, {'semseg': 19, 'depth': 1})
    out = model(torch.randn(2, 3, 64, 128))
    assert out['semseg'].shape == (2, 19, 64, 128)
    assert out['depth'].shape == (2, 1, 64, 128)
    assert (out['depth'] >= 0.1).all() and (out['depth'] <= 300).all()


@pytest.mark.parametrize('encoder', ['resnet18', 'resnet34'])
def test_branched_model_output_shapes_and_separate_heads(encoder):
    cfg = Namespace(model_encoder_name=encoder, pretrained=False)
    model = ModelDeepLabV3PlusMultiTask(cfg, {'semseg': 19, 'depth': 1})
    out = model(torch.randn(2, 3, 64, 128))
    assert out['semseg'].shape == (2, 19, 64, 128)
    assert out['depth'].shape == (2, 1, 64, 128)
    assert (out['depth'] >= 0.1).all() and (out['depth'] <= 300).all()
    # branched architecture: semseg and depth must go through separate ASPP/decoder modules
    assert model.aspps['semseg'] is not model.aspps['depth']
    assert model.decoders['semseg'] is not model.decoders['depth']
    assert set(model.aspps['semseg'].parameters()) != set(model.aspps['depth'].parameters())


def test_branched_model_single_task_only_builds_that_head():
    cfg = Namespace(model_encoder_name='resnet18', pretrained=False)
    model = ModelDeepLabV3PlusMultiTask(cfg, {'depth': 1})
    out = model(torch.randn(2, 3, 64, 128))
    assert set(out) == {'depth'} and set(model.aspps) == {'depth'}


@pytest.mark.parametrize('kind', ['l1', 'l2'])
def test_masked_depth_loss_matches_numpy(kind):
    rng = np.random.default_rng(0)
    pred = rng.uniform(1, 100, (2, 8, 8)).astype(np.float32)
    gt = rng.uniform(1, 100, (2, 8, 8)).astype(np.float32)
    gt[0, :2] = 0.0
    gt[1, 0, 0] = np.nan
    valid = np.isfinite(gt) & (gt > 0)
    diff = pred[valid] - gt[valid]
    expected = np.abs(diff).mean() if kind == 'l1' else (diff ** 2).mean()
    loss = MaskedDepthRegressionLoss(kind)(torch.from_numpy(pred).unsqueeze(1), torch.from_numpy(gt))
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_masked_depth_loss_all_invalid_is_zero_with_gradient():
    pred = torch.rand(1, 1, 4, 4, requires_grad=True)
    loss = MaskedDepthRegressionLoss('l1')(pred, torch.zeros(1, 4, 4))
    loss.backward()
    assert loss.item() == 0.0 and torch.isfinite(pred.grad).all()


@pytest.mark.parametrize('kind', ['l1', 'l2'])
def test_masked_depth_loss_scale_normalizes_the_residual(kind):
    pred = torch.tensor([10.0, 20.0, 30.0])
    gt = torch.tensor([13.0, 24.0, 42.0])
    unscaled = MaskedDepthRegressionLoss(kind, scale=1.0)(pred, gt)
    scale = 29.1264  # e.g. DatasetMiniscapes.depth_meters_stddev
    scaled = MaskedDepthRegressionLoss(kind, scale=scale)(pred, gt)
    expected_ratio = scale if kind == 'l1' else scale ** 2
    assert scaled.item() == pytest.approx(unscaled.item() / expected_ratio, rel=1e-5)


def test_masked_depth_loss_rejects_non_positive_scale():
    with pytest.raises(ValueError):
        MaskedDepthRegressionLoss('l1', scale=0.0)


def test_visualization_image_keeps_colors_in_wandb(monkeypatch):
    """W&B renders float tensors as black images: the logged image must be 8-bit and keep its colors."""
    import wandb
    from source.utils.visualization import to_pil_image
    monkeypatch.setenv('WANDB_MODE', 'offline')
    vis = torch.zeros(3, 8, 8)
    vis[:, :, :4] = torch.tensor([128, 64, 128]).view(3, 1, 1) / 255       # Cityscapes "road" color
    logged = np.array(wandb.Image(to_pil_image(vis)).image)
    assert logged[2, 1].tolist() == [128, 64, 128]
    assert logged[2, 6].tolist() == [0, 0, 0]
