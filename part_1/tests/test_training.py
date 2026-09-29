"""End-to-end smoke tests of the training system on a tiny synthetic dataset (CPU only)."""
import csv
import json
import os
import time

import pytest
import yaml

from source.scripts.train import main

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')


def run_training(tiny_dataset, tmp_path, monkeypatch, *extra):
    monkeypatch.setenv('DATASET_ROOT', tiny_dataset)
    monkeypatch.setenv('SAVEDIR', str(tmp_path / 'out'))
    argv = [
        '--config', CONFIG_PATH,
        'experiment.tasks=[semseg, depth]',  # the repo config may describe another experiment
        'data.workers=0', 'data.workers_validation=0', 'data.batch_size_validation=2',
        'optimization.batch_size=4', 'optimization.num_epochs=2', 'optimization.optimizer_lr=0.001',
        'augmentation.aug_input_crop_size=64',
        'trainer.accelerator=cpu', 'trainer.devices=1', 'trainer.log_every_n_steps=1',
        'trainer.checkpoint_monitor=metrics_task_semseg/mean_iou', 'trainer.checkpoint_mode=max',
        'trainer.save_last_checkpoint=false',
        'wandb.mode=disabled',
        *extra,
    ]
    return main(argv)


def read_csv_column(run_dir, column):
    with open(os.path.join(run_dir, 'csv', 'version_0', 'metrics.csv')) as f:
        return [float(row[column]) for row in csv.DictReader(f) if row.get(column)]


def test_joint_run_creates_reproducible_run_directory(tiny_dataset, tmp_path, monkeypatch, cpu_only):
    run_dir = run_training(tiny_dataset, tmp_path, monkeypatch)

    for name in ('config.yaml', 'run_info.json', 'final_metrics.json'):
        assert os.path.isfile(os.path.join(run_dir, name)), name
    assert any(f.endswith('.ckpt') for f in os.listdir(os.path.join(run_dir, 'checkpoints')))
    with open(os.path.join(run_dir, 'run_info.json')) as f:
        assert json.load(f)['seed'] == 42
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        assert yaml.safe_load(f)['experiment']['seed'] == 42

    # per-task losses are logged separately, both tasks are evaluated
    for column in ('loss_train/semseg', 'loss_train/depth', 'loss_train/total', 'loss_val/semseg',
                   'metrics_task_semseg/mean_iou', 'metrics_task_depth/si_log_rmse'):
        assert read_csv_column(run_dir, column), column

    # the best checkpoint can be reloaded for evaluation
    with open(os.path.join(run_dir, 'final_metrics.json')) as f:
        best_path = json.load(f)['best_model_path']
    from source.experiments import ExperimentMultiTask
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        from source.utils.config import build_config
        cfg = build_config(yaml.safe_load(f))
    cfg.run_dir = run_dir
    assert ExperimentMultiTask.load_from_checkpoint(best_path, cfg=cfg, weights_only=False).tasks == ['semseg', 'depth']

    # test-split predictions of both tasks
    for task in ('semseg', 'depth'):
        assert len(os.listdir(os.path.join(run_dir, 'predictions', task))) == 2


def test_two_runs_never_share_a_directory(tiny_dataset, tmp_path, monkeypatch, cpu_only):
    kwargs = ['experiment.tasks=[semseg]', 'optimization.num_epochs=1', 'trainer.limit_train_batches=1',
              'trainer.test_after_fit=false']
    first = run_training(tiny_dataset, tmp_path, monkeypatch, *kwargs)
    time.sleep(1.1)  # run directories are timestamped to the second
    second = run_training(tiny_dataset, tmp_path, monkeypatch, *kwargs)
    assert first != second and os.path.isdir(first) and os.path.isdir(second)


@pytest.mark.parametrize('tasks', ['[semseg]', '[depth]'])
def test_single_task_baselines_run(tiny_dataset, tmp_path, monkeypatch, cpu_only, tasks):
    run_dir = run_training(
        tiny_dataset, tmp_path, monkeypatch, f'experiment.tasks={tasks}', 'optimization.num_epochs=1',
        'trainer.limit_train_batches=1', 'trainer.checkpoint_monitor=metrics_summary/total')
    task = tasks.strip('[]')
    assert read_csv_column(run_dir, f'loss_train/{task}')
    assert os.listdir(os.path.join(run_dir, 'predictions')) == [task]


def test_joint_model_overfits_a_tiny_dataset(tiny_dataset, tmp_path, monkeypatch, cpu_only):
    """Smoke test required by CLAUDE.md: the training loss must go down on a handful of images."""
    run_dir = run_training(
        tiny_dataset, tmp_path, monkeypatch, 'optimization.num_epochs=40', 'trainer.test_after_fit=false',
        'trainer.num_sanity_val_steps=0')

    def mean(values):
        return sum(values) / len(values)

    total = read_csv_column(run_dir, 'loss_train/total')
    semseg = read_csv_column(run_dir, 'loss_train/semseg')
    assert len(total) >= 60
    # single-step losses are noisy (batch of 4 random crops): compare short windows at start and end
    assert mean(total[-10:]) < 0.6 * mean(total[:3]), total
    assert mean(semseg[-10:]) < 0.95 * mean(semseg[:3]), semseg


def test_branched_model_runs_and_overfits_a_tiny_dataset(tiny_dataset, tmp_path, monkeypatch, cpu_only):
    """Shape/contract check (separate semseg+depth heads, correct predictions/ dir) plus the CLAUDE.md
    smoke test (loss must go down on a handful of images) for the branched architecture."""
    run_dir = run_training(
        tiny_dataset, tmp_path, monkeypatch, 'model.model_name=deeplabv3p_multitask',
        'optimization.num_epochs=40', 'trainer.num_sanity_val_steps=0')

    def mean(values):
        return sum(values) / len(values)

    total = read_csv_column(run_dir, 'loss_train/total')
    semseg = read_csv_column(run_dir, 'loss_train/semseg')
    assert len(total) >= 60
    assert mean(total[-10:]) < 0.6 * mean(total[:3]), total
    assert mean(semseg[-10:]) < 0.95 * mean(semseg[:3]), semseg

    for task in ('semseg', 'depth'):
        assert len(os.listdir(os.path.join(run_dir, 'predictions', task))) == 2


def test_depth_loss_normalization_matches_dataset_stddev(tiny_dataset, tmp_path, monkeypatch, cpu_only):
    """loss.normalize_depth_loss divides the depth residual by depth_meters_stddev (29.1264, a fixed
    DatasetMiniscapes constant) before computing L1: it should shrink loss_train/depth by exactly that
    factor on the very first step (same seed => same model init and batch order in both runs)."""
    depth_stddev = 29.1264
    common = ['optimization.num_epochs=1', 'trainer.limit_train_batches=1', 'trainer.test_after_fit=false']
    run_norm = run_training(tiny_dataset, tmp_path, monkeypatch, *common, 'loss.normalize_depth_loss=true')
    run_raw = run_training(tiny_dataset, tmp_path, monkeypatch, *common, 'loss.normalize_depth_loss=false')
    norm_depth = read_csv_column(run_norm, 'loss_train/depth')[0]
    raw_depth = read_csv_column(run_raw, 'loss_train/depth')[0]
    assert norm_depth == pytest.approx(raw_depth / depth_stddev, rel=1e-4)


def test_model_auxiliary_losses_are_added_and_logged(tiny_dataset, tmp_path, monkeypatch, cpu_only):
    """A model exposing compute_aux_losses() gets its losses added (weighted) to the total and logged apart."""
    import torch
    from source.models import ModelDeepLabV3Plus
    from source.utils import helpers

    class ModelWithAux(ModelDeepLabV3Plus):
        def compute_aux_losses(self, outputs, batch):
            return {'dummy': outputs['depth'].mean() * 0.0 + 2.0}

    monkeypatch.setitem(helpers.MODELS, 'deeplabv3p', ModelWithAux)
    run_dir = run_training(
        tiny_dataset, tmp_path, monkeypatch, 'optimization.num_epochs=1', 'trainer.limit_train_batches=1',
        'trainer.test_after_fit=false', 'loss.loss_weight_aux=3.0', 'loss.loss_weight_semseg=1.0',
        'loss.loss_weight_depth=1.0')
    aux = read_csv_column(run_dir, 'loss_train/aux_dummy')
    total = read_csv_column(run_dir, 'loss_train/total')
    semseg = read_csv_column(run_dir, 'loss_train/semseg')
    depth = read_csv_column(run_dir, 'loss_train/depth')
    assert aux[0] == pytest.approx(2.0)
    assert total[0] == pytest.approx(semseg[0] + depth[0] + 3.0 * 2.0, rel=1e-4)


def test_wandb_offline_logging_and_visualization(tiny_dataset, tmp_path, monkeypatch, cpu_only):
    """Exercises the W&B logger, prediction images and depth histograms without network access."""
    monkeypatch.setenv('WANDB_MODE', 'offline')
    monkeypatch.setenv('WANDB_SILENT', 'true')
    run_dir = run_training(
        tiny_dataset, tmp_path, monkeypatch, 'wandb.mode=offline', 'wandb.group=test', 'wandb.tags=[smoke]',
        'optimization.num_epochs=1', 'trainer.test_after_fit=false',
        'visualization.num_steps_visualization_first=0', 'visualization.num_steps_visualization_interval=1',
        'visualization.observe_train_ids=[0]', 'visualization.observe_valid_ids=[1]')
    import wandb
    wandb.finish()  # the run file is written asynchronously; close the run before reading it
    logged = b''
    for folder, _, files in os.walk(os.path.join(run_dir, 'wandb')):
        for name in files:
            if name.endswith('.wandb'):
                with open(os.path.join(folder, name), 'rb') as f:
                    logged += f.read()
    for key in (b'imgs_train/batch_crops', b'imgs_val/observed_samples', b'histograms/pred_depth_meters',
                b'loss_train/total'):
        assert key in logged, key
