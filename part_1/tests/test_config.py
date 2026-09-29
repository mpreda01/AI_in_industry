import copy
import os

import pytest
import yaml

from source.utils.config import build_config, config_to_nested, parse_overrides, load_config

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')


@pytest.fixture()
def raw(monkeypatch, tmp_path):
    monkeypatch.setenv('DATASET_ROOT', str(tmp_path / 'data'))
    monkeypatch.setenv('SAVEDIR', str(tmp_path / 'out'))
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_repo_config_is_valid_and_flattened(raw, tmp_path):
    cfg = build_config(raw)
    # the values of the repo config change with the experiment: check structure, not values
    assert set(cfg.tasks) <= {'semseg', 'depth'} and isinstance(cfg.optimizer_lr, float)
    assert cfg.wandb_mode in ('online', 'offline', 'disabled') and hasattr(cfg, 'trainer_accelerator')
    assert cfg.dataset_root == str(tmp_path / 'data')
    assert cfg.output_dir == str(tmp_path / 'out')


def test_roundtrip_nested(raw):
    cfg = build_config(raw)
    nested = config_to_nested(cfg)
    assert set(nested) == set(raw)
    assert build_config(copy.deepcopy(nested)).__dict__ == cfg.__dict__


def test_overrides_dotted_and_bare_key(raw):
    # tasks=[semseg] must be paired with a semseg-compatible monitor, whatever the repo file currently monitors
    cfg = build_config(raw, parse_overrides([
        'optimization.optimizer_lr=1e-3', 'tasks=[semseg]', 'depth_loss=l2',
        'trainer.checkpoint_monitor=metrics_task_semseg/mean_iou', 'trainer.checkpoint_mode=max']))
    assert cfg.optimizer_lr == 1e-3
    assert cfg.tasks == ['semseg']
    assert cfg.depth_loss == 'l2'


@pytest.mark.parametrize('overrides', [
    ['optimization.nope=1'],              # unknown key
    ['nope.optimizer_lr=1'],              # unknown section
    ['optimization.optimizer=rmsprop'],   # not a valid choice
    ['optimization.batch_size=abc'],      # wrong type
    ['optimization.batch_size=true'],     # bool is not an int
    ['experiment.tasks=[flow]'],          # unknown task
    ['experiment.tasks=[]'],              # no task
    # adaptive_depth only supports tasks=[depth], regardless of what the repo file currently trains
    ['experiment.tasks=[semseg, depth]', 'model.model_name=adaptive_depth'],
])
def test_invalid_config_is_rejected(raw, overrides):
    with pytest.raises((KeyError, ValueError, TypeError)):
        build_config(raw, parse_overrides(overrides))


def test_missing_key_is_rejected(raw):
    del raw['optimization']['optimizer_lr']
    with pytest.raises(KeyError, match='optimizer_lr'):
        build_config(raw)


def test_unset_environment_variable_is_an_error(raw, monkeypatch):
    monkeypatch.delenv('DATASET_ROOT')
    with pytest.raises(ValueError, match='DATASET_ROOT'):
        build_config(raw)


def test_load_config_from_command_line(monkeypatch, tmp_path):
    monkeypatch.setenv('DATASET_ROOT', str(tmp_path))
    monkeypatch.setenv('SAVEDIR', str(tmp_path))
    cfg = load_config(['--config', CONFIG_PATH, 'experiment.seed=7'])
    assert cfg.seed == 7


def test_checkpoint_monitor_must_belong_to_a_trained_task(raw):
    overrides = parse_overrides([
        'experiment.tasks=[semseg, depth]', 'trainer.checkpoint_monitor=metrics_task_semseg/mean_iou',
        'trainer.checkpoint_mode=max'])
    build_config(raw, overrides)  # semseg is trained: fine
    with pytest.raises(ValueError, match='checkpoint_monitor'):
        build_config(raw, overrides + parse_overrides(['experiment.tasks=[depth]']))


def test_checkpoint_mode_must_match_the_monitor_direction(raw):
    """Regression test: monitoring an error metric (e.g. si_log_rmse) with mode='max' silently keeps the
    WORST checkpoint instead of the best, because it looks for the highest value of a "lower is better" metric."""
    base = ['experiment.tasks=[semseg, depth]']

    # si_log_rmse is lower-is-better
    with pytest.raises(ValueError, match='checkpoint_mode'):
        build_config(raw, parse_overrides(base + [
            'trainer.checkpoint_monitor=metrics_task_depth/si_log_rmse', 'trainer.checkpoint_mode=max']))
    build_config(raw, parse_overrides(base + [
        'trainer.checkpoint_monitor=metrics_task_depth/si_log_rmse', 'trainer.checkpoint_mode=min']))

    # mean_iou is higher-is-better
    with pytest.raises(ValueError, match='checkpoint_mode'):
        build_config(raw, parse_overrides(base + [
            'trainer.checkpoint_monitor=metrics_task_semseg/mean_iou', 'trainer.checkpoint_mode=min']))
    build_config(raw, parse_overrides(base + [
        'trainer.checkpoint_monitor=metrics_task_semseg/mean_iou', 'trainer.checkpoint_mode=max']))

    # delta1 (depth) is also higher-is-better (fraction of pixels within a threshold)
    with pytest.raises(ValueError, match='checkpoint_mode'):
        build_config(raw, parse_overrides([
            'experiment.tasks=[depth]', 'trainer.checkpoint_monitor=metrics_task_depth/delta1',
            'trainer.checkpoint_mode=min']))

    # a metric whose direction is not known (not one of loss_*/metrics_*) is not checked
    build_config(raw, parse_overrides([
        'experiment.tasks=[semseg]', 'trainer.checkpoint_monitor=trainer/LR', 'trainer.checkpoint_mode=min']))
