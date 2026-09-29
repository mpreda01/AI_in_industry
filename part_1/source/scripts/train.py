"""
Generic training entry point: works for every model / task combination described by a YAML config.

    python -m source.scripts.train --config config.yaml [section.key=value ...]

Every run creates its own directory <paths.output_dir>/<experiment.name>_seed<seed>_<timestamp> and never
touches previous runs. See doc/training_system.md.
"""
import json
import os
import platform
import socket
import subprocess
from datetime import datetime

import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from source.experiments import ExperimentMultiTask
from source.utils.config import load_config, config_to_nested

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_run_dir(cfg):
    """Create a fresh run directory (fails if it exists) and store it in cfg.run_dir / cfg.run_name."""
    output_dir = os.path.abspath(cfg.output_dir)
    if output_dir == REPO_ROOT or output_dir.startswith(REPO_ROOT + os.sep):
        raise ValueError(f'paths.output_dir ({output_dir}) must be outside of the repository ({REPO_ROOT})')
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    cfg.run_name = f'{cfg.name}_seed{cfg.seed}_{timestamp}'
    cfg.run_dir = os.path.join(cfg.output_dir, cfg.run_name)
    os.makedirs(cfg.run_dir, exist_ok=False)
    return cfg.run_dir


def _git(*args):
    try:
        out = subprocess.run(['git', *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def collect_run_info(cfg):
    commit = _git('rev-parse', 'HEAD')
    status = _git('status', '--porcelain')
    return {
        'run_name': cfg.run_name,
        'seed': cfg.seed,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'git_commit': commit.strip() if commit else None,
        'git_dirty': bool(status.strip()) if status is not None else None,
        'host': socket.gethostname(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def save_run_files(cfg, run_info):
    """Store everything needed to reproduce the run next to its checkpoints."""
    with open(os.path.join(cfg.run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config_to_nested(cfg), f, sort_keys=False)
    with open(os.path.join(cfg.run_dir, 'run_info.json'), 'w') as f:
        json.dump(run_info, f, indent=4)
    if run_info['git_dirty']:
        patch = _git('diff', 'HEAD')
        if patch:
            with open(os.path.join(cfg.run_dir, 'git_diff.patch'), 'w') as f:
                f.write(patch)


def load_wandb_key(cfg):
    """WANDB_API_KEY from the environment wins; otherwise read the (untracked) key file if it has content."""
    if os.environ.get('WANDB_API_KEY') or not cfg.wandb_key_file or not os.path.isfile(cfg.wandb_key_file):
        return
    with open(cfg.wandb_key_file) as f:
        key = f.read().strip()
    if key:
        os.environ['WANDB_API_KEY'] = key


def build_loggers(cfg, run_info):
    loggers = [CSVLogger(save_dir=cfg.run_dir, name='csv', version=0)]
    if cfg.wandb_mode != 'disabled':
        load_wandb_key(cfg)
        wandb_logger = WandbLogger(
            name=cfg.run_name,
            save_dir=cfg.run_dir,
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            tags=cfg.wandb_tags,
            notes=cfg.wandb_notes,
            offline=cfg.wandb_mode == 'offline',
        )
        wandb_logger.log_hyperparams({'run_info': run_info})
        print(f'W&B run: {getattr(wandb_logger.experiment, "url", None)}')
        loggers.append(wandb_logger)
    return loggers


def _to_float(value):
    return float(value) if torch.is_tensor(value) or isinstance(value, (int, float)) else str(value)


def main(argv=None):
    cfg = load_config(argv)
    seed_everything(cfg.seed, workers=True)

    create_run_dir(cfg)
    run_info = collect_run_info(cfg)
    save_run_files(cfg, run_info)
    print(f'Run directory: {cfg.run_dir}')

    model = ExperimentMultiTask(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.run_dir, 'checkpoints'),
        filename='best-epoch{epoch:02d}',
        auto_insert_metric_name=False,
        monitor=cfg.trainer_checkpoint_monitor,
        mode=cfg.trainer_checkpoint_mode,
        save_top_k=1,
        save_last=cfg.trainer_save_last_checkpoint,
    )

    use_amp = cfg.optimizer_float_16 and torch.cuda.is_available()
    trainer = Trainer(
        default_root_dir=cfg.run_dir,
        logger=build_loggers(cfg, run_info),
        callbacks=[checkpoint_callback],
        accelerator=cfg.trainer_accelerator,
        devices=cfg.trainer_devices,
        max_epochs=cfg.num_epochs,
        precision='16-mixed' if use_amp else '32-true',
        num_sanity_val_steps=cfg.trainer_num_sanity_val_steps,
        log_every_n_steps=cfg.trainer_log_every_n_steps,
        limit_train_batches=cfg.trainer_limit_train_batches,
        limit_val_batches=cfg.trainer_limit_val_batches,
        limit_test_batches=cfg.trainer_limit_test_batches,
    )

    trainer.fit(model, ckpt_path=cfg.resume)

    final = {
        'monitor': cfg.trainer_checkpoint_monitor,
        'best_model_path': checkpoint_callback.best_model_path or None,
        'best_model_score': _to_float(checkpoint_callback.best_model_score)
        if checkpoint_callback.best_model_score is not None else None,
        'last_epoch_metrics': {k: _to_float(v) for k, v in trainer.callback_metrics.items()},
    }
    with open(os.path.join(cfg.run_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final, f, indent=4)

    if cfg.trainer_test_after_fit:
        trainer.test(model, ckpt_path=checkpoint_callback.best_model_path or None)

    return cfg.run_dir


if __name__ == '__main__':
    main()
