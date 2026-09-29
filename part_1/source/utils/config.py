"""
YAML based configuration of a training run.

One experiment = one complete YAML file (see ``config.yaml`` in the repository root). The file is grouped in
sections for readability; the loader validates it against ``SCHEMA`` below (unknown, missing or badly typed keys
are errors) and flattens it into a single ``argparse.Namespace`` so the rest of the code reads ``cfg.optimizer_lr``,
``cfg.tasks``, ... Individual values can be overridden from the command line with ``section.key=value``.
"""
import argparse
import json
import os
import re
import sys

import yaml

from source.datasets.definitions import MOD_SEMSEG, MOD_DEPTH


def expandpath(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Type specifications used by SCHEMA. A spec is (accepted types, allowed choices or None).
_STR = ((str,), None)
_STR_OPT = ((str, type(None)), None)
_INT = ((int,), None)
_INT_OPT = ((int, type(None)), None)
_FLOAT = ((int, float), None)
_NUM_OPT = ((int, float, type(None)), None)
_BOOL = ((bool,), None)
_LIST_STR = ((list,), None)
_LIST_INT = ((list,), None)


def _choice(*choices):
    return (str,), choices


# section -> key -> spec. Every key is required in every experiment file (no hidden defaults).
SCHEMA = {
    'experiment': {
        'name': _STR,                   # run name, part of the output directory and the W&B run name
        'seed': _INT,                   # seeds python, numpy, torch and the dataloader workers
        'tasks': _LIST_STR,             # subset of [semseg, depth]
        'resume': _STR_OPT,             # checkpoint to resume from, or null
    },
    'paths': {
        'dataset_root': _STR,           # contains train/ val/ test/
        'output_dir': _STR,             # every run creates its own sub-directory here
    },
    'data': {
        'dataset': _choice('miniscapes'),
        'workers': _INT,
        'workers_validation': _INT,
        'batch_size_validation': _INT,
    },
    'model': {
        'model_name': _choice('deeplabv3p', 'deeplabv3p_multitask', 'adaptive_depth'),
        'model_encoder_name': _choice('resnet18', 'resnet34'),
        'pretrained': _BOOL,
        'num_bins': _INT,               # adaptive_depth only
        'num_heads': _INT,              # adaptive_depth only
        'expansion': _INT,              # adaptive_depth only
        'num_transformer_layers': _INT,  # adaptive_depth only
    },
    'optimization': {
        'num_epochs': _INT,
        'batch_size': _INT,
        'optimizer': _choice('sgd', 'adam'),
        'optimizer_lr': _FLOAT,
        'optimizer_momentum': _FLOAT,
        'optimizer_weight_decay': _FLOAT,
        'optimizer_float_16': _BOOL,
        'lr_scheduler': _choice('poly'),
        'lr_scheduler_power': _FLOAT,
    },
    'loss': {
        'loss_weight_semseg': _FLOAT,
        'loss_weight_depth': _FLOAT,
        'loss_weight_aux': _FLOAT,      # weight of the auxiliary losses a model may define
        'depth_loss': _choice('l1', 'l2'),
        # Normalize the depth residual by the dataset's depth_meters_stddev before computing L1/L2, so the
        # depth loss and cross-entropy are on a comparable scale under equal loss weights (see
        # MaskedDepthRegressionLoss docstring). Only affects the training loss, not the reported metrics
        # (metrics_task_depth/* are always in meters).
        'normalize_depth_loss': _BOOL,
    },
    'augmentation': {
        'aug_input_crop_size': _INT,
        'aug_geom_scale_min': _FLOAT,
        'aug_geom_scale_max': _FLOAT,
        'aug_geom_tilt_max_deg': _FLOAT,
        'aug_geom_wiggle_max_ratio': _FLOAT,
        'aug_geom_reflect': _BOOL,
    },
    'trainer': {
        'accelerator': _STR,            # auto | cpu | gpu
        'devices': ((int, str), None),  # auto or a number of devices
        'log_every_n_steps': _INT,
        'num_sanity_val_steps': _INT,
        'limit_train_batches': _NUM_OPT,  # null = all; int = number of batches; float = fraction (debug only)
        'limit_val_batches': _NUM_OPT,
        'limit_test_batches': _NUM_OPT,
        'checkpoint_monitor': _STR,     # logged metric that selects the best checkpoint
        'checkpoint_mode': _choice('max', 'min'),
        'save_last_checkpoint': _BOOL,  # also keep last.ckpt (needed to resume; doubles checkpoint disk usage)
        'test_after_fit': _BOOL,        # write test-split predictions with the best checkpoint
    },
    'wandb': {
        'mode': _choice('online', 'offline', 'disabled'),
        'project': _STR,
        'entity': _STR_OPT,
        'group': _STR_OPT,              # e.g. the experiment family, to compare runs
        'tags': _LIST_STR,
        'notes': _STR_OPT,
        'key_file': _STR_OPT,           # file with the API key (WANDB_API_KEY takes precedence)
    },
    'visualization': {
        'num_steps_visualization_first': _INT,
        'num_steps_visualization_interval': _INT,
        'visualize_num_samples_in_batch': _INT,
        'visualize_img_grid_width': _INT,
        'observe_train_ids': _LIST_INT,
        'observe_valid_ids': _LIST_INT,
    },
}

# Sections whose keys get the section name as prefix in the flat namespace (cfg.wandb_mode, cfg.trainer_devices).
PREFIXED_SECTIONS = ('trainer', 'wandb')

PATH_KEYS = ('dataset_root', 'output_dir', 'resume', 'wandb_key_file')

# Task sets each model can be trained with.
MODEL_TASKS = {
    'deeplabv3p': ({MOD_SEMSEG}, {MOD_DEPTH}, {MOD_SEMSEG, MOD_DEPTH}),
    'deeplabv3p_multitask': ({MOD_SEMSEG, MOD_DEPTH},),
    'adaptive_depth': ({MOD_DEPTH},),
}


def _flat_name(section, key):
    return f'{section}_{key}' if section in PREFIXED_SECTIONS else key


# Keys that identify a run rather than an experiment setup; excluded when comparing experiment settings.
EXPERIMENT_INVARIANT_KEYS = (
    'dataset_root',
    'output_dir',
    'run_dir',
    'run_name',
    'batch_size_validation',
    'workers',
    'workers_validation',
    'num_steps_visualization_first',
    'num_steps_visualization_interval',
    'visualize_num_samples_in_batch',
    'visualize_img_grid_width',
    'observe_train_ids',
    'observe_valid_ids',
)

_UNRESOLVED_VAR = re.compile(r'\$\{?[A-Za-z_][A-Za-z0-9_]*\}?')


def _expand_env(value, where):
    if isinstance(value, dict):
        return {k: _expand_env(v, f'{where}.{k}') for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v, where) for v in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        unresolved = _UNRESOLVED_VAR.search(expanded)
        if unresolved:
            raise ValueError(
                f'Config value "{where}" references the environment variable "{unresolved.group(0)}", '
                f'which is not set (value: "{value}")'
            )
        return expanded
    return value


def _coerce(value, spec):
    # YAML 1.1 (PyYAML) reads "1e-3" as a string, only "1.0e-3" as a float: accept the former for float keys
    types, _ = spec
    if isinstance(value, str) and float in types and str not in types:
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _check_type(where, value, spec):
    types, choices = spec
    # bool is an int subclass: reject booleans where a number is expected and vice versa
    if isinstance(value, bool) and bool not in types:
        raise TypeError(f'Config value "{where}" must be of type {_type_names(types)}, got bool')
    if not isinstance(value, types):
        raise TypeError(f'Config value "{where}" must be of type {_type_names(types)}, '
                        f'got {type(value).__name__} ({value!r})')
    if choices is not None and value not in choices:
        raise ValueError(f'Config value "{where}" must be one of {list(choices)}, got {value!r}')


def _type_names(types):
    return '/'.join(t.__name__ for t in types)


def _set_by_dotted_key(raw, dotted, value):
    if '.' in dotted:
        section, key = dotted.split('.', 1)
    else:
        matches = [s for s, keys in SCHEMA.items() if dotted in keys]
        if len(matches) != 1:
            raise KeyError(f'Override "{dotted}" is ambiguous or unknown; use the form section.key '
                           f'(sections: {list(SCHEMA)})')
        section, key = matches[0], dotted
    if section not in SCHEMA or key not in SCHEMA[section]:
        raise KeyError(f'Override "{dotted}" does not match any key of the config schema')
    raw.setdefault(section, {})[key] = value


def parse_overrides(overrides):
    """['optimization.optimizer_lr=1e-3', ...] -> list of (dotted key, python value parsed as YAML)."""
    parsed = []
    for item in overrides:
        if '=' not in item:
            raise ValueError(f'Override "{item}" must have the form section.key=value')
        key, value = item.split('=', 1)
        parsed.append((key.strip(), yaml.safe_load(value)))
    return parsed


def build_config(raw, overrides=()):
    """Validate a nested config dict (already loaded from YAML), apply overrides and flatten it."""
    unknown_sections = set(raw) - set(SCHEMA)
    if unknown_sections:
        raise KeyError(f'Unknown config sections: {sorted(unknown_sections)}; valid: {list(SCHEMA)}')

    raw = {section: dict(raw.get(section) or {}) for section in SCHEMA}
    for dotted, value in overrides:
        _set_by_dotted_key(raw, dotted, value)

    flat = {}
    for section, keys in SCHEMA.items():
        unknown_keys = set(raw[section]) - set(keys)
        if unknown_keys:
            raise KeyError(f'Unknown keys in config section "{section}": {sorted(unknown_keys)}; '
                           f'valid: {list(keys)}')
        missing_keys = set(keys) - set(raw[section])
        if missing_keys:
            raise KeyError(f'Missing keys in config section "{section}": {sorted(missing_keys)}')
        for key, spec in keys.items():
            where = f'{section}.{key}'
            value = _coerce(_expand_env(raw[section][key], where), spec)
            _check_type(where, value, spec)
            name = _flat_name(section, key)
            assert name not in flat, f'Config key collision on "{name}"'
            flat[name] = value

    for key in PATH_KEYS:
        if flat[key] is not None:
            flat[key] = expandpath(flat[key])

    cfg = argparse.Namespace(**flat)
    validate_config(cfg)
    return cfg


# Whether a lower or higher value is better, keyed by the part of the metric name after the last '/'.
# Used to catch a checkpoint_monitor / checkpoint_mode mismatch, which silently keeps the WORST checkpoint
# instead of the best (e.g. monitoring an error metric like si_log_rmse with checkpoint_mode: max).
_LOWER_IS_BETTER = {'log_mae', 'mae', 'rmse', 'rel', 'rel_squared', 'inv_mae', 'inv_rmse', 'log_rmse', 'si_log_rmse'}
_HIGHER_IS_BETTER = {'mean_iou', 'delta1', 'delta2', 'delta3'}


def _expected_checkpoint_mode(monitor):
    """('min'/'max') the monitor is known to want, or None if it cannot be inferred (a custom metric)."""
    if monitor.startswith('loss_train/') or monitor.startswith('loss_val/'):
        return 'min'
    if monitor.startswith('metrics_summary/'):
        return 'max'  # max(mIoU-50, 0) and max(50-SILogRMSE, 0): higher is always better here
    if monitor.startswith('metrics_task_semseg/'):
        return 'max'  # mean_iou and every per-class IoU
    if monitor.startswith('metrics_task_depth/'):
        suffix = monitor.split('/', 1)[1]
        if suffix in _HIGHER_IS_BETTER:
            return 'max'
        if suffix in _LOWER_IS_BETTER:
            return 'min'
    return None


def validate_config(cfg):
    tasks = cfg.tasks
    if len(tasks) == 0 or len(set(tasks)) != len(tasks) or not set(tasks) <= {MOD_SEMSEG, MOD_DEPTH}:
        raise ValueError(f'experiment.tasks must be a non-empty subset of [{MOD_SEMSEG}, {MOD_DEPTH}] '
                         f'without duplicates, got {tasks}')
    if set(tasks) not in MODEL_TASKS[cfg.model_name]:
        supported = [sorted(s) for s in MODEL_TASKS[cfg.model_name]]
        raise ValueError(f'Model "{cfg.model_name}" cannot be trained with tasks {sorted(tasks)}; '
                         f'supported task sets: {supported}')
    expected_mode = _expected_checkpoint_mode(cfg.trainer_checkpoint_monitor)
    if expected_mode is not None and cfg.trainer_checkpoint_mode != expected_mode:
        raise ValueError(
            f'trainer.checkpoint_monitor="{cfg.trainer_checkpoint_monitor}" is '
            f'{"lower" if expected_mode == "min" else "higher"}-is-better, but trainer.checkpoint_mode='
            f'"{cfg.trainer_checkpoint_mode}" (expected "{expected_mode}"). This mismatch silently keeps '
            f'the WORST checkpoint instead of the best.'
        )
    if cfg.model_name == 'adaptive_depth' and round(cfg.num_bins ** 0.5) ** 2 != cfg.num_bins:
        raise ValueError(f'model.num_bins must be a perfect square for adaptive_depth, got {cfg.num_bins}')
    for name in ('loss_weight_semseg', 'loss_weight_depth', 'loss_weight_aux'):
        if getattr(cfg, name) < 0:
            raise ValueError(f'loss.{name} must be >= 0')
    if cfg.batch_size < 1 or cfg.num_epochs < 1:
        raise ValueError('optimization.batch_size and optimization.num_epochs must be >= 1')
    if not all(isinstance(i, int) and not isinstance(i, bool)
               for i in cfg.observe_train_ids + cfg.observe_valid_ids):
        raise TypeError('visualization.observe_*_ids must be lists of integers')
    if not all(isinstance(t, str) for t in cfg.wandb_tags):
        raise TypeError('wandb.tags must be a list of strings')
    # a monitored metric of an inactive task is never logged: ModelCheckpoint would crash after the first epoch
    for task in (MOD_SEMSEG, MOD_DEPTH):
        for prefix in (f'metrics_task_{task}/', f'metrics_summary/{task}', f'loss_val/{task}'):
            if cfg.trainer_checkpoint_monitor.startswith(prefix) and task not in cfg.tasks:
                raise ValueError(
                    f'trainer.checkpoint_monitor="{cfg.trainer_checkpoint_monitor}" refers to the {task} task, '
                    f'which is not trained (experiment.tasks={cfg.tasks}). For a depth-only run use '
                    f'metrics_task_depth/si_log_rmse with mode min')
    if isinstance(cfg.trainer_devices, str) and cfg.trainer_devices != 'auto':
        raise ValueError('trainer.devices must be "auto" or an integer')


def config_to_nested(cfg):
    """Inverse of the flattening: {section: {key: value}} as found in the YAML file (schema keys only)."""
    return {
        section: {key: getattr(cfg, _flat_name(section, key)) for key in keys}
        for section, keys in SCHEMA.items()
    }


def load_config(argv=None):
    """
    Parse ``--config path.yaml [section.key=value ...]`` and return the validated flat configuration.
    """
    parser = argparse.ArgumentParser(
        description='Train a model described by a YAML config file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str, required=True, help='Path to the experiment YAML file')
    parser.add_argument(
        'overrides', nargs='*', metavar='section.key=value',
        help='Override individual config values, e.g. optimization.optimizer_lr=0.001 experiment.tasks=[semseg]')
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    with open(os.path.expanduser(args.config)) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f'Config file {args.config} must contain a YAML mapping')

    cfg = build_config(raw, parse_overrides(args.overrides))
    cfg.config_path = os.path.abspath(args.config)
    return cfg


def print_config(cfg):
    print(json.dumps(config_to_nested(cfg), indent=4))
