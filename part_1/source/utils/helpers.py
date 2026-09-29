from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR

from source.datasets.dataset_miniscapes import DatasetMiniscapes
from source.models import *


# Registries: the names below are the values accepted by `data.dataset` and `model.model_name` in the config
# (see SCHEMA in source/utils/config.py). To add a model, register it here and in SCHEMA / MODEL_TASKS.
DATASETS = {
    'miniscapes': DatasetMiniscapes,
}

MODELS = {
    'deeplabv3p': ModelDeepLabV3Plus,
    'adaptive_depth': ModelAdaptiveDepth,
    'deeplabv3p_multitask': ModelDeepLabV3PlusMultiTask,
}


def resolve_dataset_class(name):
    if name not in DATASETS:
        raise KeyError(f'Unknown dataset "{name}", registered: {list(DATASETS)}')
    return DATASETS[name]


def resolve_model_class(name):
    if name not in MODELS:
        raise KeyError(f'Unknown model "{name}", registered: {list(MODELS)}')
    return MODELS[name]


def resolve_optimizer(cfg, params):
    if cfg.optimizer == 'sgd':
        return SGD(
            params,
            lr=cfg.optimizer_lr,
            momentum=cfg.optimizer_momentum,
            weight_decay=cfg.optimizer_weight_decay,
        )
    elif cfg.optimizer == 'adam':
        return Adam(
            params,
            lr=cfg.optimizer_lr,
            weight_decay=cfg.optimizer_weight_decay,
        )
    else:
        raise NotImplementedError


def resolve_lr_scheduler(cfg, optimizer):
    if cfg.lr_scheduler == 'poly':
        return LambdaLR(
            optimizer,
            lambda ep: max(1e-6, (1 - ep / cfg.num_epochs) ** cfg.lr_scheduler_power)
        )
    else:
        raise NotImplementedError

