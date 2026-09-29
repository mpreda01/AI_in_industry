from .cross_entropy import CrossEntropyLoss
from .si_log import SILogLoss
from .depth_regression import MaskedDepthRegressionLoss

__all__ = [
    'CrossEntropyLoss',
    'SILogLoss',
    'MaskedDepthRegressionLoss',
]