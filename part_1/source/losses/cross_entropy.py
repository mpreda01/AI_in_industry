from typing import Optional

import torch
from torch.nn import functional as F


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[torch.Tensor] = None, size_average=None,
                 ignore_index: int = -100, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction)
        self.ignore_index = ignore_index

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(input,
                               target,
                               weight=self.weight,
                               ignore_index=self.ignore_index,
                               reduction=self.reduction
                            )
