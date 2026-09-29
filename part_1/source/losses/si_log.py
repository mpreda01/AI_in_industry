from typing import List

import torch

@torch.jit.script
def masked_mean_var(data: torch.Tensor, mask: torch.Tensor, dim: List[int]):
    if mask is None:
        return data.mean(dim=dim), data.var(dim=dim, unbiased=False)
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(mask_sum, min=1.0)
    mask_var = torch.sum(mask * (data - mask_mean) ** 2, dim=dim, keepdim=True) / torch.clamp(mask_sum, min=1.0)
    return mask_mean, mask_var


class SILogLoss(torch.nn.Module):
    def __init__(
            self,
            gamma: float = 0.15,
            eps: float = 1e-5
        ):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.gamma: float = gamma
        self.eps = eps
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Avoid dummy channel dimension
        if input.ndim == 4:
            input = input.squeeze(1)
        if target.ndim == 4:
            target = target.squeeze(1)
        

        # TODO: Implement a proper silog loss and taking into consideration
        # the invalid pixels (target is 0).
        loss = (input - target).abs().sum()

        return loss
