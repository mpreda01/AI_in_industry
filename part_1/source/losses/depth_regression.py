import torch


class MaskedDepthRegressionLoss(torch.nn.Module):
    """
    L1 or L2 regression loss on metric depth, computed only over valid ground-truth pixels
    (finite and > 0; 0 marks out-of-range/sky pixels in the dataset).

    ``scale`` divides the residual before the loss is computed (mean(|diff / scale|) for L1,
    mean((diff / scale)^2) for L2). It exists to bring the depth loss to a magnitude comparable
    to the other task losses in a multi-task sum: raw L1-in-meters (O(1-10) m, given the dataset's
    4-300 m range) dwarfs cross-entropy (O(0.1-2) nats) under nominally equal loss weights, which
    lets depth dominate the total gradient and hurts the other task ("negative transfer"). Passing
    the dataset's depth_meters_stddev as scale turns the loss into "std-units", roughly O(1), so the
    configured loss weights (loss_weight_semseg / loss_weight_depth) reflect the intended balance
    instead of being swamped by a unit mismatch. It is a fixed constant, not a learned or adaptive
    weighting scheme.
    """

    def __init__(self, kind: str = 'l1', scale: float = 1.0):
        super().__init__()
        if kind not in ('l1', 'l2'):
            raise ValueError(f'Unknown depth loss "{kind}", expected "l1" or "l2"')
        if scale <= 0:
            raise ValueError(f'scale must be > 0, got {scale}')
        self.kind = kind
        self.scale = scale

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.float()
        target = target.float()
        if input.ndim == 4:
            input = input.squeeze(1)
        if target.ndim == 4:
            target = target.squeeze(1)

        valid = torch.isfinite(target) & (target > 0)
        if not valid.any():
            # keep the graph connected so backward() does not fail on an all-invalid batch
            return input.sum() * 0.0

        diff = (input[valid] - target[valid]) / self.scale
        return diff.abs().mean() if self.kind == 'l1' else (diff * diff).mean()
