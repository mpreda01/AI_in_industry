import torch
import torch.nn as nn
import torch.nn.functional as F

from source.models.model_parts import Encoder, get_encoder_channel_counts, DecoderDeeplabV3p, TransformerBlock, LatentsExtractor


class ModelAdaptiveDepth(torch.nn.Module):
    """
    Depth as classification with adaptive bins (AdaBins, Bhat et al. CVPR 2021).

    The DeepLabV3+ decoder produces per-pixel query features. Image-dependent "latents" are extracted by
    downsampling the queries to a sqrt(num_bins) x sqrt(num_bins) grid and refined with transformer blocks
    (self-attention among bins). Each latent is projected to a bin width: widths are normalized to sum to 1
    and scaled to the depth range, so the bin centers (cumulative sum) are ordered and adapt to the scene.
    Each pixel gets a softmax over bins (dot product between its query and the latents); the depth is the
    probability-weighted sum of the bin centers.

    Besides ``depth`` the forward returns ``bin_centers`` (B, num_bins), used by ``compute_aux_losses``
    (chamfer loss between bin centers and ground-truth depths, as in AdaBins).
    """

    MIN_DEPTH = 0.1
    MAX_DEPTH = 300.0
    AUX_MAX_PIXELS = 4096  # ground-truth pixels sampled per image for the chamfer loss

    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc
        assert list(outputs_desc) == ['depth'] and outputs_desc['depth'] == 1, 'adaptive_depth predicts depth only'
        self.num_bins = cfg.num_bins
        dim = 256

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=cfg.pretrained,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.decoder = DecoderDeeplabV3p(ch_out_encoder_bottleneck, ch_out_encoder_4x, dim)

        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

        self.latents_extractor = LatentsExtractor(cfg.num_bins)
        # learnable positional embedding: gives each latent an identity (bins are ordered)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.num_bins, dim) * 0.02)
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, num_heads=cfg.num_heads, expansion=cfg.expansion)
            for _ in range(cfg.num_transformer_layers)
        ])
        self.latents_norm = nn.LayerNorm(dim)
        self.width_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, x):
        B, _, H, W = x.shape
        input_resolution = (H, W)

        features = self.encoder(x)
        features_lowest = features[max(features.keys())]
        features_4x, _ = self.decoder(features_lowest, features[4])

        queries = self.conv3x3(features_4x)  # B C h w

        # 1. adaptive latents: extracted from the image itself
        latents = self.latents_extractor(queries)  # B N C
        # 2. self-attention among latents
        pos_embed = self.pos_embed.expand(B, -1, -1)
        for block in self.transformer:
            latents = block(latents, pos_embed)
        latents = self.latents_norm(latents)

        # 3. latent -> bin width -> bin centers within [MIN_DEPTH, MAX_DEPTH]
        widths = F.softmax(self.width_head(latents).squeeze(-1), dim=1)  # B N, sums to 1
        widths = (self.MAX_DEPTH - self.MIN_DEPTH) * widths
        edges = self.MIN_DEPTH + torch.cumsum(widths, dim=1)
        bin_centers = edges - 0.5 * widths  # B N

        # 4. per-pixel probabilities over bins and expected depth
        similarity = torch.einsum("bcij,bnc->bnij", queries, latents) / queries.shape[1] ** 0.5
        probabilities = F.softmax(similarity, dim=1)
        predictions_4x = (probabilities * bin_centers[:, :, None, None]).sum(dim=1, keepdim=True)

        predictions_1x = F.interpolate(predictions_4x, size=input_resolution, mode='bilinear', align_corners=False)

        return {'depth': predictions_1x, 'bin_centers': bin_centers}

    def compute_aux_losses(self, outputs, batch):
        """Chamfer distance (both directions) between predicted bin centers and valid ground-truth depths."""
        centers = outputs['bin_centers']  # B N
        target = batch['depth'].flatten(1)  # B P
        loss = centers.new_zeros(())
        count = 0
        for c, t in zip(centers, target):
            t = t[torch.isfinite(t) & (t > 0)].to(c.dtype)
            if t.numel() == 0:
                continue
            if t.numel() > self.AUX_MAX_PIXELS:
                t = t[torch.randint(0, t.numel(), (self.AUX_MAX_PIXELS,), device=t.device)]
            dist = (c[:, None] - t[None, :]).abs()  # N P
            loss = loss + dist.min(dim=0).values.mean() + dist.min(dim=1).values.mean()
            count += 1
        if count == 0:
            return {'bin_centers': centers.sum() * 0.0}
        # meters -> O(1) scale like the other losses
        return {'bin_centers': loss / count / self.MAX_DEPTH}
