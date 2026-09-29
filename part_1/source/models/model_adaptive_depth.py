import torch
import torch.nn as nn
import torch.nn.functional as F

from source.models.model_parts import Encoder, get_encoder_channel_counts, DecoderDeeplabV3p, TransformerBlock, LatentsExtractor


class ModelAdaptiveDepth(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc
        ch_out = sum(outputs_desc.values())

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=cfg.pretrained,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.decoder = DecoderDeeplabV3p(ch_out_encoder_bottleneck, ch_out_encoder_4x, 256)

        self.conv3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)


        self.latents = nn.Parameter(torch.randn(1, cfg.num_bins, 256), requires_grad=True)
        depth_values = torch.linspace(0.1, 100.0, cfg.num_bins).view(1, cfg.num_bins, 1, 1)
        self.depth_values = nn.Parameter(depth_values, requires_grad=False)

    def forward(self, x):
        B, _, H, W = x.shape
        input_resolution = (H, W)

        features = self.encoder(x)

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        features_4x, _ = self.decoder(features_lowest, features[4])

        queries = self.conv3x3(features_4x)

        # TODO: Implement the adaptive extraction of bins and depth computation
        # 1. extract "latents" (i.e., learnable bins) with LatentsExtractor
        # 2. process them with a few layers of TrasformerBlocks
        # 3. for each latent, project it to a scalar value to obtain the depth
        #    value each corresponds to (which is learnable based on the image)
        # 4. for each pixel, compute the probabilities that it belongs to one of 
        #    the bins: softmax of the dot product between queries and latents
        # The predefined depth_values and latents, will become learnable and 
        # determined by the specific input
        similarity = torch.einsum("bcij,bnc->bnij", queries, self.latents.repeat(B, 1, 1))
        predictions_4x = (F.softmax(similarity, dim=1) * self.depth_values.repeat(B, 1, 1, 1)).sum(dim=1, keepdim=True)

        predictions_1x = F.interpolate(predictions_4x, size=input_resolution, mode='bilinear', align_corners=False)

        out = {}
        offset = 0

        for task, num_ch in self.outputs_desc.items():
            current_precition = predictions_1x[:, offset:offset+num_ch, :, :]
            out[task] = current_precition
            offset += num_ch

        return out
