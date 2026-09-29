import torch
import torch.nn.functional as F

from source.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p


class ModelDeepLabV3PlusMultiTask(torch.nn.Module):
    """
    Branched DeepLabV3+ multi-task architecture: only the encoder is shared between tasks; each task gets
    its own ASPP module and its own decoder (contrast with ModelDeepLabV3Plus, the joint architecture, where
    everything but the last 1x1 convolution is shared). See doc/training_system.md and the handout, Problem 3.
    """

    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=cfg.pretrained,
            zero_init_residual=True,
            # Same output stride as the joint model (ModelDeepLabV3Plus), so the branched-vs-joint comparison
            # isolates the effect of per-task ASPP/decoders and does not also change the encoder's output stride.
            replace_stride_with_dilation=(False, False, False),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.aspps = torch.nn.ModuleDict({
            task: ASPP(ch_out_encoder_bottleneck, 256) for task in outputs_desc
        })
        self.decoders = torch.nn.ModuleDict({
            task: DecoderDeeplabV3p(256, ch_out_encoder_4x, num_ch) for task, num_ch in outputs_desc.items()
        })

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        out = {}

        for task in self.outputs_desc:
            features_task = self.aspps[task](features_lowest)
            predictions_4x, _ = self.decoders[task](features_task, features[4])
            predictions_1x = F.interpolate(predictions_4x, size=input_resolution, mode='bilinear', align_corners=False)
            # be sure that depth is > 0, you can use other operators than exp
            out[task] = predictions_1x.exp().clamp(0.1, 300.0) if task == 'depth' else predictions_1x

        return out
