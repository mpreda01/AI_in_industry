import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from source.datasets.definitions import *
from source.losses import CrossEntropyLoss, MaskedDepthRegressionLoss
from source.utils.metrics import MetricsSemseg, MetricsDepth
from source.utils.helpers import resolve_optimizer, resolve_dataset_class, resolve_model_class, resolve_lr_scheduler
from source.utils.transforms import get_transforms
from source.utils.visualization import compose, to_pil_image

# Fixed order of the tasks: it defines the channel layout of models with a single output tensor
# (joint architecture: first the semseg class logits, then the depth channel).
TASK_ORDER = (MOD_SEMSEG, MOD_DEPTH)


class ExperimentMultiTask(pl.LightningModule):
    """
    Training recipe shared by every model of the repository: it works for any subset of tasks
    (cfg.tasks: semseg, depth or both) and any model registered in source.utils.helpers.resolve_model_class.

    Model contract:
      * ``model = ModelClass(cfg, outputs_desc)`` with ``outputs_desc = {task: number_of_channels}``;
      * ``model(rgb)`` returns a dict with one entry per task (a tensor, or a list of tensors for deep
        supervision, in which case the last one is the final prediction). Extra keys are ignored here;
      * optionally ``model.compute_aux_losses(outputs, batch) -> {name: scalar tensor}``. The auxiliary losses
        are added to the total loss with weight cfg.loss_weight_aux and logged separately (this is where e.g.
        the bin-center loss of the adaptive-bins model goes).
    """

    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = argparse.Namespace(**cfg)
        self.cfg = cfg
        self.save_hyperparameters(vars(cfg))

        self.tasks = [t for t in TASK_ORDER if t in cfg.tasks]

        dataset_class = resolve_dataset_class(cfg.dataset)
        self.datasets = {
            split: dataset_class(cfg.dataset_root, split, integrity_check=False)
            for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST)
        }
        for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST):
            print(f'Number of samples in {split} split: {len(self.datasets[split])}')

        train_set = self.datasets[SPLIT_TRAIN]
        self.rgb_mean = train_set.rgb_mean
        self.rgb_stddev = train_set.rgb_stddev
        self.depth_meters_mean = train_set.depth_meters_mean
        self.depth_meters_stddev = train_set.depth_meters_stddev
        self.semseg_num_classes = train_set.semseg_num_classes
        self.semseg_ignore_label = train_set.semseg_ignore_label
        self.semseg_class_names = train_set.semseg_class_names
        self.semseg_class_colors = train_set.semseg_class_colors

        train_set.set_transforms(get_transforms(
            semseg_ignore_label=self.semseg_ignore_label,
            geom_scale_min=cfg.aug_geom_scale_min,
            geom_scale_max=cfg.aug_geom_scale_max,
            geom_tilt_max_deg=cfg.aug_geom_tilt_max_deg,
            geom_wiggle_max_ratio=cfg.aug_geom_wiggle_max_ratio,
            geom_reflect=cfg.aug_geom_reflect,
            crop_random=cfg.aug_input_crop_size,
            rgb_mean=self.rgb_mean,
            rgb_stddev=self.rgb_stddev,
            depth_meters_mean=self.depth_meters_mean,
            depth_meters_stddev=self.depth_meters_stddev,
        ))
        self.transforms_val_test = get_transforms(
            semseg_ignore_label=self.semseg_ignore_label,
            crop_for_passable=32,
            rgb_mean=self.rgb_mean,
            rgb_stddev=self.rgb_stddev,
            depth_meters_mean=self.depth_meters_mean,
            depth_meters_stddev=self.depth_meters_stddev,
        )
        self.datasets[SPLIT_VALID].set_transforms(self.transforms_val_test)
        self.datasets[SPLIT_TEST].set_transforms(self.transforms_val_test)

        self.loss_fns = torch.nn.ModuleDict()
        self.loss_weights = {}
        self.metrics = {}
        if MOD_SEMSEG in self.tasks:
            self.loss_fns[MOD_SEMSEG] = CrossEntropyLoss(ignore_index=self.semseg_ignore_label)
            self.loss_weights[MOD_SEMSEG] = cfg.loss_weight_semseg
            self.metrics[MOD_SEMSEG] = MetricsSemseg(
                self.semseg_num_classes, self.semseg_ignore_label, self.semseg_class_names
            )
        if MOD_DEPTH in self.tasks:
            # see MaskedDepthRegressionLoss docstring: normalizing brings the raw meters-scale loss
            # to the same order of magnitude as cross-entropy, so loss_weight_semseg/depth mean what
            # they say instead of being swamped by the unit mismatch between the two tasks.
            depth_scale = self.depth_meters_stddev if cfg.normalize_depth_loss else 1.0
            self.loss_fns[MOD_DEPTH] = MaskedDepthRegressionLoss(kind=cfg.depth_loss, scale=depth_scale)
            self.loss_weights[MOD_DEPTH] = cfg.loss_weight_depth
            self.metrics[MOD_DEPTH] = MetricsDepth()

        outputs_descriptor = {
            MOD_SEMSEG: self.semseg_num_classes,
            MOD_DEPTH: 1,
        }
        self.outputs_descriptor = {t: outputs_descriptor[t] for t in self.tasks}
        self.net = resolve_model_class(cfg.model_name)(cfg, self.outputs_descriptor)
        num_params = sum(p.numel() for p in self.net.parameters())
        print(f'Model {cfg.model_name} ({cfg.model_encoder_name}), tasks {self.tasks}: {num_params:,} parameters')

    # ------------------------------------------------------------------ forward / losses

    @staticmethod
    def final_prediction(prediction):
        """Deep supervision: models may return a list of predictions; the last one is the final."""
        return prediction[-1] if isinstance(prediction, list) else prediction

    def compute_losses(self, y_hat, batch):
        """
        :return: (dict task -> loss, dict aux name -> loss, weighted total loss)
        """
        losses = {}
        for task in self.tasks:
            target = batch[task].squeeze(1)
            prediction = y_hat[task]
            if isinstance(prediction, list):
                losses[task] = sum(self.loss_fns[task](p, target) for p in prediction) / len(prediction)
            else:
                losses[task] = self.loss_fns[task](prediction, target)
        total = sum(self.loss_weights[task] * losses[task] for task in self.tasks)

        aux_losses = {}
        if hasattr(self.net, 'compute_aux_losses'):
            aux_losses = self.net.compute_aux_losses(y_hat, batch)
            total = total + self.cfg.loss_weight_aux * sum(aux_losses.values())
        return losses, aux_losses, total

    def log_losses(self, stage, losses, aux_losses, total, batch_size, **log_kwargs):
        logs = {f'loss_{stage}/{task}': loss for task, loss in losses.items()}
        logs.update({f'loss_{stage}/aux_{name}': loss for name, loss in aux_losses.items()})
        logs[f'loss_{stage}/total'] = total
        self.log_dict(logs, batch_size=batch_size, **log_kwargs)

    # ------------------------------------------------------------------ steps

    def training_step(self, batch, batch_nb):
        rgb = batch[MOD_RGB]
        y_hat = self.net(rgb)
        losses, aux_losses, total = self.compute_losses(y_hat, batch)
        self.log_losses('train', losses, aux_losses, total, rgb.shape[0], on_step=True, on_epoch=False, prog_bar=True)

        if self.can_visualize():
            self.visualize(batch, y_hat, batch[MOD_ID], 'imgs_train/batch_crops')

        return total

    def validation_step(self, batch, batch_nb):
        rgb = batch[MOD_RGB]
        y_hat = self.net(rgb)
        losses, aux_losses, total = self.compute_losses(y_hat, batch)
        self.log_losses('val', losses, aux_losses, total, rgb.shape[0], on_step=False, on_epoch=True)

        if MOD_SEMSEG in self.tasks:
            pred_lbl = self.final_prediction(y_hat[MOD_SEMSEG]).argmax(dim=1)
            self.metrics[MOD_SEMSEG].update_batch(pred_lbl, batch[MOD_SEMSEG].squeeze(1))
        if MOD_DEPTH in self.tasks:
            self.metrics[MOD_DEPTH].update_batch(self.final_prediction(y_hat[MOD_DEPTH]), batch[MOD_DEPTH].squeeze(1))

    def on_validation_epoch_end(self):
        self.observer_step()

        scalar_logs = {}
        task_scores = []
        if MOD_SEMSEG in self.tasks and self.metrics[MOD_SEMSEG].metrics_acc is not None:
            summary = self.metrics[MOD_SEMSEG].get_metrics_summary()
            self.metrics[MOD_SEMSEG].reset()
            score = (summary['mean_iou'] - 50).clamp(min=0)
            task_scores.append(score)
            scalar_logs['metrics_summary/semseg'] = score
            scalar_logs.update({f'metrics_task_semseg/{k.replace(" ", "_")}': v for k, v in summary.items()})
        if MOD_DEPTH in self.tasks and self.metrics[MOD_DEPTH].metrics_acc is not None:
            summary = self.metrics[MOD_DEPTH].get_metrics_summary()
            self.metrics[MOD_DEPTH].reset()
            score = (50 - summary['si_log_rmse']).clamp(min=0)
            task_scores.append(score)
            scalar_logs['metrics_summary/depth'] = score
            scalar_logs.update({f'metrics_task_depth/{k}': v for k, v in summary.items()})

        if task_scores:
            scalar_logs['metrics_summary/total'] = sum(task_scores) / len(task_scores)
        scalar_logs['trainer/LR'] = torch.tensor(self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log_dict(scalar_logs, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_nb):
        y_hat = self.net(batch[MOD_RGB])
        path_pred = os.path.join(self.cfg.run_dir, 'predictions')
        dataset = self.datasets[SPLIT_TEST]
        for task in self.tasks:
            os.makedirs(os.path.join(path_pred, task), exist_ok=True)
        for i in range(batch[MOD_RGB].shape[0]):
            sample_name = dataset.name_from_index(int(batch[MOD_ID][i]))
            if MOD_SEMSEG in self.tasks:
                pred_lbl = self.final_prediction(y_hat[MOD_SEMSEG])[i].argmax(dim=0)
                dataset.save_semseg(
                    os.path.join(path_pred, MOD_SEMSEG, f'{sample_name}.png'),
                    pred_lbl, self.semseg_class_colors, self.semseg_ignore_label
                )
            if MOD_DEPTH in self.tasks:
                dataset.save_depth(
                    os.path.join(path_pred, MOD_DEPTH, f'{sample_name}.png'),
                    self.final_prediction(y_hat[MOD_DEPTH])[i], out_of_range_policy='clamp_to_range'
                )

    # ------------------------------------------------------------------ optimization / data

    def configure_optimizers(self):
        optimizer = resolve_optimizer(self.cfg, self.parameters())
        lr_scheduler = resolve_lr_scheduler(self.cfg, optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'epoch'}}

    def train_dataloader(self):
        return DataLoader(
            self.datasets[SPLIT_TRAIN],
            self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.cfg.workers > 0,
            drop_last=True,
        )

    def create_val_test_dataloader(self, split):
        return DataLoader(
            self.datasets[split],
            self.cfg.batch_size_validation,
            shuffle=False,
            num_workers=self.cfg.workers_validation,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    def val_dataloader(self):
        return self.create_val_test_dataloader(SPLIT_VALID)

    def test_dataloader(self):
        return self.create_val_test_dataloader(SPLIT_TEST)

    # ------------------------------------------------------------------ W&B visualization

    def wandb_logger(self):
        return next((lg for lg in self.loggers if isinstance(lg, WandbLogger)), None)

    def can_visualize(self):
        if self.wandb_logger() is None or not self.trainer.is_global_zero:
            return False
        step = self.global_step - self.cfg.num_steps_visualization_first
        return step >= 0 and step % self.cfg.num_steps_visualization_interval == 0

    def visualize(self, batch, y_hat, rgb_tags, tag):
        batch = {k: v.cpu().detach() for k, v in batch.items() if torch.is_tensor(v)}
        visualization_plan = [(MOD_RGB, batch[MOD_RGB], rgb_tags)]
        if MOD_SEMSEG in self.tasks:
            pred_lbl = self.final_prediction(y_hat[MOD_SEMSEG]).cpu().detach().argmax(dim=1)
            visualization_plan += [
                (MOD_SEMSEG, batch[MOD_SEMSEG], 'GT SemSeg'),
                (MOD_SEMSEG, pred_lbl, 'Prediction SemSeg'),
            ]
        if MOD_DEPTH in self.tasks:
            pred_depth = self.final_prediction(y_hat[MOD_DEPTH]).cpu().detach()
            visualization_plan += [
                (MOD_DEPTH, batch[MOD_DEPTH], 'GT Depth'),
                (MOD_DEPTH, pred_depth, 'Prediction Depth'),
            ]
        vis = compose(
            visualization_plan,
            self.cfg,
            rgb_mean=self.rgb_mean,
            rgb_stddev=self.rgb_stddev,
            semseg_color_map=self.semseg_class_colors,
            semseg_ignore_label=self.semseg_ignore_label,
        )
        self.wandb_logger().log_image(tag, [to_pil_image(vis)], step=self.global_step, caption=[tag])

    def visualize_histograms(self, batch, y_hat_depth):
        import wandb
        y_hat_depth = y_hat_depth.cpu().detach()
        y_depth = batch[MOD_DEPTH].cpu().detach()
        y_depth = y_depth[y_depth > 0]
        y_depth_normalized = (y_depth - self.depth_meters_mean) / self.depth_meters_stddev
        y_hat_depth_normalized = (y_hat_depth - self.depth_meters_mean) / self.depth_meters_stddev
        # Only lowercase letters work for the log names
        self.wandb_logger().log_metrics({
            'histograms/gt_depth_normalized': wandb.Histogram(y_depth_normalized, num_bins=64),
            'histograms/gt_depth_meters': wandb.Histogram(y_depth, num_bins=64),
            'histograms/pred_depth_normalized': wandb.Histogram(y_hat_depth_normalized, num_bins=64),
            'histograms/pred_depth_meters': wandb.Histogram(y_hat_depth, num_bins=64),
        }, step=self.global_step)

    def observer_step(self):
        """Log predictions on fixed train/validation samples at the end of every validation epoch."""
        if self.wandb_logger() is None or not self.trainer.is_global_zero:
            return
        ids_train, ids_valid = self.cfg.observe_train_ids, self.cfg.observe_valid_ids
        list_samples = [self.datasets[SPLIT_TRAIN].get(i, override_transforms=self.transforms_val_test)
                        for i in ids_train]
        list_samples += [self.datasets[SPLIT_VALID].get(i, override_transforms=self.transforms_val_test)
                         for i in ids_valid]
        if not list_samples:
            return
        list_prefix = ('imgs_train/',) * len(ids_train) + ('imgs_val/',) * len(ids_valid)
        batch = default_collate(list_samples)
        rgb_tags = [f'{prefix}{sample_id}' for prefix, sample_id in zip(list_prefix, batch[MOD_ID].tolist())]
        with torch.no_grad():
            y_hat = self.net(batch[MOD_RGB].to(self.device))
        self.visualize(batch, y_hat, rgb_tags, 'imgs_val/observed_samples')
        if MOD_DEPTH in self.tasks:
            self.visualize_histograms(batch, self.final_prediction(y_hat[MOD_DEPTH]))
