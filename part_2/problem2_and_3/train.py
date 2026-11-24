# Material for Project 2

import os
import sys
import argparse
import uuid
from datetime import datetime

import yaml
import shutil
import h5py
import wandb
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader, get_worker_info
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from model import Model
from dataset import DatasetLoader
from residual import ResidualDatasetLoader, decode_boxes   # data.residual: proposal-relative refinement

from utils.task4 import RegressionLoss, ClassificationLoss
from utils.eval import generate_final_predictions, save_detections, generate_submission, compute_map
from utils.vis import point_scene

# Leonardo port (Lightning 2.x, single A100): number of DataLoader workers = CPUs allocated by SLURM
# (os.cpu_count() returns all the node's cores).
NUM_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))

def reopen_h5(_):
    '''DataLoader worker_init_fn: give each worker its own HDF5 handle instead of the one forked from the parent.'''
    ds = get_worker_info().dataset
    ds.hf = h5py.File(ds.hf.filename, 'r')

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.residual = config['data'].get('residual', False)   # see residual.py
        self.output_dir = self.config['eval']['output_dir']
        if os.path.exists(self.output_dir): shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        self.model = Model(config['model'])
        self.reg_loss = RegressionLoss(config['loss'])
        self.cls_loss = ClassificationLoss(config['loss'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, assigned_target, iou = batch['input'], batch['assigned_target'], batch['iou']
        pred_box, pred_class = self(x)
        loss = self.reg_loss(pred_box, assigned_target, iou) \
               + self.cls_loss(pred_class, iou)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, assigned_target, iou = batch['input'], batch['assigned_target'], batch['iou']
        pred_box, pred_class = self(x)

        loss = self.reg_loss(pred_box, assigned_target, iou) \
               + self.cls_loss(pred_class, iou)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.residual:   # the network predicts corrections of the proposals: turn them into boxes
            pred_box = decode_boxes(batch['proposal'], pred_box)
        nms_pred, nms_score = generate_final_predictions(pred_box, pred_class, self.config['eval'])
        save_detections(os.path.join(self.output_dir, 'pred'), batch['frame'], nms_pred, nms_score)
        self.val_frames.append(batch['frame'])   # a single frame id (val batch_size = 1)

        # Visualization
        if batch_idx == 0:
            scene = point_scene(batch['points'], nms_pred, batch['target'], name=f'e{self.current_epoch}')
            self.logger.experiment.log(scene, commit=False)

    def on_validation_epoch_start(self):
        self.val_frames = []   # frames predicted this epoch (all of val, or a subset with limit_val_batches)

    def on_validation_epoch_end(self):   # Lightning 2.x: was validation_epoch_end(self, outputs)
        # 3D AP at the course's car IoU 0.5 (e/m/h_map) and at the official KITTI IoU 0.7 (e/m/h_map70), one pass
        maps = compute_map(self.valid_dataset.hf,
                           os.path.join(self.output_dir, 'pred'),
                           self.val_frames,   # was self.valid_dataset.frames (fails with limit_val_batches)
                           overlaps=(0.5, 0.7))
        shutil.rmtree(os.path.join(self.output_dir, 'pred'))   # was rmtree(self.output_dir, 'pred'), which deleted the whole folder
        for suffix, (easy, moderate, hard) in [('', maps[0.5]), ('70', maps[0.7])]:
            self.log(f'e_map{suffix}', easy, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'm_map{suffix}', moderate, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'h_map{suffix}', hard, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        frame, x = batch['frame'], batch['input']
        pred_box, pred_class = self(x)
        if self.residual:
            pred_box = decode_boxes(batch['proposal'], pred_box)
        nms_pred, nms_score = generate_final_predictions(pred_box, pred_class, self.config['eval'])
        save_detections(os.path.join(self.output_dir, 'test'), frame, nms_pred, nms_score)

    @property
    def submission_file(self):
        return os.path.join(self.output_dir, 'submission.zip')

    def on_test_epoch_end(self):   # Lightning 2.x: was test_epoch_end(self, outputs)
        generate_submission(os.path.join(self.output_dir, 'test'), self.submission_file)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), **self.config['optimizer'])
        scheduler = lrs.MultiStepLR(optimizer, **self.config['scheduler'])
        return [optimizer], [scheduler]

    def setup(self, stage):
        Loader = ResidualDatasetLoader if self.residual else DatasetLoader
        self.train_dataset = Loader(config=self.config['data'], split='train')
        self.valid_dataset = Loader(config=self.config['data'], split='val')
        # There is no labelled test split (KITTI test labels are not public): only build it if the files exist
        test_h5 = os.path.join(self.config['data']['root_dir'], 'project2_test.h5')
        self.test_dataset = Loader(config=self.config['data'], split='test') if os.path.exists(test_h5) else None

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.config['data']['batch_size'],
                          shuffle=True,
                          pin_memory=True,
                          num_workers=NUM_WORKERS,
                          worker_init_fn=reopen_h5,
                          persistent_workers=True,
                          collate_fn=self.train_dataset.collate_batch)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,
                          batch_size=1,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=NUM_WORKERS,
                          worker_init_fn=reopen_h5,
                          persistent_workers=True,
                          collate_fn=self.train_dataset.collate_batch)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=1,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=NUM_WORKERS,
                          worker_init_fn=reopen_h5,
                          collate_fn=self.test_dataset.collate_batch)

def train(config, run_name, ckpt_path=None):
    # Offline on Leonardo compute nodes (WANDB_MODE=offline in the SLURM script), synced later from a login node.
    # id=run_name + resume='allow' continues the same wandb run when resuming from a checkpoint.
    wandb_logger = WandbLogger(
        name=run_name,
        id=run_name,
        resume='allow',
        project='CVAIAC-Ex2',
        save_dir=os.path.join(config["trainer"]["default_root_dir"])
    )

    # last.ckpt for resuming across SLURM jobs (24 h walltime) + the best epoch by moderate AP @ IoU 0.7
    checkpoint_local_callback = ModelCheckpoint(
        dirpath=os.path.join(config["trainer"]["default_root_dir"], run_name, 'checkpoints'),
        filename='epoch{epoch:02d}-m_map70_{m_map70:.2f}',
        auto_insert_metric_name=False,
        monitor='m_map70',
        mode='max',
        save_top_k=1,
        save_last=True,
    )

    print("Start training", run_name, f"(resuming from {ckpt_path})" if ckpt_path else "")

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_local_callback, LearningRateMonitor(logging_interval='epoch')],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',   # was gpus=-1, accelerator='ddp' (Lightning 1.x)
        devices=1,
        **config['trainer']
    )
    litModel = LitModel(config)
    trainer.fit(litModel, ckpt_path=ckpt_path)   # was trainer:resume_from_checkpoint in config.yaml
    if litModel.test_dataset is not None:
        trainer.test(litModel)

    wandb.finish()

def adapt_config(config, run_name):
	user = os.getenv('USER') or os.getenv('USERNAME')
	config['data']['root_dir'] = config['data']['root_dir'].replace('$USER', user)
	config['eval']['output_dir'] = config['eval']['output_dir'].replace('$USER', user)
	config['trainer']['default_root_dir'] = config['trainer']['default_root_dir'].replace('$USER', user)
	output_dir = config['eval']['output_dir']
	os.makedirs(output_dir, exist_ok=True)
	run_output_dir = os.path.join(output_dir, run_name)
	os.makedirs(run_output_dir)
	config['eval']['output_dir'] = os.path.join(run_output_dir, 'eval')
	os.makedirs(config['eval']['output_dir'])
	config['trainer']['default_root_dir'] = os.path.join(run_output_dir, 'trainer')
	os.makedirs(config['trainer']['default_root_dir'])
	# keep the resolved config with the run, so that --resume can continue it
	with open(os.path.join(run_output_dir, 'config.yaml'), 'w') as f:
		yaml.safe_dump(config, f, sort_keys=False)
	return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path', default='config.yaml')
    parser.add_argument('--resume', default=None,
                        help='run folder (<eval.output_dir>/<run_name>) to continue from its last.ckpt')
    args = parser.parse_args()

    if args.resume:
        run_output_dir = os.path.abspath(args.resume.rstrip('/'))
        run_name = os.path.basename(run_output_dir)
        config = yaml.safe_load(open(os.path.join(run_output_dir, 'config.yaml'), 'r'))
        ckpt_path = os.path.join(config['trainer']['default_root_dir'], run_name, 'checkpoints', 'last.ckpt')
        assert os.path.exists(ckpt_path), f'no checkpoint to resume from: {ckpt_path}'
    else:
        config = yaml.safe_load(open(args.config_path, 'r'))
        timestamp = datetime.now().strftime('%m%d-%H%M')
        run_name = f'G{config["group_id"]}_{timestamp}_{config["name"]}_{str(uuid.uuid4())[:5]}'
        config = adapt_config(config, run_name)
        ckpt_path = None

    train(config, run_name, ckpt_path)
