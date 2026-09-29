# Training system

How to configure, run and reproduce a training run, and how the pieces fit together.
Everything here was checked by the tests in `tests/` (CPU, tiny synthetic dataset) unless marked **not verified**.

## 1. Big picture

```
config.yaml ──► source/utils/config.py ──► cfg (flat namespace)
                                              │
source/scripts/train.py ──────────────────────┤
   • seeds, run directory, run_info.json      │
   • loggers (CSV + W&B), checkpoint callback │
   • pl.Trainer.fit / test                    ▼
                                  source/experiments/experiment_multitask.py
                                  ExperimentMultiTask (LightningModule)
                                     │  data, losses, metrics, optimizer,
                                     │  train/val/test steps, W&B images
                                     ▼
                                  source/models/*   nn.Module: rgb -> {task: tensor}
```

| Folder | Role |
| --- | --- |
| `source/models/` | The networks (`nn.Module`): image in, `{task: prediction}` out. Nothing about training. |
| `source/experiments/` | The training recipe (`LightningModule`) wrapped around **any** model: datasets and transforms, losses and their weighted sum, metrics, optimizer and LR schedule, train/val/test steps, prediction images. |
| `source/losses/` | `CrossEntropyLoss`, `MaskedDepthRegressionLoss` (L1/L2), `SILogLoss` (still a stub). |
| `source/utils/` | `config.py` (YAML loader), `metrics.py` (mIoU, depth metrics), `helpers.py` (model / dataset / optimizer registries), transforms, visualization. |
| `source/scripts/train.py` | The single entry point for every experiment. |
| `config.yaml` | The template of an experiment: **all** training parameters live here. |
| `tests/` | pytest suite (config, shapes, losses, end-to-end smoke tests). |

There is one experiment class for all cases. What changes between experiments is only the config:

| Experiment | `experiment.tasks` | `model.model_name` |
| --- | --- | --- |
| 0. Single-task baseline (segmentation) | `[semseg]` | `deeplabv3p` |
| 0. Single-task baseline (depth) | `[depth]` | `deeplabv3p` |
| 1. Joint (shared encoder/ASPP/decoder, n+1 channels) | `[semseg, depth]` | `deeplabv3p` |
| 2. Branched (shared encoder, per-task ASPP + decoder) | `[semseg, depth]` | `deeplabv3p_multitask` |
| 3. Depth as classification with adaptive bins | `[depth]` | `adaptive_depth` (**model still a stub**) |

Unsupported combinations are rejected at start-up (e.g. `adaptive_depth` with `[semseg, depth]`).

## 2. Running

Set the two environment variables the default config reads (or write literal paths in the YAML):

```bash
export DATASET_ROOT=/scratch.hpc/matteo.preda/industry/synscapes/synscapes_processed   # contains train/ val/ test/
export SAVEDIR=/scratch.hpc/matteo.preda/industry/runs                                  # runs are created here
```

Run the experiment described by a config file:

```bash
python -m source.scripts.train --config config.yaml
```

Override single values without editing the file (`section.key=value`, value parsed as YAML; a bare `key=value`
works if the key name is unique):

```bash
python -m source.scripts.train --config config.yaml experiment.tasks=[semseg] experiment.name=baseline_semseg
python -m source.scripts.train --config config.yaml optimization.optimizer_lr=0.001 experiment.seed=1
```

**One experiment = one versioned config file.** For a real experiment copy `config.yaml` to
`configs/<experiment>.yaml`, edit it and commit it; overrides are for sweeps and quick tests.
Every key is required (no hidden defaults); unknown keys, wrong types, invalid choices, unset environment
variables and unsupported model/task combinations stop the run before anything is created.

Local debugging on CPU without W&B and without the dataset (the tests generate a tiny fake one):

```bash
python -m pytest -q
```

Local debugging on a real dataset on CPU:

```bash
python -m source.scripts.train --config config.yaml trainer.accelerator=cpu wandb.mode=disabled \
    data.workers=0 trainer.limit_train_batches=2 trainer.limit_val_batches=2 trainer.limit_test_batches=2 \
    optimization.num_epochs=1
```

Cluster (SLURM): submit [train.sh](../train.sh) from the repository root. It sets `DATASET_ROOT`, `SAVEDIR` and the
virtualenv, uses one worker per allocated CPU and runs `config.yaml`; arguments are forwarded as overrides:

```bash
sbatch train.sh                              # config.yaml as is
sbatch train.sh experiment.tasks=[semseg]    # e.g. the segmentation-only baseline
sbatch train.sh wandb.mode=offline           # compute nodes without internet, see section 5
```

**Long or expensive runs must be agreed on first** (project rule): propose the command, duration and resources.

## 3. Configuration reference

`SCHEMA` in `source/utils/config.py` is the authoritative list of keys and types; `config.yaml` documents each
one in comments. Sections are flattened into one namespace (`cfg.optimizer_lr`); keys of `trainer` and `wandb` get
the section as prefix (`cfg.trainer_accelerator`, `cfg.wandb_mode`).

| Section | Keys |
| --- | --- |
| `experiment` | `name`, `seed`, `tasks`, `resume` |
| `paths` | `dataset_root` (input), `output_dir` (output) |
| `data` | `dataset`, `workers`, `workers_validation`, `batch_size_validation` |
| `model` | `model_name`, `model_encoder_name`, `pretrained`, adaptive-bins keys (`num_bins`, `num_heads`, `expansion`, `num_transformer_layers`) |
| `optimization` | `num_epochs`, `batch_size`, `optimizer`, `optimizer_lr`, `optimizer_momentum`, `optimizer_weight_decay`, `optimizer_float_16`, `lr_scheduler`, `lr_scheduler_power` |
| `loss` | `loss_weight_semseg`, `loss_weight_depth`, `loss_weight_aux`, `depth_loss` (`l1`/`l2`), `normalize_depth_loss` |
| `augmentation` | `aug_input_crop_size`, `aug_geom_*` |
| `trainer` | `accelerator`, `devices`, `log_every_n_steps`, `num_sanity_val_steps`, `limit_*_batches` (debug), `checkpoint_monitor`, `checkpoint_mode`, `save_last_checkpoint`, `test_after_fit` |
| `wandb` | `mode` (`online`/`offline`/`disabled`), `project`, `entity`, `group`, `tags`, `notes`, `key_file` |
| `visualization` | `num_steps_visualization_*`, `visualize_*`, `observe_train_ids`, `observe_valid_ids` |

Details worth knowing:

* Environment variables `$VAR` / `${VAR}` in any string are expanded; an unset variable is an error, not an empty string.
* PyYAML reads `1e-3` as a string; float keys accept it anyway (converted), so `optimizer_lr: 1e-3` works.
* The loss weights apply to single-task runs as well (template behaviour): a `semseg`-only run with weight 0.5 optimizes `0.5 * CE`.

## 4. What a run produces

Each run creates `<output_dir>/<experiment.name>_seed<seed>_<YYYYmmdd-HHMMSS>/`. Creation fails if the directory
exists and nothing outside it is ever deleted, so **previous runs are never overwritten** (the old `train.py`
wiped its log directory).

```
<run_dir>/
  config.yaml            resolved config (env vars expanded, overrides applied); re-run with --config <this file>
  run_info.json          seed, git commit, git dirty flag, host, python/torch versions, GPU name, timestamp
  git_diff.patch         only if the working tree was dirty: `git diff HEAD` at start (code that is not in the commit)
  checkpoints/           best-epochNN.ckpt (best by trainer.checkpoint_monitor) and last.ckpt (if save_last_checkpoint)
  csv/version_0/         metrics.csv (every logged scalar) and hparams.yaml
  wandb/                 W&B files (when wandb.mode is online/offline)
  final_metrics.json     monitor name, best checkpoint path and score, last-epoch validation metrics
  predictions/<task>/    test-split predictions as PNG (if trainer.test_after_fit)
```

Reproducibility: seed (`seed_everything(..., workers=True)`, also dataloader workers), config, commit and the
dirty-tree patch are stored together with the checkpoints. Runs on different hardware are not bit-identical
(no deterministic-algorithms flag is set).

Resume: set `experiment.resume=<path to .ckpt>` (use `last.ckpt` to continue). The resumed run writes to a **new** run
directory. To evaluate or continue from a checkpoint in code:
`ExperimentMultiTask.load_from_checkpoint(path, cfg=cfg, weights_only=False)` with `cfg` built from the run's `config.yaml`
(see `tests/test_training.py`).

## 5. Weights & Biases

* `wandb.mode=online` needs a key, and it must never be committed. On the cluster run `wandb login` once (with the
  virtualenv's `wandb`) on the login node: it stores the key in `~/.netrc` (mode 600, in your home, outside the repository),
  which every job reads without any change to the scripts. Alternatives: export `WANDB_API_KEY` in the job (e.g. read from a
  `chmod 600` file in your home), or a non-empty `wandb.key_file` (default `wandb.key`, in `.gitignore`; prefer not to use it).
  The key is never printed or stored in the run directory.
* `offline` writes everything under `<run_dir>/wandb` and can be uploaded later with `wandb sync`
  (use it on cluster nodes without internet). `disabled` creates no W&B logger (CSV only), used for local tests.
* Run identity: W&B run name = `<experiment.name>_seed<seed>_<timestamp>`, plus `wandb.project`, `entity`, `group`,
  `tags`, `notes`. Use `group` to bring related runs together (e.g. the seeds of one variant).
* Logged: every config key as hyperparameter (flat, so you can filter the W&B table by e.g. `model_name`), `run_info`
  (commit, dirty flag, host), and the metrics below. Images and histograms are logged only when a W&B logger exists.

Logged metrics:

| Key | Meaning |
| --- | --- |
| `loss_train/<task>`, `loss_train/aux_<name>`, `loss_train/total` | per step; per-task losses always separate, `total` is the weighted sum |
| `loss_val/<task>`, `loss_val/total` | per epoch on the validation split (same weighting as training) |
| `metrics_task_semseg/mean_iou`, `metrics_task_semseg/<class name>` | mIoU and per-class IoU from the confusion matrix accumulated over the whole validation set |
| `metrics_task_depth/*` | `si_log_rmse`, `log_rmse`, `mae`, `rmse`, `rel`, `delta1..3`, ... |
| `metrics_summary/{semseg,depth,total}` | internal checkpoint-selection convenience, `max(mIoU-50, 0)`, `max(50-SILogRMSE, 0)` and their mean over the active tasks — not a reported project metric, see section 6 |
| `trainer/LR` | current learning rate |
| `imgs_train/batch_crops`, `imgs_val/observed_samples`, `histograms/*` | prediction images (RGB, GT, prediction per task) and depth histograms |

## 6. The experiment class and the model contract

`ExperimentMultiTask(cfg)` works for every model that follows this contract:

1. `Model(cfg, outputs_desc)` where `outputs_desc = {task: channels}` (e.g. `{semseg: 19, depth: 1}`). Tasks are always
   ordered semseg first, depth last, which defines the channel layout of the joint model (n class logits, then 1 depth).
2. `model(rgb)` returns a dict with one entry per task: a tensor, or a list of tensors for deep supervision (the last
   one is the final prediction). Extra keys are allowed and ignored by the experiment.
3. *Optional auxiliary losses:* `model.compute_aux_losses(outputs, batch) -> {name: scalar}`. They are summed, multiplied
   by `loss.loss_weight_aux`, added to the total loss and logged as `loss_*/aux_<name>`. This is the hook for the bin-center
   loss of the adaptive-bins model (return the bin centers as an extra key of `outputs`).

To add a model: implement it in `source/models/`, register it in `MODELS` (`source/utils/helpers.py`), add its name to
the `model_name` choices and its supported task sets to `MODEL_TASKS` in `source/utils/config.py`.

Loss and evaluation as implemented:

* Semantic segmentation: cross-entropy with `ignore_index=255`; mIoU from the accumulated confusion matrix, void ignored.
* Depth: `MaskedDepthRegressionLoss` on metric depth (the joint model outputs `exp(x)` clamped to [0.1, 300] m), L1 or L2
  (`loss.depth_loss`), over pixels where the ground truth is finite and > 0 (0 marks sky / out of range). If
  `loss.normalize_depth_loss` is true (the default), the residual is divided by the dataset's fixed
  `depth_meters_stddev` (29.1264 m) before the loss is computed. Without this, raw L1-in-meters (O(1-10) m) dwarfs
  cross-entropy (O(0.1-2) nats) under nominally equal loss weights, so depth dominates the total gradient and hurts
  segmentation ("negative transfer") even though the two weights are equal — observed on the real dataset: a joint
  run without normalization dropped from 83.6 to 65.8 mIoU relative to the segmentation baseline. This only rescales
  the training loss; `metrics_task_depth/*` are always reported in meters, unaffected.
* Total: `loss_weight_semseg * CE + loss_weight_depth * depth_loss (+ loss_weight_aux * aux)`. Weighted sum by default; other
  weighting schemes (e.g. learned/uncertainty weighting) are a later ablation, not the default.
* Checkpoint selection: `trainer.checkpoint_monitor`. `metrics_summary/total` (for joint/branched runs) is an internal
  convenience so `ModelCheckpoint` has one scalar to compare when mIoU and SI-logRMSE are on different scales and
  directions — it is not one of this project's reported metrics (those are per-task mIoU and SI-logRMSE, see
  `doc/descrizione progetto .txt`). It is exactly 0 until a task passes 50 mIoU, so in short or early runs every epoch
  ties and the first checkpoint is kept; for such runs, or for single-task runs, monitor `metrics_task_semseg/mean_iou`
  (max) or `metrics_task_depth/si_log_rmse` (min) directly instead.
* Test split: only predictions are written (no metrics), so the test set is never used for tuning.

## 7. Tests

`python -m pytest -q` (CPU only, about one minute; needs about 1 GB of temporary disk space for checkpoints). It covers:
config loading and rejection of invalid configs, output shapes of the joint model (19 + 1 channels), the depth loss against a
numpy reference, and end-to-end runs on a synthetic dataset in the Miniscapes layout: joint / semseg-only / depth-only runs,
run-directory contents, a loss-decreases smoke test on 8 images, the auxiliary-loss hook, and offline W&B logging including
images and histograms.

## 8. Changes with respect to the original template, and known gaps

Changes made when building this system:

* Ported to PyTorch Lightning 2.x (`on_validation_epoch_end`, `Trainer(accelerator=..., precision='16-mixed')`, `CSVLogger`
  instead of `TestTubeLogger`, `ckpt_path`, W&B images through `WandbLogger.log_image`); manual `.cuda()` calls removed.
* **`experiment_semseg.py`, `experiment_depth.py` and `experiment_semseg_with_depth.py` were merged into
  `experiment_multitask.py` and deleted** (they are in the git history). They were near copies of each other.
* Validation `loss_val/total` is now the same weighted sum as `loss_train/total` (it used to be the unweighted sum).
* `train.py` no longer wipes the log directory, no longer builds a course-grader submission archive
  (`pack_submission`; `source/utils/rules.py` is now unused, and `check_all_rules`, which the old `train.py` imported, was
  removed from it in the working tree during this work, not by me). Its check that the output directory is outside the
  repository now lives in `create_run_dir`, and no longer reads argparse flags; `source/utils/config.py` is now the YAML loader.
* `torchvision` 0.29 removed the private `resnet._resnet`: the encoder builds `ResNet` directly and loads ImageNet weights via
  the `*_Weights` enum (the pretrained path is **not verified**, it needs a download).
* `visualization.py` used `np.math` and `matplotlib.cm.get_cmap`, removed in numpy 2 / matplotlib 3.9.
* `DecoderDeeplabV3p` implemented (48-channel skip projection, concatenation, two 3x3 convs) as in DeepLabV3+.
* `ModelDeepLabV3PlusMultiTask` (branched) implemented: one `ASPP` + one `DecoderDeeplabV3p` per task (`nn.ModuleDict`s
  keyed by task name), sharing only the encoder. Its encoder uses `replace_stride_with_dilation=(False, False, False)`,
  same as the joint model (the template had it at `(False, False, True)`, a different output stride, which would have
  made the branched-vs-joint comparison also a dilation comparison — changed on request).
* `loss.normalize_depth_loss`: divides the depth residual by the dataset's `depth_meters_stddev` (29.1264 m) before
  computing L1/L2, so the depth loss is on a comparable scale to cross-entropy under equal loss weights. Without it, an
  unweighted joint run measurably lost segmentation performance (83.6 → 65.8 mIoU) to the depth task dominating the
  shared gradient ("negative transfer"); default `true`. Metrics are unaffected, always in meters.
* Config validation now rejects a `trainer.checkpoint_monitor` whose task isn't in `experiment.tasks`, and a
  `trainer.checkpoint_mode` that doesn't match the monitored metric's known direction (e.g. `si_log_rmse` needs `min`) —
  both mistakes silently kept the *worst* checkpoint instead of the best before this check existed.

Known gaps (not addressed here):

* `ModelAdaptiveDepth` (the bin logic) and `SelfAttention` are still template stubs; `SILogLoss` is a stub, so the depth
  loss is L1/L2 for now.
* The reported SI-logRMSE in `source/utils/metrics.py` hardcodes lambda = 1 and a x100 scale, and its valid mask has no
  maximum depth; CLAUDE.md asks for lambda as an explicit parameter and a `<= max depth` mask.
* The depth head is `exp(x)` clamped to [0.1, 300] m and is trained with L1 in meters: far pixels dominate the loss and the
  first steps are slow because the untrained output is about 1 m. A loss in log space would be an alternative.
* Single device only (metrics are accumulated per process and are not synchronized across GPUs).
* The GPU path (`accelerator=gpu`, `optimizer_float_16`) and the online W&B mode were **not verified**: only CPU and offline W&B
  were run. Nothing was trained on the real dataset.
* `wandb.key` is tracked by git (currently an empty file). `.gitignore` now lists it, but a tracked file stays tracked until
  `git rm --cached wandb.key`.
