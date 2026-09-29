"""
Read-only sanity check of the Miniscapes dataset: is it consistent, is the ground truth aligned with the
images, and can it be fed to a training loop?

What it checks (per split):
  1. Layout: files present per modality, id ranges, id contiguity, missing/extra files, expected split sizes.
  2. Per-sample content: image modes and sizes, semseg label set, depth decoding, exact duplicate RGB files.
  3. Ground-truth alignment, three independent tests:
       - overlays (RGB / semseg / depth) saved as images for visual inspection;
       - edge alignment: semseg and depth boundaries should sit on RGB image edges, i.e. the edge-strength
         score along the boundary must peak at zero shift;
       - semantic vs geometric consistency: sky must have invalid depth, road depth must grow with height.
  4. Loader readiness: the real DatasetMiniscapes + the real training / validation transforms + DataLoader,
     checking dtypes, shapes, label range, depth validity and a cross-entropy pass on the collated batch.

Usage (from the repo root):
    python -m source.scripts.verify_dataset --dataset_root <root> [--full] [--out_dir <dir>]

The dataset is never modified. Outputs (overlay images and report.json) go to --out_dir.
"""
import argparse
import hashlib
import json
import os
import random
from collections import defaultdict

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from source.datasets.dataset_miniscapes import DatasetMiniscapes
from source.datasets.definitions import MOD_ID, MOD_RGB, MOD_SEMSEG, MOD_DEPTH, SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST
from source.utils.transforms import get_transforms

SPLITS = (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST)
EXPECTED_SIZES = {SPLIT_TRAIN: 20000, SPLIT_VALID: 2500, SPLIT_TEST: 2500}
EXTENSIONS = {MOD_RGB: '.jpg', MOD_SEMSEG: '.png', MOD_DEPTH: '.png'}
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


class Report:
    def __init__(self):
        self.items = []

    def add(self, level, message):
        self.items.append((level, message))
        print(f'[{level}] {message}')

    def count(self, level):
        return sum(1 for lvl, _ in self.items if lvl == level)


class IndexedMiniscapes(DatasetMiniscapes):
    """DatasetMiniscapes restricted to the ids that exist on disk (the base class hardcodes the split sizes)."""

    def __init__(self, dataset_root, split, indices):
        super().__init__(dataset_root, split)
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        return self.get(self.indices[position])


class SplitStats:
    def __init__(self):
        self.n_samples = 0
        self.sizes = defaultdict(int)
        self.class_pixels = np.zeros(256, np.int64)
        self.depth_raw_hist = np.zeros(256, np.int64)
        self.invalid_depth_by_class = np.zeros(256, np.int64)
        self.sky_pixels = 0
        self.sky_pixels_invalid_depth = 0
        self.problems = []


# ----------------------------------------------------------------------------------------------------------------------
# 1. Layout
# ----------------------------------------------------------------------------------------------------------------------

def discover_split(root, split):
    ids, unexpected, dir_exists = {}, {}, {}
    for modality, ext in EXTENSIONS.items():
        folder = os.path.join(root, split, modality)
        ids[modality], unexpected[modality] = set(), []
        dir_exists[modality] = os.path.isdir(folder)
        if not dir_exists[modality]:
            continue
        for name in os.listdir(folder):
            stem, suffix = os.path.splitext(name)
            if suffix == ext and stem.isdigit():
                ids[modality].add(int(stem))
            else:
                unexpected[modality].append(name)
    return ids, unexpected, dir_exists


def check_layout(report, split, ids, unexpected, dir_exists, full):
    """Returns (usable_ids, has_gt) for the split and logs every layout problem."""
    rgb_ids = ids[MOD_RGB]
    if not rgb_ids:
        report.add('FAIL', f'{split}: no RGB images found in {split}/{MOD_RGB}')
        return [], False

    has_gt = dir_exists[MOD_SEMSEG] and dir_exists[MOD_DEPTH] and bool(ids[MOD_SEMSEG]) and bool(ids[MOD_DEPTH])
    if not has_gt:
        level = 'WARN' if split == SPLIT_TEST else 'FAIL'
        report.add(level, f'{split}: no semseg/depth ground truth found ({len(ids[MOD_SEMSEG])} semseg, '
                          f'{len(ids[MOD_DEPTH])} depth files)')

    for modality, names in unexpected.items():
        if names:
            report.add('WARN', f'{split}/{modality}: {len(names)} unexpected files, e.g. {sorted(names)[:3]}')

    if has_gt:
        usable = sorted(rgb_ids & ids[MOD_SEMSEG] & ids[MOD_DEPTH])
        for modality in (MOD_SEMSEG, MOD_DEPTH):
            missing, extra = sorted(rgb_ids - ids[modality]), sorted(ids[modality] - rgb_ids)
            if missing:
                report.add('FAIL', f'{split}: {len(missing)} RGB ids have no {modality} file, e.g. {missing[:5]}')
            if extra:
                report.add('FAIL', f'{split}: {len(extra)} {modality} ids have no RGB file, e.g. {extra[:5]}')
    else:
        usable = sorted(rgb_ids)

    n = len(rgb_ids)
    if rgb_ids != set(range(n)):
        report.add('WARN', f'{split}: RGB ids are not contiguous 0..{n - 1} (min={min(rgb_ids)}, max={max(rgb_ids)}); '
                           f'DatasetMiniscapes indexes by integer id, so it only works if ids are 0..N-1')
    expected = EXPECTED_SIZES[split]
    if n != expected:
        report.add('FAIL' if full else 'WARN',
                   f'{split}: found {n} samples, DatasetMiniscapes.__len__ hardcodes {expected}'
                   f'{"" if full else " (expected for a sample; use --full on the complete dataset)"}')
    return usable, has_gt


# ----------------------------------------------------------------------------------------------------------------------
# 2. Per-sample content
# ----------------------------------------------------------------------------------------------------------------------

def scan_sample(ds, split, idx, has_gt, stats, sky_id, hashes):
    def problem(message):
        stats.problems.append(f'{split}/{idx}: {message}')

    rgb_path = ds.get_item_path(idx, MOD_RGB)
    rgb = Image.open(rgb_path)
    rgb.load()
    if rgb.mode != 'RGB':
        problem(f'RGB mode is {rgb.mode}, expected RGB')
    stats.sizes[rgb.size] += 1
    stats.n_samples += 1
    with open(rgb_path, 'rb') as f:
        digest = hashlib.md5(f.read()).hexdigest()
    if digest in hashes:
        problem(f'RGB file is byte-identical to {hashes[digest]}')
    else:
        hashes[digest] = f'{split}/{idx}'
    if not has_gt:
        return

    sem = Image.open(ds.get_item_path(idx, MOD_SEMSEG))
    depth = Image.open(ds.get_item_path(idx, MOD_DEPTH))
    if sem.mode not in ('P', 'L'):
        problem(f'semseg mode is {sem.mode}, expected P or L')
    if depth.mode != 'L':
        problem(f'depth mode is {depth.mode}, expected L (8-bit)')
    if sem.size != rgb.size or depth.size != rgb.size:
        problem(f'size mismatch rgb={rgb.size} semseg={sem.size} depth={depth.size}')
        return

    sem_np, raw = np.array(sem), np.array(depth)
    class_hist = np.bincount(sem_np.ravel(), minlength=256)
    stats.class_pixels += class_hist
    bad_labels = [int(v) for v in np.nonzero(class_hist)[0] if v >= ds.semseg_num_classes and v != ds.semseg_ignore_label]
    if bad_labels:
        problem(f'semseg contains labels outside 0..{ds.semseg_num_classes - 1} and {ds.semseg_ignore_label}: {bad_labels}')

    stats.depth_raw_hist += np.bincount(raw.ravel(), minlength=256)
    invalid = raw == 0
    stats.invalid_depth_by_class += np.bincount(sem_np[invalid].ravel(), minlength=256)
    is_sky = sem_np == sky_id
    stats.sky_pixels += int(is_sky.sum())
    stats.sky_pixels_invalid_depth += int((is_sky & invalid).sum())


def depth_summary(hist, table):
    total = int(hist.sum())
    valid = hist.astype(np.float64).copy()
    valid[0] = 0
    n_valid = valid.sum()
    if total == 0 or n_valid == 0:
        return {'invalid_fraction': 1.0}
    meters = table.astype(np.float64)
    mean = float((valid * meters).sum() / n_valid)
    std = float(np.sqrt((valid * (meters - mean) ** 2).sum() / n_valid))
    full = hist.astype(np.float64)
    mean_all = float((full * meters).sum() / total)
    std_all = float(np.sqrt((full * (meters - mean_all) ** 2).sum() / total))
    order = np.argsort(meters[1:]) + 1
    cum = np.cumsum(valid[order]) / n_valid
    pct = {f'p{p}': float(meters[order][min(np.searchsorted(cum, p / 100), len(order) - 1)]) for p in PERCENTILES}
    present = np.nonzero(valid)[0]
    return {
        'invalid_fraction': float(hist[0] / total),
        'valid_mean_m': mean, 'valid_std_m': std,
        'mean_incl_invalid_zeros_m': mean_all, 'std_incl_invalid_zeros_m': std_all,
        'valid_min_m': float(meters[present].min()), 'valid_max_m': float(meters[present].max()),
        'fraction_at_nearest_level': float(hist[255] / total), 'fraction_at_farthest_level': float(hist[1] / total),
        'percentiles_m': pct,
    }


# ----------------------------------------------------------------------------------------------------------------------
# 3. Ground-truth alignment
# ----------------------------------------------------------------------------------------------------------------------

def _pair_mask(diff_h, diff_v, shape):
    mask = np.zeros(shape, bool)
    mask[:, 1:] |= diff_h
    mask[:, :-1] |= diff_h
    mask[1:, :] |= diff_v
    mask[:-1, :] |= diff_v
    return mask


def semseg_boundaries(sem, ignore_label):
    valid = sem != ignore_label
    dh = (sem[:, 1:] != sem[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    dv = (sem[1:, :] != sem[:-1, :]) & valid[1:, :] & valid[:-1, :]
    return _pair_mask(dh, dv, sem.shape)


def depth_boundaries(meters, log_threshold):
    valid = meters > 0
    log_depth = np.log(np.where(valid, meters, 1.0))
    dh = (np.abs(log_depth[:, 1:] - log_depth[:, :-1]) > log_threshold) & valid[:, 1:] & valid[:, :-1]
    dv = (np.abs(log_depth[1:, :] - log_depth[:-1, :]) > log_threshold) & valid[1:, :] & valid[:-1, :]
    return _pair_mask(dh, dv, meters.shape)


def shift_scores(edge_strength, boundary, radius, rng, max_points=20000):
    """Mean RGB edge strength at the boundary points displaced by every (dy, dx) in [-radius, radius]^2."""
    ys, xs = np.nonzero(boundary)
    if len(ys) < 200:
        return None
    if len(ys) > max_points:
        keep = rng.choice(len(ys), max_points, replace=False)
        ys, xs = ys[keep], xs[keep]
    h, w = edge_strength.shape
    scores = np.zeros((2 * radius + 1, 2 * radius + 1))
    for i, dy in enumerate(range(-radius, radius + 1)):
        yy = np.clip(ys + dy, 0, h - 1)
        for j, dx in enumerate(range(-radius, radius + 1)):
            scores[i, j] = edge_strength[yy, np.clip(xs + dx, 0, w - 1)].mean()
    return scores


def summarize_shift(scores, radius):
    i, j = np.unravel_index(np.argmax(scores), scores.shape)
    yy, xx = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), indexing='ij')
    ring = np.maximum(np.abs(yy), np.abs(xx)) == radius
    return {'best_dy': int(i - radius), 'best_dx': int(j - radius),
            'contrast': float(scores[radius, radius] / scores[ring].mean())}


def alignment_tests(ds, split, idx, args, rng, table):
    rgb = Image.open(ds.get_item_path(idx, MOD_RGB)).convert('RGB')
    gray = np.asarray(rgb.convert('L'), dtype=np.float32)
    gy, gx = np.gradient(gray)
    edge_strength = np.hypot(gx, gy)
    sem = np.array(Image.open(ds.get_item_path(idx, MOD_SEMSEG)))
    raw = np.array(Image.open(ds.get_item_path(idx, MOD_DEPTH)))
    meters = table[raw]

    out = {}
    scores = shift_scores(edge_strength, semseg_boundaries(sem, ds.semseg_ignore_label), args.shift_radius, rng)
    if scores is not None:
        out['semseg'] = summarize_shift(scores, args.shift_radius)
    scores = shift_scores(edge_strength, depth_boundaries(meters, args.depth_edge_log_threshold), args.shift_radius, rng)
    if scores is not None:
        out['depth'] = summarize_shift(scores, args.shift_radius)

    names = ds.semseg_class_names
    road_id = names.index('road')
    road = (sem == road_id) & (raw > 0)
    if road.sum() > 500:
        ys, _ = np.nonzero(road)
        out['road_row_vs_logdepth_corr'] = float(np.corrcoef(ys, np.log(meters[road]))[0, 1])
    return out


def aggregate_alignment(report, split, per_sample):
    for key in ('semseg', 'depth'):
        entries = [s[key] for s in per_sample if key in s]
        if not entries:
            report.add('WARN', f'{split}: not enough {key} boundary pixels for the edge-alignment test')
            continue
        aligned = np.mean([abs(e['best_dy']) <= 1 and abs(e['best_dx']) <= 1 for e in entries])
        contrast = np.mean([e['contrast'] for e in entries])
        level = 'PASS' if aligned >= 0.9 and contrast > 1.0 else 'FAIL'
        report.add(level, f'{split}: {key} boundaries vs RGB edges: peak within +-1 px of zero shift in '
                          f'{aligned * 100:.0f}% of {len(entries)} samples, mean edge contrast (zero shift / far shift) '
                          f'{contrast:.2f} (aligned GT gives >1 and a zero-shift peak)')
    corrs = [s['road_row_vs_logdepth_corr'] for s in per_sample if 'road_row_vs_logdepth_corr' in s]
    if corrs:
        mean_corr = float(np.mean(corrs))
        report.add('PASS' if mean_corr < -0.5 else 'FAIL',
                   f'{split}: road pixels, correlation(row, log depth) = {mean_corr:.2f} over {len(corrs)} samples '
                   f'(strongly negative expected: lower in the image means closer)')


def render_overlay(ds, split, idx, out_path, table, tile_width, log_threshold):
    rgb = np.asarray(Image.open(ds.get_item_path(idx, MOD_RGB)).convert('RGB'))
    sem = np.array(Image.open(ds.get_item_path(idx, MOD_SEMSEG)))
    raw = np.array(Image.open(ds.get_item_path(idx, MOD_DEPTH)))
    meters = table[raw]

    palette = np.zeros((256, 3), np.uint8)
    palette[:ds.semseg_num_classes] = np.array(ds.semseg_class_colors, dtype=np.uint8)
    sem_color = palette[sem]
    blend = (0.5 * rgb + 0.5 * sem_color).astype(np.uint8)

    def thick(mask):
        out = mask.copy()
        out[1:, :] |= mask[:-1, :]
        out[:, 1:] |= mask[:, :-1]
        return out

    sem_edges = rgb.copy()
    sem_edges[thick(semseg_boundaries(sem, ds.semseg_ignore_label))] = (255, 0, 0)
    depth_edges = rgb.copy()
    depth_edges[thick(depth_boundaries(meters, log_threshold))] = (255, 255, 0)

    lo, hi = np.log(ds.depth_meters_min), np.log(ds.depth_meters_max)
    t = np.clip((np.log(np.where(meters > 0, meters, 1.0)) - lo) / (hi - lo), 0, 1)
    depth_color = (matplotlib.colormaps['plasma_r'](t)[..., :3] * 255).astype(np.uint8)
    depth_color[raw == 0] = 0

    tiles = []
    for tile in (sem_edges, blend, depth_color, depth_edges):
        image = Image.fromarray(tile)
        tiles.append(image.resize((tile_width, round(image.height * tile_width / image.width)), Image.BILINEAR))
    canvas = Image.new('RGB', (tile_width * len(tiles), tiles[0].height))
    for k, tile in enumerate(tiles):
        canvas.paste(tile, (k * tile_width, 0))
    canvas.save(out_path)


# ----------------------------------------------------------------------------------------------------------------------
# 4. Loader readiness
# ----------------------------------------------------------------------------------------------------------------------

def check_batch(report, tag, batch, ds):
    n_cls, ignore = ds.semseg_num_classes, ds.semseg_ignore_label
    rgb, sem, depth = batch[MOD_RGB], batch[MOD_SEMSEG], batch[MOD_DEPTH]
    problems = []
    if not (rgb.dtype == torch.float32 and rgb.dim() == 4 and rgb.shape[1] == 3):
        problems.append(f'rgb must be float32 [B,3,H,W], got {rgb.dtype} {tuple(rgb.shape)}')
    elif not torch.isfinite(rgb).all():
        problems.append('rgb contains non-finite values')
    if not (sem.dtype == torch.long and sem.dim() == 4 and sem.shape[1] == 1):
        problems.append(f'semseg must be int64 [B,1,H,W], got {sem.dtype} {tuple(sem.shape)}')
    else:
        values = torch.unique(sem)
        bad = [int(v) for v in values if not (0 <= v < n_cls or v == ignore)]
        if bad:
            problems.append(f'semseg labels outside 0..{n_cls - 1}/{ignore}: {bad}')
    if not (depth.dtype == torch.float32 and depth.dim() == 4 and depth.shape[1] == 1):
        problems.append(f'depth must be float32 [B,1,H,W], got {depth.dtype} {tuple(depth.shape)}')
    if rgb.shape[-2:] != sem.shape[-2:] or rgb.shape[-2:] != depth.shape[-2:]:
        problems.append(f'spatial size mismatch rgb={tuple(rgb.shape)} semseg={tuple(sem.shape)} depth={tuple(depth.shape)}')

    if not problems:
        n_nan, n_inf = int(torch.isnan(depth).sum()), int(torch.isinf(depth).sum())
        n_neg = int((depth < 0).sum())
        valid = (depth > 0) & (depth <= ds.depth_meters_max)
        if n_inf or n_neg:
            problems.append(f'depth has {n_inf} inf and {n_neg} negative pixels')
        if valid.sum() == 0:
            problems.append('batch has no valid depth pixels')
        if n_nan:
            report.add('WARN', f'{tag}: depth target contains {n_nan} NaN pixels (warp fill). The depth loss and metric '
                               f'must mask with (d > 0), which is False for NaN')
        logits = torch.randn(sem.shape[0], n_cls, *sem.shape[-2:])
        loss = F.cross_entropy(logits, sem.squeeze(1), ignore_index=ignore)
        if not torch.isfinite(loss):
            problems.append('cross-entropy on the batch is not finite')
        report.add('INFO', f'{tag}: batch rgb{tuple(rgb.shape)} semseg{tuple(sem.shape)} depth{tuple(depth.shape)}, '
                           f'valid depth pixels {valid.float().mean().item() * 100:.1f}%, '
                           f'ignored semseg pixels {(sem == ignore).float().mean().item() * 100:.1f}%')
    for p in problems:
        report.add('FAIL', f'{tag}: {p}')
    return not problems


def check_loader(report, root, split, indices, args, train):
    reference = IndexedMiniscapes(root, split, indices)
    common = dict(
        semseg_ignore_label=reference.semseg_ignore_label,
        rgb_mean=reference.rgb_mean, rgb_stddev=reference.rgb_stddev,
        depth_meters_mean=reference.depth_meters_mean, depth_meters_stddev=reference.depth_meters_stddev,
    )
    if train:
        # Same call as ExperimentDepthSemseg.__init__ for the training split
        transforms = get_transforms(
            geom_scale_min=args.aug_geom_scale_min, geom_scale_max=args.aug_geom_scale_max,
            geom_tilt_max_deg=args.aug_geom_tilt_max_deg, geom_wiggle_max_ratio=args.aug_geom_wiggle_max_ratio,
            geom_reflect=args.aug_geom_reflect, crop_random=args.crop_size, **common)
        batch_size = args.batch_size
    else:
        transforms = get_transforms(crop_for_passable=32, **common)
        batch_size = args.batch_size_validation
    reference.set_transforms(transforms)
    tag = f'loader[{split}, {"train" if train else "val"} transforms' \
          f'{", stress augmentation" if train and args.aug_geom_scale_max != 1.0 else ""}]'
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(reference, batch_size=min(batch_size, len(reference)), shuffle=train, generator=generator,
                        num_workers=args.workers, drop_last=False)
    ok = True
    try:
        for b, batch in enumerate(loader):
            ok &= check_batch(report, tag, batch, reference)
            if b + 1 >= args.loader_batches:
                break
    except Exception as e:  # any exception here means the loop would crash too
        report.add('FAIL', f'{tag}: DataLoader raised {type(e).__name__}: {e}')
        return
    if ok:
        report.add('PASS', f'{tag}: {args.loader_batches} batches OK')


# ----------------------------------------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset_root', default=os.environ.get('DATASET_ROOT'), help='Folder containing train/ val/ test/')
    p.add_argument('--out_dir', default=None, help='Where overlays and report.json go (default: <root>/../dataset_check)')
    p.add_argument('--full', action='store_true', help='Complete dataset: split sizes must be exactly 20000/2500/2500')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_samples', type=int, default=0, help='Content-scan at most N samples per split (0 = all)')
    p.add_argument('--align_samples', type=int, default=50, help='Samples per split for the alignment tests')
    p.add_argument('--num_overlays', type=int, default=8, help='Overlay images per split')
    p.add_argument('--overlay_tile_width', type=int, default=640)
    p.add_argument('--shift_radius', type=int, default=6)
    p.add_argument('--depth_edge_log_threshold', type=float, default=0.2, help='|d log depth| defining a depth edge')
    p.add_argument('--crop_size', type=int, default=256, help='Training crop, as in config.yaml')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--batch_size_validation', type=int, default=8)
    p.add_argument('--loader_batches', type=int, default=3)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--aug_geom_scale_min', type=float, default=1.0)
    p.add_argument('--aug_geom_scale_max', type=float, default=1.0)
    p.add_argument('--aug_geom_tilt_max_deg', type=float, default=0.0)
    p.add_argument('--aug_geom_wiggle_max_ratio', type=float, default=0.0)
    p.add_argument('--aug_geom_reflect', action='store_true')
    args = p.parse_args()
    if not args.dataset_root:
        p.error('--dataset_root is required (or set DATASET_ROOT)')
    return args


def main():
    args = parse_args()
    root = os.path.abspath(os.path.expanduser(os.path.expandvars(args.dataset_root)))
    out_dir = os.path.abspath(args.out_dir or os.path.join(os.path.dirname(root), 'dataset_check'))
    os.makedirs(out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    report = Report()
    ds_ref = DatasetMiniscapes(root, SPLIT_TRAIN)
    sky_id = ds_ref.semseg_class_names.index('sky')
    table = ds_ref.depth_disparity_uint8_to_meters_float32(np.arange(256, dtype=np.uint8), False)
    if not np.isfinite(table).all():
        report.add('FAIL', 'depth decoding table contains non-finite values')
    report.add('INFO', f'dataset_root={root}')
    report.add('INFO', f'classes ({ds_ref.semseg_num_classes}): {ds_ref.semseg_class_names}, '
                       f'ignore label {ds_ref.semseg_ignore_label}; depth range {ds_ref.depth_meters_min}-'
                       f'{ds_ref.depth_meters_max} m')

    results, split_indices, hashes = {}, {}, {}
    for split in SPLITS:
        print(f'\n===== {split} =====')
        ids, unexpected, dir_exists = discover_split(root, split)
        usable, has_gt = check_layout(report, split, ids, unexpected, dir_exists, args.full)
        split_indices[split] = (usable, has_gt)
        results[split] = {
            'files': {m: len(ids[m]) for m in EXTENSIONS},
            'id_range': [min(ids[MOD_RGB]), max(ids[MOD_RGB])] if ids[MOD_RGB] else None,
            'usable_samples': len(usable), 'has_ground_truth': has_gt,
        }
        if not usable:
            continue

        scan_ids = usable if args.max_samples <= 0 else usable[:args.max_samples]
        ds = IndexedMiniscapes(root, split, usable)
        stats = SplitStats()
        for idx in tqdm(scan_ids, desc=f'scan {split}'):
            try:
                scan_sample(ds, split, idx, has_gt, stats, sky_id, hashes)
            except Exception as e:  # unreadable/corrupt file
                stats.problems.append(f'{split}/{idx}: could not be read ({type(e).__name__}: {e})')
        for message in stats.problems[:20]:
            report.add('FAIL', message)
        if len(stats.problems) > 20:
            report.add('FAIL', f'{split}: {len(stats.problems) - 20} more content problems not shown')
        if not stats.problems:
            report.add('PASS', f'{split}: {stats.n_samples} samples readable, modes/sizes/labels consistent, '
                               f'no duplicate RGB files')
        report.add('INFO', f'{split}: image sizes (W,H): {dict(stats.sizes)}')
        results[split]['image_sizes'] = {f'{w}x{h}': c for (w, h), c in stats.sizes.items()}

        if has_gt:
            total_px = stats.class_pixels.sum()
            freq = {ds_ref.semseg_class_names[c]: float(stats.class_pixels[c] / total_px) for c in range(ds_ref.semseg_num_classes)}
            freq['ignore(255)'] = float(stats.class_pixels[ds_ref.semseg_ignore_label] / total_px)
            results[split]['class_pixel_fraction'] = freq
            absent = [n for n, v in freq.items() if v == 0 and n != 'ignore(255)']
            if absent:
                report.add('WARN', f'{split}: classes with zero pixels in the scanned samples: {absent}')

            summary = depth_summary(stats.depth_raw_hist, table)
            results[split]['depth'] = summary
            report.add('INFO', f'{split}: depth invalid (0) {summary["invalid_fraction"] * 100:.1f}% of pixels; valid '
                               f'mean {summary.get("valid_mean_m", float("nan")):.2f} m, std '
                               f'{summary.get("valid_std_m", float("nan")):.2f} m, range '
                               f'[{summary.get("valid_min_m", float("nan")):.2f}, {summary.get("valid_max_m", float("nan")):.2f}] m; '
                               f'incl. invalid zeros: mean {summary.get("mean_incl_invalid_zeros_m", float("nan")):.2f} m, std '
                               f'{summary.get("std_incl_invalid_zeros_m", float("nan")):.2f} m '
                               f'(hardcoded in the dataset: {ds_ref.depth_meters_mean} / {ds_ref.depth_meters_stddev})')
            invalid_total = stats.invalid_depth_by_class.sum()
            if invalid_total:
                top = np.argsort(stats.invalid_depth_by_class)[::-1][:4]
                names = lambda c: 'ignore' if c == ds_ref.semseg_ignore_label else ds_ref.semseg_class_names[c]
                report.add('INFO', f'{split}: invalid depth pixels by class: ' + ', '.join(
                    f'{names(c)} {stats.invalid_depth_by_class[c] / invalid_total * 100:.0f}%' for c in top))
            if stats.sky_pixels:
                frac = stats.sky_pixels_invalid_depth / stats.sky_pixels
                report.add('PASS' if frac > 0.9 else 'WARN',
                           f'{split}: {frac * 100:.1f}% of sky pixels have invalid depth (sky should have no depth)')

            k = min(args.align_samples, len(usable))
            align_ids = sorted(random.Random(args.seed).sample(usable, k))
            per_sample = []
            for idx in tqdm(align_ids, desc=f'align {split}'):
                per_sample.append(alignment_tests(ds, split, idx, args, rng, table))
            aggregate_alignment(report, split, per_sample)
            results[split]['alignment_per_sample'] = dict(zip(map(str, align_ids), per_sample))

            overlay_ids = sorted(random.Random(args.seed + 1).sample(usable, min(args.num_overlays, len(usable))))
            overlay_dir = os.path.join(out_dir, 'overlays', split)
            os.makedirs(overlay_dir, exist_ok=True)
            for idx in overlay_ids:
                render_overlay(ds, split, idx, os.path.join(overlay_dir, f'{idx}.png'), table,
                               args.overlay_tile_width, args.depth_edge_log_threshold)
            report.add('INFO', f'{split}: {len(overlay_ids)} overlays written to {overlay_dir} '
                               f'(tiles: RGB+semseg edges | semseg blend | depth | RGB+depth edges)')

    print('\n===== loader readiness =====')
    for split, train in ((SPLIT_TRAIN, True), (SPLIT_TRAIN, False), (SPLIT_VALID, False)):
        indices, has_gt = split_indices.get(split, ([], False))
        if indices and has_gt:
            check_loader(report, root, split, indices, args, train)
    stress = argparse.Namespace(**vars(args))
    stress.aug_geom_scale_min, stress.aug_geom_scale_max = 0.75, 1.5
    stress.aug_geom_tilt_max_deg, stress.aug_geom_wiggle_max_ratio, stress.aug_geom_reflect = 10.0, 0.1, True
    indices, has_gt = split_indices.get(SPLIT_TRAIN, ([], False))
    if indices and has_gt:
        check_loader(report, root, SPLIT_TRAIN, indices, stress, True)

    print('\n===== summary =====')
    print(f'PASS {report.count("PASS")} | WARN {report.count("WARN")} | FAIL {report.count("FAIL")}')
    with open(os.path.join(out_dir, 'report.json'), 'w') as f:
        json.dump({'args': vars(args), 'results': results, 'log': report.items}, f, indent=2, default=float)
    print(f'report: {os.path.join(out_dir, "report.json")}')
    raise SystemExit(1 if report.count('FAIL') else 0)


if __name__ == '__main__':
    main()
