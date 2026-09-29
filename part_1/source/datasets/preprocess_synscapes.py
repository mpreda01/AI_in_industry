#!/usr/bin/env python3
"""
Offline preprocessing for the Synscapes dataset (https://synscapes.on.liu.se/):
converts it into the exact on-disk layout DatasetMiniscapes
(source/datasets/dataset_miniscapes.py) expects, so this repo can train on it.

What it does:
  1. Splits the 25000 Synscapes indices into train/val/test (20000/2500/2500,
     matching DatasetMiniscapes.__len__) with a seeded shuffle, and renumbers
     each split's images to 0..N-1 (DatasetMiniscapes indexes by contiguous int).
  2. RGB: img/<rgb-source>/[i].png -> <split>/rgb/[new_idx].jpg
  3. Semantic segmentation: img/class/[i].png (Cityscapes label ids) ->
     <split>/semseg/[new_idx].png (Cityscapes train ids 0-18, 255=ignore), via
     the LUT built from labels.py.
  4. Depth: img/depth/[i].exr (planar depth in meters, float32) ->
     <split>/depth/[new_idx].png, encoded by calling
     DatasetMiniscapes.depth_meters_float32_to_disparity_uint8 directly (8-bit
     quantized disparity, 4-300 m range, 0 = invalid), so the on-disk encoding
     is guaranteed to match the repo's own decoder (DatasetMiniscapes.load_depth)
     instead of re-deriving the formula here and risking drift.

What it does NOT do:
  - It never writes to --synscapes-root.
  - It does not invent the depth range or the per-split counts: both are read
    from / matched against DatasetMiniscapes itself (source/datasets/dataset_miniscapes.py),
    not hardcoded independently in this script.
  - It does not silently pick the RGB source resolution or the split seed: both
    are required CLI flags (--rgb-source, --split-seed), so the choice is explicit
    and gets logged in the manifest, per the project's "do not invent values" rule.
  - It does not support --rgb-source rgb-2k yet: Synscapes only ships class/ and
    depth/ at native 1440x720, so using the 2048x1024 rgb-2k images would need
    semseg/depth to be upscaled to match, and DatasetMiniscapes asserts
    semseg.size == rgb.size / depth.size == rgb.size. That resize strategy is a
    modeling decision (interpolation choice for labels and for pre-quantization
    depth), not just an I/O detail, so it is refused explicitly rather than done
    silently - ask if you need it.

Must run against the ORIGINAL Synscapes root (img/rgb/, img/class/, img/depth/),
not against a previous run of this script's --output-root: an older flat
"labels/ + depth/" output has no RGB at all, and if its depth was quantized with
a different range/encoding it cannot be losslessly reused - point --synscapes-root
at the original data again.

Typical usage:
  python3 preprocess_synscapes.py \
      --synscapes-root /path/to/synscapes \
      --output-root    /path/to/miniscapes \
      --rgb-source rgb \
      --split-seed 42 \
      --workers 8

  # Quick smoke test of the pipeline (still requires all 25000 raw images to be
  # present, since the split is always computed over the full set; only the
  # first N tasks are actually processed):
  python3 preprocess_synscapes.py --synscapes-root ... --output-root ... --rgb-source rgb --split-seed 42 --limit 200
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

# This script lives at the repo root, next to `source/`: make sure it can be
# imported even when run directly (`python3 preprocess_synscapes.py`) rather
# than as `python -m`.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from source.datasets.dataset_miniscapes import DatasetMiniscapes
from source.datasets.definitions import SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST

# Must match DatasetMiniscapes.__len__ exactly (source/datasets/dataset_miniscapes.py):
# the loader indexes each split 0..count-1, so any mismatch here breaks
# integrity_check=True and normal training alike.
SPLIT_COUNTS = {SPLIT_TRAIN: 20000, SPLIT_VALID: 2500, SPLIT_TEST: 2500}


# labels.py must be next to this script, or pass --labels-py to point elsewhere.
def load_labels_module(labels_py_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cityscapes_labels", labels_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_lut(labels_module) -> np.ndarray:
    """
    Builds a 256-element LUT: lut[labelId] = trainId.
    Labels not present in labels.py (unknown id) map to 255 (ignore).
    id -1 ('license plate') is skipped: it cannot index an array, and its
    trainId is -1 anyway, which we already treat as ignore (255).
    """
    lut = np.full(256, 255, dtype=np.uint8)
    for label in labels_module.labels:
        if label.id < 0:
            continue  # e.g. 'license plate' (id=-1): never appears in the PNGs
        train_id = label.trainId
        if train_id < 0 or train_id > 254:
            train_id = 255  # safety net, should not happen for id >= 0
        lut[label.id] = train_id
    return lut


def find_indices(synscapes_root: Path, rgb_source: str) -> list:
    """Finds the available image indices by looking at img/<rgb_source> (source of truth)."""
    rgb_dir = synscapes_root / "img" / rgb_source
    indices = []
    for p in rgb_dir.glob("*.png"):
        try:
            indices.append(int(p.stem))
        except ValueError:
            continue
    return sorted(indices)


def compute_split_assignment(indices: list, seed: int) -> dict:
    """
    Randomly (seeded) assigns each Synscapes index to a split, matching
    SPLIT_COUNTS exactly, and renumbers each split's images to 0..count-1.
    Returns {split: [(old_idx, new_idx), ...]}.
    """
    n_expected = sum(SPLIT_COUNTS.values())
    if len(indices) != n_expected:
        raise SystemExit(
            f'Found {len(indices)} images under img/<rgb-source>, but DatasetMiniscapes '
            f'requires exactly {n_expected} total ({SPLIT_COUNTS}). Check --synscapes-root '
            f'and --rgb-source; this check runs even with --limit, since the split is '
            f'always computed over the full dataset.'
        )
    rng = np.random.default_rng(seed)
    shuffled = np.array(indices)
    rng.shuffle(shuffled)

    assignment = {}
    pos = 0
    for split, count in SPLIT_COUNTS.items():
        chunk = shuffled[pos:pos + count]
        assignment[split] = list(zip(chunk.tolist(), range(count)))
        pos += count
    return assignment


def read_exr_depth(path: Path) -> np.ndarray:
    """
    Reads a single-channel EXR (planar depth in meters) as a float32 HxW array.
    Tries OpenEXR/Imath first (if installed), falls back to imageio.
    """
    try:
        import OpenEXR
        import Imath

        exr_file = OpenEXR.InputFile(str(path))
        header = exr_file.header()
        dw = header["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1

        # The channel may be named 'R', 'Y' or 'Z' depending on how it was
        # exported: take the first available among the usual candidates.
        channel_names = header["channels"].keys()
        for candidate in ("R", "Y", "Z"):
            if candidate in channel_names:
                chosen = candidate
                break
        else:
            chosen = list(channel_names)[0]

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        raw = exr_file.channel(chosen, pt)
        depth = np.frombuffer(raw, dtype=np.float32).reshape(h, w)
        return depth.copy()

    except ImportError:
        # Fallback: imageio with the freeimage plugin (downloaded on first use if missing).
        arr = iio.imread(path, plugin="EXR-FI")
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]  # single channel replicated across 3: take one
        return arr


def encode_depth(depth_meters: np.ndarray, ds: DatasetMiniscapes) -> np.ndarray:
    """
    Encodes planar depth in meters into the exact 8-bit quantized-disparity
    format DatasetMiniscapes.load_depth decodes, by calling the repo's own
    encoder instead of re-deriving the formula here (single source of truth).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        encoded = ds.depth_meters_float32_to_disparity_uint8(depth_meters, out_of_range_policy='invalidate')
    # Belt-and-braces: depth_meters_float32_to_disparity_uint8 already maps
    # anything non-finite or outside [depth_meters_min, depth_meters_max] to 0,
    # but we enforce it explicitly too rather than relying only on the 1/x
    # edge-case behavior for zero/negative/inf inputs.
    valid = (
        np.isfinite(depth_meters)
        & (depth_meters >= ds.depth_meters_min)
        & (depth_meters <= ds.depth_meters_max)
    )
    encoded[~valid] = 0
    return encoded


_worker_ds = None


def _init_worker():
    global _worker_ds
    # dataset_root/split are irrelevant here: only the encoding method and the
    # depth_meters_min/max properties are used, neither of which touches disk.
    _worker_ds = DatasetMiniscapes(dataset_root='unused', split=SPLIT_TRAIN, integrity_check=False)


def process_one(args):
    (old_idx, new_idx, split, synscapes_root, output_root, rgb_source, jpg_quality, lut, overwrite) = args

    synscapes_root = Path(synscapes_root)
    output_root = Path(output_root)

    rgb_in = synscapes_root / "img" / rgb_source / f"{old_idx}.png"
    class_in = synscapes_root / "img" / "class" / f"{old_idx}.png"
    depth_in = synscapes_root / "img" / "depth" / f"{old_idx}.exr"

    rgb_out = output_root / split / "rgb" / f"{new_idx}.jpg"
    semseg_out = output_root / split / "semseg" / f"{new_idx}.png"
    depth_out = output_root / split / "depth" / f"{new_idx}.png"

    result = {"old_idx": old_idx, "new_idx": new_idx, "split": split, "ok": True, "error": None}

    if rgb_out.exists() and semseg_out.exists() and depth_out.exists() and not overwrite:
        result["skipped"] = True
        return result
    result["skipped"] = False

    try:
        # ---- RGB: re-encode as jpg ----
        if not rgb_in.exists():
            raise FileNotFoundError(f"missing {rgb_in}")
        rgb_img = cv2.imread(str(rgb_in), cv2.IMREAD_COLOR)
        if rgb_img is None:
            raise RuntimeError(f"could not read {rgb_in}")
        rgb_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(rgb_out), rgb_img, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])

        # ---- Segmentation: labelId -> trainId via LUT ----
        if not class_in.exists():
            raise FileNotFoundError(f"missing {class_in}")
        class_img = cv2.imread(str(class_in), cv2.IMREAD_UNCHANGED)
        if class_img is None:
            raise RuntimeError(f"could not read {class_in}")
        if class_img.ndim == 3:
            # should not happen for a single-channel PNG, but just in case
            class_img = class_img[..., 0]
        class_img = class_img.astype(np.uint8)

        train_id_img = lut[class_img]  # vectorized LUT application
        semseg_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(semseg_out), train_id_img)

        # ---- Depth: EXR meters -> 8-bit quantized disparity (repo encoding) ----
        if not depth_in.exists():
            raise FileNotFoundError(f"missing {depth_in}")
        depth = read_exr_depth(depth_in)
        depth_u8 = encode_depth(depth, _worker_ds)

        depth_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(depth_out), depth_u8)

    except Exception as e:
        result["ok"] = False
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synscapes-root", required=True, type=Path,
                        help="Root of the ORIGINAL Synscapes dataset (contains img/ and meta/). Never written to.")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="Output root, filled with the Miniscapes layout (<split>/{rgb,semseg,depth}/).")
    parser.add_argument("--labels-py", type=Path, default=Path(__file__).parent / "labels.py",
                        help="Path to Cityscapes' labels.py (default: next to this script).")
    parser.add_argument("--rgb-source", choices=["rgb", "rgb-2k"], required=True,
                        help="Which Synscapes RGB folder to use. Only 'rgb' (native 1440x720) is "
                             "implemented; 'rgb-2k' is refused for now, see the module docstring.")
    parser.add_argument("--split-seed", type=int, required=True,
                        help="Seed for the random train/val/test split (20000/2500/2500). Required "
                             "and logged in the manifest: report this seed with your results.")
    parser.add_argument("--jpg-quality", type=int, default=95,
                        help="JPEG quality for the re-encoded RGB (default 95).")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1),
                        help="Number of parallel worker processes.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only actually process the first N (old_idx, new_idx) tasks, for a quick "
                             "pipeline smoke test. The split is still computed over the full 25000 "
                             "images, so this does not exercise the real per-split proportions.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate files that already exist in the output (default: skip).")
    args = parser.parse_args()

    if not args.synscapes_root.exists():
        sys.exit(f"Error: --synscapes-root does not exist: {args.synscapes_root}")
    if not args.labels_py.exists():
        sys.exit(f"Error: labels.py not found at {args.labels_py} (use --labels-py to point elsewhere).")
    if args.rgb_source == "rgb-2k":
        sys.exit(
            "Error: --rgb-source rgb-2k is not supported yet. Synscapes only ships class/ and depth/ "
            "at native 1440x720, so using the 2048x1024 rgb-2k images would require upscaling semseg/"
            "depth to match, and DatasetMiniscapes asserts semseg.size == rgb.size / depth.size == "
            "rgb.size. That resize strategy (interpolation for labels, and for depth before "
            "quantization) is a modeling decision, not just an I/O detail, so it isn't done silently "
            "here. Use --rgb-source rgb, or ask for rgb-2k support to be added."
        )

    labels_module = load_labels_module(args.labels_py)
    lut = build_lut(labels_module)

    indices = find_indices(args.synscapes_root, args.rgb_source)
    assignment = compute_split_assignment(indices, args.split_seed)

    args.output_root.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_COUNTS:
        for modality in ("rgb", "semseg", "depth"):
            (args.output_root / split / modality).mkdir(parents=True, exist_ok=True)

    # Save the manifest and the split assignment alongside the output: if the
    # seed, rgb-source or labels mapping ever changes, this records what
    # produced the data on disk. Depth range/counts are read from
    # DatasetMiniscapes, not re-typed here, so they can't drift from the loader.
    probe_ds = DatasetMiniscapes(dataset_root='unused', split=SPLIT_TRAIN, integrity_check=False)
    manifest = {
        "synscapes_root": str(args.synscapes_root),
        "rgb_source": args.rgb_source,
        "jpg_quality": args.jpg_quality,
        "split_seed": args.split_seed,
        "split_counts": SPLIT_COUNTS,
        "depth_meters_min": probe_ds.depth_meters_min,
        "depth_meters_max": probe_ds.depth_meters_max,
        "depth_encoding": "8-bit PNG, quantized disparity via "
                           "DatasetMiniscapes.depth_meters_float32_to_disparity_uint8; 0 = invalid",
        "label_encoding": "uint8 PNG, Cityscapes trainId (0-18), 255 = ignore",
        "lut_labelId_to_trainId": {int(i): int(v) for i, v in enumerate(lut)},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.output_root / "preprocess_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # split_assignment[split][new_idx] == old_idx. Regenerable from split_seed
    # alone, but kept explicit for auditability/debugging.
    split_assignment_readable = {
        split: [old_idx for old_idx, _new_idx in pairs]
        for split, pairs in assignment.items()
    }
    with open(args.output_root / "split_assignment.json", "w") as f:
        json.dump(split_assignment_readable, f)

    tasks = [
        (old_idx, new_idx, split, str(args.synscapes_root), str(args.output_root),
         args.rgb_source, args.jpg_quality, lut, args.overwrite)
        for split, pairs in assignment.items()
        for old_idx, new_idx in pairs
    ]
    if args.limit is not None:
        tasks = tasks[:args.limit]

    print(f"Found {len(indices)} images. Output -> {args.output_root}")
    print(f"rgb_source={args.rgb_source} | split_seed={args.split_seed} | workers={args.workers}")
    print(f"Processing {len(tasks)} tasks "
          f"({'all splits' if args.limit is None else f'first {len(tasks)}, --limit set'}).")

    n_done, n_skipped, n_error = 0, 0, 0
    errors = []
    t0 = time.time()

    with mp.Pool(processes=args.workers, initializer=_init_worker) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one, tasks, chunksize=16), start=1):
            if result["error"]:
                n_error += 1
                errors.append((result["split"], result["old_idx"], result["new_idx"], result["error"]))
            elif result["skipped"]:
                n_skipped += 1
            else:
                n_done += 1

            if i % 500 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  [{i}/{len(tasks)}] done={n_done} skipped={n_skipped} "
                      f"errors={n_error} ({rate:.1f} img/s)", flush=True)

    print(f"\nCompleted in {time.time() - t0:.1f}s")
    print(f"  OK: {n_done}  |  Skipped (already existed): {n_skipped}  |  Errors: {n_error}")

    if errors:
        err_log = args.output_root / "preprocess_errors.log"
        with open(err_log, "w") as f:
            for split, old_idx, new_idx, msg in errors:
                f.write(f"{split} old={old_idx} new={new_idx}: {msg}\n")
        print(f"  Error details saved to: {err_log}")


if __name__ == "__main__":
    main()
