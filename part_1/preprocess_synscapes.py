#!/usr/bin/env python3
"""
Preprocessing offline per il dataset Synscapes.

Cosa fa:
  1. Converte img/class/[i].png (label ID, convenzione Cityscapes) in
     processed/labels/[i].png con i trainId (0-18, ignore=255), usando
     la LUT costruita da labels.py.
  2. Converte img/depth/[i].exr (planar depth in metri, float32) in
     processed/depth/[i].png uint16, clippata e quantizzata su [0, d_max].
     Il valore 0 nel PNG16 codifica "depth non valida" (sky o oltre d_max),
     quindi la depth reale minima rappresentabile è > 0.

Cosa NON fa (di proposito):
  - Non tocca né copia img/rgb, img/rgb-2k, img/instance, meta/*.json:
    quei file si leggono direttamente dalla root originale in training,
    perché non richiedono nessuna trasformazione.
  - Non sovrascrive mai la root originale: scrive solo sotto --output-root.

Uso tipico:
  python3 preprocess_synscapes.py \
      --synscapes-root /path/to/synscapes \
      --output-root    /path/to/synscapes_processed \
      --d-max 80.0 \
      --workers 8

  # Per un test veloce su poche immagini prima del run completo:
  python3 preprocess_synscapes.py --synscapes-root ... --output-root ... --limit 200
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

# labels.py deve stare nella stessa cartella di questo script, oppure
# passa --labels-py per puntare a un percorso diverso.
def load_labels_module(labels_py_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cityscapes_labels", labels_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_lut(labels_module) -> np.ndarray:
    """
    Costruisce una LUT array di 256 elementi: lut[labelId] = trainId.
    Le label non presenti in labels.py (id sconosciuto) mappano a 255 (ignore).
    L'id -1 ('license plate') viene ignorato: non può indicizzare un array,
    e comunque il suo trainId è -1 -> lo trattiamo come ignore (255).
    """
    lut = np.full(256, 255, dtype=np.uint8)
    for label in labels_module.labels:
        if label.id < 0:
            continue  # es. 'license plate' (id=-1): non compare mai nei PNG
        train_id = label.trainId
        if train_id < 0 or train_id > 254:
            train_id = 255  # sicurezza, non dovrebbe succedere per id>=0
        lut[label.id] = train_id
    return lut


def find_indices(synscapes_root: Path) -> list:
    """Trova gli indici immagine disponibili guardando img/rgb (fonte di verità)."""
    rgb_dir = synscapes_root / "img" / "rgb"
    indices = []
    for p in rgb_dir.glob("*.png"):
        try:
            indices.append(int(p.stem))
        except ValueError:
            continue
    return sorted(indices)


def read_exr_depth(path: Path) -> np.ndarray:
    """
    Legge un EXR a canale singolo (planar depth in metri) come float32 HxW.
    Prova prima OpenEXR/Imath (se installati), altrimenti usa imageio.
    """
    try:
        import OpenEXR
        import Imath

        exr_file = OpenEXR.InputFile(str(path))
        header = exr_file.header()
        dw = header["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1

        # Il canale può chiamarsi 'R', 'Y' o 'Z' a seconda di come è stato
        # esportato: prendiamo il primo canale disponibile fra quelli tipici.
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
        # Fallback: imageio con plugin freeimage (lo scarica al primo uso se manca).
        arr = iio.imread(path, plugin="EXR-FI")
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]  # canale singolo replicato su 3: prendine uno
        return arr


def process_one(args):
    (idx, synscapes_root, output_root, d_max, lut, overwrite) = args

    synscapes_root = Path(synscapes_root)
    output_root = Path(output_root)

    class_in = synscapes_root / "img" / "class" / f"{idx}.png"
    depth_in = synscapes_root / "img" / "depth" / f"{idx}.exr"

    label_out = output_root / "labels" / f"{idx}.png"
    depth_out = output_root / "depth" / f"{idx}.png"

    result = {"idx": idx, "ok": True, "error": None}

    if label_out.exists() and depth_out.exists() and not overwrite:
        result["skipped"] = True
        return result
    result["skipped"] = False

    try:
        # ---- Segmentazione: labelId -> trainId via LUT ----
        if not class_in.exists():
            raise FileNotFoundError(f"manca {class_in}")
        class_img = cv2.imread(str(class_in), cv2.IMREAD_UNCHANGED)
        if class_img is None:
            raise RuntimeError(f"impossibile leggere {class_in}")
        if class_img.ndim == 3:
            # non dovrebbe succedere per un single-channel, ma per sicurezza
            class_img = class_img[..., 0]
        class_img = class_img.astype(np.uint8)

        train_id_img = lut[class_img]  # applicazione vettorizzata della LUT
        label_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(label_out), train_id_img)

        # ---- Depth: EXR metri -> PNG16 quantizzato ----
        if not depth_in.exists():
            raise FileNotFoundError(f"manca {depth_in}")
        depth = read_exr_depth(depth_in)

        valid = (depth > 0) & (depth < d_max) & np.isfinite(depth)
        depth_clipped = np.clip(depth, 0, d_max)
        depth_u16 = np.round(depth_clipped / d_max * 65535.0).astype(np.uint16)
        # I pixel non validi (cielo, oltre d_max, inf/nan) vanno esplicitamente
        # a 0, cosi' in training basta il check "> 0" per mascherarli, senza
        # bisogno di un file di maschera separato.
        depth_u16[~valid] = 0

        depth_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(depth_out), depth_u16)

    except Exception as e:
        result["ok"] = False
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synscapes-root", required=True, type=Path,
                        help="Root del dataset originale (contiene img/ e meta/). Mai scritta.")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="Root di output per i dati processati (labels/, depth/).")
    parser.add_argument("--labels-py", type=Path, default=Path(__file__).parent / "labels.py",
                        help="Percorso al file labels.py di Cityscapes (default: stessa cartella dello script).")
    parser.add_argument("--d-max", type=float, default=80.0,
                        help="Depth massima in metri usata per clip e quantizzazione (default 80.0).")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1),
                        help="Numero di processi paralleli.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Processa solo le prime N immagini (utile per un test veloce).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rigenera anche i file gia' presenti in output (default: skip).")
    args = parser.parse_args()

    if not args.synscapes_root.exists():
        sys.exit(f"Errore: --synscapes-root non esiste: {args.synscapes_root}")
    if not args.labels_py.exists():
        sys.exit(f"Errore: labels.py non trovato in {args.labels_py} (usa --labels-py per indicarne uno).")

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "labels").mkdir(exist_ok=True)
    (args.output_root / "depth").mkdir(exist_ok=True)

    labels_module = load_labels_module(args.labels_py)
    lut = build_lut(labels_module)

    # Salvo la LUT e i parametri usati insieme all'output: se in futuro cambi
    # d_max o la mappatura trainId, questo file ti dice con cosa e' stato
    # generato il dataset processato che hai su disco.
    manifest = {
        "synscapes_root": str(args.synscapes_root),
        "d_max": args.d_max,
        "depth_encoding": "uint16 PNG = round(clip(depth_m, 0, d_max) / d_max * 65535); 0 = invalid",
        "label_encoding": "uint8 PNG, Cityscapes trainId (0-18), 255 = ignore",
        "lut_labelId_to_trainId": {int(i): int(v) for i, v in enumerate(lut)},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.output_root / "preprocess_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    indices = find_indices(args.synscapes_root)
    if args.limit is not None:
        indices = indices[: args.limit]

    if not indices:
        sys.exit(f"Nessuna immagine trovata sotto {args.synscapes_root / 'img' / 'rgb'}")

    print(f"Trovate {len(indices)} immagini. Output -> {args.output_root}")
    print(f"d_max = {args.d_max} m | workers = {args.workers}")

    tasks = [
        (idx, str(args.synscapes_root), str(args.output_root), args.d_max, lut, args.overwrite)
        for idx in indices
    ]

    n_done, n_skipped, n_error = 0, 0, 0
    errors = []
    t0 = time.time()

    with mp.Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one, tasks, chunksize=16), start=1):
            if result["error"]:
                n_error += 1
                errors.append((result["idx"], result["error"]))
            elif result["skipped"]:
                n_skipped += 1
            else:
                n_done += 1

            if i % 500 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  [{i}/{len(tasks)}] processate={n_done} skip={n_skipped} "
                      f"errori={n_error} ({rate:.1f} img/s)", flush=True)

    print(f"\nCompletato in {time.time() - t0:.1f}s")
    print(f"  OK: {n_done}  |  Skippate (gia' esistenti): {n_skipped}  |  Errori: {n_error}")

    if errors:
        err_log = args.output_root / "preprocess_errors.log"
        with open(err_log, "w") as f:
            for idx, msg in errors:
                f.write(f"{idx}: {msg}\n")
        print(f"  Dettaglio errori salvato in: {err_log}")


if __name__ == "__main__":
    main()
