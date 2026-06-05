"""
benchmark.py — Forensic accuracy benchmark for the detection engine.

What it does
------------
Walks a labeled dataset directory of videos/images, runs each through the
PluginManager + audio analyzers, and computes standard binary-classification
metrics PER PLUGIN and GLOBALLY:
  - AUC (area under ROC curve)
  - F1, precision, recall
  - Confusion matrix at the dashboard threshold (0.6)
  - Per-class score histogram (printable)
Output is written to engine/tests/benchmark_results/ as CSV + Markdown.

Expected dataset layout
-----------------------
    datasets/
      real/        # ground-truth REAL videos & images
        clip01.mp4
        photo02.jpg
      fake/        # ground-truth FAKE / deepfake / AI-generated content
        clip03.mp4
        ...

Public datasets to try
----------------------
  • FaceForensics++ (https://github.com/ondyari/FaceForensics) — needs license
  • Celeb-DF v2     (https://github.com/yuezunli/celeb-deepfakeforensics)
  • DFDC subset     (Kaggle DeepFake Detection Challenge)

Usage
-----
    cd engine
    python benchmark.py --dataset ../datasets --limit 50

Limit is useful for quick smoke runs. Without it, every file under
real/ and fake/ is processed (can take hours on a large dataset).

The metrics are reported in BENCHMARKS.md at the repo root — fill in the
result columns by running this script and copying numbers from the output.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Make the engine package importable when running as a script
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np

from core.plugin_manager import PluginManager
from core.lip_sync_analyzer import LipSyncAnalyzer
from core.audio_deepfake_analyzer import AudioDeepfakeAnalyzer

logging.basicConfig(level=logging.WARNING)  # quiet noise during benchmarking
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
FAKE_THRESHOLD = 0.6


# ── Metric helpers ──────────────────────────────────────────────────────────

def roc_auc(labels: list[int], scores: list[float]) -> float:
    """
    AUC via Mann–Whitney U (no sklearn dependency).
    labels: 1 = positive (fake), 0 = negative (real)
    """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def precision_recall_f1(labels: list[int], scores: list[float], threshold: float):
    tp = fp = fn = tn = 0
    for l, s in zip(labels, scores):
        pred = 1 if s > threshold else 0
        if pred == 1 and l == 1: tp += 1
        elif pred == 1 and l == 0: fp += 1
        elif pred == 0 and l == 1: fn += 1
        else: tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, (tp, fp, fn, tn)


# ── Analysis driver ─────────────────────────────────────────────────────────

def _imread_unicode(path: Path):
    """Read an image whose path may contain non-ASCII characters.

    cv2.imread() goes through fopen() on Windows and silently returns None
    for paths with non-ASCII characters (e.g. "3ºAno"). np.fromfile uses
    Python's IO, which handles any path, then cv2.imdecode parses the bytes.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _extract_frames_quick(path: Path, max_frames: int = 60) -> list:
    """Sample up to `max_frames` evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    skip = max(1, total // max_frames)
    frames = []
    idx = 0
    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += skip
    cap.release()
    return frames


def analyze_file(path: Path, pm: PluginManager, lsa: LipSyncAnalyzer, ada: AudioDeepfakeAnalyzer) -> dict:
    """Return scores for one file: overall + per-plugin + audio breakdown."""
    ext = path.suffix.lower()
    is_image = ext in IMAGE_EXTENSIONS

    if is_image:
        img = _imread_unicode(path)
        frames = [img] if img is not None else []
    else:
        frames = _extract_frames_quick(path)

    if not frames:
        return {"error": "no_frames"}

    results = pm.run_analysis(frames, fps=30.0, mode="local")
    visual_score = results.get("overall_score", 0.5)
    per_plugin = {p["name"]: p["average_score"] for p in results.get("plugins", [])}

    lip_score = None
    audio_score = None
    if not is_image:
        try:
            lr = lsa.analyze_video(str(path))
            if lr and not lr.get("inconclusive"):
                lip_score = lr["lip_sync_score"]
        except Exception:
            pass
        try:
            ar = ada.analyze_audio(str(path))
            if ar:
                audio_score = ar["audio_fake_score"]
        except Exception:
            pass

    audio_scores = [s for s in [lip_score, audio_score] if s is not None]
    audio_max = max(audio_scores) if audio_scores else None
    overall = max(visual_score, audio_max) if audio_max is not None else visual_score

    return {
        "overall": overall,
        "visual": visual_score,
        "audio": audio_max,
        "lip_sync": lip_score,
        "wavlm": audio_score,
        **{f"plugin:{k}": v for k, v in per_plugin.items()},
    }


# ── Reporting ───────────────────────────────────────────────────────────────

def write_csv(out_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def compute_metrics(rows: list[dict], score_field: str) -> dict | None:
    labels, scores = [], []
    for r in rows:
        s = r.get(score_field)
        if s is None or (isinstance(s, float) and (np.isnan(s) or s != s)):
            continue
        labels.append(r["label"])
        scores.append(float(s))
    if not labels:
        return None
    auc = roc_auc(labels, scores)
    p, rcl, f1, cm = precision_recall_f1(labels, scores, FAKE_THRESHOLD)
    return {
        "field": score_field,
        "n": len(labels),
        "auc": auc,
        "precision": p,
        "recall": rcl,
        "f1": f1,
        "confusion": cm,
    }


def write_markdown(out_path: Path, all_metrics: list[dict], n_real: int, n_fake: int, dataset_path: str) -> None:
    lines = [
        "# Benchmark Results",
        "",
        f"_Generated by `engine/benchmark.py` at {time.strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Real samples: **{n_real}**",
        f"- Fake samples: **{n_fake}**",
        f"- Threshold for confusion matrix / F1: **{FAKE_THRESHOLD}**",
        "",
        "## Metrics by score source",
        "",
        "| Source | N | AUC | Precision | Recall | F1 | TP | FP | FN | TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in all_metrics:
        tp, fp, fn, tn = m["confusion"]
        lines.append(
            f"| `{m['field']}` | {m['n']} | {m['auc']:.3f} | {m['precision']:.3f} | "
            f"{m['recall']:.3f} | {m['f1']:.3f} | {tp} | {fp} | {fn} | {tn} |"
        )
    lines += [
        "",
        "## How to read",
        "- **AUC** is threshold-independent. >0.9 = strong, 0.7–0.9 = useful, <0.6 = barely better than random.",
        "- **F1** is at the production threshold (0.6). Tune the threshold per use case.",
        "- **Per-plugin** rows let you see which detector carries the signal.",
        "",
        "## Reproducing",
        "",
        "```sh",
        "cd engine",
        f"python benchmark.py --dataset {dataset_path}",
        "```",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the engine over a labeled dataset and print metrics.")
    parser.add_argument("--dataset", required=True, help="Path to dir with real/ and fake/ subdirs")
    parser.add_argument("--limit", type=int, default=None, help="Cap total files per class (smoke runs)")
    parser.add_argument("--out", default="tests/benchmark_results", help="Output directory")
    args = parser.parse_args()

    dataset_root = Path(args.dataset).resolve()
    real_dir = dataset_root / "real"
    fake_dir = dataset_root / "fake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        sys.exit(f"Expected '{real_dir}' and '{fake_dir}' to exist.")

    def collect(d: Path) -> list[Path]:
        files = [p for p in sorted(d.iterdir()) if p.suffix.lower() in (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)]
        return files[: args.limit] if args.limit else files

    real_files = collect(real_dir)
    fake_files = collect(fake_dir)
    logger.info(f"Loaded {len(real_files)} real + {len(fake_files)} fake files")

    pm = PluginManager()
    lsa = LipSyncAnalyzer()
    ada = AudioDeepfakeAnalyzer()

    rows: list[dict] = []
    for label, files in [(0, real_files), (1, fake_files)]:
        for i, path in enumerate(files, 1):
            logger.info(f"[{'REAL' if label == 0 else 'FAKE'}] {i}/{len(files)} {path.name}")
            t0 = time.time()
            try:
                scores = analyze_file(path, pm, lsa, ada)
            except Exception as e:
                logger.warning(f"  failed: {e}")
                scores = {"error": str(e)}
            elapsed = round(time.time() - t0, 2)
            rows.append({
                "file": str(path.relative_to(dataset_root)),
                "label": label,
                "label_name": "fake" if label == 1 else "real",
                "elapsed_s": elapsed,
                **scores,
            })

    out_dir = Path(args.out)
    write_csv(out_dir / "per_file_scores.csv", rows)
    logger.info(f"Per-file CSV written to {out_dir / 'per_file_scores.csv'}")

    # Compute metrics for the global score + per-plugin + each modality
    sample_keys = set()
    for r in rows:
        for k in r.keys():
            if k in {"file", "label", "label_name", "elapsed_s", "error"}:
                continue
            sample_keys.add(k)
    metric_fields = ["overall", "visual", "audio", "lip_sync", "wavlm"] + sorted(
        k for k in sample_keys if k.startswith("plugin:")
    )

    all_metrics: list[dict] = []
    for field in metric_fields:
        m = compute_metrics(rows, field)
        if m is not None:
            all_metrics.append(m)

    write_markdown(
        out_dir / "BENCHMARKS.md", all_metrics,
        n_real=len(real_files), n_fake=len(fake_files),
        dataset_path=str(dataset_root),
    )
    logger.info(f"Markdown report written to {out_dir / 'BENCHMARKS.md'}")

    # Print summary to stdout so the user sees the headline numbers immediately
    print("\n=== Summary ===")
    print(f"Real: {len(real_files)}  Fake: {len(fake_files)}")
    for m in all_metrics:
        print(f"  {m['field']:40s}  AUC={m['auc']:.3f}  F1={m['f1']:.3f}  N={m['n']}")
    print(f"\nFull report: {out_dir / 'BENCHMARKS.md'}")


if __name__ == "__main__":
    main()
