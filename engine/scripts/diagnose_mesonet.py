"""
diagnose_mesonet.py — isolate MesoNet from the full pipeline.

What this does
--------------
Picks 3 real + 3 fake videos from a labeled dataset, extracts ~10 frames
from each, runs MesoNet directly on the face crops (bypassing the
ensemble weighted-mean and the frame-level MEAN aggregation), and prints
the raw per-frame scores.

This tells us:
  • Is MesoNet ACTUALLY scoring fakes higher than reals at the frame level?
    (separates the model from the aggregation/ensemble dilution problem)
  • Are face detections succeeding on this dataset?
    (separates the model from MTCNN failures)
  • What's the score distribution?
    (tells us if the threshold is wrong or the model is wrong)

If MesoNet reliably scores fakes higher per-frame here, the engine's
aggregation logic is what's burying the signal. If it doesn't, the
weights or preprocessing have a more subtle bug than the NHWC fix
already caught.

Usage
-----
    python engine/scripts/diagnose_mesonet.py \
        --dataset "C:\\path\\to\\dataset_with_real_and_fake"

The dataset must follow the engine/benchmark.py layout: real/ and fake/
subfolders with .mp4 files inside.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from statistics import mean, stdev

import cv2
import numpy as np

# Make the engine package importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.plugin_manager import FacePreProcessor
from plugins.mesonet_detector import MesoNetDetectorPlugin


def _sample_frames(video_path: Path, n: int = 10) -> list[np.ndarray]:
    """Pull `n` evenly-spaced frames from the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, total // n)
    frames = []
    idx = 0
    while len(frames) < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        idx += step
    cap.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, help="Folder with real/ + fake/ subdirs")
    parser.add_argument("--per-class", type=int, default=3, help="Videos per class (default 3)")
    parser.add_argument("--frames", type=int, default=10, help="Frames per video (default 10)")
    args = parser.parse_args()

    plugin = MesoNetDetectorPlugin()
    if not plugin.is_configured():
        sys.exit("MesoNet weights not loaded — run download_mesonet_weights.py first.")

    face_det = FacePreProcessor()

    dataset = Path(args.dataset)
    all_real: list[float] = []
    all_fake: list[float] = []
    no_face_count = 0

    for label, sub in [("REAL", "real"), ("FAKE", "fake")]:
        folder = dataset / sub
        if not folder.is_dir():
            sys.exit(f"Missing folder: {folder}")
        videos = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".mp4"])[: args.per_class]

        for video in videos:
            print(f"\n[{label}] {video.name}")
            frames = _sample_frames(video, args.frames)
            if not frames:
                print("  (couldn't read frames)")
                continue

            scores_this_video: list[float] = []
            no_face_this_video = 0
            for i, frame in enumerate(frames):
                faces, _, found = face_det.process(frame)
                if not found:
                    no_face_this_video += 1
                    no_face_count += 1
                    continue
                # Score the LARGEST face only (mimics what frame_max_score does)
                face_roi, bbox = max(
                    faces,
                    key=lambda fb: (fb[1]["w"] * fb[1]["h"]) if fb[1] else 0,
                )
                if face_roi is None:
                    no_face_this_video += 1
                    no_face_count += 1
                    continue
                score = plugin.analyze_frame(frame, face_roi=face_roi)
                scores_this_video.append(score)

            if scores_this_video:
                avg = mean(scores_this_video)
                rng = (min(scores_this_video), max(scores_this_video))
                print(
                    f"  frames_with_face={len(scores_this_video)}/{len(frames)}  "
                    f"mean={avg:.3f}  range=[{rng[0]:.3f}, {rng[1]:.3f}]  "
                    f"no_face={no_face_this_video}"
                )
                (all_real if label == "REAL" else all_fake).extend(scores_this_video)

    # ── Summary across all frames ──────────────────────────────────────
    def _summary(name: str, scores: list[float]) -> None:
        if not scores:
            print(f"  {name}: (no scores)")
            return
        m = mean(scores)
        s = stdev(scores) if len(scores) > 1 else 0.0
        print(f"  {name}: n={len(scores)}  mean={m:.3f}  std={s:.3f}  "
              f"min={min(scores):.3f}  max={max(scores):.3f}")

    print("\n=== Per-frame distributions ===")
    _summary("REAL frames", all_real)
    _summary("FAKE frames", all_fake)
    print(f"\nTotal frames with no face detected: {no_face_count}")

    # ── Verdict on the engine's chain ───────────────────────────────────
    if all_real and all_fake:
        margin = mean(all_fake) - mean(all_real)
        print(f"\nMesoNet mean(fake) - mean(real) = {margin:+.3f}")
        if margin > 0.10:
            print("  → MesoNet IS discriminating at the frame level.")
            print("    The pipeline's MEAN aggregation is likely diluting the signal.")
        elif margin > 0.02:
            print("  → Weak discrimination. Could be N too small OR genuine model limit on this dataset.")
        else:
            print("  → No discrimination. Either weights are still wrong OR this dataset is OOD for MesoNet.")


if __name__ == "__main__":
    main()
