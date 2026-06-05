"""
Temporal Coherence Analyzer — video-level temporal-consistency checks.

Rationale
---------
Deepfake generators usually operate on isolated frames or short clips and
have no notion of "the previous frame". This leaks into measurable temporal
artifacts that real cameras don't produce:

  * **Face jitter**     — bounding box centroid jumping between frames at
                          rates inconsistent with the body's actual motion.
  * **Score volatility** — per-frame fake-probability swings wildly even
                           though the scene is visually continuous; real
                           manipulations are usually consistent within a shot.
  * **Brightness jumps** — sudden frame-mean changes that don't match a
                           camera's auto-exposure ramp.

We feed in the per-frame results that PluginManager already computed (so
no extra ML inference), plus the raw frames for brightness analysis. The
analyzer returns a single `temporal_score` in [0,1] alongside human-readable
sub-signals.

This is intentionally lightweight — heavier methods (optical-flow consistency,
phase-based residuals) exist in the literature but require significant compute.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class TemporalCoherenceAnalyzer:
    """Stateless. Call `analyze(frames, frame_details)` per video."""

    # Tunables — adjust if calibration suggests it
    MIN_FRAMES = 6                # need at least this many frames for stats
    BBOX_JITTER_NORMALIZE = 0.05  # bbox center jumps of >5% of frame are "bad"
    SCORE_VOLATILITY_NORMALIZE = 0.25  # per-frame score std above this is suspect

    def analyze(
        self,
        frames: list,
        frame_details: list[dict],
    ) -> dict | None:
        """
        Args:
            frames:        the BGR numpy arrays already used in the main pipeline
            frame_details: the per-frame dicts emitted by PluginManager, each
                           containing `faces[]` with `face_bbox` and `overall_score`

        Returns:
            { "temporal_score": float in [0,1],
              "signals": [...],
              "frame_count": int,
              "verdict": "CONSISTENT" | "VOLATILE" | "INCONCLUSIVE" }
            or None if there's not enough data to be meaningful.
        """
        if not frames or not frame_details or len(frame_details) < self.MIN_FRAMES:
            return None

        n = min(len(frames), len(frame_details))
        frames = frames[:n]
        frame_details = frame_details[:n]

        signals: list[str] = []
        components: list[tuple[str, float]] = []

        # ── Component 1: face centroid jitter ───────────────────────────
        # Track the DOMINANT (largest) face per frame. If many frames lose
        # the face entirely, that itself is volatility (people don't blink
        # out of existence).
        centroids: list[tuple[float, float, int, int] | None] = []
        for f, fd in zip(frames, frame_details):
            h, w = f.shape[:2]
            faces = fd.get("faces") or []
            biggest = None
            biggest_area = 0
            for face in faces:
                bbox = face.get("face_bbox")
                if not bbox:
                    continue
                try:
                    bw, bh = int(bbox["w"]), int(bbox["h"])
                    bx, by = int(bbox["x"]), int(bbox["y"])
                except (KeyError, TypeError, ValueError):
                    continue
                area = bw * bh
                if area > biggest_area:
                    biggest_area = area
                    biggest = (bx + bw / 2.0, by + bh / 2.0, w, h)
            centroids.append(biggest)

        present = [c for c in centroids if c is not None]
        if len(present) >= 3:
            # Normalize displacements by the frame diagonal so 720p and 4K
            # videos give comparable numbers.
            diag = np.sqrt(present[0][2] ** 2 + present[0][3] ** 2)
            displacements: list[float] = []
            for prev, curr in zip(present, present[1:]):
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                displacements.append(float(np.sqrt(dx * dx + dy * dy) / diag))
            if displacements:
                jitter_mean = float(np.mean(displacements))
                jitter_score = float(np.clip(jitter_mean / self.BBOX_JITTER_NORMALIZE, 0.0, 1.0))
                components.append(("face_jitter", jitter_score))
                if jitter_score > 0.6:
                    signals.append("face_bbox_jittering")

        # Frames where the face disappears even though it was just there
        face_loss_events = 0
        for prev, curr in zip(centroids, centroids[1:]):
            if prev is not None and curr is None:
                face_loss_events += 1
        if len(centroids) > 1:
            loss_rate = face_loss_events / max(len(centroids) - 1, 1)
            if loss_rate > 0.20:
                signals.append("face_disappearing_frames")
                components.append(("face_loss", float(np.clip(loss_rate * 2, 0.0, 1.0))))

        # ── Component 2: score volatility ───────────────────────────────
        scores = [
            fd.get("overall_score", 0.5)
            for fd in frame_details if isinstance(fd.get("overall_score"), (int, float))
        ]
        if len(scores) >= self.MIN_FRAMES:
            score_std = float(np.std(scores))
            volatility = float(np.clip(score_std / self.SCORE_VOLATILITY_NORMALIZE, 0.0, 1.0))
            components.append(("score_volatility", volatility))
            if volatility > 0.7:
                signals.append("score_swings_between_frames")

        # ── Component 3: brightness jumps ───────────────────────────────
        # Real auto-exposure ramps smoothly. Edit cuts and synthesized frames
        # often produce step changes in mean brightness.
        means = [float(np.mean(f)) for f in frames]
        if len(means) >= 3:
            diffs = np.abs(np.diff(means))
            # Normalize by typical pixel scale (255)
            big_jumps = np.sum(diffs > 25.0)
            jump_rate = big_jumps / max(len(diffs), 1)
            components.append(("brightness_jumps", float(np.clip(jump_rate * 3, 0.0, 1.0))))
            if jump_rate > 0.2:
                signals.append("abrupt_brightness_jumps")

        if not components:
            return None

        # Weighted mean of components — equal weights for now; calibrate later.
        weights = {
            "face_jitter": 0.40,
            "face_loss": 0.20,
            "score_volatility": 0.25,
            "brightness_jumps": 0.15,
        }
        total_w = 0.0
        weighted = 0.0
        for name, val in components:
            w = weights.get(name, 0.0)
            weighted += val * w
            total_w += w
        temporal_score = weighted / total_w if total_w > 0 else 0.5

        verdict = "VOLATILE" if temporal_score > 0.6 else (
            "INCONCLUSIVE" if temporal_score > 0.4 else "CONSISTENT"
        )

        return {
            "temporal_score": round(float(temporal_score), 4),
            "signals": signals,
            "frame_count": n,
            "components": {name: round(v, 4) for name, v in components},
            "verdict": verdict,
        }
