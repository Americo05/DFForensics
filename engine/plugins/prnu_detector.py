"""
Plugin: PRNU-style Noise Residue Detector.

PRNU (Photo-Response Non-Uniformity) is the camera sensor's unique noise
fingerprint. A face that was synthesized or pasted from another image will
carry DIFFERENT sensor noise statistics than the background pixels of the
same frame. Real cameras imprint consistent noise across the entire image.

A "true" PRNU pipeline requires a per-camera reference fingerprint built
from dozens of flat-field images. We don't have that — but we can use a
weaker, single-frame proxy:

  1. Estimate the noise residue of the whole frame via a denoising filter
     (high-pass = original − denoised).
  2. Compute noise *energy* (variance) in the FACE region and in the
     BACKGROUND region separately.
  3. Real cameras: face and background noise variance are close (ratio ~1).
  4. Face-swapped: ratio diverges — pasted face has different noise statistics.

This is a per-frame plugin that runs in CROPPED_FACE and FACE_IN_SCENE
scenes only — it's pointless without a face boundary.

Dependencies:
    cv2, numpy (already required)

Version: 1.0.0
"""

import cv2
import logging
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.plugin_base import BaseDetectorPlugin
from core.plugin_names import PluginNames  # noqa: E402

logger = logging.getLogger(__name__)


class PRNUNoiseResidueDetector(BaseDetectorPlugin):

    @property
    def plugin_name(self) -> str:
        return PluginNames.PRNU_NOISE

    @property
    def plugin_description(self) -> str:
        return (
            "Compares high-pass noise variance inside the face ROI against "
            "the surrounding background. Real cameras imprint consistent "
            "sensor noise; face-swapped regions carry foreign noise "
            "statistics. Single-frame proxy for full PRNU analysis."
        )

    @property
    def plugin_version(self) -> str:
        return "1.0.0"

    @property
    def plugin_weight(self) -> float:
        # Conservative weight: this is a heuristic, not the real PRNU pipeline.
        return 0.15

    # ── Tuning ─────────────────────────────────────────────────────────
    MIN_ROI_SIDE = 48  # don't run on tiny crops — noise stats unstable

    def analyze_frame(self, frame: np.ndarray, face_roi: np.ndarray | None = None) -> float:
        # Without a face boundary there's no "inside vs outside" to compare,
        # so the plugin is intentionally unhelpful in NO_FACE scenes — and
        # the scene router doesn't include it there anyway.
        if frame is None or frame.size == 0:
            return 0.5
        if face_roi is None or face_roi.size == 0:
            return 0.5
        if frame.ndim != 3 or frame.shape[2] < 3:
            return 0.5

        try:
            return self._noise_ratio_score(frame, face_roi)
        except Exception as e:
            logger.warning(f"PRNU analysis failed: {e}")
            return 0.5

    def _noise_ratio_score(self, frame: np.ndarray, face_roi: np.ndarray) -> float:
        h_frame, w_frame = frame.shape[:2]

        # Locate the face ROI within the frame by simple template-position
        # search: since FacePreProcessor crops the face_roi from `frame`
        # by integer indexing, the exact pixels match. We could pass the
        # bbox here too, but the plugin API only gives us the ROI. So we
        # use a fast cross-correlation to find it.
        bbox = self._locate_roi(frame, face_roi)
        if bbox is None:
            return 0.5
        x, y, w, h = bbox
        if w < self.MIN_ROI_SIDE or h < self.MIN_ROI_SIDE:
            return 0.5

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # High-pass: original − Gaussian-blurred = noise residue.
        # OpenCV's GaussianBlur is much faster than the wavelet denoising
        # used in academic PRNU pipelines; quality is enough for a ratio
        # comparison (we're not extracting an actual fingerprint).
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
        residue = gray.astype(np.float32) - blurred.astype(np.float32)

        # Face ROI residue stats
        face_residue = residue[y:y + h, x:x + w]

        # Background: erase the face region; use the rest of the frame
        bg_mask = np.ones_like(residue, dtype=bool)
        bg_mask[y:y + h, x:x + w] = False
        # Trim a small margin around the face to avoid edge bleed
        margin = max(4, min(w, h) // 20)
        y0 = max(0, y - margin)
        y1 = min(h_frame, y + h + margin)
        x0 = max(0, x - margin)
        x1 = min(w_frame, x + w + margin)
        bg_mask[y0:y1, x0:x1] = False
        bg_residue = residue[bg_mask]

        if bg_residue.size < 1000 or face_residue.size < 1000:
            # Too little background to compare against — fail safe to neutral
            return 0.5

        face_var = float(np.var(face_residue))
        bg_var = float(np.var(bg_residue))
        if bg_var < 1e-6:
            return 0.5

        # Ratio. Real cameras give r ≈ 1 (±25%). Synth/pasted faces drift.
        ratio = face_var / bg_var
        # Convert to a "how far from 1" score. log makes the metric symmetric:
        # ratio=0.5 and ratio=2.0 contribute equally.
        deviation = abs(np.log(max(ratio, 1e-3)))
        # Empirical normalization: log(2) ≈ 0.69 is "borderline suspect";
        # log(4) ≈ 1.38 is "very suspicious".
        score = float(np.clip((deviation - 0.35) / 1.0, 0.0, 1.0))
        return round(score, 4)

    def _locate_roi(self, frame: np.ndarray, roi: np.ndarray) -> tuple[int, int, int, int] | None:
        """Find roi's top-left in frame via template matching. Returns (x, y, w, h)."""
        if roi.shape[0] > frame.shape[0] or roi.shape[1] > frame.shape[1]:
            return None
        # Downscale both for speed if the frame is huge
        scale = 1.0
        max_dim = max(frame.shape[0], frame.shape[1])
        if max_dim > 1280:
            scale = 1280 / max_dim
            new_frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
            new_roi = cv2.resize(roi, (max(1, int(roi.shape[1] * scale)),
                                       max(1, int(roi.shape[0] * scale))))
        else:
            new_frame = frame
            new_roi = roi
        if new_roi.shape[0] > new_frame.shape[0] or new_roi.shape[1] > new_frame.shape[1]:
            return None
        result = cv2.matchTemplate(new_frame, new_roi, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < 0.6:  # poor match — bail
            return None
        x = int(max_loc[0] / scale)
        y = int(max_loc[1] / scale)
        w = roi.shape[1]
        h = roi.shape[0]
        return (x, y, w, h)
