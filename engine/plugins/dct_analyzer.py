"""
Plugin: DCT / FFT Frequency Analyzer — v2

Technique:
    Natural photographic images obey the 1/f² power law: spectral power
    decays as ~frequency^(-2) across all spatial frequencies. This arises
    from the statistics of scenes in nature and is INDEPENDENT of content.

    Neural-network-generated or face-swapped images deviate from this law:
    - GAN / diffusion decoders create spectral peaks at stride-artifact
      frequencies (N/4, N/8 from centre in the 2D FFT power spectrum).
    - Some generators are over-smooth (power falls too fast, α < −2.5).
    - Face-swap decoders introduce non-linearities (peaks at specific
      frequencies from the decoder's upsampling layers).

    This plugin:
      1. Fits the actual power-law exponent (α) of the image's 2D FFT.
      2. Measures deviation of α from the expected −2.0 ± 0.5.
      3. Measures non-linearity (residual std) → detects spectral peaks.
      4. Detects cross-shaped peaks at stride-2 artifact frequencies.

    It works on ANY image (face crop, full frame, scene without a face)
    with NO face detection and NO calibration-data requirement, because
    it compares the image against its own fitted spectral model.

Dependencies:
    numpy, opencv-python (cv2)  — already in the venv, no extras needed.

Version: 2.0.0
"""

import cv2
import numpy as np
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.plugin_base import BaseDetectorPlugin
from core.plugin_names import PluginNames

logger = logging.getLogger(__name__)


class DCTFrequencyAnalyzerPlugin(BaseDetectorPlugin):

    @property
    def plugin_name(self) -> str:
        return PluginNames.DCT_FREQUENCY

    @property
    def plugin_description(self) -> str:
        return (
            "Detects GAN/decoder spectral artifacts and power-law deviations. "
            "Natural images obey a 1/f² spectral decay; neural face-swap decoders "
            "deviate through over-smoothing, spectral peaks at stride frequencies, "
            "or non-linear residuals. Works on any image without face detection."
        )

    @property
    def plugin_version(self) -> str:
        return "2.0.0"

    @property
    def plugin_weight(self) -> float:
        # Full weight — this plugin is meant to contribute real signal.
        # It complements ViT (learned features) with physics-based frequency analysis.
        return 0.25

    def analyze_frame(self, frame: np.ndarray, face_roi: np.ndarray | None = None) -> float:
        if frame is None or frame.size == 0:
            return 0.5

        # Prefer face_roi: the manipulated region has the strongest signal.
        # Fall back to full frame (still useful for fully AI-generated scenes).
        target = face_roi if (face_roi is not None and face_roi.size > 0) else frame

        try:
            return self._frequency_score(target)
        except Exception as e:
            logger.error(f"DCT Frequency analysis failed: {e}")
            return 0.5

    # ──────────────────────────────────────────────────────────────────────────

    def _frequency_score(self, img: np.ndarray) -> float:
        """
        Combine three complementary frequency-domain signals into one score.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        s1 = self._power_law_deviation(gray)     # 1/f² fit quality
        s2 = self._stride_artifact_score(gray)   # cross-shaped GAN peaks
        s3 = self._spectral_flatness_score(gray) # over-smoothness detector

        combined = 0.50 * s1 + 0.35 * s2 + 0.15 * s3
        return round(float(np.clip(combined, 0.0, 1.0)), 4)

    # ── Signal 1: Power-law exponent deviation ────────────────────────────────

    def _power_law_deviation(self, gray: np.ndarray) -> float:
        """
        Fit α in log(Power) ≈ α·log(f) + β over the azimuthal power spectrum.

        Natural images: α ≈ −2.0 ± 0.5
        NN-generated / face-swap:
          • Over-smooth decoder   → α < −2.8  (power falls too fast)
          • Artifact-heavy output → α > −1.2  (power doesn't decay enough)
          • Non-linear residuals  → residual std > 0.4 (spectral peaks)

        Returns a score in [0, 1]:  0 = natural,  1 = highly suspicious.
        """
        n = 128
        resized = cv2.resize(gray, (n, n))
        f = np.float32(resized) / 255.0

        # 2D FFT → shifted power spectrum
        F = np.fft.fft2(f)
        Fshift = np.fft.fftshift(F)
        power = np.abs(Fshift) ** 2

        center = n // 2
        yx = np.mgrid[-center:center, -center:center]
        r = np.sqrt(yx[0] ** 2 + yx[1] ** 2).ravel()
        p = power.ravel()

        # Bin into frequency rings, compute average power
        n_bins = 24
        edges = np.linspace(1.0, center * 0.9, n_bins + 1)
        log_f, log_p = [], []

        for i in range(n_bins):
            mask = (r >= edges[i]) & (r < edges[i + 1])
            if mask.sum() > 0:
                avg_p = p[mask].mean()
                avg_r = r[mask].mean()
                if avg_p > 1e-12 and avg_r > 0:
                    log_f.append(np.log(avg_r))
                    log_p.append(np.log(avg_p))

        if len(log_f) < 6:
            return 0.5

        log_f = np.array(log_f)
        log_p = np.array(log_p)

        # Linear regression in log-log space: log_p = alpha * log_f + beta
        A = np.vstack([log_f, np.ones_like(log_f)]).T
        solution, *_ = np.linalg.lstsq(A, log_p, rcond=None)
        alpha, beta = float(solution[0]), float(solution[1])
        fitted = log_f * alpha + beta
        residuals = log_p - fitted
        residual_std = float(np.std(residuals))

        # Score 1a: deviation of α from expected −2.0
        expected_alpha = -2.0
        alpha_dev = abs(float(alpha) - expected_alpha)
        # Tolerance ±0.7 is natural variation; beyond that = suspicious
        score_alpha = float(np.clip((alpha_dev - 0.7) / 1.5, 0.0, 1.0))

        # Score 1b: non-linearity → residual std
        # Natural: residual_std ≈ 0.1–0.3; NN peaks: > 0.4
        score_nonlinear = float(np.clip((residual_std - 0.3) / 0.6, 0.0, 1.0))

        return float(0.55 * score_alpha + 0.45 * score_nonlinear)

    # ── Signal 2: Stride-2 / checkerboard artifact detection ─────────────────

    def _stride_artifact_score(self, gray: np.ndarray) -> float:
        """
        Detect cross-shaped peaks in the FFT power spectrum at frequencies
        corresponding to stride-2 transposed-conv upsampling artifacts:
          N/4 and N/8 from the spectrum center on the horizontal/vertical axes.

        A peak prominently above the azimuthal background → suspicious.
        """
        n = 128
        resized = cv2.resize(gray, (n, n))
        f = np.float32(resized) / 255.0

        F = np.fft.fft2(f)
        Fsh = np.fft.fftshift(F)
        log_mag = np.log1p(np.abs(Fsh))

        center = n // 2

        # Azimuthal average for background estimate
        yx = np.mgrid[-center:center, -center:center]
        r_idx = np.round(np.sqrt(yx[0] ** 2 + yx[1] ** 2)).astype(int).ravel()
        r_idx = np.clip(r_idx, 0, center)
        lm_flat = log_mag.ravel()
        bg = np.bincount(r_idx, lm_flat, minlength=center + 1)
        cnt = np.bincount(r_idx, minlength=center + 1)
        radial_avg = bg / (cnt + 1e-10)

        max_prominence = 0.0

        for dist in [n // 8, n // 4]:
            if dist <= 0 or dist >= center:
                continue

            # Axis values at this distance
            positions = [
                (center, center + dist),
                (center, center - dist),
                (center + dist, center),
                (center - dist, center),
            ]
            axis_vals = [
                log_mag[r, c]
                for r, c in positions
                if 0 <= r < n and 0 <= c < n
            ]
            if not axis_vals:
                continue

            axis_mean = float(np.mean(axis_vals))
            bg_at_dist = float(radial_avg[min(dist, center)])

            if bg_at_dist > 0:
                prominence = axis_mean / bg_at_dist - 1.0
                max_prominence = max(max_prominence, prominence)

        # Natural: prominence ≈ 0 (no peaks above background)
        # GAN with stride artifacts: prominence 0.3–1.5+
        return float(np.clip(max_prominence / 1.0, 0.0, 1.0))

    # ── Signal 3: Spectral flatness (over-smoothness) ─────────────────────────

    def _spectral_flatness_score(self, gray: np.ndarray) -> float:
        """
        Spectral flatness (Wiener entropy) — ratio of geometric mean to
        arithmetic mean of the power spectrum.

        Natural images: flatness ~0.01–0.10 (low — energy concentrated at DC)
        Over-smooth GAN images: flatness near 0 (even lower → score high)
        White noise / artifact-heavy: flatness near 1 (score high)
        """
        n = 64
        resized = cv2.resize(gray, (n, n))
        f = np.float32(resized) / 255.0

        dct = cv2.dct(f)
        power = dct ** 2 + 1e-12
        power[0, 0] = 0.0  # exclude DC

        p_flat = power.ravel()
        p_flat = p_flat[p_flat > 1e-12]
        if len(p_flat) == 0:
            return 0.5

        geo_mean = np.exp(np.mean(np.log(p_flat)))
        arith_mean = np.mean(p_flat)
        flatness = geo_mean / (arith_mean + 1e-10)

        # Natural faces: flatness typically 0.02–0.12
        # Score: deviation from the natural range
        if flatness < 0.02:
            score = float(np.clip((0.02 - flatness) / 0.02, 0.0, 1.0))
        elif flatness > 0.15:
            score = float(np.clip((flatness - 0.15) / 0.35, 0.0, 1.0))
        else:
            score = 0.0

        return score
