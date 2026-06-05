"""
rPPG (Remote Photoplethysmography) Analyzer — physiological signal forensics.

What it does
------------
Real human skin reflects light differently as blood flows through capillaries
with each heartbeat. The change is invisible to the naked eye but shows up as
a periodic intensity oscillation in the GREEN channel of the face region,
in the 0.7–4.0 Hz band (42–240 BPM).

Deepfake generators don't model circulation. A synthesized face either:
  * Carries no rPPG signal at all (flat power in the 0.7–4 Hz band), OR
  * Inherits the source video's pulse, which is desynchronized with what
    the audio/scene suggests.

Single-signal heuristic
-----------------------
We compute the green-channel mean of the face ROI per frame, detrend it,
band-pass filter to 0.7–4 Hz, and measure the energy *concentration*
around a single peak. Real faces produce a sharp peak (the heart rate);
synthesized faces produce a flat or noisy spectrum.

Output is in [0,1]; higher = more suspicious. This is intentionally a
fallback signal in the ensemble — rPPG is unreliable when:
  * The face is poorly lit, moving rapidly, or partially occluded
  * The video is heavily compressed (the subtle green signal gets crushed)
  * FPS is low (< ~15 fps Nyquist-limits the band of interest)

Dependencies:
    cv2, numpy (already required)

Version: 1.0.0
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class rPPGAnalyzer:
    """Stateless. Call `analyze(frames, fps)` per video."""

    MIN_FRAMES = 90                 # ~3 seconds at 30fps; less and FFT is too noisy
    BAND_HZ = (0.7, 4.0)            # 42-240 BPM
    SAMPLE_FPS = 30.0               # We resample mean signal to this rate

    def analyze(
        self,
        frames: list,
        fps: float,
        frame_details: list[dict] | None = None,
    ) -> dict | None:
        """
        Args:
            frames:        BGR numpy frames
            fps:           source video framerate (so we can interpret time)
            frame_details: optional per-frame face bboxes (saves re-detecting)

        Returns dict with rppg_score and details, or None if not enough signal.
        """
        if not frames or len(frames) < self.MIN_FRAMES:
            return None
        if fps < 10:
            # Below Nyquist for 2x our band's max → can't reliably detect
            return None

        try:
            green_signal = self._extract_face_green(frames, frame_details)
        except Exception as e:
            logger.warning(f"rPPG extraction failed: {e}")
            return None

        if green_signal is None or len(green_signal) < self.MIN_FRAMES:
            return None

        return self._score_spectrum(green_signal, fps)

    # ── Signal extraction ──────────────────────────────────────────────

    def _extract_face_green(
        self,
        frames: list,
        frame_details: list[dict] | None,
    ) -> np.ndarray | None:
        """
        Compute the mean of the GREEN channel inside the face ROI for each
        frame. Returns a 1D array of length len(frames), or None if too many
        frames have no usable face.
        """
        means: list[float] = []
        for i, frame in enumerate(frames):
            if frame is None or frame.size == 0:
                means.append(np.nan)
                continue

            bbox = None
            if frame_details and i < len(frame_details):
                faces = frame_details[i].get("faces") or []
                # Pick the largest face if multiple
                biggest_area = 0
                for face in faces:
                    bb = face.get("face_bbox")
                    if not bb:
                        continue
                    try:
                        bw, bh = int(bb["w"]), int(bb["h"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if bw * bh > biggest_area:
                        biggest_area = bw * bh
                        bbox = bb

            if bbox is None:
                means.append(np.nan)
                continue

            try:
                x, y = int(bbox["x"]), int(bbox["y"])
                w, h = int(bbox["w"]), int(bbox["h"])
            except (KeyError, TypeError, ValueError):
                means.append(np.nan)
                continue

            # Use the central 60% of the bbox — avoids hair, jawline, and
            # background bleed that contaminate the rPPG signal.
            inset_w, inset_h = int(w * 0.2), int(h * 0.2)
            x0 = max(0, x + inset_w)
            y0 = max(0, y + inset_h)
            x1 = min(frame.shape[1], x + w - inset_w)
            y1 = min(frame.shape[0], y + h - inset_h)
            if x1 <= x0 or y1 <= y0:
                means.append(np.nan)
                continue

            roi = frame[y0:y1, x0:x1]
            if roi.size == 0:
                means.append(np.nan)
                continue

            # BGR → green channel is index 1
            green = roi[:, :, 1].astype(np.float32)
            means.append(float(np.mean(green)))

        arr = np.array(means, dtype=np.float32)

        # Fail if more than half the frames have no face
        nan_ratio = float(np.mean(np.isnan(arr)))
        if nan_ratio > 0.5:
            return None

        # Linear interpolation across short gaps
        valid_idx = np.where(~np.isnan(arr))[0]
        if len(valid_idx) < self.MIN_FRAMES:
            return None
        arr = np.interp(np.arange(len(arr)), valid_idx, arr[valid_idx])
        return arr

    # ── Spectral scoring ───────────────────────────────────────────────

    def _score_spectrum(self, signal: np.ndarray, fps: float) -> dict:
        """
        FFT the signal, look at power inside the heart-rate band, measure
        peak prominence vs in-band median noise. Real face → sharp peak.
        Synthesized → flat or scattered.
        """
        # Resample to a fixed sample rate so band indices are predictable
        target_len = int(len(signal) * self.SAMPLE_FPS / fps)
        if target_len < self.MIN_FRAMES:
            target_len = self.MIN_FRAMES
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, target_len)
        resampled = np.interp(x_new, x_old, signal)

        # Detrend (remove slow drift / lighting changes)
        detrended = resampled - cv2.GaussianBlur(
            resampled.reshape(-1, 1).astype(np.float32),
            (1, max(5, int(self.SAMPLE_FPS * 0.5)) | 1),  # ~0.5s smoothing
            0,
        ).ravel()

        # FFT
        spectrum = np.abs(np.fft.rfft(detrended))
        freqs = np.fft.rfftfreq(len(detrended), d=1.0 / self.SAMPLE_FPS)

        band_mask = (freqs >= self.BAND_HZ[0]) & (freqs <= self.BAND_HZ[1])
        if not band_mask.any():
            return {
                "rppg_score": 0.5,
                "verdict": "INCONCLUSIVE",
                "reason": "no_band_bins",
            }

        band_power = spectrum[band_mask]
        if len(band_power) < 4:
            return {
                "rppg_score": 0.5,
                "verdict": "INCONCLUSIVE",
                "reason": "too_few_bins",
            }

        peak_idx = int(np.argmax(band_power))
        peak_freq = float(freqs[band_mask][peak_idx])
        peak_val = float(band_power[peak_idx])
        median_val = float(np.median(band_power))
        # Prominence: peak / median. Real faces give 3-10x; synthesized give 1-2x.
        prominence = peak_val / (median_val + 1e-9)

        bpm = peak_freq * 60.0

        # Convert prominence → fake_score (HIGH prominence = REAL = low fake)
        # Empirical breakpoints; calibrate against a dataset to tighten.
        if prominence > 4.0:
            fake_score = float(np.clip((6.0 - prominence) / 8.0, 0.0, 0.3))
        elif prominence > 2.0:
            fake_score = float(np.clip((4.0 - prominence) / 2.0, 0.0, 0.7))
        else:
            fake_score = float(np.clip(0.7 + (2.0 - prominence) * 0.15, 0.7, 1.0))

        verdict = "SUSPICIOUS" if fake_score > 0.6 else (
            "INCONCLUSIVE" if fake_score > 0.4 else "PULSE_DETECTED"
        )

        return {
            "rppg_score": round(fake_score, 4),
            "estimated_bpm": round(bpm, 1),
            "peak_prominence": round(prominence, 2),
            "verdict": verdict,
            "frames_analyzed": int(len(signal)),
        }
