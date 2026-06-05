"""
Plugin: Deepfake Detector — Fine-tuned ViT (dima806/deepfake_vs_real_image_detection)

Technique:
    Uses a Vision Transformer fine-tuned on 140K+ real and deepfake/GAN face
    images, achieving 99.27% accuracy. The model is specifically designed to
    classify face crops as "Fake" or "Real".

    For face-swap deepfakes (FaceForensics++, etc.), only the face region is
    manipulated — so we analyze the face_roi to focus on the swapped area.
    If no face ROI is available, falls back to the full frame (useful for fully
    AI-generated videos like Sora/Runway where the whole scene is synthetic).

Pre-Processor contract:
    The PluginManager's FacePreProcessor detects the face ONCE per frame
    and passes the cropped face_roi here via analyze_frame(frame, face_roi).

Model labels: "Fake" → fake probability, "Real" → real (inverted)

Dependencies:
    pip install torch torchvision transformers pillow

Version: 3.0.0
"""

import cv2
import numpy as np
import sys
import os
import logging

try:
    from PIL import Image
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.plugin_base import BaseDetectorPlugin
from core.plugin_names import PluginNames

logger = logging.getLogger(__name__)


class ViTDetectorPlugin(BaseDetectorPlugin):

    # Enables batched inference path in PluginManager. The HF image-classification
    # pipeline accepts a list and runs the transformer with batch_size>1, which
    # is materially faster on GPU and still helps on CPU by amortizing overhead.
    SUPPORTS_BATCH = True

    # Empirical score-range recalibration.
    # The dima806 model is over-confident: on the Celeb-DF v2 benchmark (N=200)
    # raw fake-probability outputs concentrate in [0.50, 1.00] for both real and
    # fake images, producing F1 ≈ 0.66 — the "everything is fake" pattern. The
    # residual separation in that compressed range gets diluted by the ensemble's
    # weighted mean with broader-range plugins.
    #
    # We stretch [RAW_LOW, RAW_HIGH] → [0.0, 1.0] linearly. Since the transform
    # is strictly monotone, AUC is unchanged (rank order preserved), but the
    # plugin now uses the full [0, 1] range so the production threshold (0.6)
    # becomes a meaningful decision boundary instead of "always fake".
    # To disable, set RAW_LOW=0.0 and RAW_HIGH=1.0.
    RAW_LOW: float = 0.50
    RAW_HIGH: float = 1.00

    @classmethod
    def _calibrate(cls, raw: float) -> float:
        """Stretch [RAW_LOW, RAW_HIGH] → [0, 1]. Monotone, so AUC is preserved."""
        span = cls.RAW_HIGH - cls.RAW_LOW
        if span <= 0:
            return raw
        return max(0.0, min(1.0, (raw - cls.RAW_LOW) / span))

    @property
    def plugin_name(self) -> str:
        return PluginNames.VIT_DETECTOR

    @property
    def plugin_description(self) -> str:
        return (
            "Fine-tuned Vision Transformer (dima806/deepfake_vs_real_image_detection) "
            "trained on 140K+ real/fake face images with 99.27% accuracy. "
            "Analyzes the face ROI for face-swap detection; falls back to the full "
            "frame for fully AI-generated content."
        )

    @property
    def plugin_version(self) -> str:
        return "3.0.0"

    @property
    def plugin_weight(self) -> float:
        return 0.60  # High confidence — purpose-built deepfake model

    def __init__(self):
        self._pipe = None
        self._load_error: str | None = None
        if not TRANSFORMERS_AVAILABLE:
            self._load_error = "transformers/torch/pillow not installed"
            logger.error(f"{self._load_error}. Run: pip install torch transformers pillow")
            return

        model_name = "dima806/deepfake_vs_real_image_detection"
        logger.info(f"Loading deepfake ViT model: {model_name}...")

        device = 0 if torch.cuda.is_available() else -1
        try:
            self._pipe = pipeline("image-classification", model=model_name, device=device)
            logger.info(f"✅ Deepfake ViT model loaded (device={'GPU' if device == 0 else 'CPU'})")
        except Exception as e:
            self._load_error = f"model load failed: {e}"
            logger.error(f"Failed to load deepfake ViT model: {e}")
            self._pipe = None

    def is_configured(self) -> bool:
        return self._pipe is not None

    def analyze_frame(self, frame: np.ndarray, face_roi: np.ndarray | None = None) -> float:
        if self._pipe is None:
            return 0.5

        if frame is None or frame.size == 0:
            return 0.5

        # For face-swap deepfakes: analyze the face region (only the face is manipulated).
        # For fully AI-generated content: fall back to the full frame.
        target = face_roi if (face_roi is not None and face_roi.size > 0) else frame

        try:
            rgb = self._to_rgb(target)
            if rgb is None:
                return 0.5
            pil_img = Image.fromarray(rgb)

            results = self._pipe(pil_img)

            if not isinstance(results, list) or len(results) == 0:
                logger.warning("ViT pipeline returned an empty or unexpected result.")
                return 0.5

            # dima806 model labels: "Fake" and "Real"
            for item in results:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "")).upper()
                try:
                    score = float(item.get("score", 0.5))
                except (TypeError, ValueError):
                    continue

                if "FAKE" in label or "DEEPFAKE" in label or label == "LABEL_1":
                    return round(self._calibrate(score), 4)
                if "REAL" in label or label == "LABEL_0":
                    return round(self._calibrate(1.0 - score), 4)

            logger.warning(
                f"Unknown labels from ViT model: {[r.get('label') for r in results if isinstance(r, dict)]}"
            )
            first = results[0] if isinstance(results[0], dict) else {}
            return round(self._calibrate(float(first.get("score", 0.5))), 4)

        except Exception as e:
            logger.error(f"Deepfake ViT analysis failed: {e}")
            return 0.5

    @staticmethod
    def _to_rgb(img: np.ndarray) -> np.ndarray | None:
        """Normalize input to a 3-channel RGB array regardless of source layout."""
        if img is None or img.size == 0:
            return None
        if img.ndim == 2:
            # Grayscale → RGB
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.ndim == 3:
            ch = img.shape[2]
            if ch == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if ch == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            if ch == 1:
                return cv2.cvtColor(img.squeeze(-1), cv2.COLOR_GRAY2RGB)
        return None

    # ── Batched inference path ─────────────────────────────────────────────
    def analyze_frames_batch(
        self, items: list[tuple[np.ndarray, np.ndarray | None]]
    ) -> list[float]:
        """
        Score multiple (frame, face_roi) pairs in a single pipeline call.

        The HF image-classification pipeline accepts a list of PIL images and
        runs the transformer with batch_size>1, which is materially faster on
        GPU and still saves dispatch overhead on CPU. We preserve input order
        and fall back to neutral 0.5 for any item whose preprocessing fails.
        """
        if self._pipe is None or not items:
            return [0.5] * len(items)

        # Build PIL inputs preserving order; remember which slots are valid.
        pil_inputs: list[Image.Image | None] = []
        for frame, face_roi in items:
            target = face_roi if (face_roi is not None and face_roi.size > 0) else frame
            if target is None or target.size == 0:
                pil_inputs.append(None)
                continue
            rgb = self._to_rgb(target)
            if rgb is None:
                pil_inputs.append(None)
                continue
            try:
                pil_inputs.append(Image.fromarray(rgb))
            except Exception:
                pil_inputs.append(None)

        valid_idx = [i for i, p in enumerate(pil_inputs) if p is not None]
        if not valid_idx:
            return [0.5] * len(items)

        valid_pils = [pil_inputs[i] for i in valid_idx]
        try:
            # batch_size=len(valid_pils) tells HF to batch them in one forward pass
            batch_results = self._pipe(valid_pils, batch_size=len(valid_pils))
        except Exception as e:
            logger.warning(f"ViT batch inference failed, falling back to per-item: {e}")
            return [self.analyze_frame(frame, face_roi=roi) for frame, roi in items]

        # HF returns a list where each element is itself a list of {label, score}
        scores = [0.5] * len(items)
        for slot, classification in zip(valid_idx, batch_results):
            scores[slot] = self._extract_fake_score(classification)
        return scores

    def _extract_fake_score(self, classification) -> float:
        """Pull the fake probability out of one HF classification result."""
        if not isinstance(classification, list) or not classification:
            return 0.5
        for item in classification:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).upper()
            try:
                score = float(item.get("score", 0.5))
            except (TypeError, ValueError):
                continue
            if "FAKE" in label or "DEEPFAKE" in label or label == "LABEL_1":
                return round(self._calibrate(score), 4)
            if "REAL" in label or label == "LABEL_0":
                return round(self._calibrate(1.0 - score), 4)
        first = classification[0] if isinstance(classification[0], dict) else {}
        return round(self._calibrate(float(first.get("score", 0.5))), 4)
