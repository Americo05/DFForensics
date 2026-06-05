"""
Plugin: MesoNet — compact CNN for deepfake detection.

Technique
---------
Afchar et al., "MesoNet: a Compact Facial Video Forgery Detection Network"
(IEEE WIFS 2018). A purpose-built mesoscopic CNN with only ~30k parameters
that beat much larger general-purpose classifiers on the FaceForensics++
benchmark by focusing on mid-level facial features (the mesoscopic scale
between micro-textures and global semantics, where face-swap algorithms
typically leave the most visible artefacts).

Architecture (Meso-4 variant)
-----------------------------
Input:  256×256×3 RGB face crop, pixels in [0, 1]
Conv2D(8,  3×3)  + ReLU + BN + MaxPool(2)  → 128×128×8
Conv2D(8,  5×5)  + ReLU + BN + MaxPool(2)  →  64×64×8
Conv2D(16, 5×5)  + ReLU + BN + MaxPool(2)  →  32×32×16
Conv2D(16, 5×5)  + ReLU + BN + MaxPool(4)  →   8×8×16
Flatten + Dropout(0.5)
Dense(16) + LeakyReLU(0.1)
Dropout(0.5)
Dense(1)  + Sigmoid   → output ∈ [0, 1]

The original Keras output represents the REAL probability (label encoding
in the paper: 1 = pristine, 0 = forged). We invert it so this plugin
returns the FAKE probability, matching the rest of the engine.

Weights
-------
Pre-trained weights are not bundled (license + size considerations).
Run `python engine/scripts/download_mesonet_weights.py` to download the
official Meso4_DF.h5 from the paper authors' GitHub and convert to a
PyTorch state_dict at `engine/models/mesonet_meso4_df.pt`. Without that
file the plugin loads but reports `is_configured() == False` and returns
neutral 0.5 from `analyze_frame` — same pattern as Sightengine without
an API key.

Pre-processor contract
----------------------
Uses face_roi if available; falls back to full frame. Resizes to 256×256.
The original paper crops slightly tighter than MTCNN by default — close
enough that the empirical hit is small. If we ever do per-frame face-
alignment we should expand the crop margin to match the FaceForensics++
preprocessing pipeline.

Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.plugin_base import BaseDetectorPlugin
from core.plugin_names import PluginNames

logger = logging.getLogger(__name__)


# Path conventions (relative to engine/) — the download script writes here.
_DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "mesonet_meso4_df.pt"
)


# ── Architecture ──────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class _Meso4(nn.Module):
        """PyTorch port of the Meso-4 architecture from the original paper."""

        def __init__(self) -> None:
            super().__init__()
            # 4 conv blocks: each is Conv → ReLU → BN → MaxPool.
            # The original Keras model uses BN AFTER activation; mirror that
            # ordering so the converted weights load with matching statistics.
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(8)
            self.pool1 = nn.MaxPool2d(2)

            self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm2d(8)
            self.pool2 = nn.MaxPool2d(2)

            self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
            self.bn3 = nn.BatchNorm2d(16)
            self.pool3 = nn.MaxPool2d(2)

            self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
            self.bn4 = nn.BatchNorm2d(16)
            self.pool4 = nn.MaxPool2d(4)

            # 16 channels × 8×8 spatial = 1024 features
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(16 * 8 * 8, 16)
            self.fc2 = nn.Linear(16, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.pool1(self.bn1(F.relu(self.conv1(x))))
            x = self.pool2(self.bn2(F.relu(self.conv2(x))))
            x = self.pool3(self.bn3(F.relu(self.conv3(x))))
            x = self.pool4(self.bn4(F.relu(self.conv4(x))))
            # Keras flattens NHWC (channels last) — channels iterate fastest.
            # PyTorch's native NCHW.flatten iterates spatial fastest, producing
            # a DIFFERENT permutation of the same 1024 values. fc1's pre-trained
            # weights were learned on Keras's order, so we must permute to NHWC
            # before flatten to align with the loaded dense-layer weights.
            # Forgetting this produces a model that runs and outputs plausible
            # ~0.5 scores but is essentially random — the dense layer sees
            # shuffled features.
            x = x.permute(0, 2, 3, 1).contiguous()  # NCHW → NHWC
            x = x.flatten(start_dim=1)
            x = self.dropout(x)
            x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
            x = self.dropout(x)
            x = torch.sigmoid(self.fc2(x))
            return x.squeeze(-1)


# ── Plugin ────────────────────────────────────────────────────────────────


class MesoNetDetectorPlugin(BaseDetectorPlugin):

    # Batched inference is meaningful here: the network is tiny (~30k params)
    # and the dominant cost per call is the Python/Torch dispatch overhead,
    # not the actual convolutions. Batching 8 faces in one forward pass is
    # ~5× faster than 8 sequential calls on CPU.
    SUPPORTS_BATCH = True

    INPUT_SIZE: int = 256  # paper input size (do not change without retraining)

    @property
    def plugin_name(self) -> str:
        return PluginNames.MESONET

    @property
    def plugin_description(self) -> str:
        return (
            "Compact mesoscopic CNN (Afchar et al., WIFS 2018) purpose-built "
            "for face-swap deepfake detection. ~30k parameters; trained on "
            "FaceForensics++ Deepfakes. Strong in-distribution performance; "
            "fallback to neutral 0.5 if the weights file is missing."
        )

    @property
    def plugin_version(self) -> str:
        return "1.0.0"

    @property
    def plugin_weight(self) -> float:
        # Authoritative weights live in scene_classifier.SCENE_PLUGIN_WEIGHTS;
        # this is just metadata for plugins that consult the per-plugin weight
        # (currently none do). Kept high to signal this is the strongest plugin.
        return 0.40

    def __init__(self, weights_path: str | os.PathLike | None = None) -> None:
        self._model = None
        self._device = None
        self._weights_path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS_PATH
        self._load_error: str | None = None

        if not TORCH_AVAILABLE:
            self._load_error = "torch not installed"
            logger.error("MesoNet: torch not available — install pytorch")
            return

        if not self._weights_path.is_file():
            self._load_error = f"weights file missing at {self._weights_path}"
            logger.warning(
                f"⚠️  MesoNet: weights not found at {self._weights_path}. "
                "Run: python engine/scripts/download_mesonet_weights.py"
            )
            return

        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = _Meso4().to(self._device)
            state = torch.load(self._weights_path, map_location=self._device)
            # Allow both bare state_dicts and {"state_dict": ...} wrappers
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self._model.load_state_dict(state, strict=True)
            self._model.eval()
            device_str = "GPU" if self._device.type == "cuda" else "CPU"
            logger.info(f"✅ MesoNet loaded from {self._weights_path.name} (device={device_str})")
        except Exception as e:
            self._load_error = f"model load failed: {e}"
            logger.error(f"MesoNet: failed to load weights — {e}")
            self._model = None

    def is_configured(self) -> bool:
        return self._model is not None

    # ── Preprocessing ───────────────────────────────────────────────────

    @classmethod
    def _preprocess(cls, img: np.ndarray) -> np.ndarray | None:
        """BGR → RGB → 256×256 → float32 [0, 1] → CHW."""
        if img is None or img.size == 0:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            return None
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (cls.INPUT_SIZE, cls.INPUT_SIZE))
        arr = resized.astype(np.float32) / 255.0
        # HWC → CHW for PyTorch
        return np.transpose(arr, (2, 0, 1))

    # ── Single-frame path ───────────────────────────────────────────────

    def analyze_frame(self, frame: np.ndarray, face_roi: np.ndarray | None = None) -> float:
        if self._model is None:
            return 0.5
        target = face_roi if (face_roi is not None and face_roi.size > 0) else frame
        pre = self._preprocess(target)
        if pre is None:
            return 0.5
        try:
            with torch.no_grad():
                x = torch.from_numpy(pre).unsqueeze(0).to(self._device)
                real_prob = float(self._model(x).item())
            # Network output = P(real); we report P(fake) for consistency.
            return round(max(0.0, min(1.0, 1.0 - real_prob)), 4)
        except Exception as e:
            logger.error(f"MesoNet inference failed: {e}")
            return 0.5

    # ── Batched path ────────────────────────────────────────────────────

    def analyze_frames_batch(
        self, items: "list[tuple[np.ndarray, np.ndarray | None]]"
    ) -> "list[float]":
        if self._model is None or not items:
            return [0.5] * len(items)

        preps: list[np.ndarray | None] = []
        for frame, face_roi in items:
            target = face_roi if (face_roi is not None and face_roi.size > 0) else frame
            preps.append(self._preprocess(target))

        valid_idx = [i for i, p in enumerate(preps) if p is not None]
        if not valid_idx:
            return [0.5] * len(items)

        try:
            batch_np = np.stack([preps[i] for i in valid_idx], axis=0)
            with torch.no_grad():
                x = torch.from_numpy(batch_np).to(self._device)
                real_probs = self._model(x).detach().cpu().numpy()
        except Exception as e:
            logger.warning(f"MesoNet batch inference failed, falling back to per-item: {e}")
            return [self.analyze_frame(frame, face_roi=roi) for frame, roi in items]

        scores = [0.5] * len(items)
        for slot, rp in zip(valid_idx, real_probs):
            scores[slot] = round(max(0.0, min(1.0, 1.0 - float(rp))), 4)
        return scores
