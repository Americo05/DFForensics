"""
Plugin: Sightengine Cloud Deepfake Detector

Technique:
    Uses the Sightengine commercial API for deepfake and AI-generated
    content detection. Cloud-based detector with state-of-the-art accuracy.

    Two models are called in a single request:
      - deepfake: detects face swaps, reenactments, lip-sync manipulations
      - genai:    detects fully AI-generated images (GANs, diffusion, etc.)

    Free tier: 2000 operations/month (deepfake costs 5 ops each = ~400 checks)

    Configuration: API keys are loaded from engine/.env file (never committed).
    Toggle: Set SIGHTENGINE_ENABLED=true/false in .env to enable/disable.

    API Keys: Sign up at https://dashboard.sightengine.com/signup

Version: 1.0.0
"""

import cv2
import numpy as np
import requests
import logging
import sys
import os

# Load .env file from engine/ directory
from pathlib import Path
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed — fall back to os.environ

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.plugin_base import BaseDetectorPlugin
from core.plugin_names import PluginNames

logger = logging.getLogger(__name__)

# ── Read from .env (engine/.env) — NEVER hardcode keys here ──────────────────
SIGHTENGINE_API_USER   = os.environ.get("SIGHTENGINE_API_USER", "")
SIGHTENGINE_API_SECRET = os.environ.get("SIGHTENGINE_API_SECRET", "")
SIGHTENGINE_ENABLED    = os.environ.get("SIGHTENGINE_ENABLED", "true").lower() == "true"

API_URL = "https://api.sightengine.com/1.0/check.json"

# Only call the API every N frames to conserve free tier quota.
CALL_EVERY_N_FRAMES = 10


class SightengineDeepfakeDetector(BaseDetectorPlugin):

    def __init__(self):
        self._frame_counter = 0
        self._cached_score = 0.5
        self._api_configured = bool(
            SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET and SIGHTENGINE_ENABLED
        )
        if not SIGHTENGINE_ENABLED:
            logger.info("☁️ Sightengine Cloud Detector: DISABLED by config (SIGHTENGINE_ENABLED=false)")
        elif self._api_configured:
            logger.info("✅ Sightengine Cloud Detector: API keys loaded from .env")
        else:
            logger.warning("⚠️  Sightengine Cloud Detector: API keys NOT found in .env")

    def is_configured(self) -> bool:
        return self._api_configured

    @property
    def plugin_name(self) -> str:
        return PluginNames.SIGHTENGINE_CLOUD

    @property
    def plugin_description(self) -> str:
        return (
            "Cloud-based deepfake and AI-generated content detection via "
            "Sightengine API. Uses proprietary multi-model analysis for "
            "face swaps, reenactments, lip-sync, and fully AI-generated "
            "images. Free tier: ~400 deepfake checks/month."
        )

    @property
    def plugin_version(self) -> str:
        return "1.0.0"

    @property
    def plugin_weight(self) -> float:
        return 0.40

    def analyze_frame(self, frame: np.ndarray, face_roi: np.ndarray | None = None) -> float:
        if not self._api_configured:
            return 0.5

        if frame is None or frame.size == 0:
            return 0.5

        self._frame_counter += 1

        # Rate limit: only call API every N frames, reuse cached score otherwise
        if self._frame_counter % CALL_EVERY_N_FRAMES != 1 and self._frame_counter > 1:
            return self._cached_score

        try:
            score = self._cloud_score(frame)
            self._cached_score = score
            return score
        except Exception as e:
            logger.error(f"Sightengine API error: {e}")
            return self._cached_score

    def _cloud_score(self, frame: np.ndarray) -> float:
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            return 0.5

        files = {"media": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
        params = {
            "models": "deepfake,genai",
            "api_user": SIGHTENGINE_API_USER,
            "api_secret": SIGHTENGINE_API_SECRET,
        }

        response = requests.post(API_URL, files=files, data=params, timeout=15)
        result = response.json()

        if result.get("status") != "success":
            error_msg = result.get("error", {}).get("message", "Unknown error")
            logger.warning(f"Sightengine API error: {error_msg}")
            return 0.5

        # Parse scores
        deepfake_score = float(result.get("deepfake", {}).get("score", 0.0))
        genai_score = float(result.get("type", {}).get("ai_generated", 0.0))
        combined = max(deepfake_score, genai_score)

        logger.info(f"☁️ Sightengine: deepfake={deepfake_score:.3f}, "
                    f"ai_generated={genai_score:.3f} → {combined:.3f}")

        return round(float(np.clip(combined, 0.0, 1.0)), 4)

    def reset(self):
        self._frame_counter = 0
        self._cached_score = 0.5
