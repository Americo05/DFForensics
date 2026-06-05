"""
Plugin: Edge Blending Boundary Detector

Technique:
    Deepfake face-swap algorithms paste a synthesized face onto the original
    frame, creating a blending boundary where the two regions meet. This
    boundary often exhibits:
      - Unnatural gradient transitions (Sobel edge discontinuities)
      - Color channel shifts in HSV (hue/saturation jumps)
      - Illumination inconsistencies (lighting direction mismatch)

    This plugin analyzes the peripheral ring (~15% outer border) of the
    face ROI where blending artifacts are most concentrated.

    Based on: "Face X-ray for More General Face Forgery Detection" (Li et al.,
    CVPR 2020) and the forensic document's discussion of "artefactos de
    blending em bordas de máscaras e discrepâncias de iluminação facial."

Pre-Processor contract:
    Requires both frame and face_roi from FacePreProcessor.
    Returns neutral 0.5 if either is missing.

Dependencies:
    pip install opencv-python numpy

Version: 1.0.0
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


class EdgeBlendingDetector(BaseDetectorPlugin):

    @property
    def plugin_name(self) -> str:
        return PluginNames.EDGE_BLENDING

    @property
    def plugin_description(self) -> str:
        return (
            "Analyzes the peripheral boundary of the face ROI for blending "
            "artifacts: gradient discontinuities, color channel shifts, and "
            "illumination inconsistencies that occur when a synthesized face "
            "is pasted onto the original frame. Based on Face X-ray concepts."
        )

    @property
    def plugin_version(self) -> str:
        return "1.0.0"

    @property
    def plugin_weight(self) -> float:
        return 0.15

    def analyze_frame(self, frame: np.ndarray, face_roi: np.ndarray | None = None) -> float:
        if frame is None or frame.size == 0:
            return 0.5
        if face_roi is None or face_roi.size == 0:
            return 0.5

        try:
            return self._blending_score(face_roi)
        except Exception as e:
            logger.error(f"Edge blending analysis failed: {e}")
            return 0.5

    def _blending_score(self, face: np.ndarray) -> float:
        h, w = face.shape[:2]
        if h < 32 or w < 32:
            return 0.5

        s1 = self._gradient_discontinuity_score(face)
        s2 = self._color_boundary_score(face)
        s3 = self._illumination_consistency_score(face)

        combined = 0.40 * s1 + 0.35 * s2 + 0.25 * s3
        return round(float(np.clip(combined, 0.0, 1.0)), 4)

    def _gradient_discontinuity_score(self, face: np.ndarray) -> float:
        """
        Compares edge gradient magnitude in the peripheral ring vs. the center.
        Blending creates unnatural edge transitions at the face boundary.
        """
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        border = max(int(min(h, w) * 0.15), 4)

        # Compute Sobel edge magnitude
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)

        # Create masks for peripheral ring and center
        mask_outer = np.zeros_like(gray, dtype=bool)
        mask_outer[:border, :] = True
        mask_outer[-border:, :] = True
        mask_outer[:, :border] = True
        mask_outer[:, -border:] = True

        mask_inner = np.zeros_like(gray, dtype=bool)
        inner_b = border * 2
        if inner_b < h and inner_b < w:
            mask_inner[inner_b:-inner_b, inner_b:-inner_b] = True
        else:
            return 0.5

        outer_mean = float(mag[mask_outer].mean()) if mask_outer.sum() > 0 else 0
        inner_mean = float(mag[mask_inner].mean()) if mask_inner.sum() > 0 else 0

        if inner_mean < 1e-6:
            return 0.5

        # Ratio of peripheral edges to center edges
        ratio = outer_mean / (inner_mean + 1e-6)

        # Natural images: ratio ~0.7-1.4 (edges everywhere)
        # Blended faces: ratio > 1.8 (strong edges at boundary)
        # Or ratio < 0.4 (over-blurred boundary)
        if ratio > 1.6:
            return float(np.clip((ratio - 1.6) / 1.5, 0.0, 1.0))
        elif ratio < 0.4:
            return float(np.clip((0.4 - ratio) / 0.4, 0.0, 0.8))
        else:
            return 0.1

    def _color_boundary_score(self, face: np.ndarray) -> float:
        """
        Measures Hue/Saturation shift between the outer ring and inner core.
        Face-swaps often have mismatched skin tones at the blend boundary.
        """
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_full, w_full = hsv.shape[:2]
        border = max(int(min(h_full, w_full) * 0.15), 4)

        # Outer ring Hue/Saturation
        outer_hue_vals = []
        outer_sat_vals = []

        # Top strip
        outer_hue_vals.append(hsv[:border, :, 0].ravel())
        outer_sat_vals.append(hsv[:border, :, 1].ravel())
        # Bottom strip
        outer_hue_vals.append(hsv[-border:, :, 0].ravel())
        outer_sat_vals.append(hsv[-border:, :, 1].ravel())
        # Left strip
        outer_hue_vals.append(hsv[border:-border, :border, 0].ravel())
        outer_sat_vals.append(hsv[border:-border, :border, 1].ravel())
        # Right strip
        outer_hue_vals.append(hsv[border:-border, -border:, 0].ravel())
        outer_sat_vals.append(hsv[border:-border, -border:, 1].ravel())

        outer_hue = np.concatenate(outer_hue_vals)
        outer_sat = np.concatenate(outer_sat_vals)

        # Inner core
        inner_b = border * 2
        if inner_b >= h_full or inner_b >= w_full:
            return 0.5
        inner_hue = hsv[inner_b:-inner_b, inner_b:-inner_b, 0].ravel()
        inner_sat = hsv[inner_b:-inner_b, inner_b:-inner_b, 1].ravel()

        if len(outer_hue) == 0 or len(inner_hue) == 0:
            return 0.5

        # Hue difference (circular — hue wraps at 180 in OpenCV)
        hue_diff = abs(float(np.median(outer_hue)) - float(np.median(inner_hue)))
        hue_diff = min(hue_diff, 180 - hue_diff)  # Handle circular wrap

        # Saturation difference
        sat_diff = abs(float(np.mean(outer_sat)) - float(np.mean(inner_sat)))

        # Natural: hue_diff < 5, sat_diff < 15
        hue_score = float(np.clip((hue_diff - 4) / 15, 0.0, 1.0))
        sat_score = float(np.clip((sat_diff - 12) / 30, 0.0, 1.0))

        return 0.6 * hue_score + 0.4 * sat_score

    def _illumination_consistency_score(self, face: np.ndarray) -> float:
        """
        Checks if the illumination gradient direction is consistent across
        the face. In natural images, light comes from one direction.
        In blended faces, the pasted face may have different lighting.
        """
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        # Divide face into 4 quadrants, compute mean brightness
        mid_h, mid_w = h // 2, w // 2
        q_tl = float(gray[:mid_h, :mid_w].mean())
        q_tr = float(gray[:mid_h, mid_w:].mean())
        q_bl = float(gray[mid_h:, :mid_w].mean())
        q_br = float(gray[mid_h:, mid_w:].mean())

        # Compute horizontal and vertical gradients
        h_grad_top = q_tr - q_tl
        h_grad_bot = q_br - q_bl
        v_grad_left = q_bl - q_tl
        v_grad_right = q_br - q_tr

        # In natural lighting, gradients should be consistent
        h_inconsistency = abs(h_grad_top - h_grad_bot)
        v_inconsistency = abs(v_grad_left - v_grad_right)

        total_inconsistency = (h_inconsistency + v_inconsistency) / 2.0

        # Normalize by overall brightness range to be scale-independent
        brightness_range = max(abs(q_tl - q_br), abs(q_tr - q_bl), 1.0)
        normalized = total_inconsistency / brightness_range

        # Natural: normalized < 0.3
        # Inconsistent lighting (blended): normalized > 0.5
        return float(np.clip((normalized - 0.25) / 0.5, 0.0, 1.0))
