"""
SceneClassifier — determines the content type of each frame to route
analysis to the appropriate detectors.

Scene Types
───────────
CROPPED_FACE
    A face fills the majority of the image (>50% of total area).
    Typical for: pre-cropped face datasets, profile-picture-style inputs.
    Active plugins (in order of weight): MesoNet > Sightengine > ViT > DCT
                                       > Edge Blending > PRNU.

FACE_IN_SCENE
    A face was detected but occupies ≤50% of the frame.
    Typical for: standard video frames, interviews, social media videos.
    Same plugin set as CROPPED_FACE; the per-scene weight table tweaks
    Edge Blending/PRNU slightly upwards because they benefit from having
    a real background to compare against.

NO_FACE
    No face was detected by the face detector.
    Typical for: AI-generated landscapes, objects, non-frontal shots.
    Active plugins: Sightengine + DCT only.
    MesoNet, ViT, Edge Blending, PRNU are all face-conditional and
    therefore SKIPPED here:
      • MesoNet + ViT are face-crop classifiers — applying them to a
        landscape produces unreliable scores.
      • Edge Blending + PRNU need a face boundary to compute the
        center-vs-periphery / face-vs-background ratios.

Weight calibration history
──────────────────────────
Initial weights (v1) gave PRNU and Edge Blending ~0.15 each on the
hypothesis that they would catch face-swap blending artefacts. Empirical
benchmark on Celeb-DF v2 (N=200) and FF++ (N=40) showed both plugins at
AUC ≤ 0.45 — they were actively diluting the signal. v2 weights (current)
downshift them to 0.05–0.09 and promote the newly added MesoNet to the
dominant 0.38–0.40 slot. The heuristic plugins remain in the ensemble
because they catch failure modes neural detectors don't (non-NN
manipulations, simple swaps) — just with reduced say in the final score.
"""

from enum import Enum
import numpy as np

from core.plugin_names import PluginNames


class SceneType(str, Enum):
    CROPPED_FACE  = "CROPPED_FACE"
    FACE_IN_SCENE = "FACE_IN_SCENE"
    NO_FACE       = "NO_FACE"


# ── Per-scene plugin weight tables ────────────────────────────────────────────
# Keys are PluginNames constants — keep them in sync with plugin `plugin_name`
# properties (which also reference PluginNames). Plugins NOT listed for a
# scene type are SKIPPED entirely (not executed). Weights MUST sum to 1.0
# per scene type.
#
# NOTE: In "cloud" mode only Sightengine runs; in "local" mode only the
# plugins below are used. Sightengine weight only applies in "all" mode.

SCENE_PLUGIN_WEIGHTS: dict[str, dict[str, float]] = {
    SceneType.CROPPED_FACE: {
        # MesoNet (Afchar et al., WIFS 2018) carries the heaviest weight here:
        # it's the only purpose-trained deepfake detector in the local plugin
        # set (pre-trained on FF++ Deepfakes). Heuristic plugins (DCT, Edge,
        # PRNU) are downweighted because the empirical benchmark (Celeb-DF v2
        # N=200, FF++ N=40) showed AUC ≤ 0.45 for them on modern face-swap
        # content — they were diluting MesoNet's signal. They stay in the
        # table because they catch failure modes MesoNet doesn't (frequency
        # artefacts from non-NN manipulations, blending artefacts on simpler
        # swaps), but with reduced say in the final score.
        PluginNames.MESONET:           0.40,
        PluginNames.SIGHTENGINE_CLOUD: 0.20,
        PluginNames.VIT_DETECTOR:      0.18,
        PluginNames.DCT_FREQUENCY:     0.10,
        PluginNames.EDGE_BLENDING:     0.07,
        PluginNames.PRNU_NOISE:        0.05,
    },
    SceneType.FACE_IN_SCENE: {
        # Same logic as CROPPED_FACE. PRNU/Edge get a slightly larger slice
        # here because they benefit from a real background to compare against
        # — but still capped well below MesoNet.
        PluginNames.MESONET:           0.38,
        PluginNames.SIGHTENGINE_CLOUD: 0.18,
        PluginNames.VIT_DETECTOR:      0.18,
        PluginNames.DCT_FREQUENCY:     0.10,
        PluginNames.EDGE_BLENDING:     0.09,
        PluginNames.PRNU_NOISE:        0.07,
    },
    SceneType.NO_FACE: {
        # MesoNet, ViT, Edge Blending, PRNU all need a face → SKIPPED here.
        # Cloud + frequency stay as the only signals for fully synthetic scenes
        # (AI-generated landscapes, objects, non-frontal shots).
        PluginNames.SIGHTENGINE_CLOUD: 0.50,
        PluginNames.DCT_FREQUENCY:     0.50,
    },
}


class SceneClassifier:
    """
    Classifies a video frame into a SceneType based on face detection output.

    This runs AFTER FacePreProcessor (which already ran detection once).
    No additional ML model is needed — classification is pure geometry.
    """

    # If face_area / frame_area exceeds this, we treat it as a tight face crop.
    CROPPED_FACE_RATIO_THRESHOLD = 0.50

    @staticmethod
    def classify(
        frame: np.ndarray,
        face_bbox: dict | None,
        face_actually_detected: bool,
    ) -> SceneType:
        """
        Parameters
        ──────────
        frame                  Full BGR frame from the video/image.
        face_bbox              Dict {x, y, w, h} from FacePreProcessor, or None.
                               When FacePreProcessor uses the fallback (no face
                               found by OpenCV), face_actually_detected is False.
        face_actually_detected True  → face detector found the face.
                               False → fallback bbox covering the whole frame.

        Returns
        ───────
        SceneType enum value.
        """
        if not face_actually_detected or face_bbox is None:
            return SceneType.NO_FACE

        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            return SceneType.NO_FACE

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w
        if frame_area <= 0:
            return SceneType.NO_FACE

        # Validate bbox shape — malformed entries shouldn't crash the pipeline
        try:
            w = float(face_bbox["w"])
            h = float(face_bbox["h"])
        except (KeyError, TypeError, ValueError):
            return SceneType.NO_FACE

        if w <= 0 or h <= 0:
            return SceneType.NO_FACE

        face_ratio = (w * h) / frame_area

        if face_ratio >= SceneClassifier.CROPPED_FACE_RATIO_THRESHOLD:
            return SceneType.CROPPED_FACE
        else:
            return SceneType.FACE_IN_SCENE

    @staticmethod
    def get_active_plugins_and_weights(
        scene_type: SceneType,
        all_plugins: list,
    ) -> list[tuple]:
        """
        Returns (plugin, weight) pairs for plugins that should run for the
        given scene type. Plugins not in the weight table are SKIPPED.

        Parameters
        ──────────
        scene_type   The classified scene type for this frame.
        all_plugins  List of loaded BaseDetectorPlugin instances.

        Returns
        ───────
        List of (plugin, weight) tuples — only active plugins.
        """
        weight_table = SCENE_PLUGIN_WEIGHTS.get(scene_type, {})
        active = []
        for plugin in all_plugins:
            w = weight_table.get(plugin.plugin_name)
            if w is not None and w > 0.0:
                active.append((plugin, w))
        return active
