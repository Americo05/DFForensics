"""Unit tests for SceneClassifier — pure geometry, no models needed."""

import numpy as np
import pytest

from core.scene_classifier import SceneClassifier, SceneType, SCENE_PLUGIN_WEIGHTS


def _frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestClassify:
    def test_no_face_when_not_detected(self):
        assert SceneClassifier.classify(_frame(), None, False) == SceneType.NO_FACE
        assert SceneClassifier.classify(_frame(), {"x": 0, "y": 0, "w": 100, "h": 100}, False) == SceneType.NO_FACE

    def test_no_face_when_bbox_none(self):
        assert SceneClassifier.classify(_frame(), None, True) == SceneType.NO_FACE

    def test_cropped_face_when_ratio_above_threshold(self):
        # 480x640 frame = 307200 px. Need >= 50% face = >=153600 px.
        bbox = {"x": 0, "y": 0, "w": 500, "h": 400}  # 200000 px = ~65%
        assert SceneClassifier.classify(_frame(), bbox, True) == SceneType.CROPPED_FACE

    def test_face_in_scene_when_ratio_below_threshold(self):
        bbox = {"x": 0, "y": 0, "w": 100, "h": 100}  # 10000 px = ~3%
        assert SceneClassifier.classify(_frame(), bbox, True) == SceneType.FACE_IN_SCENE

    # ── Regression tests for bug #2: malformed bbox should not crash ────
    def test_malformed_bbox_missing_keys(self):
        bbox = {"x": 0, "y": 0}  # no w/h
        assert SceneClassifier.classify(_frame(), bbox, True) == SceneType.NO_FACE

    def test_malformed_bbox_non_numeric(self):
        bbox = {"x": 0, "y": 0, "w": "huge", "h": None}
        assert SceneClassifier.classify(_frame(), bbox, True) == SceneType.NO_FACE

    def test_zero_size_bbox(self):
        bbox = {"x": 10, "y": 10, "w": 0, "h": 0}
        assert SceneClassifier.classify(_frame(), bbox, True) == SceneType.NO_FACE

    def test_zero_frame_area_does_not_crash(self):
        # 0x0 frame
        assert SceneClassifier.classify(np.zeros((0, 0, 3), dtype=np.uint8),
                                        {"x": 0, "y": 0, "w": 1, "h": 1}, True) == SceneType.NO_FACE


class TestActivePluginsAndWeights:
    """Mock plugin objects with just a plugin_name attribute — enough for routing."""

    class _Plug:
        def __init__(self, name):
            self.plugin_name = name

    def test_only_listed_plugins_get_returned(self):
        plugins = [
            self._Plug("Deepfake-Specific ViT Detector"),
            self._Plug("DCT Frequency Analyzer"),
            self._Plug("Sightengine Cloud Detector"),
            self._Plug("Edge Blending Boundary Detector"),
            self._Plug("PRNU Noise Residue Detector"),
        ]
        active = SceneClassifier.get_active_plugins_and_weights(SceneType.NO_FACE, plugins)
        names = {p.plugin_name for p, _ in active}
        # Regression for bug #2 — ViT must NOT be routed in NO_FACE
        assert "Deepfake-Specific ViT Detector" not in names
        # Edge Blending + PRNU both require a face — also excluded from NO_FACE
        assert "Edge Blending Boundary Detector" not in names
        assert "PRNU Noise Residue Detector" not in names
        # The two that remain
        assert names == {"DCT Frequency Analyzer", "Sightengine Cloud Detector"}

    def test_prnu_routed_in_face_scenes(self):
        plugins = [self._Plug("PRNU Noise Residue Detector")]
        for scene in (SceneType.CROPPED_FACE, SceneType.FACE_IN_SCENE):
            active = SceneClassifier.get_active_plugins_and_weights(scene, plugins)
            names = {p.plugin_name for p, _ in active}
            assert "PRNU Noise Residue Detector" in names, f"PRNU missing from {scene}"

    def test_unknown_plugin_skipped_silently(self):
        plugins = [self._Plug("UnknownPlugin")]
        assert SceneClassifier.get_active_plugins_and_weights(SceneType.CROPPED_FACE, plugins) == []


class TestSceneWeightTablesIntegrity:
    """The weight tables are configuration — keep them sane."""

    def test_all_scenes_have_a_weight_table(self):
        for scene in SceneType:
            assert scene in SCENE_PLUGIN_WEIGHTS

    def test_weights_sum_to_one_per_scene(self):
        for scene, weights in SCENE_PLUGIN_WEIGHTS.items():
            total = sum(weights.values())
            assert total == pytest.approx(1.0, abs=1e-3), f"{scene} weights sum to {total}, not 1.0"

    def test_weights_are_positive(self):
        for scene, weights in SCENE_PLUGIN_WEIGHTS.items():
            for name, w in weights.items():
                assert w > 0, f"{scene} → {name} has non-positive weight {w}"
