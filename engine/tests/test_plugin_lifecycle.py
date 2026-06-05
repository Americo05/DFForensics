"""
Regression tests for the plugin lifecycle bugs we fixed:

  - Bug #1: Sightengine state leak — reset() must be called between analyses
  - Bug #2: face_roi=None must propagate when no face is detected
  - Bug #3: dominant-only mode removed; small faces analyzed but filtered out
            of the frame verdict

These tests stub out the heavy plugins/preprocessor so we can verify control
flow without loading any ML model.
"""

import numpy as np
import pytest

from core.plugin_base import BaseDetectorPlugin
from core.plugin_manager import PluginManager
from core.plugin_names import PluginNames
from core.scene_classifier import SceneType


def _frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ── Test doubles ──────────────────────────────────────────────────────────

class _CountingPlugin(BaseDetectorPlugin):
    """A plugin that tracks how many times analyze_frame and reset are called."""

    def __init__(self, name: str, fixed_score: float = 0.5):
        self._name = name
        self._fixed = fixed_score
        self.calls = 0
        self.reset_calls = 0
        self.received_face_roi: list = []

    @property
    def plugin_name(self) -> str: return self._name
    @property
    def plugin_description(self) -> str: return "test plugin"
    @property
    def plugin_version(self) -> str: return "0.0.0"
    @property
    def plugin_weight(self) -> float: return 1.0

    def analyze_frame(self, frame, face_roi=None) -> float:
        self.calls += 1
        self.received_face_roi.append(face_roi)
        return self._fixed

    def reset(self) -> None:
        self.reset_calls += 1


class _StatefulPlugin(_CountingPlugin):
    """Mimics the Sightengine bug: counter persists across analyses unless reset()."""

    def __init__(self, name: str):
        super().__init__(name, fixed_score=0.5)
        self._counter = 0
        self._cache = 0.5

    def analyze_frame(self, frame, face_roi=None) -> float:
        self.calls += 1
        self._counter += 1
        # Returns cache for first 4 calls, fresh value on the 5th — same shape
        # as the real Sightengine rate-limit logic.
        if self._counter % 5 != 1:
            return self._cache
        self._cache = 0.9  # "fresh" suspicious score
        return self._cache

    def reset(self) -> None:
        super().reset()
        self._counter = 0
        self._cache = 0.5


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def manager_with_plugins(monkeypatch):
    """A PluginManager with stubbed plugins and a no-op face preprocessor."""
    pm = PluginManager.__new__(PluginManager)  # skip __init__ (avoids MTCNN load)
    pm._plugins = []
    import threading
    pm._analysis_lock = threading.Lock()
    return pm


# ── Tests ─────────────────────────────────────────────────────────────────

class TestReset:
    """Bug #1: reset() must run on every plugin before each analysis."""

    def test_reset_called_before_each_analysis(self, manager_with_plugins):
        pm = manager_with_plugins
        # Use names listed in CROPPED_FACE so the plugin actually gets executed
        plugin = _CountingPlugin(PluginNames.DCT_FREQUENCY, fixed_score=0.4)
        pm._plugins = [plugin]

        # Inject a fake preprocessor that returns one big face → CROPPED_FACE scene
        class _FakePP:
            def process(self, frame):
                h, w = frame.shape[:2]
                bbox = {"x": 0, "y": 0, "w": int(w * 0.9), "h": int(h * 0.9)}
                return [(frame, bbox)], frame[..., 0], True
        pm._preprocessor = _FakePP()

        # First analysis
        pm.run_analysis([_frame()], fps=30.0, mode="local")
        assert plugin.reset_calls == 1
        # Second analysis
        pm.run_analysis([_frame()], fps=30.0, mode="local")
        assert plugin.reset_calls == 2

    def test_stateful_plugin_does_not_leak_across_analyses(self, manager_with_plugins):
        """Direct regression of the Sightengine bug."""
        pm = manager_with_plugins
        plugin = _StatefulPlugin(PluginNames.DCT_FREQUENCY)
        pm._plugins = [plugin]

        class _FakePP:
            def process(self, frame):
                h, w = frame.shape[:2]
                bbox = {"x": 0, "y": 0, "w": int(w * 0.9), "h": int(h * 0.9)}
                return [(frame, bbox)], frame[..., 0], True
        pm._preprocessor = _FakePP()

        # First analysis: 5 frames. Counter advances 1..5; cache is set on
        # frame 1 (counter%5==1) to 0.9, then re-used for frames 2..5.
        pm.run_analysis([_frame() for _ in range(5)], fps=30.0, mode="local")
        assert plugin._counter == 5, "Counter should have advanced during analysis"
        assert plugin._cache == 0.9, "Plugin should have refreshed cache on first frame"

        # Simulate "leftover dirty state" from the previous run.
        plugin._cache = 0.7  # value that must NOT leak into the next analysis

        # Second analysis. reset() runs BEFORE the work → counter back to 0,
        # cache back to 0.5. First analyze_frame call then sets counter=1
        # (which is %5==1), executes the "fresh" branch, and writes 0.9.
        # If reset hadn't fired, counter would be 6 (6%5==1 too, coincidentally),
        # but cache would have stayed at 0.7 because no branch resets it from
        # 0.7 to 0.5 outside of reset(). So 0.9 confirms reset ran.
        pm.run_analysis([_frame()], fps=30.0, mode="local")
        assert plugin.reset_calls == 2, "reset() should have been called twice"
        assert plugin._counter == 1, "Counter must restart from 0 each analysis"
        assert plugin._cache == 0.9, "Cache must be refreshed, not the leaked 0.7"

    def test_reset_failure_does_not_stop_analysis(self, manager_with_plugins):
        """A plugin with a broken reset() must not abort the whole pipeline."""
        pm = manager_with_plugins

        class _BadResetPlugin(_CountingPlugin):
            def reset(self):
                raise RuntimeError("boom")

        bad = _BadResetPlugin(PluginNames.DCT_FREQUENCY)
        pm._plugins = [bad]

        class _FakePP:
            def process(self, frame):
                h, w = frame.shape[:2]
                bbox = {"x": 0, "y": 0, "w": int(w * 0.9), "h": int(h * 0.9)}
                return [(frame, bbox)], frame[..., 0], True
        pm._preprocessor = _FakePP()

        # Must not raise
        out = pm.run_analysis([_frame()], fps=30.0, mode="local")
        assert "overall_score" in out


class TestFaceRoiNoneSemantics:
    """Bug #2: when no face is detected, face_roi must be None (not full frame)."""

    def test_face_roi_is_none_when_no_face_detected(self, manager_with_plugins):
        pm = manager_with_plugins
        plugin = _CountingPlugin(PluginNames.DCT_FREQUENCY)
        pm._plugins = [plugin]

        class _NoFacePP:
            def process(self, frame):
                # Mirror the post-fix contract
                return [(None, None)], frame[..., 0], False

        pm._preprocessor = _NoFacePP()

        pm.run_analysis([_frame()], fps=30.0, mode="local")
        assert plugin.calls == 1
        assert plugin.received_face_roi == [None], (
            "Plugin received non-None face_roi when no face was detected — "
            "regression of bug #2"
        )

    def test_vit_not_called_in_no_face_scene(self, manager_with_plugins):
        """ViT must be SKIPPED via the scene routing table when NO_FACE."""
        pm = manager_with_plugins
        vit = _CountingPlugin(PluginNames.VIT_DETECTOR)
        dct = _CountingPlugin(PluginNames.DCT_FREQUENCY)
        pm._plugins = [vit, dct]

        class _NoFacePP:
            def process(self, frame):
                return [(None, None)], frame[..., 0], False
        pm._preprocessor = _NoFacePP()

        pm.run_analysis([_frame()], fps=30.0, mode="local")
        assert vit.calls == 0, "ViT should not run in NO_FACE scene"
        assert dct.calls == 1, "DCT must still run in NO_FACE scene"


class TestMultiFaceVerdict:
    """Bug #3: all faces analyzed, but only large enough faces vote on the verdict."""

    def test_tiny_face_does_not_dominate_verdict(self, manager_with_plugins):
        pm = manager_with_plugins
        plugin = _CountingPlugin(PluginNames.DCT_FREQUENCY, fixed_score=0.95)
        # We use a custom plugin that returns DIFFERENT scores per face
        class _PerFaceScorer(BaseDetectorPlugin):
            def __init__(self):
                self.calls = 0
            @property
            def plugin_name(self): return PluginNames.DCT_FREQUENCY
            @property
            def plugin_description(self): return "test"
            @property
            def plugin_version(self): return "0.0.0"
            @property
            def plugin_weight(self): return 1.0
            def analyze_frame(self, frame, face_roi=None):
                self.calls += 1
                # First face = small + suspicious; second face = large + clean
                return 0.95 if self.calls == 1 else 0.1

        scorer = _PerFaceScorer()
        pm._plugins = [scorer]

        class _TwoFacePP:
            def process(self, frame):
                h, w = frame.shape[:2]
                # Face 1: TINY (1% of frame) → analyzed but should NOT count for verdict
                tiny = {"x": 0, "y": 0, "w": int(w * 0.1), "h": int(h * 0.1)}
                # Face 2: LARGE (~80% of frame) → counts for verdict
                large = {"x": 0, "y": 0, "w": int(w * 0.9), "h": int(h * 0.9)}
                return [(frame, tiny), (frame, large)], frame[..., 0], True
        pm._preprocessor = _TwoFacePP()

        out = pm.run_analysis([_frame()], fps=30.0, mode="local")
        frame_score = out["frame_details"][0]["overall_score"]

        # Both faces should be in the output (multi-face restored)
        assert len(out["frame_details"][0]["faces"]) == 2

        # The frame verdict should reflect the LARGE face (clean, ~0.1),
        # not the tiny suspicious one (~0.95).
        assert frame_score < 0.5, (
            f"Frame verdict was {frame_score} — tiny suspicious face appears "
            "to be dominating the verdict (bug #3 regression)"
        )

    def test_two_large_faces_use_max(self, manager_with_plugins):
        """Both faces big enough → MAX takes the suspicious one."""
        pm = manager_with_plugins
        class _PerFaceScorer(BaseDetectorPlugin):
            def __init__(self): self.calls = 0
            @property
            def plugin_name(self): return PluginNames.DCT_FREQUENCY
            @property
            def plugin_description(self): return "test"
            @property
            def plugin_version(self): return "0.0.0"
            @property
            def plugin_weight(self): return 1.0
            def analyze_frame(self, frame, face_roi=None):
                self.calls += 1
                return 0.95 if self.calls == 1 else 0.1

        pm._plugins = [_PerFaceScorer()]

        class _TwoLargePP:
            def process(self, frame):
                h, w = frame.shape[:2]
                a = {"x": 0,           "y": 0, "w": int(w * 0.4), "h": int(h * 0.8)}
                b = {"x": int(w * 0.5), "y": 0, "w": int(w * 0.4), "h": int(h * 0.8)}
                return [(frame, a), (frame, b)], frame[..., 0], True
        pm._preprocessor = _TwoLargePP()

        out = pm.run_analysis([_frame()], fps=30.0, mode="local")
        assert out["frame_details"][0]["overall_score"] > 0.9, "MAX should pick the suspicious face"
