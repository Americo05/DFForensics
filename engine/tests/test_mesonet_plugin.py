"""
Tests for the MesoNet plugin.

What these guard
----------------
1. The plugin loads cleanly even WITHOUT the weights file (the user hasn't
   run the download script yet on a fresh checkout) — `is_configured`
   reports False and `analyze_frame` returns neutral 0.5 rather than
   raising.
2. The Meso-4 architecture has the exact tensor shapes the converter
   produces. If the layer ordering, channel counts, or kernel sizes drift,
   weight loading would silently succeed-then-produce-garbage; this test
   catches that.
3. Preprocessing handles the input shapes the plugin manager actually
   feeds it (BGR uint8, varying sizes, grayscale, RGBA, None).
4. The plugin registers in plugin_names + the SCENE_PLUGIN_WEIGHTS table.
5. Both inference paths (single-frame and batch) produce well-formed
   outputs WHEN weights ARE present — this is gated to skip if the user
   hasn't downloaded weights yet, so the suite stays green either way.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.plugin_names import PluginNames
from core.scene_classifier import SCENE_PLUGIN_WEIGHTS, SceneType
from plugins.mesonet_detector import MesoNetDetectorPlugin, _DEFAULT_WEIGHTS_PATH


# ── Setup helpers ─────────────────────────────────────────────────────────


def _fake_bgr(h: int = 240, w: int = 320) -> np.ndarray:
    """A deterministic 'fake' BGR frame for plugin tests."""
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ── Configuration / registration ─────────────────────────────────────────


def test_name_constant_registered():
    """The plugin name string is stable across renames."""
    assert PluginNames.MESONET == "MesoNet (Afchar et al., WIFS 2018)"


def test_scene_table_includes_mesonet_for_face_scenes():
    """MesoNet must be routed for face-containing scenes and SKIPPED for NO_FACE."""
    assert PluginNames.MESONET in SCENE_PLUGIN_WEIGHTS[SceneType.CROPPED_FACE]
    assert PluginNames.MESONET in SCENE_PLUGIN_WEIGHTS[SceneType.FACE_IN_SCENE]
    # NO_FACE must NOT include MesoNet — applying a face-trained network to
    # landscapes/objects produces garbage scores; if this assertion fails,
    # someone added it without thinking about scene routing.
    assert PluginNames.MESONET not in SCENE_PLUGIN_WEIGHTS[SceneType.NO_FACE]


def test_scene_weights_still_sum_to_one():
    """Adding MesoNet must not break the sum-to-1 invariant."""
    for scene, weights in SCENE_PLUGIN_WEIGHTS.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"Scene {scene} weights sum to {total}, not 1.0"


# ── Loading without weights ──────────────────────────────────────────────


def test_loads_without_weights_file_gracefully(tmp_path, monkeypatch):
    """Plugin must not raise when the weights file is missing."""
    missing = tmp_path / "definitely_not_there.pt"
    plugin = MesoNetDetectorPlugin(weights_path=missing)
    assert plugin.is_configured() is False
    # Neutral score: matches the "no API key" pattern in Sightengine.
    score = plugin.analyze_frame(_fake_bgr())
    assert score == 0.5


def test_batch_returns_neutral_when_unconfigured(tmp_path):
    """The batch path must also fall back to neutral, not raise."""
    plugin = MesoNetDetectorPlugin(weights_path=tmp_path / "nope.pt")
    items = [(_fake_bgr(), None), (_fake_bgr(), _fake_bgr(64, 64))]
    scores = plugin.analyze_frames_batch(items)
    assert scores == [0.5, 0.5]


# ── Architecture sanity ──────────────────────────────────────────────────


def test_meso4_architecture_shapes():
    """
    Validate the architecture independently of weights.

    If someone changes a kernel size, channel count, or pool stride, the
    flattened-feature dimension feeding fc1 changes — and weights loaded
    from a converted .h5 would silently land at fc1 with the wrong shape
    (PyTorch raises strict=True in load_state_dict, which is what we want;
    this test asserts the shape matches the paper before that loading
    even happens).
    """
    torch = pytest.importorskip("torch")
    from plugins.mesonet_detector import _Meso4

    model = _Meso4().eval()
    # Paper input: 256×256 RGB.
    x = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1,), f"Expected scalar output per item, got {tuple(y.shape)}"

    # Confirm the post-Pool4 spatial size = 8×8 — flattened to 1024 features
    # which is what fc1's input layer expects.
    assert model.fc1.in_features == 16 * 8 * 8, (
        f"fc1 expects {model.fc1.in_features} features; if you changed the "
        f"conv/pool stack, recompute and update."
    )


# ── Preprocessing ────────────────────────────────────────────────────────


def test_preprocess_normal_bgr():
    arr = MesoNetDetectorPlugin._preprocess(_fake_bgr(120, 180))
    assert arr is not None
    # CHW layout, 3 channels, 256×256 spatial
    assert arr.shape == (3, 256, 256)
    assert arr.dtype == np.float32
    assert 0.0 <= float(arr.min()) and float(arr.max()) <= 1.0


def test_preprocess_handles_grayscale():
    gray = np.full((100, 100), 128, dtype=np.uint8)
    arr = MesoNetDetectorPlugin._preprocess(gray)
    # Grayscale must be promoted to 3 channels; otherwise the conv layer
    # would receive 1 channel and crash with a cryptic torch error.
    assert arr is not None and arr.shape == (3, 256, 256)


def test_preprocess_handles_rgba():
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    arr = MesoNetDetectorPlugin._preprocess(rgba)
    assert arr is not None and arr.shape == (3, 256, 256)


def test_preprocess_rejects_invalid_inputs():
    """Empty arrays and weird shapes must return None (caller produces 0.5)."""
    assert MesoNetDetectorPlugin._preprocess(None) is None
    assert MesoNetDetectorPlugin._preprocess(np.zeros((0, 0, 3), dtype=np.uint8)) is None
    # 5 channels — not BGR, not BGRA. Defensive return.
    assert MesoNetDetectorPlugin._preprocess(np.zeros((50, 50, 5), dtype=np.uint8)) is None


# ── Inference (gated on weights availability) ────────────────────────────


@pytest.mark.skipif(
    not _DEFAULT_WEIGHTS_PATH.is_file(),
    reason="MesoNet weights not downloaded — run engine/scripts/download_mesonet_weights.py",
)
def test_inference_with_real_weights_returns_probability():
    plugin = MesoNetDetectorPlugin()
    assert plugin.is_configured(), "weights exist but plugin failed to load them"
    score = plugin.analyze_frame(_fake_bgr())
    assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"


@pytest.mark.skipif(
    not _DEFAULT_WEIGHTS_PATH.is_file(),
    reason="MesoNet weights not downloaded",
)
def test_batch_inference_matches_single_inference_shape():
    """Batched and per-item paths must return same-length output lists."""
    plugin = MesoNetDetectorPlugin()
    items = [(_fake_bgr(), None) for _ in range(4)]
    scores = plugin.analyze_frames_batch(items)
    assert len(scores) == 4
    for s in scores:
        assert 0.0 <= s <= 1.0
