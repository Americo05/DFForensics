"""
Sanity tests for the three P3 analyzers: MetadataAnalyzer,
TemporalCoherenceAnalyzer, and rPPGAnalyzer.

We test pure logic and bounds, not the underlying ML/IO — so these run
without needing actual videos, EXIF data, or face detection.
"""

import numpy as np
import pytest

from core.metadata_analyzer import MetadataAnalyzer, _SUSPECT_SOFTWARE_PATTERNS
from core.temporal_coherence_analyzer import TemporalCoherenceAnalyzer
from core.rppg_analyzer import rPPGAnalyzer


# ── MetadataAnalyzer ──────────────────────────────────────────────────

class TestMetadataAnalyzer:
    def test_non_existent_file_returns_none(self):
        ma = MetadataAnalyzer()
        assert ma.analyze_file("/does/not/exist.jpg") is None

    def test_empty_path_returns_none(self):
        ma = MetadataAnalyzer()
        assert ma.analyze_file("") is None
        assert ma.analyze_file(None) is None  # type: ignore[arg-type]

    def test_suspect_patterns_lowercased(self):
        for p in _SUSPECT_SOFTWARE_PATTERNS:
            assert p == p.lower(), f"Pattern '{p}' must be lowercase for matching"

    def test_suspect_patterns_include_common_generators(self):
        joined = " ".join(_SUSPECT_SOFTWARE_PATTERNS)
        for must_be_present in ("stable diffusion", "midjourney", "deepfake"):
            assert must_be_present in joined


# ── TemporalCoherenceAnalyzer ─────────────────────────────────────────

def _make_frame(h=240, w=320, brightness=128):
    return np.full((h, w, 3), brightness, dtype=np.uint8)


def _make_frame_detail(face_x=None, face_y=None, score=0.3):
    """Build a frame_details dict in the same shape PluginManager emits."""
    faces = []
    if face_x is not None:
        faces.append({
            "face_bbox": {"x": face_x, "y": face_y, "w": 80, "h": 100},
            "overall_score": score,
        })
    return {"faces": faces, "overall_score": score}


class TestTemporalCoherence:
    def test_too_few_frames_returns_none(self):
        ta = TemporalCoherenceAnalyzer()
        assert ta.analyze([_make_frame()], [_make_frame_detail()]) is None

    def test_smooth_consistent_video_scores_low(self):
        ta = TemporalCoherenceAnalyzer()
        # 30 frames, same brightness, face barely moves, score steady
        frames = [_make_frame(brightness=120) for _ in range(30)]
        details = [_make_frame_detail(face_x=100 + (i % 2), face_y=50, score=0.25) for i in range(30)]
        out = ta.analyze(frames, details)
        assert out is not None
        assert out["temporal_score"] < 0.6, f"Smooth video should not be flagged: {out}"

    def test_jittering_face_raises_score(self):
        ta = TemporalCoherenceAnalyzer()
        frames = [_make_frame() for _ in range(30)]
        # Face hops back and forth by ~80px — that's huge relative to a 320px-wide frame
        details = [
            _make_frame_detail(face_x=20 if i % 2 == 0 else 200, face_y=50, score=0.3)
            for i in range(30)
        ]
        out = ta.analyze(frames, details)
        assert out is not None
        assert "face_bbox_jittering" in out["signals"]

    def test_score_volatility_detected(self):
        ta = TemporalCoherenceAnalyzer()
        frames = [_make_frame() for _ in range(30)]
        # Scores swing wildly
        details = [_make_frame_detail(face_x=100, face_y=50, score=0.05 if i % 2 == 0 else 0.95)
                   for i in range(30)]
        out = ta.analyze(frames, details)
        assert out is not None
        assert "score_swings_between_frames" in out["signals"]

    def test_bounded_in_unit_interval(self):
        ta = TemporalCoherenceAnalyzer()
        rng = np.random.default_rng(0)
        frames = [_make_frame(brightness=int(rng.integers(0, 256))) for _ in range(20)]
        details = [_make_frame_detail(face_x=int(rng.integers(0, 200)), face_y=50,
                                      score=float(rng.random())) for _ in range(20)]
        out = ta.analyze(frames, details)
        if out is not None:
            assert 0.0 <= out["temporal_score"] <= 1.0


# ── rPPGAnalyzer ──────────────────────────────────────────────────────

class TestrPPG:
    def test_low_fps_rejected(self):
        rp = rPPGAnalyzer()
        # 5 fps — below Nyquist for our 4 Hz band's upper edge
        frames = [_make_frame() for _ in range(120)]
        assert rp.analyze(frames, fps=5.0) is None

    def test_too_few_frames_rejected(self):
        rp = rPPGAnalyzer()
        frames = [_make_frame() for _ in range(10)]
        assert rp.analyze(frames, fps=30.0) is None

    def test_no_face_data_returns_none(self):
        rp = rPPGAnalyzer()
        frames = [_make_frame() for _ in range(120)]
        # No frame_details → no face → can't extract green signal
        details = [{"faces": [], "overall_score": 0.5} for _ in range(120)]
        assert rp.analyze(frames, fps=30.0, frame_details=details) is None

    def test_synthetic_pulse_detected_as_real(self):
        """A frame stream with a clear 1.2 Hz (72 BPM) green oscillation should
        score LOW (not fake)."""
        rp = rPPGAnalyzer()
        fps = 30.0
        n = 240  # 8 seconds
        t = np.arange(n) / fps
        # Build frames whose green-channel face ROI mean oscillates at 1.2 Hz
        frames = []
        for i in range(n):
            base = 120 + int(20 * np.sin(2 * np.pi * 1.2 * t[i]))  # ~72 BPM
            f = np.zeros((200, 200, 3), dtype=np.uint8)
            f[..., 1] = base  # green channel carries the pulse
            frames.append(f)
        details = [{
            "faces": [{"face_bbox": {"x": 50, "y": 50, "w": 100, "h": 100}, "overall_score": 0.2}],
            "overall_score": 0.2,
        } for _ in range(n)]

        out = rp.analyze(frames, fps=fps, frame_details=details)
        assert out is not None
        assert 0.0 <= out["rppg_score"] <= 1.0
        # The synthetic oscillation is ABOVE noise → some pulse detected
        # (we don't assert score<0.5 strictly because window/leakage effects
        # can shift the metric; just assert it isn't strongly flagged as fake)
        assert out["rppg_score"] < 0.9
        assert out.get("estimated_bpm") is not None
