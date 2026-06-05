"""
Unit tests for LipSyncAnalyzer logic — covers `_compute_sync_score`,
`_compute_audio_energy`, and the new sliding-window scaffolding without
requiring MediaPipe or librosa to actually run (no real video needed).
"""

import numpy as np
import pytest

from core.lip_sync_analyzer import LipSyncAnalyzer


@pytest.fixture(scope="module")
def lsa():
    # The analyzer's __init__ tries to load MediaPipe; that's fine — if it
    # fails, _face_mesh stays None and the methods we test don't depend on it.
    return LipSyncAnalyzer()


class TestComputeSyncScore:
    def test_perfect_correlation_yields_low_fake_score(self, lsa):
        # Energy and mouth move together → real sync
        energy = np.array([0.1, 0.3, 0.6, 0.9, 0.7, 0.5, 0.2, 0.1, 0.4, 0.8])
        mouth  = np.array([0.1, 0.3, 0.6, 0.9, 0.7, 0.5, 0.2, 0.1, 0.4, 0.8])
        score, details = lsa._compute_sync_score(energy, mouth)
        assert score < 0.4, f"Perfectly correlated streams should score authentic, got {score}"
        assert not details.get("inconclusive")

    def test_anticorrelation_yields_high_fake_score(self, lsa):
        # Mouth moves OPPOSITE to audio — clearly desynchronized
        energy = np.array([0.1, 0.3, 0.6, 0.9, 0.7, 0.5, 0.2, 0.1, 0.4, 0.8])
        mouth  = np.array([0.9, 0.7, 0.4, 0.1, 0.3, 0.5, 0.8, 0.9, 0.6, 0.2])
        score, _ = lsa._compute_sync_score(energy, mouth)
        assert score > 0.6, f"Anti-correlated streams should score suspicious, got {score}"

    def test_silent_audio_marks_inconclusive(self, lsa):
        energy = np.zeros(15)
        mouth = np.linspace(0, 1, 15)
        score, details = lsa._compute_sync_score(energy, mouth)
        assert details.get("inconclusive") is True

    def test_no_mouth_movement_marks_high_score(self, lsa):
        """Regression: audio fluctuates but mouth never moves → bad lip sync."""
        energy = np.array([0.1, 0.5, 0.9, 0.6, 0.3, 0.7, 0.4, 0.2, 0.8, 0.5])
        mouth = np.full(10, 0.5)  # Mouth detected but never opens (constant)
        score, _ = lsa._compute_sync_score(energy, mouth)
        # Constant mouth = std ≈ 0 = "audio fluctuating but mouth still" → 1.0
        assert score >= 0.9

    def test_too_few_mouth_detections_marks_inconclusive(self, lsa):
        energy = np.linspace(0, 1, 20)
        mouth = np.zeros(20)  # Mouth never detected (all zeros)
        score, details = lsa._compute_sync_score(energy, mouth)
        assert details.get("inconclusive") is True

    def test_score_bounded(self, lsa):
        """For arbitrary inputs, fake_score must stay in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            energy = rng.random(30)
            mouth = rng.random(30)
            score, _ = lsa._compute_sync_score(energy, mouth)
            assert 0.0 <= score <= 1.0


class TestComputeAudioEnergy:
    def test_silent_signal_returns_zeros(self, lsa):
        signal = np.zeros(16000)  # 1 second of silence
        energy = lsa._compute_audio_energy(signal, sr=16000, n_frames=30, fps=30.0)
        assert len(energy) == 30
        assert np.all(energy == 0)

    def test_loud_uniform_signal_normalized_to_one(self, lsa):
        signal = np.ones(16000, dtype=np.float32) * 0.5
        energy = lsa._compute_audio_energy(signal, sr=16000, n_frames=30, fps=30.0)
        # Normalized to [0,1] with max=1
        assert energy.max() == pytest.approx(1.0, abs=1e-6)

    def test_length_matches_n_frames(self, lsa):
        signal = np.zeros(16000)
        for n in (10, 30, 60, 100):
            energy = lsa._compute_audio_energy(signal, sr=16000, n_frames=n, fps=30.0)
            assert len(energy) == n


class TestSlidingWindowParams:
    """The window constants are part of the algorithm — guard them."""

    def test_window_seconds_positive(self, lsa):
        assert lsa.WINDOW_SECONDS > 0
        assert lsa.WINDOW_STRIDE_SECONDS > 0
        assert lsa.WINDOW_STRIDE_SECONDS < lsa.WINDOW_SECONDS, "Stride must be smaller than window for overlap"

    def test_max_total_seconds_bounded(self, lsa):
        assert lsa.MAX_TOTAL_SECONDS > lsa.WINDOW_SECONDS
        assert lsa.MAX_TOTAL_SECONDS <= 600  # Sanity: no more than 10 min
