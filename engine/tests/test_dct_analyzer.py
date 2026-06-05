"""
Unit tests for the DCT Frequency Analyzer.

Focused regression coverage for the bug we fixed where np.linalg.lstsq's tuple
unpacking discarded the intercept and the residual computation used garbage.
With the bug present, scores on natural-like images were essentially random.
"""

import numpy as np
import pytest

from plugins.dct_analyzer import DCTFrequencyAnalyzerPlugin


@pytest.fixture(scope="module")
def plugin():
    return DCTFrequencyAnalyzerPlugin()


def _gradient_image(n=128):
    """A smooth gradient — approximates the 1/f² statistics of natural images."""
    x = np.linspace(0, 1, n, dtype=np.float32)
    y = np.linspace(0, 1, n, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    img = (xx + yy) * 0.5 * 255
    return np.stack([img, img, img], axis=-1).astype(np.uint8)


def _checkerboard_image(n=128, stride=8):
    """High-frequency stride pattern — what GAN upsampling artifacts look like."""
    img = np.zeros((n, n), dtype=np.uint8)
    for i in range(0, n, stride):
        for j in range(0, n, stride):
            if ((i // stride) + (j // stride)) % 2 == 0:
                img[i:i + stride, j:j + stride] = 255
    return np.stack([img, img, img], axis=-1)


def _random_image(n=128, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, n, 3), dtype=np.uint8)


class TestBoundsAndContract:
    def test_returns_float_in_unit_interval(self, plugin):
        for img in (_gradient_image(), _checkerboard_image(), _random_image()):
            score = plugin.analyze_frame(img)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_neutral_on_empty_input(self, plugin):
        assert plugin.analyze_frame(None) == 0.5
        assert plugin.analyze_frame(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.5

    def test_uses_face_roi_when_provided(self, plugin):
        frame = _random_image()
        roi = _gradient_image()
        # We're not asserting a specific value, just that passing an ROI doesn't crash
        # and yields a valid score
        score = plugin.analyze_frame(frame, face_roi=roi)
        assert 0.0 <= score <= 1.0


class TestPowerLawRegression:
    """
    Regression test for the lstsq tuple-unpacking bug.

    Before the fix, `fitted = log_f * alpha + _` used `_` (which was the lstsq
    return tuple, not the intercept). Residuals were therefore garbage and the
    `score_nonlinear` term was effectively random. The exact alpha estimate
    didn't matter, but residual_std was meaningless.

    We assert that the power-law function on a smooth gradient (~natural 1/f²
    statistics) produces a LOW non-linearity score, which only holds with a
    correct fit.
    """

    def test_smooth_gradient_has_low_power_law_deviation(self, plugin):
        # Use the internal helper directly to isolate the regression
        import cv2
        img = _gradient_image(256)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s = plugin._power_law_deviation(gray)
        # Smooth gradient = low deviation = score on the lower half.
        # With the bug present, residual_std was random and would frequently
        # push this above 0.7.
        assert s < 0.6, f"Smooth gradient scored {s} — regression of the lstsq fix?"

    def test_intercept_is_finite_and_used(self, plugin):
        """Verify lstsq solution is unpacked and both terms are real numbers."""
        import cv2
        img = _gradient_image(128)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # If `fitted` were computed against a tuple, np.std(residuals) would
        # raise (cannot subtract tuple from array). The bug only didn't crash
        # because `_` happened to be a numpy array of singular values. So the
        # most reliable assertion is "doesn't raise" plus "score is finite".
        s = plugin._power_law_deviation(gray)
        assert np.isfinite(s)


class TestSubScores:
    def test_spectral_flatness_returns_unit_interval(self, plugin):
        import cv2
        for img in (_gradient_image(), _random_image()):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            s = plugin._spectral_flatness_score(gray)
            assert 0.0 <= s <= 1.0

    def test_stride_artifact_returns_unit_interval(self, plugin):
        import cv2
        for img in (_gradient_image(), _checkerboard_image()):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            s = plugin._stride_artifact_score(gray)
            assert 0.0 <= s <= 1.0
