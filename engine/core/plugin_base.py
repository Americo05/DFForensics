"""
Base class for all DeepFake Detection Plugins.

Every plugin placed in the engine/plugins/ folder MUST inherit from this class 
and implement all abstract methods. This is the "contract" that guarantees the 
PluginManager can talk to any plugin interchangeably.

To create a new plugin:
1. Create a new .py file in engine/plugins/
2. Create a class that inherits from BaseDetectorPlugin
3. Implement all abstract methods
4. The PluginManager will auto-discover and load it on the next server start.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseDetectorPlugin(ABC):
    """
    Abstract Base Class for all Deepfake Detector Plugins.
    """

    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """A short, unique name for this plugin. e.g. 'Laplacian Variance Detector'"""
        pass

    @property
    @abstractmethod
    def plugin_description(self) -> str:
        """A description of the technique this plugin uses, ideally citing a paper."""
        pass

    @property
    @abstractmethod
    def plugin_version(self) -> str:
        """Version of this plugin. e.g. '1.0.0'"""
        pass

    @property
    @abstractmethod
    def plugin_weight(self) -> float:
        """
        Reliability weight (0.0 to 1.0) determining how much this plugin's score 
        influences the final overall probability.
        For example, a high-accuracy ML model might have a weight of 0.8, 
        while a heuristic edge-sharpness detector might only have 0.2.
        """
        pass

    @abstractmethod
    def analyze_frame(self, frame: np.ndarray, face_roi: np.ndarray | None = None) -> float:
        """
        Analyzes a single video frame (as a NumPy array from OpenCV).

        Args:
            frame:    Full BGR frame as a NumPy array (H, W, 3).
            face_roi: Pre-cropped face region (BGR) supplied by the FacePreProcessor
                      in PluginManager. When provided, plugins MUST use this instead
                      of running their own face detection — the face is detected ONCE
                      per frame and shared across all plugins.
                      Is None when no face was detected in the frame.

        Returns:
            A float between 0.0 and 1.0 representing the probability
            that this frame contains a deepfake manipulation.
            (0.0 = definitely real, 1.0 = definitely fake)
        """
        pass

    def get_plugin_info(self) -> dict:
        """Returns a dictionary with plugin metadata. Used by the PluginManager."""
        return {
            "name": self.plugin_name,
            "description": self.plugin_description,
            "version": self.plugin_version,
            "configured": self.is_configured(),
            "supports_batch": getattr(self, "SUPPORTS_BATCH", False),
        }

    def is_configured(self) -> bool:
        """
        Whether the plugin is ready to score (model loaded, API key present, etc.).

        Default `True` covers stateless plugins like DCT/Edge Blending that have
        no external dependencies. Plugins that need an ML model, an API key,
        or any setup step should override this to return False when something
        is missing — the /health endpoint surfaces this so operators see
        partial-functionality states without having to read logs.
        """
        return True

    def reset(self) -> None:
        """
        Reset any per-analysis state held on the plugin instance.

        The PluginManager calls this on every plugin before each `run_analysis`.
        Stateless plugins (the common case) should leave this as the default no-op.
        Plugins that cache scores, count frames, throttle API calls, or otherwise
        carry state across `analyze_frame` calls MUST override this to clear that
        state — otherwise leftovers from a previous analysis will leak into the
        next one.
        """
        return None

    # ── Optional batch API ──────────────────────────────────────────────────
    # Override `analyze_frames_batch` only when the plugin can do real batched
    # inference (e.g. neural networks where batch_size>1 is much faster).
    # The default implementation just loops `analyze_frame`, so plugins that
    # don't override pay no penalty and the PluginManager doesn't need to
    # special-case anything.
    SUPPORTS_BATCH: bool = False

    def analyze_frames_batch(
        self,
        items: "list[tuple[np.ndarray, np.ndarray | None]]",
    ) -> "list[float]":
        """
        Score a batch of (frame, face_roi) pairs in one call.

        Default implementation: loops `analyze_frame`. Override on plugins
        where batched inference is meaningfully faster than per-frame calls.

        Args:
            items: list of (frame, face_roi) tuples. `face_roi` may be None.

        Returns:
            list of fake-probability floats, same length and order as `items`.
        """
        return [self.analyze_frame(frame, face_roi=face_roi) for frame, face_roi in items]
