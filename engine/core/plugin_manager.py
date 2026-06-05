"""
Plugin Manager — Auto-discovers, loads, and orchestrates all detector plugins.

Key optimisations:
  1. FacePreProcessor — runs face detection ONCE per frame (O(1) regardless of
     plugin count) and shares the result with all plugins.

  2. SceneClassifier — after face detection, classifies the frame as:
       CROPPED_FACE  → face fills >50% of image (tight crop / profile pic)
       FACE_IN_SCENE → face detected but small relative to the frame
       NO_FACE       → no face found by OpenCV

  3. Conditional execution — only the plugins relevant to the detected scene
     type are executed. Irrelevant plugins are SKIPPED entirely (not called),
     saving CPU time throughout the analysis pipeline.

  4. Scene-specific weights — each scene type has its own weight table
     (defined in scene_classifier.py). Weights are normalised automatically.

Adding a new plugin:
    Drop a .py file in engine/plugins/ — no changes here required.
    Optionally add the plugin_name to the SCENE_PLUGIN_WEIGHTS table in
    scene_classifier.py to control when it is active.
"""

import os
import cv2
import threading
import numpy as np
import importlib.util
import logging
from typing import List
from core.plugin_base import BaseDetectorPlugin
from core.plugin_names import PluginNames
from core.scene_classifier import SceneClassifier, SCENE_PLUGIN_WEIGHTS

logger = logging.getLogger(__name__)

PLUGINS_DIR = os.path.join(os.path.dirname(__file__), "..", "plugins")


# ── Face Pre-Processor ────────────────────────────────────────────────────────

import urllib.request

try:
    from facenet_pytorch import MTCNN as FacenetMTCNN
    import torch
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

class FacePreProcessor:
    """
    Detects the primary face using MTCNN (Multi-task Cascaded Convolutional
    Networks) — the standard recommended by the forensic literature for
    deepfake detection pipelines.

    MTCNN uses three cascaded networks (P-Net, R-Net, O-Net) for:
      1. Face detection with bounding boxes
      2. Face alignment via 5-point landmarks
      3. Built-in NMS (no manual deduplication needed)

    Falls back to OpenCV SSD ResNet-10 if facenet-pytorch is not installed.

    Reference: Zhang et al., "Joint Face Detection and Alignment Using
    Multi-task Cascaded Convolutional Networks" (IEEE SPL, 2016)
    """

    def __init__(self):
        self._use_mtcnn = False

        if MTCNN_AVAILABLE:
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self._mtcnn = FacenetMTCNN(
                    keep_all=True,
                    device=device,
                    min_face_size=40,
                    thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net
                    post_process=False,
                )
                self._use_mtcnn = True
                logger.info(f"✅ MTCNN face detector loaded (device={device})")
            except Exception as e:
                logger.warning(f"MTCNN init failed ({e}), falling back to SSD")
                self._init_ssd_fallback()
        else:
            logger.warning("facenet-pytorch not installed, using SSD fallback")
            self._init_ssd_fallback()

    def _init_ssd_fallback(self):
        """Initialize the old SSD ResNet-10 face detector as fallback."""
        self.protxt_path = os.path.abspath(os.path.join(PLUGINS_DIR, "..", "core", "deploy.prototxt"))
        self.model_path  = os.path.abspath(os.path.join(PLUGINS_DIR, "..", "core", "res10_300x300_ssd_iter_140000.caffemodel"))

        if not os.path.exists(self.protxt_path):
            logger.info("Downloading SSD Face Detector prototxt...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                self.protxt_path
            )
        if not os.path.exists(self.model_path):
            logger.info("Downloading SSD Face Detector weights (10MB)...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                self.model_path
            )

        with open(self.protxt_path, "rb") as f:
            proto_buf = bytearray(f.read())
        with open(self.model_path, "rb") as f:
            model_buf = bytearray(f.read())

        self._net = cv2.dnn.readNetFromCaffe(proto_buf, model_buf)

    def process(self, frame: np.ndarray):
        """
        Detect faces in a frame.
        Returns: (faces_list, gray_frame, face_actually_detected)
          faces_list: list of (roi_crop, bbox_dict) tuples
          gray_frame: grayscale version of the input frame
          face_actually_detected: True if at least one face was found
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_full, w_full = frame.shape[:2]
        frame_area = h_full * w_full

        if self._use_mtcnn:
            return self._process_mtcnn(frame, gray, h_full, w_full, frame_area)
        else:
            return self._process_ssd(frame, gray, h_full, w_full, frame_area)

    def _process_mtcnn(self, frame, gray, h_full, w_full, frame_area):
        """MTCNN-based face detection."""
        import PIL.Image

        # MTCNN expects RGB PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(rgb)

        try:
            boxes, probs, landmarks = self._mtcnn.detect(pil_img, landmarks=True)
        except Exception as e:
            logger.error(f"MTCNN detection failed: {e}")
            # Signal "no face" with (None, None) so plugins know there is no ROI.
            # Previous behavior returned the full frame as a fake face_roi, which
            # fed non-face content to face-trained models (ViT) and corrupted scores.
            return [(None, None)], gray, False

        if boxes is None or len(boxes) == 0:
            logger.info("[FACE] MTCNN: no faces detected")
            return [(None, None)], gray, False

        # Convert MTCNN boxes (x1, y1, x2, y2) to our format (x, y, w, h)
        final_boxes = []
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob < 0.5:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_full, x2), min(h_full, y2)
            w_b, h_b = x2 - x1, y2 - y1

            if w_b > 0 and h_b > 0 and w_b * h_b >= frame_area * 0.002:
                ratio = w_b / h_b
                if 0.3 <= ratio <= 2.0:
                    final_boxes.append({
                        "x": x1, "y": y1, "w": w_b, "h": h_b,
                        "conf": float(prob)
                    })
                    logger.info(f"[FACE] MTCNN: conf={prob:.3f} x={x1} y={y1} w={w_b} h={h_b}")

        # Keep ALL detected faces (up to MTCNN's max_num_faces=6). The previous
        # "close-up mode" silently discarded background faces whenever the largest
        # face exceeded 10% of the frame — which is almost any interview shot.
        # That contradicted the multi-face analysis the UI advertises. Small or
        # noisy faces are still filtered out of the FRAME VERDICT downstream
        # (see MIN_FACE_AREA_FOR_VERDICT_RATIO in _run_analysis_locked) but they
        # remain visible in the per-face output for transparency.

        extracted_faces = []
        for bbox in final_boxes:
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            roi = frame[y: y + h, x: x + w]
            if roi.size > 0:
                extracted_faces.append((roi, bbox))

        if not extracted_faces:
            return [(None, None)], gray, False

        if len(extracted_faces) > 1:
            logger.info(f"[FACE] MTCNN: kept {len(extracted_faces)} faces (multi-face mode)")

        return extracted_faces, gray, True

    def _process_ssd(self, frame, gray, h_full, w_full, frame_area):
        """SSD ResNet-10 fallback face detection (original implementation)."""
        def _run_detector(target_size, thresh):
            blob = cv2.dnn.blobFromImage(
                frame, 1.0, target_size, (104.0, 177.0, 123.0), swapRB=False, crop=False
            )
            self._net.setInput(blob)
            dets = self._net.forward()
            boxes = []
            for i in range(dets.shape[2]):
                conf = float(dets[0, 0, i, 2])
                if conf > thresh:
                    box = dets[0, 0, i, 3:7] * np.array([w_full, h_full, w_full, h_full])
                    sx, sy, ex, ey = box.astype("int")
                    sx, sy = max(0, sx), max(0, sy)
                    ex, ey = min(w_full, ex), min(h_full, ey)
                    w_b, h_b = ex - sx, ey - sy
                    if w_b > 0 and h_b > 0:
                        boxes.append({"x": int(sx), "y": int(sy), "w": int(w_b), "h": int(h_b), "conf": conf})
            return boxes

        pass1 = _run_detector((300, 300), 0.50)

        combined = pass1
        combined = [b for b in combined if b['w'] * b['h'] >= frame_area * 0.002]
        combined = [b for b in combined if 0.3 <= (b['w'] / b['h']) <= 2.0]
        combined.sort(key=lambda b: b['conf'], reverse=True)

        # Simple NMS
        final_boxes = []
        for box in combined:
            discard = False
            for fb in final_boxes:
                xl = max(box['x'], fb['x']); yt = max(box['y'], fb['y'])
                xr = min(box['x'] + box['w'], fb['x'] + fb['w'])
                yb = min(box['y'] + box['h'], fb['y'] + fb['h'])
                if xr > xl and yb > yt:
                    inter = (xr - xl) * (yb - yt)
                    union = box['w'] * box['h'] + fb['w'] * fb['h'] - inter
                    if inter / float(union) > 0.25:
                        discard = True
                        break
            if not discard:
                final_boxes.append(box)

        # Multi-face mode (see _process_mtcnn for the rationale): keep all detected
        # faces; small/noisy faces are filtered from the frame verdict downstream,
        # not from the detection list.

        extracted_faces = []
        for bbox in final_boxes:
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            roi = frame[y: y + h, x: x + w]
            if roi.size > 0:
                extracted_faces.append((roi, bbox))

        if not extracted_faces:
            return [(None, None)], gray, False

        return extracted_faces, gray, True


# ── Plugin Manager ────────────────────────────────────────────────────────────

class PluginManager:
    """
    Discovers, loads, and orchestrates detector plugins with scene-aware routing.
    """

    def __init__(self):
        self._plugins: List[BaseDetectorPlugin] = []
        self._preprocessor = FacePreProcessor()
        # Serializes per-frame analysis. The underlying ML models (PyTorch/CUDA)
        # aren't safe to call from multiple FastAPI worker threads on the same
        # pipeline objects, and the per-request `global_plugin_scores` dict is
        # appended-to from inside the loop — concurrent runs would corrupt it.
        self._analysis_lock = threading.Lock()
        self._load_all_plugins()

    def get_plugins(self) -> List[BaseDetectorPlugin]:
        """Return the list of loaded plugins (for external configuration)."""
        return list(self._plugins)  # defensive copy

    # ── Plugin discovery ──────────────────────────────────────────────────────

    def _load_all_plugins(self):
        plugins_path = os.path.abspath(PLUGINS_DIR)
        if not os.path.isdir(plugins_path):
            logger.warning(f"Plugins directory not found: {plugins_path}")
            return

        for filename in sorted(os.listdir(plugins_path)):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue

            module_path = os.path.join(plugins_path, filename)
            module_name = filename[:-3]

            try:
                spec   = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseDetectorPlugin)
                        and attr is not BaseDetectorPlugin
                    ):
                        instance = attr()
                        self._plugins.append(instance)
                        logger.info(
                            f"✅ Loaded plugin: [{instance.plugin_name}] v{instance.plugin_version}"
                        )

            except Exception as e:
                logger.error(f"❌ Failed to load plugin from '{filename}': {e}")

        logger.info(f"--- PluginManager ready. {len(self._plugins)} plugin(s) active. ---")

        # Log scene routing table for visibility at startup
        for scene, weights in SCENE_PLUGIN_WEIGHTS.items():
            active = [name for name in weights]
            logger.info(f"   Scene [{scene}] → {active}")

    # ── Main analysis ─────────────────────────────────────────────────────────

    CLOUD_PLUGIN_NAME = PluginNames.SIGHTENGINE_CLOUD

    # Faces smaller than this fraction of the frame are analyzed and surfaced in
    # the per-face output, but DO NOT influence frame_max_score (the per-frame
    # verdict). Tiny background faces yield noisy scores from face-trained models
    # and shouldn't be able to single-handedly flag a frame as suspicious.
    MIN_FACE_AREA_FOR_VERDICT_RATIO = 0.03

    def run_analysis(self, frames: list, fps: float = 30.0, mode: str = "all", progress_callback=None) -> dict:
        """
        mode: "cloud"  → only Sightengine API
              "local"  → only local plugins (no API)
              "all"    → everything
        """
        if not frames:
            return {"error": "No frames to analyze"}

        # Serialize: the per-request `global_plugin_scores` is read/written in
        # this loop and the ML model pipelines aren't reentrant.
        with self._analysis_lock:
            return self._run_analysis_locked(frames, fps, mode, progress_callback)

    def _run_analysis_locked(self, frames: list, fps: float, mode: str, progress_callback) -> dict:
        # Clear per-analysis state on every plugin. Without this, stateful plugins
        # (e.g. Sightengine, which throttles API calls and caches the last score)
        # leak data from the previous analysis into the next.
        for plugin in self._plugins:
            try:
                plugin.reset()
            except Exception as reset_err:
                logger.warning(f"reset() failed on plugin '{plugin.plugin_name}': {reset_err}")

        frame_details_list = []

        # Filter plugins based on mode
        if mode == "cloud":
            active_plugins = [p for p in self._plugins if p.plugin_name == self.CLOUD_PLUGIN_NAME]
        elif mode == "local":
            active_plugins = [p for p in self._plugins if p.plugin_name != self.CLOUD_PLUGIN_NAME]
        else:
            active_plugins = list(self._plugins)

        logger.info(f"Analysis mode: {mode} → {len(active_plugins)} plugin(s) active")
        global_plugin_scores: dict[str, list[float]] = {p.plugin_name: [] for p in active_plugins}
        # Track which plugins have crashed so we don't log the same error every frame
        plugin_errors: dict[str, int] = {}

        for i, frame in enumerate(frames):
            if progress_callback:
                progress_callback(i, len(frames))
            timestamp = i / fps if fps > 0 else float(i)
            faces_tuples, _, face_detected = self._preprocessor.process(frame)

            frame_h, frame_w = frame.shape[:2]
            frame_area_pixels = max(frame_h * frame_w, 1)

            frame_faces_data = []
            frame_max_score = None  # MAX over faces that pass the size filter

            # Pre-compute scene routing for every face in this frame.
            face_meta = []  # list[(scene_type, weight_map, face_roi, face_bbox)]
            for face_roi, face_bbox in faces_tuples:
                scene_type = SceneClassifier.classify(frame, face_bbox, face_detected)
                active_pairs = SceneClassifier.get_active_plugins_and_weights(
                    scene_type, self._plugins
                )
                weight_map = {p.plugin_name: w for p, w in active_pairs}
                face_meta.append((scene_type, weight_map, face_roi, face_bbox))

            # ── Per-frame batching for plugins that opt in via SUPPORTS_BATCH ──
            # When the same plugin scores 2+ faces in this frame, we collect
            # those (frame, face_roi) pairs and submit them in one call. The
            # plugin's batch implementation (e.g. ViT's HF pipeline with
            # batch_size>N) amortizes overhead and, on GPU, parallelizes the
            # transformer forward pass. Single-face frames bypass this path.
            batch_score_cache: dict[str, dict[int, float]] = {}
            for plugin in active_plugins:
                if not getattr(plugin, "SUPPORTS_BATCH", False):
                    continue
                items: list[tuple] = []
                face_indices: list[int] = []
                for face_idx, (_, weight_map, face_roi, _) in enumerate(face_meta):
                    if plugin.plugin_name in weight_map:
                        items.append((frame, face_roi))
                        face_indices.append(face_idx)
                if len(items) < 2:
                    continue  # not worth the batch ceremony for 0 or 1 items
                try:
                    scores = plugin.analyze_frames_batch(items)
                    if len(scores) == len(face_indices):
                        batch_score_cache[plugin.plugin_name] = dict(zip(face_indices, scores))
                except Exception as batch_err:
                    plugin_errors[plugin.plugin_name] = plugin_errors.get(plugin.plugin_name, 0) + 1
                    if plugin_errors[plugin.plugin_name] == 1:
                        logger.error(
                            f"Batch path for plugin '{plugin.plugin_name}' crashed on frame {i}: {batch_err}",
                            exc_info=True,
                        )
                    # Fall through — per-face loop below will call analyze_frame.

            for face_idx, (scene_type, weight_map, face_roi, face_bbox) in enumerate(face_meta):
                # When no face was detected, (face_roi, face_bbox) is (None, None).
                # Plugins that fall back to the full frame (DCT, ViT) handle face_roi=None
                # internally; face-only plugins (Edge Blending) are skipped via the
                # NO_FACE scene routing table.
                face_plugin_dict = {}
                face_weighted_sum = 0.0
                face_total_weight = 0.0

                for plugin in active_plugins:
                    if plugin.plugin_name not in weight_map:
                        continue

                    # Prefer the batched result if one was computed for this face.
                    cached = batch_score_cache.get(plugin.plugin_name, {}).get(face_idx)
                    if cached is not None:
                        try:
                            if cached is None or (isinstance(cached, float) and np.isnan(cached)):
                                score = 0.5
                            else:
                                score = float(min(max(float(cached), 0.0), 1.0))
                        except (TypeError, ValueError):
                            score = 0.5
                    else:
                        try:
                            raw = plugin.analyze_frame(frame, face_roi=face_roi)
                            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                                score = 0.5
                            else:
                                score = float(min(max(float(raw), 0.0), 1.0))
                        except Exception as plugin_err:
                            plugin_errors[plugin.plugin_name] = plugin_errors.get(plugin.plugin_name, 0) + 1
                            # Log only the first failure per plugin to avoid log flooding
                            if plugin_errors[plugin.plugin_name] == 1:
                                logger.error(
                                    f"Plugin '{plugin.plugin_name}' crashed on frame {i}: {plugin_err}",
                                    exc_info=True,
                                )
                            score = 0.5

                    face_plugin_dict[plugin.plugin_name] = score
                    global_plugin_scores[plugin.plugin_name].append(score)

                    w = weight_map[plugin.plugin_name]
                    face_weighted_sum += (score * w)
                    face_total_weight += w

                if face_total_weight > 0:
                    face_overall_score = round(face_weighted_sum / face_total_weight, 4)
                else:
                    face_overall_score = 0.5

                # Only faces that are big enough to be analyzed reliably get to
                # influence the frame-level verdict. Smaller faces are still
                # reported in frame_faces_data so the UI can show them.
                if face_bbox is not None:
                    face_area_ratio = (face_bbox["w"] * face_bbox["h"]) / frame_area_pixels
                else:
                    # No face was detected (NO_FACE scene): the score IS the frame
                    # verdict — there's nothing else to compare against.
                    face_area_ratio = 1.0

                if face_area_ratio >= self.MIN_FACE_AREA_FOR_VERDICT_RATIO:
                    if frame_max_score is None or face_overall_score > frame_max_score:
                        frame_max_score = face_overall_score

                frame_faces_data.append({
                    "face_bbox": face_bbox,
                    "scene_detected": scene_type.value,
                    "overall_score": face_overall_score,
                    "plugin_scores": face_plugin_dict
                })

            frame_details_list.append({
                "frame_index": i,
                "timestamp_seconds": timestamp,
                "faces": frame_faces_data,
                # 0.5 = neutral fallback only when no faces were scored at all
                "overall_score": frame_max_score if frame_max_score is not None else 0.5
            })

        # Summarize plugin failure counts (if any) so the operator sees them
        for name, count in plugin_errors.items():
            logger.warning(f"Plugin '{name}' failed on {count} frame(s); used neutral fallback (0.5).")

        all_frame_scores = [f["overall_score"] for f in frame_details_list]
        global_overall_score = round(sum(all_frame_scores) / len(all_frame_scores), 4) if all_frame_scores else 0.5

        plugin_summaries = []
        for plugin in active_plugins:
            scores = global_plugin_scores[plugin.plugin_name]
            if not scores:
                continue
            avg_score = round(sum(scores) / len(scores), 4)
            plugin_summaries.append({
                "name": plugin.plugin_name,
                "average_score": avg_score,
                "frames_analyzed": len(scores)
            })

        return {
            "overall_score": global_overall_score,
            "plugins": plugin_summaries,
            "frame_details": frame_details_list
        }
