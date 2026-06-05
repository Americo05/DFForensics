"""
Metadata Analyzer — File-level EXIF / codec / container forensics.

Why this is a separate analyzer (not a plugin)
----------------------------------------------
The per-frame plugin contract sees only `(frame, face_roi)` numpy arrays —
no path, no codec, no EXIF. Re-encoded social-media uploads, re-saved
images, and AI-generated content have characteristic metadata signatures
that disappear from the pixel buffer. So this runs once per file, like
LipSyncAnalyzer / AudioDeepfakeAnalyzer.

Signals we look for
-------------------
Images (EXIF):
  * EXIF block completely stripped               → very common in AI-generated images
  * Missing camera Make/Model but has dimensions → re-saved (e.g. screenshot/upload)
  * Software field mentions a generative tool    → "Stable Diffusion", "Midjourney", etc.
  * DateTimeOriginal absent or set to epoch      → manipulated metadata

Videos (FFmpeg ffprobe):
  * No camera-style metadata, generic encoder    → re-encoded or generated
  * Encoder string mentions a known editor       → "HandBrake", "iMovie", AI tools
  * Multiple re-encode passes detectable via QP  → not in this version (advanced)

This is a HEURISTIC signal — false-positive prone. We deliberately keep
the weight low when this analyzer is wired into the aggregation.

Dependencies:
    pip install Pillow  (already required by ViT plugin)
    FFmpeg via imageio-ffmpeg (already bundled)

Version: 1.0.0
"""

import logging
import os
import re
import subprocess

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# Strings commonly found in generated-image or heavy-editor metadata.
# Lowercased before matching. Conservative — common phone editors are NOT here.
_SUSPECT_SOFTWARE_PATTERNS = [
    "stable diffusion", "stablediffusion", "midjourney", "dall-e", "dall·e",
    "dalle", "openai", "comfyui", "automatic1111", "invokeai", "fooocus",
    "leonardo.ai", "runwayml", "novelai", "civitai",
    "deepfake", "faceswap", "deepfacelab", "facefusion",
]

# Encoder strings that indicate the video has been re-encoded (not raw camera
# output). The presence of one of these doesn't prove fakeness on its own, but
# combined with absent camera metadata it's a meaningful signal.
_REENCODE_ENCODER_PATTERNS = [
    "lavf",        # FFmpeg
    "handbrake",
    "imovie",
    "premiere",
    "after effects",
    "davinci",
    "kdenlive",
    "shotcut",
]


class MetadataAnalyzer:
    """File-level metadata forensics. Lazy — only inspects when asked."""

    def __init__(self):
        self._ffprobe = self._find_ffprobe()
        if self._ffprobe:
            logger.info(f"Metadata Analyzer: ffprobe at {self._ffprobe}")
        else:
            logger.info("Metadata Analyzer: ffprobe not found, video analysis limited")

    @staticmethod
    def _find_ffprobe() -> str | None:
        """Look for ffprobe next to the bundled ffmpeg, then on PATH."""
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            # ffprobe is usually shipped alongside ffmpeg with the same prefix
            candidate = ffmpeg.replace("ffmpeg", "ffprobe")
            if os.path.isfile(candidate):
                return candidate
        except Exception:
            pass
        import shutil as _shutil
        sys_ffprobe = _shutil.which("ffprobe")
        return sys_ffprobe

    # ── Public API ──────────────────────────────────────────────────────

    def analyze_file(self, file_path: str) -> dict | None:
        """
        Inspect the file and return a dict like:
          { "metadata_score": 0.0..1.0,  # higher = more suspicious
            "signals": ["exif_stripped", "suspect_software:midjourney", ...],
            "details": { ... } }
        Returns None if the file type isn't supported or inspection fails.
        """
        if not file_path or not os.path.isfile(file_path):
            return None
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext in IMAGE_EXTENSIONS:
                return self._analyze_image(file_path)
            return self._analyze_video(file_path)
        except Exception as e:
            logger.warning(f"Metadata analysis failed: {e}")
            return None

    # ── Image (EXIF) ────────────────────────────────────────────────────

    def _analyze_image(self, path: str) -> dict | None:
        if not PIL_AVAILABLE:
            return None

        try:
            img = Image.open(path)
            exif_raw = img.getexif()
        except Exception as e:
            logger.warning(f"Could not read EXIF from {path}: {e}")
            return None

        signals: list[str] = []
        score = 0.0
        details: dict = {"format": img.format, "size": list(img.size)}

        exif: dict[str, object] = {}
        for tag_id, value in (exif_raw or {}).items():
            tag = TAGS.get(tag_id, str(tag_id))
            try:
                exif[tag] = value if isinstance(value, (int, float, str)) else str(value)
            except Exception:
                pass

        details["exif_tag_count"] = len(exif)

        if not exif:
            # Nothing at all — common in PNGs but also in many generated images
            signals.append("exif_completely_stripped")
            score += 0.35
        else:
            has_make = "Make" in exif
            has_model = "Model" in exif
            has_datetime = "DateTimeOriginal" in exif or "DateTime" in exif

            if not (has_make and has_model):
                signals.append("missing_camera_make_model")
                score += 0.20

            if not has_datetime:
                signals.append("missing_datetime")
                score += 0.10

            software = str(exif.get("Software", "")).lower()
            if software:
                details["software"] = exif.get("Software")
                for pattern in _SUSPECT_SOFTWARE_PATTERNS:
                    if pattern in software:
                        signals.append(f"suspect_software:{pattern}")
                        score += 0.60  # strong signal
                        break

        # PNG often carries a textual `parameters` chunk in AI-generated images
        # (e.g. Stable Diffusion saves the prompt there). Pillow exposes that
        # via img.text on PNG.
        try:
            png_text = getattr(img, "text", {}) or {}
        except Exception:
            png_text = {}
        for key, val in png_text.items():
            joined = f"{key} {val}".lower()
            for pattern in _SUSPECT_SOFTWARE_PATTERNS:
                if pattern in joined:
                    signals.append(f"png_text_mentions:{pattern}")
                    score += 0.50
                    break

        return {
            "metadata_score": round(min(score, 1.0), 4),
            "signals": signals,
            "details": details,
            "verdict": "SUSPICIOUS" if score > 0.6 else ("INCONCLUSIVE" if score > 0.3 else "CLEAN"),
        }

    # ── Video (ffprobe) ─────────────────────────────────────────────────

    def _analyze_video(self, path: str) -> dict | None:
        if not self._ffprobe:
            return None

        cmd = [
            self._ffprobe, "-v", "error",
            "-show_format", "-show_streams",
            "-of", "default=noprint_wrappers=0",
            path,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"ffprobe failed: {e}")
            return None

        if result.returncode != 0:
            return None

        text = result.stdout.decode("utf-8", errors="replace").lower()

        signals: list[str] = []
        score = 0.0
        details: dict = {}

        # Encoder field — common in re-encoded files
        encoder_match = re.search(r"\bencoder\s*=\s*([^\n\r]+)", text)
        encoder = encoder_match.group(1).strip() if encoder_match else ""
        if encoder:
            details["encoder"] = encoder
            for pattern in _REENCODE_ENCODER_PATTERNS:
                if pattern in encoder:
                    signals.append(f"reencoded_with:{pattern}")
                    score += 0.20
                    break

        # Camera-y metadata that real phones/cameras include
        has_make = bool(re.search(r"\b(make|com\.apple\.quicktime\.make)\s*=", text))
        has_model = bool(re.search(r"\b(model|com\.apple\.quicktime\.model)\s*=", text))
        has_creation = bool(re.search(r"\b(creation_time|com\.apple\.quicktime\.creationdate)\s*=", text))

        if not (has_make or has_model):
            signals.append("no_camera_metadata")
            score += 0.15
        if not has_creation:
            signals.append("no_creation_time")
            score += 0.10

        # Generative tool names anywhere in the metadata blob
        for pattern in _SUSPECT_SOFTWARE_PATTERNS:
            if pattern in text:
                signals.append(f"metadata_mentions:{pattern}")
                score += 0.50
                break

        return {
            "metadata_score": round(min(score, 1.0), 4),
            "signals": signals,
            "details": details,
            "verdict": "SUSPICIOUS" if score > 0.6 else ("INCONCLUSIVE" if score > 0.3 else "CLEAN"),
        }
