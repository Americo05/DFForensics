"""
Audio Deepfake Analyzer — WavLM-based Voice Cloning Detection

Technique:
    Extracts the audio track from the video and processes it using a WavLM 
    (Waveform Learning Model) fine-tuned for Anti-Spoofing / Deepfake detection.
    WavLM is an SSL model that excels at capturing phonetic and speaker-specific
    features, allowing it to robustly distinguish between genuine human speech 
    and AI-generated/cloned voices (e.g., VALL-E, ElevenLabs).

    Model: abhishtagatya/wavlm-base-960h-itw-deepfake
    This is an "In-The-Wild" deepfake audio classifier.

Dependencies:
    pip install transformers torch librosa imageio-ffmpeg

Version: 1.0.0
"""

import os
import logging
import tempfile
import subprocess
import numpy as np

try:
    import librosa
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from core.voice_activity import contains_speech

logger = logging.getLogger(__name__)


class AudioDeepfakeAnalyzer:
    """
    Video-level audio deepfake (voice cloning) consistency analyzer.
    Analyzes the entire audio track.
    """

    def __init__(self):
        self._pipe = None
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers/torch/librosa not installed. Audio Deepfake analysis disabled.")
            return

        model_name = "abhishtagatya/wavlm-base-960h-itw-deepfake"
        logger.info(f"Loading WavLM Audio Deepfake model: {model_name}...")

        device = 0 if torch.cuda.is_available() else -1
        try:
            self._pipe = pipeline("audio-classification", model=model_name, device=device)
            logger.info(f"✅ WavLM Audio Deepfake model loaded (device={'GPU' if device == 0 else 'CPU'})")
        except Exception as e:
            logger.error(f"Failed to load WavLM model: {e}")
            self._pipe = None

    # ── Chunking parameters ────────────────────────────────────────────────
    # WavLM has a finite context window; >30s passes risk OOM on common hardware.
    # We chunk to 10s segments and report the WORST (highest fake score) because
    # a single cloned segment is enough evidence — the rest being real doesn't
    # absolve the fake. Cap total audio analysed to keep latency bounded.
    CHUNK_SECONDS = 10.0
    CHUNK_STRIDE_SECONDS = 8.0   # 2s overlap so boundary fakes aren't split
    MAX_TOTAL_SECONDS = 180.0    # ≤ 3 minutes of audio analysed
    MIN_CHUNK_SECONDS = 1.0

    # ── Speech-presence thresholds ─────────────────────────────────────────
    # WavLM was trained on REAL speech vs CLONED speech. Feeding it silence,
    # background music, or just ambient noise produces garbage scores (often
    # very high "spoof" probability) because none of that matches its training
    # distribution. We pre-check the audio for actual human speech and bail
    # out as INCONCLUSIVE when there's none.
    SPEECH_RMS_THRESHOLD = 0.015        # below this = silence
    SPEECH_MIN_ACTIVE_RATIO = 0.10      # need ≥10% of frames above the threshold
    SPEECH_MIN_RMS_VARIABILITY = 0.005  # speech has ups and downs; constant tone = no speech

    def analyze_audio(self, video_path: str) -> dict | None:
        """
        Extracts audio from a video and analyzes it for voice cloning across
        the WHOLE track via chunked sliding-window inference.

        Pipeline:
          1. Extract audio (handled by _extract_audio).
          2. Resample to 16kHz if needed.
          3. Slide CHUNK_SECONDS windows with CHUNK_STRIDE_SECONDS stride.
          4. For each chunk: run WavLM, extract fake probability.
          5. Return MAX fake score across chunks (worst-case) plus per-chunk
             details so the caller can see where the suspicion peaked.

        Returns a dict with the score, or None if no analysable audio.
        """
        if self._pipe is None:
            return None
        if not video_path or not os.path.isfile(video_path):
            return None

        try:
            audio_data = self._extract_audio(video_path)
            if audio_data is None:
                logger.info("No audio track found — Audio Deepfake analysis skipped")
                return None

            signal, sr = audio_data

            if sr != 16000:
                signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Hard cap to keep latency bounded
            max_samples = int(self.MAX_TOTAL_SECONDS * sr)
            if len(signal) > max_samples:
                signal = signal[:max_samples]

            if len(signal) < sr * self.MIN_CHUNK_SECONDS:
                return None  # Too short

            if np.max(np.abs(signal)) < 0.01:
                logger.info("Audio track is silent — Audio Deepfake analysis skipped")
                return None

            # Speech-presence gate: skip if the audio doesn't contain human speech.
            # Without this, WavLM happily emits 99% "spoof" on silent/music tracks.
            if not contains_speech(
                signal, sr,
                rms_threshold=self.SPEECH_RMS_THRESHOLD,
                min_active_ratio=self.SPEECH_MIN_ACTIVE_RATIO,
                min_rms_variability=self.SPEECH_MIN_RMS_VARIABILITY,
            ):
                logger.info("WavLM: no detectable human speech — returning inconclusive")
                return {
                    "audio_fake_score": 0.5,
                    "chunks_evaluated": 0,
                    "inconclusive": True,
                    "reason": "no_speech_detected",
                    "verdict": "INCONCLUSIVE",
                }

            chunk_size = int(self.CHUNK_SECONDS * sr)
            stride = max(int(self.CHUNK_STRIDE_SECONDS * sr), 1)
            min_chunk_samples = int(self.MIN_CHUNK_SECONDS * sr)

            chunk_scores: list[tuple[float, float, float]] = []  # (start_s, end_s, score)
            worst_score = 0.0

            for start in range(0, len(signal), stride):
                end = min(start + chunk_size, len(signal))
                if end - start < min_chunk_samples:
                    break

                chunk = signal[start:end]
                if np.max(np.abs(chunk)) < 0.01:
                    continue  # silent slice, no info

                fake_score = self._score_chunk(chunk)
                if fake_score is None:
                    continue

                start_s = round(start / sr, 2)
                end_s = round(end / sr, 2)
                chunk_scores.append((start_s, end_s, fake_score))
                if fake_score > worst_score:
                    worst_score = fake_score

                # Don't slide past the end if the last chunk reached it
                if end >= len(signal):
                    break

            if not chunk_scores:
                logger.warning("WavLM: no chunks produced a score")
                return None

            worst_start_s, worst_end_s, _ = max(chunk_scores, key=lambda c: c[2])
            logger.info(
                f"WavLM: {len(chunk_scores)} chunk(s); worst score "
                f"{worst_score:.3f} at {worst_start_s}s-{worst_end_s}s"
            )

            return {
                "audio_fake_score": round(worst_score, 4),
                "chunks_evaluated": len(chunk_scores),
                "worst_chunk_start_s": worst_start_s,
                "worst_chunk_end_s": worst_end_s,
                "verdict": "SUSPICIOUS" if worst_score > 0.6 else "AUTHENTIC",
            }

        except Exception as e:
            logger.error(f"Audio Deepfake analysis failed: {e}")
            return None

    def _score_chunk(self, chunk: np.ndarray) -> float | None:
        """Run WavLM on one audio chunk; return fake probability in [0,1]."""
        try:
            results = self._pipe(chunk)
        except Exception as e:
            logger.warning(f"WavLM chunk inference failed: {e}")
            return None

        if not isinstance(results, list) or not results:
            return None

        fake_score = 0.5
        for item in results:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).lower()
            try:
                score = float(item.get("score", 0.5))
            except (TypeError, ValueError):
                continue
            if "spoof" in label or "fake" in label:
                fake_score = score
                break
            if "bonafide" in label or "real" in label:
                fake_score = 1.0 - score
        return fake_score

    def _extract_audio(self, video_path: str) -> tuple | None:
        """Extract audio from video file using FFmpeg, return (signal, sr)."""
        ffmpeg_exe = self._get_ffmpeg_path()
        if not ffmpeg_exe:
            logger.warning("FFmpeg binary not found — Audio Deepfake skipped")
            return None

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            cmd = [
                ffmpeg_exe, "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                tmp_path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )

            if result.returncode != 0 or not os.path.exists(tmp_path):
                return None

            if os.path.getsize(tmp_path) < 1000:
                return None

            signal, sr = librosa.load(tmp_path, sr=16000, mono=True)
            return signal, sr

        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return None
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _get_ffmpeg_path() -> str | None:
        try:
            import imageio_ffmpeg
            path = imageio_ffmpeg.get_ffmpeg_exe()
            if os.path.isfile(path):
                return path
        except ImportError:
            pass

        import shutil
        sys_ffmpeg = shutil.which("ffmpeg")
        if sys_ffmpeg:
            return sys_ffmpeg

        return None
