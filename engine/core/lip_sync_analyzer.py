"""
Lip Sync Analyzer — Video-level audio-visual consistency check.

Technique:
    Analyzes the correlation between audio energy and lip/mouth movement
    across video frames. In authentic videos, mouth opening correlates
    strongly with audio amplitude. In lip-synced deepfakes, there is a
    measurable desynchronization between audio and visual lip motion.

    Based on: "Lips Are Lying: Spotting the Temporal Inconsistency between
    Audio and Visual in Lip-Syncing DeepFakes" (NeurIPS 2024, ref. 46)
    and LipFD methodology (ref. 46).

    This module operates at the VIDEO level (not per-frame) and is called
    separately from the per-frame plugin pipeline. It requires:
      - Multiple sequential frames from the video
      - Audio track extracted via FFmpeg

    For image-only inputs, this analysis is SKIPPED and returns None.

Dependencies:
    pip install librosa numpy opencv-python
    FFmpeg must be available on the system (for audio extraction)

Version: 1.0.0
"""

import cv2
import numpy as np
import logging
import subprocess
import tempfile
import os

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from core.voice_activity import contains_speech

logger = logging.getLogger(__name__)


class LipSyncAnalyzer:
    """
    Video-level lip sync consistency analyzer.
    NOT a plugin (doesn't inherit BaseDetectorPlugin) because it operates
    on the entire video, not individual frames.
    """

    def __init__(self):
        # MediaPipe FaceMesh initialization
        if MEDIAPIPE_AVAILABLE:
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=True,   # Must be True: our frames are sampled at 2fps, NOT consecutive video
                max_num_faces=6,          # Support up to 6 people in scene
                refine_landmarks=True,
                min_detection_confidence=0.4,
            )
            logger.info("✅ Lip Sync Analyzer: MediaPipe FaceMesh loaded")
        else:
            self._face_mesh = None
            logger.warning("Lip Sync Analyzer: MediaPipe not available. Analysis will fail.")

    # ── Windowing parameters ───────────────────────────────────────────────
    # 4-second windows match the original LipFD design. 1-second overlap means a
    # boundary deepfake (e.g. a 2-second clip stitched at second 8) is covered
    # by two windows so we don't miss it. Tune cautiously — bigger windows hurt
    # locality, smaller ones hurt correlation reliability.
    WINDOW_SECONDS = 4.0
    WINDOW_STRIDE_SECONDS = 3.0  # = WINDOW_SECONDS - 1s overlap
    MIN_WINDOW_FRAMES = 10       # need at least this many frames per window
    # Cap how much video we load to keep RAM bounded on huge inputs.
    MAX_TOTAL_SECONDS = 120.0    # analyse up to 2 minutes (covers most clips)

    def analyze_video(self, video_path: str) -> dict | None:
        """
        Analyze audio-visual lip sync consistency across the WHOLE video using
        a sliding window.

        Pipeline:
          1. Extract continuous frames at native FPS (capped at MAX_TOTAL_SECONDS).
          2. Extract the matching audio track.
          3. Slide a WINDOW_SECONDS window with stride WINDOW_STRIDE_SECONDS.
          4. For each window: compute mouth movement per face + audio energy,
             then correlate them.
          5. Return the WORST (highest fake_score) conclusive window — a single
             desynchronized segment is enough to flag the video.

        Returns:
            Dict with lip_sync_score and details, or None if analysis
            cannot be performed (no audio, not a video, etc.)
        """
        if not LIBROSA_AVAILABLE:
            logger.warning("librosa not installed — lip sync analysis skipped")
            return None

        if not video_path or not os.path.isfile(video_path):
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames_total = int(min(self.MAX_TOTAL_SECONDS * fps, max(total_video_frames, 1)))

        continuous_frames: list = []
        try:
            while len(continuous_frames) < max_frames_total:
                ret, frame = cap.read()
                if not ret:
                    break
                continuous_frames.append(frame)
        finally:
            cap.release()

        if len(continuous_frames) < self.MIN_WINDOW_FRAMES:
            logger.info(
                f"Too few frames for lip sync ({len(continuous_frames)} < {self.MIN_WINDOW_FRAMES})"
            )
            return None

        try:
            audio_data = self._extract_audio(video_path)
            if audio_data is None:
                logger.info("No audio track found — lip sync analysis skipped")
                return None

            audio_signal, sr = audio_data

            if np.max(np.abs(audio_signal)) < 0.01:
                logger.info("Audio track is silent — lip sync analysis skipped")
                return None

            # Speech-presence gate: lip sync correlates AUDIO energy with mouth
            # movement. Without actual human speech the correlation is between
            # ambient noise variations and small involuntary lip movements (e.g.
            # breathing, blinking artifacts) — which lands near zero or even
            # negative, and the score formula reads that as desynchronization.
            # That's not what's happening — there's just nothing to correlate.
            if not contains_speech(audio_signal, sr):
                logger.info("Lip sync: no detectable human speech — returning inconclusive")
                return {
                    "lip_sync_score": 0.5,
                    "correlation": 0.0,
                    "frames_analyzed": 0,
                    "windows_evaluated": 0,
                    "inconclusive": True,
                    "reason": "no_speech_detected",
                    "verdict": "INCONCLUSIVE",
                }

            # Audio energy per video frame for the entire loaded segment
            audio_energy = self._compute_audio_energy(
                audio_signal, sr, len(continuous_frames), fps,
            )

            # Mouth openness per face for the entire loaded segment
            faces_openness = self._compute_mouth_movement(continuous_frames)

            # Slide a window across the timeline; keep the WORST conclusive score.
            window_size = int(self.WINDOW_SECONDS * fps)
            stride = max(int(self.WINDOW_STRIDE_SECONDS * fps), 1)
            n_total = len(continuous_frames)

            worst_score: float | None = None
            worst_details: dict | None = None
            inconclusive_fallback: tuple[float, dict] | None = None
            windows_evaluated = 0

            for start in range(0, max(n_total - window_size + 1, 1), stride):
                end = min(start + window_size, n_total)
                if end - start < self.MIN_WINDOW_FRAMES:
                    break

                audio_slice = audio_energy[start:end]
                # Best (lowest fake) score among faces in this window
                window_best_score: float | None = None
                window_best_details: dict | None = None
                window_inconclusive: tuple[float, dict] | None = None

                for mouth in faces_openness:
                    mouth_slice = mouth[start:end]
                    score, details = self._compute_sync_score(audio_slice, mouth_slice)
                    if details.get("inconclusive"):
                        if window_inconclusive is None:
                            window_inconclusive = (score, details)
                        continue
                    if window_best_score is None or score < window_best_score:
                        window_best_score = score
                        window_best_details = details

                if window_best_score is not None and window_best_details is not None:
                    windows_evaluated += 1
                    # Worst across windows = highest fake_score
                    if worst_score is None or window_best_score > worst_score:
                        worst_score = window_best_score
                        worst_details = {
                            **window_best_details,
                            "window_start_seconds": round(start / fps, 2),
                            "window_end_seconds": round(end / fps, 2),
                            "windows_evaluated": windows_evaluated,
                        }
                elif window_inconclusive is not None and inconclusive_fallback is None:
                    inconclusive_fallback = window_inconclusive

            if worst_score is not None and worst_details is not None:
                logger.info(
                    f"Lip sync: {windows_evaluated} window(s) evaluated; "
                    f"worst score {worst_score:.3f} at "
                    f"{worst_details.get('window_start_seconds')}s-"
                    f"{worst_details.get('window_end_seconds')}s"
                )
                return {
                    "lip_sync_score": round(worst_score, 4),
                    "correlation": round(worst_details["correlation"], 4),
                    "frames_analyzed": worst_details["frames_with_mouth"],
                    "windows_evaluated": windows_evaluated,
                    "worst_window_start_s": worst_details.get("window_start_seconds"),
                    "worst_window_end_s": worst_details.get("window_end_seconds"),
                    "inconclusive": False,
                    "verdict": "SUSPICIOUS" if worst_score > 0.6 else "CONSISTENT",
                }

            if inconclusive_fallback is not None:
                fb_score, fb_details = inconclusive_fallback
                return {
                    "lip_sync_score": round(fb_score, 4),
                    "correlation": round(fb_details["correlation"], 4),
                    "frames_analyzed": fb_details["frames_with_mouth"],
                    "windows_evaluated": 0,
                    "inconclusive": True,
                    "verdict": "INCONCLUSIVE",
                }

            return None

        except Exception as e:
            logger.error(f"Lip sync analysis failed: {e}")
            return None

    def _extract_audio(self, video_path: str) -> tuple | None:
        """Extract audio from video file using FFmpeg, return (signal, sr)."""
        ffmpeg_exe = self._get_ffmpeg_path()
        if not ffmpeg_exe:
            logger.warning("FFmpeg binary not found — lip sync skipped")
            return None

        # Create the temp file path but guarantee cleanup via try/finally
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
                logger.warning(f"FFmpeg audio extraction failed (code {result.returncode})")
                return None

            # Tiny file = effectively no audio data
            if os.path.getsize(tmp_path) < 1000:
                return None

            signal, sr = librosa.load(tmp_path, sr=16000, mono=True)

            if len(signal) < sr * 0.5:  # Less than 0.5 seconds
                return None

            return signal, sr

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"FFmpeg error: {e} — lip sync skipped")
            return None
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
        """Find FFmpeg binary: prefer imageio-ffmpeg, fall back to PATH."""
        # 1. Try imageio-ffmpeg (bundled binary)
        try:
            import imageio_ffmpeg
            path = imageio_ffmpeg.get_ffmpeg_exe()
            if os.path.isfile(path):
                return path
        except ImportError:
            pass

        # 2. Try system PATH
        import shutil
        sys_ffmpeg = shutil.which("ffmpeg")
        if sys_ffmpeg:
            return sys_ffmpeg

        return None

    def _compute_audio_energy(
        self, signal: np.ndarray, sr: int, n_frames: int, fps: float
    ) -> np.ndarray:
        """
        Compute a speech-band energy envelope, aligned to each video frame.

        Why mel-band energy instead of plain RMS:
          * SyncNet and other audio-visual sync models use mel-spectrogram
            features because human speech energy correlates with lip movement
            ONLY in the voice band (~300-3400 Hz). Plain RMS picks up wind,
            mic rumble, electrical hum — none of which the lips produce.
          * librosa's `melspectrogram` aggregates power into perceptually-
            spaced bins. Summing the bins that overlap the voice band gives
            a single "how loud is speech in this window" number per frame.

        Falls back to RMS if librosa fails (no extra deps; this module
        already imports librosa for audio extraction).
        """
        if fps <= 0:
            fps = 30.0
        frame_duration = 1.0 / fps
        hop_length = max(int(sr * frame_duration), 1)

        try:
            # Mel-spectrogram with the hop length matched to one video frame
            mel = librosa.feature.melspectrogram(
                y=signal.astype(np.float32),
                sr=sr,
                n_mels=40,
                fmin=200.0,    # below typical voice fundamental
                fmax=3800.0,   # above formant region; clips highs (claps, hiss)
                hop_length=hop_length,
                n_fft=max(hop_length * 2, 1024),
                power=2.0,
            )
            # Convert to dB so dynamic range is comparable across recordings,
            # then average across bins to get one scalar per frame.
            mel_db = librosa.power_to_db(mel + 1e-10, ref=np.max)
            energy = np.mean(mel_db, axis=0)
            # Rescale to [0, 1]: high (close to 0 dB) = loud speech;
            # very negative = silence.
            energy = energy - energy.min()
            if energy.max() > 0:
                energy = energy / energy.max()

            # Resample to exactly n_frames if hop_length didn't land us there.
            if len(energy) != n_frames:
                idx = np.linspace(0, max(len(energy) - 1, 1), n_frames)
                energy = np.interp(idx, np.arange(len(energy)), energy)
            return energy.astype(np.float32)

        except Exception as e:
            logger.warning(f"Mel-band energy failed ({e}); falling back to RMS")
            energies = []
            for i in range(n_frames):
                start_sample = int(i * frame_duration * sr)
                end_sample = int((i + 1) * frame_duration * sr)
                if start_sample >= len(signal):
                    energies.append(0.0)
                    continue
                end_sample = min(end_sample, len(signal))
                chunk = signal[start_sample:end_sample]
                rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
                energies.append(rms)
            energy = np.array(energies, dtype=np.float32)
            if energy.max() > 0:
                energy = energy / energy.max()
            return energy

    def _compute_mouth_movement(self, frames: list) -> list[np.ndarray]:
        """
        Estimate mouth openness using MediaPipe Facial Landmarks in 3D.
        Returns a list of 1D numpy arrays (one array per face index), normalized [0, 1].
        """
        if not self._face_mesh:
            return [np.zeros(len(frames))]

        # Store openness values per face index: face_idx -> list of distances
        faces_openness = {}

        for frame_idx, frame in enumerate(frames):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                continue

            h, w, _ = frame.shape

            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                upper_lip = face_landmarks.landmark[13]
                lower_lip = face_landmarks.landmark[14]

                ux, uy = upper_lip.x * w, upper_lip.y * h
                lx, ly = lower_lip.x * w, lower_lip.y * h

                distance = np.sqrt((ux - lx)**2 + (uy - ly)**2)
                
                if face_idx not in faces_openness:
                    # Pad with 0 for previous frames where this face wasn't detected
                    faces_openness[face_idx] = [0.0] * frame_idx
                
                faces_openness[face_idx].append(distance)

            # Pad faces that were lost in this frame
            for face_idx in faces_openness.keys():
                if len(faces_openness[face_idx]) <= frame_idx:
                    faces_openness[face_idx].append(0.0)

        if not faces_openness:
            return [np.zeros(len(frames))]

        results = []
        for face_idx, openness_list in faces_openness.items():
            arr = np.array(openness_list)
            if arr.max() > 0:
                arr = arr / arr.max()
            results.append(arr)

        return results

    def _compute_sync_score(
        self, audio_energy: np.ndarray, mouth_openness: np.ndarray
    ) -> tuple:
        """
        Compute synchronization score between audio and lip movement.

        Returns:
            (fake_probability, details_dict)
            Higher score = more likely desynchronized (fake lip sync)
        """
        n = min(len(audio_energy), len(mouth_openness))
        audio_energy = audio_energy[:n]
        mouth_openness = mouth_openness[:n]

        # Count frames where mouth was detected
        frames_with_mouth = int(np.sum(mouth_openness > 0))

        if frames_with_mouth < 5:
            return 0.5, {"correlation": 0.0, "frames_with_mouth": frames_with_mouth, "inconclusive": True}

        # Only consider frames where there's audio activity
        active_mask = audio_energy > 0.05
        if active_mask.sum() < 3:
            # Very little audio — can't judge sync
            return 0.3, {"correlation": 0.0, "frames_with_mouth": frames_with_mouth, "inconclusive": True}

        # Pearson correlation between audio energy and mouth movement
        a = audio_energy[active_mask]
        m = mouth_openness[active_mask]

        # The old heuristic "audio fluctuates + mouth still = BAD LIP SYNC"
        # only makes sense IF there is actual human speech in the audio.
        # A normal recording where the person isn't speaking will fail it
        # spuriously (audio has ambient variation, lips are naturally still).
        #
        # New rule:
        #   * audio flat (silence/constant tone) → INCONCLUSIVE
        #   * mouth flat AND audio shows clear speech-like variability → SUSPICIOUS
        #   * mouth flat AND audio is mostly quiet/low-variability → INCONCLUSIVE
        AUDIO_FLAT_STD = 1e-6
        MOUTH_FLAT_STD = 1e-6
        # "Speech-like" = audio std is high AND mean is meaningfully above floor.
        SPEECH_AUDIO_STD = 0.10
        SPEECH_AUDIO_MEAN = 0.10

        if a.std() < AUDIO_FLAT_STD:
            return 0.5, {"correlation": 0.0, "frames_with_mouth": frames_with_mouth, "inconclusive": True}

        if m.std() < MOUTH_FLAT_STD:
            audio_looks_like_speech = (a.std() >= SPEECH_AUDIO_STD and a.mean() >= SPEECH_AUDIO_MEAN)
            if audio_looks_like_speech:
                # Real speech but no mouth movement at all → desynchronization.
                return 1.0, {"correlation": 0.0, "frames_with_mouth": frames_with_mouth}
            # No speech in audio either — the person isn't talking. Normal.
            return 0.5, {
                "correlation": 0.0,
                "frames_with_mouth": frames_with_mouth,
                "inconclusive": True,
            }

        correlation = float(np.corrcoef(a, m)[0, 1])
        if np.isnan(correlation):
            return 0.5, {"correlation": 0.0, "frames_with_mouth": frames_with_mouth, "inconclusive": True}

        # ── Lag exploration ±MAX_LAG frames ────────────────────────────────
        # SyncNet and similar audio-visual sync models recognise that real
        # videos exhibit small but non-zero audio-visual offsets (typical
        # range: ±5 frames at 25-30fps, i.e. up to ~150 ms). The original
        # zero/±1 lag check was too narrow and penalised honest recordings
        # whose sync had a small inherent offset. We now search a wider window
        # and take the BEST correlation across all lags as the sync confidence.
        MAX_LAG = 5
        best_corr = correlation
        best_lag = 0
        if len(a) >= 2 * MAX_LAG + 5:
            for lag in range(-MAX_LAG, MAX_LAG + 1):
                if lag == 0:
                    continue
                try:
                    if lag > 0:
                        # Audio leads — mouth follows by `lag` frames
                        slice_a = a[:-lag]
                        slice_m = m[lag:]
                    else:
                        # Mouth leads (rare, but possible with playback delay)
                        slice_a = a[-lag:]
                        slice_m = m[:lag]
                    if len(slice_a) < 5:
                        continue
                    c = float(np.corrcoef(slice_a, slice_m)[0, 1])
                    if not np.isnan(c) and c > best_corr:
                        best_corr = c
                        best_lag = lag
                except Exception:
                    continue
            if best_lag != 0:
                logger.debug(f"Lip sync: best correlation {best_corr:.3f} at lag {best_lag} frames")

        # High positive correlation = audio matches lip movement = REAL
        # Low or negative correlation = desynchronized = FAKE
        # Natural: correlation > 0.4
        # Lip-synced fake: correlation < 0.15
        if best_corr > 0.4:
            fake_score = float(np.clip((0.5 - best_corr) / 0.5, 0.0, 0.3))
        elif best_corr > 0.15:
            fake_score = float(np.clip((0.4 - best_corr) / 0.4, 0.0, 0.7))
        else:
            fake_score = float(np.clip(0.7 + (0.15 - best_corr) * 2, 0.7, 1.0))

        details = {
            "correlation": best_corr,
            "frames_with_mouth": frames_with_mouth,
        }

        return fake_score, details
