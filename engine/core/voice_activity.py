"""
Voice-activity detector (VAD) — Silero with heuristic fallback.

Used by both AudioDeepfakeAnalyzer (WavLM) and LipSyncAnalyzer to gate their
inference: both produce garbage scores when there's no actual human speech
in the audio. Examples:
  * WavLM was trained on real-speech vs cloned-speech — silence/music/claps
    make it output ~99% "spoof".
  * Lip sync correlates audio energy with mouth openness — without speech,
    the correlation is between unrelated signals and lands near zero, which
    the score formula misreads as "desynchronized".

Both should call `contains_speech()` first and treat the no-speech case as
INCONCLUSIVE, not as fake.

How the detector works
----------------------
Primary: Silero VAD via torch.hub. ~1.5 MB neural network trained on 6000+
hours of multilingual speech. Distinguishes speech from claps, music,
breathing, animals, etc. Lazy-loaded on first call.

Fallback: pure-numpy heuristic combining three checks — energy, voice-band
spectral concentration (300-3400 Hz), and harmonicity (autocorrelation in
the pitch range). Activates if Silero can't be loaded (offline, network
error, torch issue).
"""

import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)

# ── Silero VAD ────────────────────────────────────────────────────────────
# Lazy-loaded singleton: the torch.hub download happens on the first
# contains_speech() call, not at import time, so the engine can boot offline.

_SILERO_MODEL = None              # torch module
_SILERO_GET_TIMESTAMPS = None     # helper from the Silero utils tuple
_SILERO_LOAD_LOCK = threading.Lock()
_SILERO_LOAD_ATTEMPTED = False    # so we only try (and fail) once
_SILERO_SR = 16000                # Silero supports 16k and 8k; we use 16k everywhere

# Threshold on the model's per-chunk speech probability. Silero's docs
# recommend 0.5 as a balanced default; lower = more sensitive.
SILERO_SPEECH_PROB_THRESHOLD = 0.5
# Minimum total speech duration in the clip to declare "has speech".
SILERO_MIN_SPEECH_SECONDS = 0.30


def _load_silero():
    """Try to load Silero VAD; return (model, get_timestamps_fn) or (None, None)."""
    global _SILERO_MODEL, _SILERO_GET_TIMESTAMPS, _SILERO_LOAD_ATTEMPTED

    with _SILERO_LOAD_LOCK:
        if _SILERO_LOAD_ATTEMPTED:
            return _SILERO_MODEL, _SILERO_GET_TIMESTAMPS

        _SILERO_LOAD_ATTEMPTED = True
        try:
            import torch  # noqa: F401  — silero needs torch
            # torch.hub caches the model under ~/.cache/torch/hub after the
            # first download (~1.5 MB). Subsequent calls are instant.
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            get_speech_timestamps = utils[0]  # (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
            _SILERO_MODEL = model
            _SILERO_GET_TIMESTAMPS = get_speech_timestamps
            logger.info("✅ Silero VAD loaded")
        except Exception as e:
            logger.warning(
                f"Silero VAD unavailable ({e!s}). Falling back to heuristic VAD."
            )
            _SILERO_MODEL = None
            _SILERO_GET_TIMESTAMPS = None
        return _SILERO_MODEL, _SILERO_GET_TIMESTAMPS


def _contains_speech_silero(signal: np.ndarray, sr: int) -> bool | None:
    """
    Run Silero VAD on `signal`. Returns:
      True/False if it ran successfully
      None      if the model couldn't be used (caller should fall back)
    """
    model, get_speech_timestamps = _load_silero()
    if model is None or get_speech_timestamps is None:
        return None

    try:
        import torch

        # Silero wants float32 mono at 16k or 8k
        if sr not in (8000, 16000):
            # Resample to 16k for the VAD call only; we don't mutate the input
            try:
                import librosa
                resampled = librosa.resample(
                    signal.astype(np.float32), orig_sr=sr, target_sr=_SILERO_SR,
                )
                vad_sr = _SILERO_SR
            except Exception:
                # No librosa or it failed — give up on Silero for this call
                return None
        else:
            resampled = signal.astype(np.float32)
            vad_sr = sr

        tensor = torch.from_numpy(resampled)
        ts = get_speech_timestamps(
            tensor,
            model,
            sampling_rate=vad_sr,
            threshold=SILERO_SPEECH_PROB_THRESHOLD,
            return_seconds=True,
        )

        total_speech_s = sum((seg["end"] - seg["start"]) for seg in ts)
        result = total_speech_s >= SILERO_MIN_SPEECH_SECONDS
        logger.info(
            f"VAD(Silero): {len(ts)} speech segment(s), total {total_speech_s:.2f}s "
            f"(need ≥{SILERO_MIN_SPEECH_SECONDS}s) → {'speech' if result else 'no speech'}"
        )
        return result
    except Exception as e:
        logger.warning(f"Silero VAD inference failed ({e!s}); falling back to heuristic")
        return None


def speech_timestamps(signal: np.ndarray, sr: int) -> list[dict] | None:
    """
    Return per-segment speech timestamps via Silero, e.g.
        [{"start": 0.3, "end": 1.8}, {"start": 2.4, "end": 4.1}]
    Returns None if Silero isn't usable. Handy for downstream callers that
    want to crop to actual speech regions before further analysis.
    """
    model, get_ts = _load_silero()
    if model is None or get_ts is None:
        return None
    try:
        import torch
        if sr not in (8000, 16000):
            import librosa
            resampled = librosa.resample(
                signal.astype(np.float32), orig_sr=sr, target_sr=_SILERO_SR,
            )
            vad_sr = _SILERO_SR
        else:
            resampled = signal.astype(np.float32)
            vad_sr = sr
        tensor = torch.from_numpy(resampled)
        return list(get_ts(tensor, model, sampling_rate=vad_sr, return_seconds=True))
    except Exception:
        return None

# ── Default thresholds (tunable by callers) ───────────────────────────────
DEFAULT_RMS_THRESHOLD = 0.015        # below this RMS = silence
DEFAULT_MIN_ACTIVE_RATIO = 0.10      # at least 10% of windows must be above threshold
DEFAULT_MIN_RMS_VARIABILITY = 0.005  # speech ramps up/down; constant tones are flat
DEFAULT_WINDOW_SECONDS = 0.05        # 50 ms windows — syllable-level granularity

# Voice-band fraction: how much spectral energy must sit in 300–3400 Hz.
# Claps/music typically <0.35. Clean phone speech is ~0.7+; recorded speech
# from a phone/laptop mic typically 0.45–0.85.
DEFAULT_MIN_VOICE_BAND_ENERGY_RATIO = 0.35

# Harmonicity: autocorrelation peak in the pitch range (80–400 Hz) divided by
# the zero-lag energy. Clean speech ~0.5+; claps ~0.05; constant tones ~1.0
# (we cap at 0.95 to avoid catching pure sine waves).
DEFAULT_MIN_HARMONICITY = 0.20
DEFAULT_MAX_HARMONICITY = 0.95


def _voice_band_ratio(signal: np.ndarray, sr: int) -> float:
    """
    Fraction of spectral energy in the 300–3400 Hz human-voice band.
    Computed on a single FFT of the whole signal — adequate for the
    typical clip lengths we see (1-180s) and avoids dependencies.
    """
    n = len(signal)
    if n < 32:
        return 0.0
    spectrum = np.abs(np.fft.rfft(signal.astype(np.float32)))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    total = float(np.sum(spectrum ** 2)) + 1e-12
    band = (freqs >= 300.0) & (freqs <= 3400.0)
    band_energy = float(np.sum(spectrum[band] ** 2))
    return band_energy / total


def _harmonicity(signal: np.ndarray, sr: int) -> float:
    """
    Peak autocorrelation value in the pitch-period band (80–400 Hz fundamental).
    Speech is periodic at the pitch rate → strong peak. Claps/noise/percussion
    are aperiodic → flat autocorrelation. Pure tones saturate near 1.0 (we
    cap them out at the caller via DEFAULT_MAX_HARMONICITY).

    Works on at most the central 2 seconds of audio to keep this cheap.
    """
    if sr <= 0 or len(signal) < sr // 10:
        return 0.0
    # Trim to a window in the middle for speed; remove DC.
    max_samples = sr * 2
    if len(signal) > max_samples:
        mid = len(signal) // 2
        chunk = signal[max(0, mid - max_samples // 2): mid + max_samples // 2]
    else:
        chunk = signal
    x = chunk.astype(np.float32) - float(np.mean(chunk))
    energy = float(np.dot(x, x)) + 1e-12

    # Pitch period range: 80 Hz → sr/80, 400 Hz → sr/400
    min_lag = max(int(sr / 400.0), 1)
    max_lag = min(int(sr / 80.0), len(x) - 1)
    if max_lag <= min_lag:
        return 0.0

    # Compute autocorrelation only in the relevant lag range
    best = 0.0
    for lag in range(min_lag, max_lag + 1, max(1, (max_lag - min_lag) // 64)):
        # Subsample lags for speed (64 evenly-spaced samples through the range)
        ac = float(np.dot(x[:len(x) - lag], x[lag:])) / energy
        if ac > best:
            best = ac
    return float(np.clip(best, 0.0, 1.0))


def contains_speech(
    signal: np.ndarray,
    sr: int,
    rms_threshold: float = DEFAULT_RMS_THRESHOLD,
    min_active_ratio: float = DEFAULT_MIN_ACTIVE_RATIO,
    min_rms_variability: float = DEFAULT_MIN_RMS_VARIABILITY,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    min_voice_band_ratio: float = DEFAULT_MIN_VOICE_BAND_ENERGY_RATIO,
    min_harmonicity: float = DEFAULT_MIN_HARMONICITY,
    max_harmonicity: float = DEFAULT_MAX_HARMONICITY,
) -> bool:
    """
    Return True iff `signal` plausibly contains human speech.

    Strategy: try Silero VAD (neural network, multilingual, robust to claps/
    music/noise). If Silero can't be loaded — torch.hub network error,
    offline, torch missing — fall back to the pure-numpy heuristic.

    The heuristic kwargs are only consulted when the fallback runs.
    """
    if signal is None or sr <= 0:
        return False

    silero_result = _contains_speech_silero(signal, sr)
    if silero_result is not None:
        return silero_result

    return _contains_speech_heuristic(
        signal, sr,
        rms_threshold=rms_threshold,
        min_active_ratio=min_active_ratio,
        min_rms_variability=min_rms_variability,
        window_seconds=window_seconds,
        min_voice_band_ratio=min_voice_band_ratio,
        min_harmonicity=min_harmonicity,
        max_harmonicity=max_harmonicity,
    )


def _contains_speech_heuristic(
    signal: np.ndarray,
    sr: int,
    rms_threshold: float = DEFAULT_RMS_THRESHOLD,
    min_active_ratio: float = DEFAULT_MIN_ACTIVE_RATIO,
    min_rms_variability: float = DEFAULT_MIN_RMS_VARIABILITY,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    min_voice_band_ratio: float = DEFAULT_MIN_VOICE_BAND_ENERGY_RATIO,
    min_harmonicity: float = DEFAULT_MIN_HARMONICITY,
    max_harmonicity: float = DEFAULT_MAX_HARMONICITY,
) -> bool:
    """
    Pure-numpy fallback when Silero VAD is unavailable.

    Three conditions must hold:
      1. Energy — meaningful fraction of windows above an RMS floor
         (eliminates silence).
      2. Voice-band concentration — enough spectral energy in 300-3400 Hz
         (eliminates claps, percussion, music focused outside this band).
      3. Harmonicity — periodic structure typical of voiced speech
         (eliminates broadband noise / impulsive sounds).
    """
    win = max(int(window_seconds * sr), 1)
    if len(signal) < win * 4:
        return False

    n_windows = len(signal) // win
    if n_windows < 4:
        return False

    trimmed = signal[: n_windows * win].reshape(n_windows, win)
    rms = np.sqrt(np.mean(trimmed.astype(np.float32) ** 2, axis=1))

    active_ratio = float(np.mean(rms > rms_threshold))
    rms_std = float(np.std(rms))
    has_active = active_ratio >= min_active_ratio
    has_variability = rms_std >= min_rms_variability

    band_ratio = _voice_band_ratio(signal, sr)
    has_voice_band = band_ratio >= min_voice_band_ratio

    harm = _harmonicity(signal, sr)
    has_harmonicity = min_harmonicity <= harm <= max_harmonicity

    result = has_active and has_variability and has_voice_band and has_harmonicity

    logger.info(
        f"VAD(heuristic): active={active_ratio:.2f}/{min_active_ratio:.2f} "
        f"std={rms_std:.4f}/{min_rms_variability:.4f} "
        f"band={band_ratio:.2f}/{min_voice_band_ratio:.2f} "
        f"harm={harm:.2f}∈[{min_harmonicity:.2f},{max_harmonicity:.2f}] "
        f"→ {'speech' if result else 'no speech'}"
    )
    return result
