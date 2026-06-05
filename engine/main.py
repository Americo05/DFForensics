import asyncio
import json
import os
import uuid
import shutil
import threading
import cv2
import logging
import tempfile
import time
from collections import OrderedDict, deque
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from core.plugin_manager import PluginManager
from core.lip_sync_analyzer import LipSyncAnalyzer
from core.audio_deepfake_analyzer import AudioDeepfakeAnalyzer
from core.metadata_analyzer import MetadataAnalyzer
from core.temporal_coherence_analyzer import TemporalCoherenceAnalyzer
from core.rppg_analyzer import rPPGAnalyzer
from core import history_store

logging.basicConfig(level=logging.INFO) # Auto-reload trigger
logger = logging.getLogger(__name__)

app = FastAPI(title="Deepfake Forensics Engine — Plugin System")

# CORS — explicit origins only. Adding "*" alongside specific origins is
# misleading: the wildcard alone matches everything and disables credentialed
# auth. Add new dev hostnames here as needed.
_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://192.168.113.50:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all plugins at server startup
plugin_manager = PluginManager()
lip_sync_analyzer = LipSyncAnalyzer()
audio_deepfake_analyzer = AudioDeepfakeAnalyzer()
metadata_analyzer = MetadataAnalyzer()
temporal_analyzer = TemporalCoherenceAnalyzer()
rppg_analyzer = rPPGAnalyzer()

# Initialise the local SQLite history DB. Idempotent — creates ~/.deepfake-forensics/
# and the analyses table only if missing. Failure here is non-fatal: the engine
# still serves analyses, just without persistence across restarts.
try:
    history_store.init_db()
except Exception as _hist_err:
    logger.warning(f"History DB init failed ({_hist_err}); persistence disabled this session.")

# ── Auth / Rate limit / Upload size limit ─────────────────────────────────
# Auth: comma-separated API keys in env ENGINE_API_KEYS. Empty → auth DISABLED
#       (dev default; never deploy publicly with this empty).
# Rate limit: simple in-process sliding window per client identity (API key
#       when present, otherwise client IP). Sufficient for single-process
#       FastAPI; a multi-worker deploy would need Redis. Free-tier policy
#       protects the GPU and the Sightengine quota from abuse.
# Upload size: bounded by MAX_UPLOAD_BYTES via Content-Length probe + a
#       streaming size cap during write, so an attacker can't lie in the
#       header and dump a 10GB file to disk.

_API_KEYS: set[str] = {
    k.strip() for k in os.environ.get("ENGINE_API_KEYS", "").split(",") if k.strip()
}
_AUTH_ENABLED = bool(_API_KEYS)

RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("ENGINE_RATE_LIMIT_PER_MIN", "10"))
MAX_UPLOAD_BYTES = int(os.environ.get("ENGINE_MAX_UPLOAD_MB", "200")) * 1024 * 1024

_rate_buckets: dict[str, deque] = {}
_RATE_LOCK = threading.Lock()

if _AUTH_ENABLED:
    logger.info(f"Auth ENABLED ({len(_API_KEYS)} key(s) loaded from ENGINE_API_KEYS)")
else:
    logger.warning("Auth DISABLED — set ENGINE_API_KEYS to lock the engine down")
logger.info(f"Rate limit: {RATE_LIMIT_MAX_REQUESTS} req/{RATE_LIMIT_WINDOW_SECONDS}s per client")
logger.info(f"Max upload size: {MAX_UPLOAD_BYTES // (1024*1024)} MB")


def _client_identity(request: Request, api_key: str | None) -> str:
    """Identify the client for rate-limiting purposes."""
    if api_key:
        return f"key:{api_key[:8]}"  # truncate so logs don't echo full keys
    # request.client may be None in some test setups
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


def _enforce_rate_limit(identity: str) -> None:
    """Raise HTTPException 429 if `identity` is over the per-minute budget."""
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    with _RATE_LOCK:
        bucket = _rate_buckets.setdefault(identity, deque())
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = int(RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0])) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({RATE_LIMIT_MAX_REQUESTS}/{RATE_LIMIT_WINDOW_SECONDS}s). Retry in {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)


def _check_auth(api_key: str | None) -> None:
    """Raise 401 if auth is enabled and the API key is missing/invalid."""
    if not _AUTH_ENABLED:
        return
    if not api_key or api_key not in _API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")

# ── In-memory frame cache ──────────────────────────────────────────────
# Stores extracted frames keyed by analysis_id for the frame viewer endpoint.
# Entry: { 'frames': [...numpy arrays...], 'expires': timestamp }
# Bounded by FRAME_CACHE_MAX_ENTRIES so the server can't run out of RAM
# if no new uploads trigger the periodic cleanup.
FRAME_CACHE: "OrderedDict[str, dict]" = OrderedDict()
FRAME_CACHE_TTL = 3600  # 1 hour
FRAME_CACHE_MAX_ENTRIES = 20
_FRAME_CACHE_LOCK = threading.Lock()

# ── Analysis Progress Tracker ──────────────────────────────────────────
# Polled by the frontend to show real-time progress during analysis.
# Maps task_id -> progress dict. Makes the progress bar thread-safe!
ANALYSIS_PROGRESS: dict = {}
_PROGRESS_LOCK = threading.Lock()
PROGRESS_RETENTION_SECONDS = 60  # keep last status briefly for late polls

# ── Analysis Results Store ─────────────────────────────────────────────
# Background analyses write their final payload here. The frontend polls
# /api/progress for status; when stage == "done" it fetches /api/result.
# Bounded + LRU so it can't grow unbounded if no one fetches results.
RESULTS_STORE: "OrderedDict[str, dict]" = OrderedDict()
RESULTS_STORE_TTL = 3600  # 1 hour
RESULTS_STORE_MAX_ENTRIES = 50
_RESULTS_LOCK = threading.Lock()

def _store_result(task_id: str, payload: dict) -> None:
    if not task_id:
        logger.warning("[RESULTS_STORE] _store_result called with empty task_id — entry dropped")
        return
    expires = time.time() + RESULTS_STORE_TTL
    with _RESULTS_LOCK:
        RESULTS_STORE[task_id] = {"payload": payload, "expires": expires}
        RESULTS_STORE.move_to_end(task_id)
        # Drop expired entries + LRU evict over the size cap
        now = time.time()
        expired = [k for k, v in RESULTS_STORE.items() if v["expires"] < now]
        for k in expired:
            del RESULTS_STORE[k]
        while len(RESULTS_STORE) > RESULTS_STORE_MAX_ENTRIES:
            RESULTS_STORE.popitem(last=False)
        size = len(RESULTS_STORE)
    logger.info(f"[RESULTS_STORE] STORED task_id={task_id!r} (status={payload.get('status')}, store_size={size})")

def _take_result(task_id: str) -> dict | None:
    """Fetch a stored result. Non-destructive — same task_id can be polled twice."""
    if not task_id:
        return None
    with _RESULTS_LOCK:
        entry = RESULTS_STORE.get(task_id)
        if entry is None:
            known_keys = list(RESULTS_STORE.keys())
            logger.warning(
                f"[RESULTS_STORE] MISS task_id={task_id!r} "
                f"(store has {len(known_keys)} entries: {known_keys[:5]}{'...' if len(known_keys) > 5 else ''})"
            )
            return None
        if entry["expires"] < time.time():
            del RESULTS_STORE[task_id]
            logger.warning(f"[RESULTS_STORE] EXPIRED task_id={task_id!r}")
            return None
        logger.info(f"[RESULTS_STORE] HIT task_id={task_id!r}")
        return entry["payload"]


# ── History persistence helper ──────────────────────────────────────────
# Bridges the in-memory analysis payload (built in _run_analysis_background)
# to history_store's flat schema. Failures here are caught by the caller —
# the in-memory flow keeps working even if SQLite is unreachable.

THUMBNAIL_MAX_DIM = 320  # px on the longest side; keeps blobs ~10–20 KB

def _build_thumbnail_jpeg(frames: list) -> bytes | None:
    """
    Encode the first available frame as a downscaled JPEG.

    Returns None when frames is empty or encoding fails — caller stores
    the row without a thumbnail and the UI falls back to a placeholder.
    """
    if not frames:
        return None
    first = frames[0]
    if first is None or not hasattr(first, "shape"):
        return None
    h, w = first.shape[:2]
    scale = THUMBNAIL_MAX_DIM / max(h, w, 1)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        thumb = cv2.resize(first, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        thumb = first
    ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return bytes(buf) if ok else None


def _persist_to_history(
    analysis_id: str,
    original_filename: str,
    is_image: bool,
    frames: list,
    video_info: dict,
    final_payload: dict,
) -> None:
    """Flatten the analysis payload and hand it to history_store.save_analysis."""
    results = final_payload.get("results", {}) or {}
    overall_score = float(results.get("overall_score", 0.5))
    plugin_summaries = results.get("plugins", []) or []
    plugins_dict = {
        p.get("name", "?"): float(p.get("average_score", 0.0))
        for p in plugin_summaries
        if isinstance(p, dict)
    }
    frame_details = results.get("frame_details", []) or []
    duration = video_info.get("duration_seconds") if not is_image else None

    history_store.save_analysis(
        analysis_id=analysis_id,
        filename=original_filename,
        overall_score=overall_score,
        is_image=is_image,
        frame_count=len(frames),
        plugins=plugins_dict,
        frame_details=frame_details,
        thumbnail_jpeg=_build_thumbnail_jpeg(frames),
        duration_secs=duration,
    )
    logger.info(f"[HISTORY] Persisted analysis {analysis_id} ({original_filename})")


def _evict_expired_cache():
    """Remove expired cache entries to avoid memory leaks."""
    now = time.time()
    with _FRAME_CACHE_LOCK:
        expired = [k for k, v in FRAME_CACHE.items() if v['expires'] < now]
        for k in expired:
            del FRAME_CACHE[k]
            logger.info(f"Cache evicted (expired): {k}")
        # LRU eviction once over the size cap
        while len(FRAME_CACHE) > FRAME_CACHE_MAX_ENTRIES:
            oldest_key, _ = FRAME_CACHE.popitem(last=False)
            logger.info(f"Cache evicted (LRU): {oldest_key}")

def _update_progress(task_id: str, patch: dict) -> None:
    """Thread-safe shallow update of the progress dict for `task_id`."""
    if not task_id:
        return
    with _PROGRESS_LOCK:
        current = ANALYSIS_PROGRESS.get(task_id, {})
        current.update(patch)
        ANALYSIS_PROGRESS[task_id] = current

def _schedule_progress_cleanup(task_id: str) -> None:
    """Mark progress entry as expired and let _evict_progress drop it later."""
    if not task_id:
        return
    expires_at = time.time() + PROGRESS_RETENTION_SECONDS
    _update_progress(task_id, {"_expires_at": expires_at})

def _evict_progress() -> None:
    now = time.time()
    with _PROGRESS_LOCK:
        expired = [k for k, v in ANALYSIS_PROGRESS.items()
                   if v.get("_expires_at") is not None and v["_expires_at"] < now]
        for k in expired:
            del ANALYSIS_PROGRESS[k]

def get_ffmpeg_exe() -> str:
    """Return the path to the bundled FFmpeg binary from imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"  # Fall back to system ffmpeg if available

def has_audio_track(video_path: str) -> bool:
    """Quick FFmpeg probe to check if the video file contains an audio stream."""
    import subprocess
    try:
        ffmpeg_exe = get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-i", video_path,
            "-hide_banner",
        ]
        result = subprocess.run(
            cmd, capture_output=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        # FFmpeg prints stream info to stderr
        output = result.stderr.decode("utf-8", errors="replace")
        has_audio = "Audio:" in output
        logger.info(f"Audio probe: {'audio track found' if has_audio else 'NO audio track'}")
        return has_audio
    except Exception as e:
        logger.warning(f"Audio probe failed: {e}")
        return False

def extract_keyframes(video_path: str, target_fps: float = 2.0) -> tuple[list, dict]:
    """
    Extract frames from a video at a target rate (default: 2 frames/second).

    For a 2-minute video at 2fps, this extracts ~240 frames — enough for:
      - Proper per-frame deepfake analysis across the entire video
      - Lip sync correlation (needs ≥10 frames with temporal continuity)
      - Temporal consistency checks

    Strategy:
      1. Extract frames at target_fps using FFmpeg's fps filter (fast, accurate)
      2. Fallback to OpenCV sampling if FFmpeg fails
      3. Cap at 300 frames max to prevent memory issues

    Returns:
        (frames, video_info)
        frames     — list of BGR numpy arrays
        video_info — dict with fps, total_frames, resolution, extraction_method
    """
    import subprocess

    MAX_FRAMES = 300  # Safety cap to prevent OOM on very long videos

    # ── Step 1: Get video metadata via OpenCV ────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = total_frames / fps if fps > 0 else 0
    expected_frames = min(int(duration * target_fps), MAX_FRAMES)
    expected_frames = max(expected_frames, 8)  # At least 8 frames

    video_info = {
        "fps": fps,
        "total_frames": total_frames,
        "resolution": f"{width}x{height}",
        "duration_seconds": round(duration, 2),
    }

    logger.info(f"Video: {duration:.1f}s @ {fps:.1f}fps → extracting ~{expected_frames} frames at {target_fps}fps")

    # ── Step 2: Extract frames with FFmpeg fps filter ────────────────────────
    ffmpeg_exe = get_ffmpeg_exe()
    frames = []
    extraction_method = f"FFmpeg {target_fps}fps sampling"

    # Adjust target_fps if video is very long (to stay under MAX_FRAMES)
    actual_fps = target_fps
    if duration * target_fps > MAX_FRAMES:
        actual_fps = MAX_FRAMES / duration
        logger.info(f"Long video — reducing extraction to {actual_fps:.2f}fps to cap at {MAX_FRAMES} frames")

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_pattern = os.path.join(tmp_dir, "frame_%06d.png")

        cmd = [
            ffmpeg_exe,
            "-i", video_path,
            "-vf", f"fps={actual_fps}",
            "-f", "image2",
            output_pattern,
            "-loglevel", "error",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[:200])

            frame_files = sorted([
                f for f in os.listdir(tmp_dir) if f.endswith(".png")
            ])
            for fname in frame_files:
                img = cv2.imread(os.path.join(tmp_dir, fname))
                if img is not None:
                    frames.append(img)

            logger.info(f"FFmpeg extracted {len(frames)} frame(s) at {actual_fps:.2f}fps")

        except Exception as e:
            logger.warning(f"FFmpeg extraction failed: {e}. Using OpenCV fallback.")

    # ── Step 3: Fallback to OpenCV if FFmpeg produced nothing ─────────────────
    if len(frames) < 8:
        needed = max(expected_frames, 8)
        logger.info(f"Only {len(frames)} frame(s) from FFmpeg — sampling {needed} via OpenCV")
        frames = _opencv_sample(video_path, n=needed)
        extraction_method = f"OpenCV {needed}-frame sampling"

    video_info["extraction_method"] = extraction_method
    logger.info(f"Final frame count for analysis: {len(frames)} ({extraction_method})")
    return frames, video_info



def _opencv_sample(video_path: str, n: int = 60) -> list:
    """Evenly-spaced frame sampling with OpenCV."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, total // n)
    current = 0
    while len(frames) < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        current += skip
    cap.release()
    logger.info(f"OpenCV sampled {len(frames)} frame(s) from {total} total.")
    return frames



IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


@app.get("/api/progress")
def get_progress(
    task_id: str = Query(""),
    x_api_key: str | None = Header(default=None),
):
    """Returns current analysis progress for the frontend progress bar."""
    _check_auth(x_api_key)
    # No rate limit here — the frontend polls at 400ms during analysis and
    # rate-limiting that would just spam 429s and break the UI.
    if not task_id:
        return {"active": False, "message": "No task ID provided"}
    _evict_progress()
    with _PROGRESS_LOCK:
        entry = ANALYSIS_PROGRESS.get(task_id)
        if entry is None:
            return {"active": False, "message": ""}
        # Don't leak the internal expiry marker to the client
        return {k: v for k, v in entry.items() if k != "_expires_at"}


def _snapshot_progress(task_id: str) -> dict | None:
    """Single-shot, lock-protected read of progress (omits internal markers)."""
    with _PROGRESS_LOCK:
        entry = ANALYSIS_PROGRESS.get(task_id)
        if entry is None:
            return None
        return {k: v for k, v in entry.items() if k != "_expires_at"}


@app.get("/api/progress/stream")
async def stream_progress(
    task_id: str = Query(...),
    x_api_key: str | None = Header(default=None),
):
    """
    Server-Sent Events stream for analysis progress.

    Cheaper than polling — one persistent HTTP connection instead of
    2-3 req/s. The client should:
        const es = new EventSource(`/api/progress/stream?task_id=...`);
        es.onmessage = e => { const data = JSON.parse(e.data); ... };

    The server emits one event whenever the progress dict changes (compared
    to the last sent snapshot), plus a heartbeat comment every 15s so proxies
    don't kill an idle connection. The stream closes automatically when:
      - The task reaches stage "done" or "error"
      - The task entry has been evicted (60s after completion)
      - The client disconnects
    """
    _check_auth(x_api_key)

    async def event_gen():
        last_payload: dict | None = None
        last_heartbeat = time.time()
        # Cap total stream lifetime so a buggy client can't hold a connection
        # forever; 30 min is more than any analysis we'd reasonably run.
        deadline = time.time() + 30 * 60

        # Send a comment immediately so the client sees the connection is up
        yield ": connected\n\n"

        # State for end-event reason logic:
        #   saw_terminal_stage — we observed stage="done"/"error" on this stream
        #   ever_saw_entry     — at any point the task had a progress record
        # The end-event reason tells the client whether to fetch the result.
        saw_terminal_stage = False
        ever_saw_entry = False

        # Grace period: with the post-fix frontend the POST completes BEFORE
        # the SSE connects, so the task entry should already exist by the
        # time we look. This small grace just covers tiny timing wobbles
        # (a heavily loaded server taking >1s to register the entry, etc.).
        GRACE_SECONDS = 3
        grace_deadline = time.time() + GRACE_SECONDS

        while time.time() < deadline:
            _evict_progress()
            snap = _snapshot_progress(task_id)

            if snap is None:
                if ever_saw_entry:
                    # We had the task and now it's gone → evicted after completion.
                    # If we saw a terminal stage just before eviction, the result
                    # is in the results store. Otherwise the task somehow ended
                    # without flipping to done/error (worker crashed, etc.).
                    reason = "completed" if saw_terminal_stage else "evicted"
                    yield f"event: end\ndata: {json.dumps({'reason': reason})}\n\n"
                    return
                if time.time() > grace_deadline:
                    # Grace expired without ever seeing the task — genuinely
                    # unknown (e.g. SSE opened with a stale task_id after a
                    # server restart).
                    yield f"event: end\ndata: {json.dumps({'reason': 'unknown_task'})}\n\n"
                    return
                # Still inside grace — keep waiting silently.
                await asyncio.sleep(0.4)
                continue

            ever_saw_entry = True

            if snap != last_payload:
                yield f"data: {json.dumps(snap)}\n\n"
                last_payload = snap
                # Close as soon as we observe a terminal stage.
                if snap.get("stage") in ("done", "error"):
                    saw_terminal_stage = True
                    yield f"event: end\ndata: {json.dumps({'reason': 'completed'})}\n\n"
                    return

            # Heartbeat every 15s to defeat idle proxy timeouts
            if time.time() - last_heartbeat > 15:
                yield ": heartbeat\n\n"
                last_heartbeat = time.time()

            await asyncio.sleep(0.4)

        # Deadline hit — tell the client to fall back to polling
        yield f"event: end\ndata: {json.dumps({'reason': 'deadline'})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
        },
    )


@app.post("/api/analyze")
def analyze_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    cloud: bool = Query(True),
    task_id: str = Query(""),
    x_api_key: str | None = Header(default=None),
):
    """
    Receives a media file, persists it, then queues the heavy analysis work
    in a background task and returns immediately.

    The client then:
      1. polls GET /api/progress?task_id=... until stage == "done" or "error"
      2. fetches GET /api/result/{task_id} for the final payload

    Why async: the previous synchronous version blocked an entire FastAPI
    worker for the full duration of analysis (minutes), making the server
    unable to accept a second upload. BackgroundTasks runs the work in
    Starlette's thread pool, freeing the request handler immediately.

    Security:
      - X-API-Key header required when ENGINE_API_KEYS env var is set.
      - Sliding-window rate limit per client (key or IP).
      - Upload size capped both via Content-Length probe and streaming.
    """
    _check_auth(x_api_key)
    _enforce_rate_limit(_client_identity(request, x_api_key))

    # Reject early if the client declares a too-large payload.
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit",
                )
        except ValueError:
            pass  # malformed header, ignore — streaming cap below catches abuse

    logger.info(f"====== UPLOAD ACCEPTED: {file.filename} (Task: {task_id}) ======")
    analysis_id = task_id or str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "upload")[1].lower()
    is_image = ext in IMAGE_EXTENSIONS

    _update_progress(task_id, {
        "active": True, "stage": "upload",
        "current_frame": 0, "total_frames": 0,
        "message": "A guardar ficheiro...",
    })

    # Persist the upload SYNCHRONOUSLY before returning. UploadFile is a stream
    # tied to the request lifecycle — we can't read it from a background task
    # that runs after the request is closed.
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"upload{ext}")
    try:
        bytes_written = 0
        chunk_size = 1024 * 1024  # 1 MB
        with open(tmp_path, "wb") as f:
            while True:
                chunk = file.file.read(chunk_size)
                if not chunk:
                    break
                bytes_written += len(chunk)
                # Streaming guard: even if Content-Length lied, we won't dump
                # more than MAX_UPLOAD_BYTES to disk.
                if bytes_written > MAX_UPLOAD_BYTES:
                    f.close()
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit",
                    )
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _update_progress(task_id, {"active": False, "stage": "error", "message": "Upload too large"})
        _schedule_progress_cleanup(task_id)
        raise
    except Exception as save_err:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _update_progress(task_id, {"active": False, "stage": "error", "message": str(save_err)})
        _schedule_progress_cleanup(task_id)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {save_err}")

    file_size = os.path.getsize(tmp_path)
    logger.info(f"Saved upload to {tmp_path} ({file_size} bytes); queueing analysis")

    background_tasks.add_task(
        _run_analysis_background,
        analysis_id=analysis_id,
        task_id=task_id,
        tmp_dir=tmp_dir,
        tmp_path=tmp_path,
        original_filename=file.filename or "upload",
        file_size=file_size,
        is_image=is_image,
        cloud=cloud,
    )

    return {
        "analysis_id": analysis_id,
        "task_id": task_id,
        "status": "queued",
        "message": "Análise em fila. Polling /api/progress para acompanhar.",
    }


@app.get("/api/result/{task_id}")
def get_result(
    task_id: str,
    x_api_key: str | None = Header(default=None),
):
    """
    Fetch the final result of a queued analysis.

    Returns 404 if the task doesn't exist, expired, or hasn't finished yet —
    clients should rely on /api/progress to know when to call this.
    """
    _check_auth(x_api_key)
    payload = _take_result(task_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Result not found or expired. Poll /api/progress.")
    return payload


def _run_analysis_background(
    analysis_id: str,
    task_id: str,
    tmp_dir: str,
    tmp_path: str,
    original_filename: str,
    file_size: int,
    is_image: bool,
    cloud: bool,
) -> None:
    """The heavy lifting that used to live in `analyze_file`."""
    try:
        # ── Stage 2: Extract frames
        _update_progress(task_id, {
            "active": True, "stage": "extracting",
            "current_frame": 0, "total_frames": 0,
            "message": "A extrair frames do vídeo...",
        })

        video_info: dict = {}
        if is_image:
            frame = cv2.imread(tmp_path)
            frames = [frame] if frame is not None else []
        else:
            frames, video_info = extract_keyframes(tmp_path)
        logger.info(f"Extracted {len(frames)} frame(s) for analysis.")
        _update_progress(task_id, {
            "total_frames": len(frames),
            "message": f"{len(frames)} frames extraídos",
        })

        mode = "cloud" if cloud else "local"
        logger.info(f"Analysis mode: {mode}")

        # ── Stage 3: Run plugins
        def on_frame_progress(current: int, total: int):
            _update_progress(task_id, {
                "stage": "analyzing",
                "current_frame": current + 1,
                "total_frames": total,
                "message": f"A analisar frame {current + 1} de {total}...",
            })

        analysis_results = plugin_manager.run_analysis(
            frames, fps=video_info.get("fps", 30.0), mode=mode,
            progress_callback=on_frame_progress,
        )

        # ── Stage 4: Audio analyses
        lip_sync_result = None
        audio_deepfake_result = None

        if not is_image:
            _update_progress(task_id, {"stage": "audio", "message": "A verificar faixa de áudio..."})
            video_has_audio = has_audio_track(tmp_path)

            if video_has_audio:
                _update_progress(task_id, {"message": "A analisar sincronia labial (MediaPipe)..."})
                lip_sync_result = lip_sync_analyzer.analyze_video(tmp_path)
                if lip_sync_result:
                    logger.info(
                        f"Lip sync analysis: score={lip_sync_result['lip_sync_score']}, "
                        f"correlation={lip_sync_result['correlation']}"
                    )

                _update_progress(task_id, {"message": "A detetar clonagem de voz (WavLM)..."})
                audio_deepfake_result = audio_deepfake_analyzer.analyze_audio(tmp_path)
                if audio_deepfake_result:
                    logger.info(
                        f"Audio deepfake analysis: score={audio_deepfake_result['audio_fake_score']}"
                    )
                else:
                    logger.warning("Audio deepfake analysis returned None (audio too short or silent).")
            else:
                logger.info("Video has no audio track — skipping all audio analysis.")

        # ── Stage 4b: File-level + temporal analyzers (cheap, don't gate on audio) ──
        _update_progress(task_id, {"stage": "extras", "message": "A inspecionar metadata, coerência temporal e rPPG..."})

        # Metadata: EXIF for images, ffprobe for videos
        metadata_result = metadata_analyzer.analyze_file(tmp_path)
        if metadata_result:
            logger.info(
                f"Metadata: score={metadata_result['metadata_score']}, "
                f"signals={metadata_result['signals']}"
            )

        # Temporal coherence: only meaningful on videos with multiple frames
        temporal_result = None
        if not is_image and len(frames) >= 6:
            try:
                temporal_result = temporal_analyzer.analyze(
                    frames, analysis_results.get("frame_details", []),
                )
                if temporal_result:
                    logger.info(
                        f"Temporal coherence: score={temporal_result['temporal_score']}, "
                        f"signals={temporal_result['signals']}"
                    )
            except Exception as te:
                logger.warning(f"Temporal coherence analysis failed: {te}")

        # rPPG: needs enough frames AND a face for the signal to be present
        rppg_result = None
        if not is_image and len(frames) >= 90:
            try:
                rppg_result = rppg_analyzer.analyze(
                    frames, fps=video_info.get("fps", 30.0),
                    frame_details=analysis_results.get("frame_details", []),
                )
                if rppg_result:
                    logger.info(
                        f"rPPG: score={rppg_result['rppg_score']}, "
                        f"bpm={rppg_result.get('estimated_bpm')}"
                    )
            except Exception as re:
                logger.warning(f"rPPG analysis failed: {re}")

        # ── Score separation ──
        if "overall_score" in analysis_results:
            visual_score = analysis_results["overall_score"]
            analysis_results["visual_score"] = visual_score

            # Track WHICH modality contributed each candidate score so the UI
            # can tell the user "the audio is what raised the alarm" rather
            # than just showing a combined number.
            #
            # Both WavLM and Lip Sync must declare themselves CONCLUSIVE to
            # contribute to the audio_score. Without this, a noisy/silent
            # video (e.g. no speech) makes WavLM return a meaningless 99%
            # which then dominates the MAX aggregation.
            score_sources: list[tuple[str, float]] = [("visual", visual_score)]
            if audio_deepfake_result and not audio_deepfake_result.get("inconclusive", False):
                score_sources.append(("wavlm", audio_deepfake_result["audio_fake_score"]))
            if lip_sync_result and not lip_sync_result.get("inconclusive", False):
                score_sources.append(("lip_sync", lip_sync_result["lip_sync_score"]))

            audio_scores = [s for src, s in score_sources if src in ("wavlm", "lip_sync")]
            if audio_scores:
                audio_score = max(audio_scores)
                analysis_results["audio_score"] = audio_score
                analysis_results["overall_score"] = max(visual_score, audio_score)
            else:
                analysis_results["audio_score"] = None

            # Pick the modality whose score equals the final overall (the one
            # that "won" the MAX). Ties resolved by source order above.
            overall = analysis_results["overall_score"]
            triggered_by, triggered_score = max(score_sources, key=lambda kv: kv[1])
            analysis_results["triggered_by"] = triggered_by
            analysis_results["triggered_by_score"] = round(triggered_score, 4)
            logger.info(
                f"Scores -> {dict(score_sources)} | overall={overall} "
                f"| triggered_by={triggered_by}"
            )

        # Cache frames for the per-frame viewer endpoint
        with _FRAME_CACHE_LOCK:
            FRAME_CACHE[analysis_id] = {
                "frames": frames,
                "expires": time.time() + FRAME_CACHE_TTL,
            }
            FRAME_CACHE.move_to_end(analysis_id)
        _evict_expired_cache()

        final_payload = {
            "analysis_id": analysis_id,
            "status": "completed",
            "metadata": {
                "filename": original_filename,
                "filesize_bytes": file_size,
                "format": "image" if is_image else "video",
                "frames_analyzed": len(frames),
                "total_frames": video_info.get("total_frames", len(frames)),
                "fps": video_info.get("fps"),
                "resolution": video_info.get("resolution"),
                "duration_seconds": video_info.get("duration_seconds"),
                "extraction_method": (
                    video_info.get("extraction_method", "unknown")
                    if not is_image else "single image"
                ),
                "analysis_mode": mode,
                "active_plugins": len(plugin_manager.get_plugins()),
            },
            "results": analysis_results,
            "lip_sync": lip_sync_result,
            "audio_deepfake": audio_deepfake_result,
            "metadata_forensics": metadata_result,
            "temporal_coherence": temporal_result,
            "rppg": rppg_result,
        }

        # ── Stage 5: Persist + done
        # Save to local SQLite history. Failures here are non-fatal — the
        # in-memory result still works for the immediate UI flow; only the
        # /historico page would miss this entry.
        try:
            _persist_to_history(
                analysis_id=analysis_id,
                original_filename=original_filename,
                is_image=is_image,
                frames=frames,
                video_info=video_info,
                final_payload=final_payload,
            )
        except Exception as hist_err:
            logger.warning(f"History persist failed for {analysis_id}: {hist_err}")

        # IMPORTANT ordering: store the result BEFORE advertising "done".
        # The frontend reacts to stage=="done" by immediately calling
        # /api/result/{task_id}; if we flipped the order it would race the
        # store and get a 404 ("Resultado indisponível").
        _store_result(task_id or analysis_id, final_payload)
        _update_progress(task_id, {"stage": "done", "message": "Análise completa!"})

    except Exception as e:
        logger.error(f"Background analysis failed: {e}", exc_info=True)
        # Same ordering rule on the error path.
        _store_result(task_id or analysis_id, {
            "analysis_id": analysis_id,
            "status": "error",
            "error": str(e),
        })
        _update_progress(task_id, {"active": False, "stage": "error", "message": str(e)})

    finally:
        _update_progress(task_id, {"active": False})
        _schedule_progress_cleanup(task_id)
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as cleanup_err:
            logger.warning(f"Temp dir cleanup failed for {tmp_dir}: {cleanup_err}")


@app.get("/health")
def health_check():
    """
    Detailed health snapshot. Surfaces partial-functionality states so an
    operator doesn't have to grep logs to find a missing model or API key.

    Status is "degraded" when at least one plugin is not configured (e.g.
    Sightengine API key missing, ViT model failed to load). The engine still
    runs in that state — other plugins continue to score — but the routing
    weights are skewed.
    """
    plugin_infos = []
    for p in plugin_manager.get_plugins():
        info = p.get_plugin_info()
        # Side-channel: pull the load_error attribute that some plugins set
        load_err = getattr(p, "_load_error", None)
        if load_err:
            info["load_error"] = load_err
        plugin_infos.append(info)

    n_total = len(plugin_infos)
    n_ok = sum(1 for i in plugin_infos if i.get("configured"))
    status = "ok" if n_ok == n_total else ("degraded" if n_ok > 0 else "down")

    return {
        "status": status,
        "engine_version": "2.1.0",
        "plugins_total": n_total,
        "plugins_configured": n_ok,
        "auth_enabled": _AUTH_ENABLED,
        "rate_limit_per_minute": RATE_LIMIT_MAX_REQUESTS,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "active_plugins": plugin_infos,
        "video_analyzers": {
            "lip_sync": {
                "available": lip_sync_analyzer._face_mesh is not None,
                "window_seconds": lip_sync_analyzer.WINDOW_SECONDS,
            },
            "audio_deepfake": {
                "available": audio_deepfake_analyzer._pipe is not None,
                "chunk_seconds": audio_deepfake_analyzer.CHUNK_SECONDS,
            },
        },
    }


@app.get("/api/frame/{analysis_id}/{frame_index}")
def get_frame(analysis_id: str, frame_index: int):
    """
    Serves an extracted video frame as a JPEG image.
    The frame must have been cached during a prior /api/analyze call.
    Used by the VideoForensicsPlayer frontend component to display frames
    alongside bounding-box overlays without re-processing the video.
    """
    entry = FRAME_CACHE.get(analysis_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Analysis not found or expired")
    
    frames = entry["frames"]
    if frame_index < 0 or frame_index >= len(frames):
        raise HTTPException(
            status_code=404,
            detail=f"Frame index {frame_index} out of range (0–{len(frames) - 1})"
        )

    frame = frames[frame_index]
    # Encode as JPEG in memory (quality=85 — good balance of size vs quality)
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode frame")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


# ── History endpoints ─────────────────────────────────────────────────────
# Backed by core/history_store (SQLite at ~/.deepfake-forensics/history.db).
# Auth uses the same X-API-Key gate as the analysis endpoints — when the
# engine is exposed publicly the history is per-key; when run locally
# without ENGINE_API_KEYS it's unauthenticated, which matches expectations
# for a single-user desktop install.

@app.get("/api/history")
def history_list(
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    x_api_key: str | None = Header(default=None),
):
    """
    Return analyses in newest-first order. Excludes heavy fields
    (frame_details, thumbnail BLOB) so the list page loads quickly even
    with hundreds of entries.

    The `has_thumbnail` flag tells the UI whether to render the thumbnail
    endpoint or fall back to a placeholder.
    """
    _check_auth(x_api_key)
    try:
        items = history_store.list_analyses(limit=limit, offset=offset)
        total = history_store.count_analyses()
    except Exception as e:
        logger.error(f"history_list failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="History store unavailable")
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@app.get("/api/history/{analysis_id}")
def history_get(analysis_id: str, x_api_key: str | None = Header(default=None)):
    """
    Return the full analysis payload (including frame_details) for the
    /historico detail view. Same shape as a fresh /api/result, so the
    frontend can reuse the existing VideoForensicsPlayer + report UI.
    """
    _check_auth(x_api_key)
    try:
        record = history_store.get_analysis(analysis_id)
    except Exception as e:
        logger.error(f"history_get failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="History store unavailable")
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not in history")
    return record


@app.get("/api/history/{analysis_id}/thumbnail")
def history_thumbnail(analysis_id: str):
    """
    Serve the stored thumbnail as a JPEG. Public (no auth) — the JPEG
    leaks no more than the listing payload already does, and `<img src>`
    can't easily attach headers without extra plumbing.
    """
    try:
        blob = history_store.get_thumbnail(analysis_id)
    except Exception as e:
        logger.error(f"history_thumbnail failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="History store unavailable")
    if blob is None:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return Response(content=blob, media_type="image/jpeg")


@app.delete("/api/history/{analysis_id}")
def history_delete(analysis_id: str, x_api_key: str | None = Header(default=None)):
    """Remove one analysis from the local history."""
    _check_auth(x_api_key)
    try:
        removed = history_store.delete_analysis(analysis_id)
    except Exception as e:
        logger.error(f"history_delete failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="History store unavailable")
    if not removed:
        raise HTTPException(status_code=404, detail="Analysis not in history")
    return {"deleted": analysis_id}


@app.delete("/api/history")
def history_clear(x_api_key: str | None = Header(default=None)):
    """Wipe the entire local history. Returns the number of rows removed."""
    _check_auth(x_api_key)
    try:
        n = history_store.clear_all()
    except Exception as e:
        logger.error(f"history_clear failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="History store unavailable")
    return {"deleted_count": n}
