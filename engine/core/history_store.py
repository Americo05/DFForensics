"""
history_store.py — Local SQLite persistence for completed analyses.

Why
---
Without persistence, results live for 5 minutes in an in-memory OrderedDict
and vanish on container restart. The local-install model is supposed to
behave like a real app: the user wants to come back tomorrow and review
yesterday's analysis. SQLite is the standard answer for embedded persistence
— zero extra dependencies (stdlib `sqlite3`), single-file storage, ACID.

Storage location
----------------
~/.deepfake-forensics/history.db

Lives outside the project directory so:
  - `docker compose down -v` doesn't wipe it
  - upgrading the engine (git pull + rebuild) keeps history intact
  - it's per-user on the host machine (multi-user systems get separate DBs)

The Docker engine container needs the host's home directory bind-mounted —
see docker-compose.yml `volumes`. Without the mount the DB lives inside the
container and gets nuked with the container; the engine still works, just
without persistence across restarts.

Schema rationale
----------------
- `frame_details` is stored as JSON text (not normalized into a frames
  table). The data is read whole when displaying a result; relational
  queries against per-frame scores aren't needed in practice. JSON keeps
  the schema flat and the round-trip simple.
- `plugins_json` mirrors the same trade-off for per-plugin averages.
- `thumbnail_jpeg` is a small BLOB (~10 KB target) so the /historico list
  page loads thumbnails inline without a second round-trip.

Concurrency model
-----------------
SQLite is fine with one writer + many readers. FastAPI dispatches handlers
across a thread pool, so each request gets its own connection
(`sqlite3.connect` per call). Connections are cheap; no pool needed at this
scale. Writes happen at most once per analysis (~minutes apart), reads on
demand from the /historico page.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Storage path ────────────────────────────────────────────────────────

DEFAULT_DB_DIR = Path.home() / ".deepfake-forensics"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "history.db"


# ── Schema ──────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id              TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    overall_score   REAL NOT NULL,
    verdict         TEXT NOT NULL,
    is_image        INTEGER NOT NULL,
    frame_count     INTEGER NOT NULL,
    duration_secs   REAL,
    plugins_json    TEXT NOT NULL,
    frame_details   TEXT NOT NULL,
    thumbnail_jpeg  BLOB
);

CREATE INDEX IF NOT EXISTS idx_analyses_created_at
    ON analyses(created_at DESC);
"""


# ── Verdict helper (mirrors src/utils/verdict.ts logic) ─────────────────

def score_to_verdict(score: float) -> str:
    """
    Map a 0..1 fake-probability score to the verdict label the UI shows.

    Thresholds match the frontend convention:
      < 0.5  → AUTÊNTICO
      < 0.75 → INCERTO
      ≥ 0.75 → SUSPEITO
    """
    if score >= 0.75:
        return "SUSPEITO"
    if score >= 0.5:
        return "INCERTO"
    return "AUTÊNTICO"


# ── Connection management ───────────────────────────────────────────────

@contextmanager
def _connect(db_path: Path = DEFAULT_DB_PATH):
    """
    Open a SQLite connection, ensuring the parent directory exists and
    foreign keys are on. Closes the connection on exit.

    Using a context manager so every call site is exception-safe — the
    connection always closes even if a query raises.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Reasonable defaults for a local app:
    #   - WAL gives concurrent readers while a writer is active.
    #   - synchronous=NORMAL is the WAL-recommended trade-off (durable
    #     across crashes, doesn't fsync on every commit).
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create the schema if it doesn't exist. Idempotent — safe to call at startup."""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()
    logger.info(f"History DB ready at {db_path}")


# ── CRUD ────────────────────────────────────────────────────────────────

def save_analysis(
    analysis_id: str,
    filename: str,
    overall_score: float,
    is_image: bool,
    frame_count: int,
    plugins: dict[str, float],
    frame_details: list[dict],
    thumbnail_jpeg: bytes | None = None,
    duration_secs: float | None = None,
    created_at: int | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """
    Persist a completed analysis. `analysis_id` doubles as primary key —
    re-saving with the same id replaces the row (INSERT OR REPLACE).
    """
    ts = created_at if created_at is not None else int(time.time())
    verdict = score_to_verdict(overall_score)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO analyses (
                id, filename, created_at, overall_score, verdict,
                is_image, frame_count, duration_secs,
                plugins_json, frame_details, thumbnail_jpeg
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                filename,
                ts,
                float(overall_score),
                verdict,
                1 if is_image else 0,
                int(frame_count),
                duration_secs,
                json.dumps(plugins),
                json.dumps(frame_details),
                thumbnail_jpeg,
            ),
        )
        conn.commit()


def list_analyses(
    limit: int = 200,
    offset: int = 0,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[dict]:
    """
    Return analyses in newest-first order, WITHOUT frame_details (heavy)
    and WITHOUT thumbnail BLOB (fetched separately via the thumbnail route).

    Designed for the /historico list page: small payload, fast load.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, filename, created_at, overall_score, verdict,
                   is_image, frame_count, duration_secs, plugins_json,
                   thumbnail_jpeg IS NOT NULL AS has_thumbnail
            FROM analyses
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return [_row_to_summary(r) for r in rows]


def get_analysis(analysis_id: str, db_path: Path = DEFAULT_DB_PATH) -> dict | None:
    """Return the full analysis (including frame_details). None if not found."""
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, filename, created_at, overall_score, verdict,
                   is_image, frame_count, duration_secs, plugins_json,
                   frame_details, thumbnail_jpeg IS NOT NULL AS has_thumbnail
            FROM analyses
            WHERE id = ?
            """,
            (analysis_id,),
        ).fetchone()
    if row is None:
        return None
    summary = _row_to_summary(row)
    summary["frame_details"] = json.loads(row["frame_details"])
    return summary


def get_thumbnail(analysis_id: str, db_path: Path = DEFAULT_DB_PATH) -> bytes | None:
    """Return raw JPEG bytes of the thumbnail, or None."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT thumbnail_jpeg FROM analyses WHERE id = ?",
            (analysis_id,),
        ).fetchone()
    if row is None or row["thumbnail_jpeg"] is None:
        return None
    return bytes(row["thumbnail_jpeg"])


def delete_analysis(analysis_id: str, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """Delete by id. Returns True if a row was removed."""
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        conn.commit()
        return cur.rowcount > 0


def clear_all(db_path: Path = DEFAULT_DB_PATH) -> int:
    """Wipe the entire history. Returns the number of rows removed."""
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM analyses")
        conn.commit()
        return cur.rowcount


def count_analyses(db_path: Path = DEFAULT_DB_PATH) -> int:
    """Cheap COUNT(*) for the UI badge."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM analyses").fetchone()
    return int(row["n"])


# ── Row → API dict ──────────────────────────────────────────────────────

def _row_to_summary(row: sqlite3.Row) -> dict:
    """
    Convert a sqlite3.Row into a JSON-serializable dict in the shape the
    frontend expects. Keeps the API stable even if the schema gains
    columns later.
    """
    return {
        "id": row["id"],
        "filename": row["filename"],
        "created_at": int(row["created_at"]),
        "overall_score": float(row["overall_score"]),
        "verdict": row["verdict"],
        "is_image": bool(row["is_image"]),
        "frame_count": int(row["frame_count"]),
        "duration_secs": (
            float(row["duration_secs"]) if row["duration_secs"] is not None else None
        ),
        "plugins": json.loads(row["plugins_json"]),
        "has_thumbnail": bool(row["has_thumbnail"]),
    }
