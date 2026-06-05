"""
Unit tests for engine/core/history_store.py.

Uses a per-test tmp DB so tests are independent and don't touch the user's
real ~/.deepfake-forensics/history.db.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from core.history_store import (
    clear_all,
    count_analyses,
    delete_analysis,
    get_analysis,
    get_thumbnail,
    init_db,
    list_analyses,
    save_analysis,
    score_to_verdict,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Fresh SQLite file per test; init_db creates the schema."""
    path = tmp_path / "history.db"
    init_db(path)
    return path


# ── score_to_verdict ─────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "score,expected",
    [
        (0.00, "AUTÊNTICO"),
        (0.49, "AUTÊNTICO"),
        (0.50, "INCERTO"),
        (0.74, "INCERTO"),
        (0.75, "SUSPEITO"),
        (1.00, "SUSPEITO"),
    ],
)
def test_score_to_verdict_thresholds(score, expected):
    assert score_to_verdict(score) == expected


# ── save / get round trip ────────────────────────────────────────────────

def test_save_and_get_round_trip(db_path: Path):
    save_analysis(
        analysis_id="abc-123",
        filename="video.mp4",
        overall_score=0.81,
        is_image=False,
        frame_count=60,
        plugins={"MesoNet": 0.78, "ViT": 0.72},
        frame_details=[{"frame_index": 0, "overall_score": 0.8}],
        thumbnail_jpeg=b"\xff\xd8\xff\xe0fake jpeg bytes",
        duration_secs=12.5,
        db_path=db_path,
    )

    result = get_analysis("abc-123", db_path=db_path)
    assert result is not None
    assert result["id"] == "abc-123"
    assert result["filename"] == "video.mp4"
    assert result["overall_score"] == pytest.approx(0.81)
    assert result["verdict"] == "SUSPEITO"
    assert result["is_image"] is False
    assert result["frame_count"] == 60
    assert result["duration_secs"] == pytest.approx(12.5)
    assert result["plugins"] == {"MesoNet": 0.78, "ViT": 0.72}
    assert result["frame_details"] == [{"frame_index": 0, "overall_score": 0.8}]
    assert result["has_thumbnail"] is True


def test_get_thumbnail_returns_raw_bytes(db_path: Path):
    blob = b"\xff\xd8\xff\xe0jpeg-content"
    save_analysis(
        analysis_id="thumb-1",
        filename="x.jpg",
        overall_score=0.3,
        is_image=True,
        frame_count=1,
        plugins={},
        frame_details=[],
        thumbnail_jpeg=blob,
        db_path=db_path,
    )
    assert get_thumbnail("thumb-1", db_path=db_path) == blob


def test_save_without_thumbnail_marks_has_thumbnail_false(db_path: Path):
    save_analysis(
        analysis_id="no-thumb",
        filename="audio-only.wav",
        overall_score=0.1,
        is_image=False,
        frame_count=0,
        plugins={},
        frame_details=[],
        thumbnail_jpeg=None,
        db_path=db_path,
    )
    result = get_analysis("no-thumb", db_path=db_path)
    assert result is not None
    assert result["has_thumbnail"] is False
    assert get_thumbnail("no-thumb", db_path=db_path) is None


def test_get_missing_id_returns_none(db_path: Path):
    assert get_analysis("does-not-exist", db_path=db_path) is None
    assert get_thumbnail("does-not-exist", db_path=db_path) is None


# ── list ordering / pagination ───────────────────────────────────────────

def test_list_returns_newest_first(db_path: Path):
    now = int(time.time())
    for i, delta in enumerate([0, -10, -20, -5]):
        save_analysis(
            analysis_id=f"id-{i}",
            filename=f"f{i}.mp4",
            overall_score=0.5,
            is_image=False,
            frame_count=1,
            plugins={},
            frame_details=[],
            created_at=now + delta,
            db_path=db_path,
        )
    rows = list_analyses(db_path=db_path)
    # Expected order by created_at DESC: id-0 (now), id-3 (now-5), id-1 (-10), id-2 (-20)
    assert [r["id"] for r in rows] == ["id-0", "id-3", "id-1", "id-2"]


def test_list_respects_limit_and_offset(db_path: Path):
    now = int(time.time())
    for i in range(5):
        save_analysis(
            analysis_id=f"id-{i}",
            filename=f"f{i}.mp4",
            overall_score=0.5,
            is_image=False,
            frame_count=1,
            plugins={},
            frame_details=[],
            created_at=now - i,  # decreasing → id-0 newest, id-4 oldest
            db_path=db_path,
        )
    page1 = list_analyses(limit=2, offset=0, db_path=db_path)
    page2 = list_analyses(limit=2, offset=2, db_path=db_path)
    assert [r["id"] for r in page1] == ["id-0", "id-1"]
    assert [r["id"] for r in page2] == ["id-2", "id-3"]


def test_list_excludes_frame_details(db_path: Path):
    """List endpoint must NOT ship the heavy frame_details payload."""
    save_analysis(
        analysis_id="heavy",
        filename="x.mp4",
        overall_score=0.5,
        is_image=False,
        frame_count=60,
        plugins={},
        frame_details=[{"frame_index": i} for i in range(60)],
        db_path=db_path,
    )
    rows = list_analyses(db_path=db_path)
    assert len(rows) == 1
    assert "frame_details" not in rows[0]


# ── delete / replace / clear ─────────────────────────────────────────────

def test_save_with_existing_id_replaces(db_path: Path):
    """INSERT OR REPLACE — re-saving the same id should update, not duplicate."""
    save_analysis(
        analysis_id="dup",
        filename="old.mp4",
        overall_score=0.2,
        is_image=False,
        frame_count=1,
        plugins={},
        frame_details=[],
        db_path=db_path,
    )
    save_analysis(
        analysis_id="dup",
        filename="new.mp4",
        overall_score=0.9,
        is_image=False,
        frame_count=2,
        plugins={"MesoNet": 0.9},
        frame_details=[],
        db_path=db_path,
    )
    assert count_analyses(db_path=db_path) == 1
    result = get_analysis("dup", db_path=db_path)
    assert result["filename"] == "new.mp4"
    assert result["overall_score"] == pytest.approx(0.9)


def test_delete_returns_true_on_hit_false_on_miss(db_path: Path):
    save_analysis(
        analysis_id="del-me",
        filename="x.mp4",
        overall_score=0.5,
        is_image=False,
        frame_count=1,
        plugins={},
        frame_details=[],
        db_path=db_path,
    )
    assert delete_analysis("del-me", db_path=db_path) is True
    assert delete_analysis("del-me", db_path=db_path) is False  # already gone
    assert get_analysis("del-me", db_path=db_path) is None


def test_clear_all_wipes_everything(db_path: Path):
    for i in range(3):
        save_analysis(
            analysis_id=f"clr-{i}",
            filename=f"f{i}.mp4",
            overall_score=0.5,
            is_image=False,
            frame_count=1,
            plugins={},
            frame_details=[],
            db_path=db_path,
        )
    assert count_analyses(db_path=db_path) == 3
    removed = clear_all(db_path=db_path)
    assert removed == 3
    assert count_analyses(db_path=db_path) == 0


# ── init_db idempotence ──────────────────────────────────────────────────

def test_init_db_is_idempotent(db_path: Path):
    """Calling init_db twice must not error and must not lose data."""
    save_analysis(
        analysis_id="survives",
        filename="x.mp4",
        overall_score=0.5,
        is_image=False,
        frame_count=1,
        plugins={},
        frame_details=[],
        db_path=db_path,
    )
    init_db(db_path)  # second call — schema CREATE IF NOT EXISTS
    assert get_analysis("survives", db_path=db_path) is not None


# ── JSON round-trip preserves nested structure ──────────────────────────

def test_complex_frame_details_round_trip(db_path: Path):
    """Realistic frame_details payload survives the JSON round-trip intact."""
    complex_details = [
        {
            "frame_index": 0,
            "timestamp_seconds": 0.0,
            "overall_score": 0.72,
            "faces": [
                {
                    "face_bbox": {"x": 100, "y": 50, "w": 200, "h": 250},
                    "scene_detected": "FACE_IN_SCENE",
                    "overall_score": 0.72,
                    "plugin_scores": {"MesoNet": 0.78, "ViT": 0.65},
                }
            ],
        }
    ]
    save_analysis(
        analysis_id="rich",
        filename="rich.mp4",
        overall_score=0.72,
        is_image=False,
        frame_count=1,
        plugins={"MesoNet": 0.78, "ViT": 0.65},
        frame_details=complex_details,
        db_path=db_path,
    )
    result = get_analysis("rich", db_path=db_path)
    assert result["frame_details"] == complex_details
