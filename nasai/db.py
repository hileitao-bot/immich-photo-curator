from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS assets (
          asset_id TEXT PRIMARY KEY,
          asset_type TEXT NOT NULL,
          original_file_name TEXT,
          original_path TEXT,
          visibility TEXT,
          is_archived INTEGER NOT NULL DEFAULT 0,
          width INTEGER,
          height INTEGER,
          duration TEXT,
          file_created_at TEXT,
          updated_at TEXT,
          exif_description TEXT,
          metadata_json TEXT NOT NULL,
          thumbnail_cache_path TEXT,
          raw_score REAL,
          normalized_score REAL,
          percentile REAL,
          grade TEXT,
          suggested_action TEXT,
          chinese_tags_json TEXT,
          embedding_model TEXT,
          embedding_blob BLOB,
          scored_at TEXT,
          trial_synced_at TEXT
        );

        CREATE TABLE IF NOT EXISTS kv (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS progress_snapshots (
          captured_at TEXT PRIMARY KEY,
          total INTEGER NOT NULL,
          scored INTEGER NOT NULL,
          images INTEGER NOT NULL,
          videos INTEGER NOT NULL,
          burst_picks INTEGER NOT NULL
        );
        """
    )
    _ensure_column(conn, "assets", "vision_labels_json", "TEXT")
    _ensure_column(conn, "assets", "ocr_text_json", "TEXT")
    _ensure_column(conn, "assets", "search_text", "TEXT")
    _ensure_column(conn, "assets", "burst_group_id", "TEXT")
    _ensure_column(conn, "assets", "burst_group_size", "INTEGER")
    _ensure_column(conn, "assets", "burst_rank", "INTEGER")
    _ensure_column(conn, "assets", "is_burst_pick", "INTEGER NOT NULL DEFAULT 1")
    _ensure_column(conn, "assets", "score_attempts", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "assets", "score_last_error", "TEXT")
    _ensure_column(conn, "assets", "score_last_attempt_at", "TEXT")
    _ensure_column(conn, "assets", "score_failed_at", "TEXT")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_progress_snapshots_captured_at
        ON progress_snapshots (captured_at DESC)
        """
    )
    conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def upsert_assets(conn: sqlite3.Connection, assets: Iterable[Dict[str, Any]]) -> int:
    rows = [
        (
            asset["id"],
            asset["type"],
            asset.get("originalFileName"),
            asset.get("originalPath"),
            asset.get("visibility") or ("archive" if asset.get("isArchived") else "timeline"),
            1 if asset.get("isArchived") else 0,
            asset.get("width"),
            asset.get("height"),
            asset.get("duration"),
            asset.get("fileCreatedAt"),
            asset.get("updatedAt"),
            ((asset.get("exifInfo") or {}).get("description")),
            json.dumps(asset, ensure_ascii=False),
        )
        for asset in assets
    ]
    conn.executemany(
        """
        INSERT INTO assets (
          asset_id, asset_type, original_file_name, original_path, visibility, is_archived,
          width, height, duration, file_created_at, updated_at, exif_description, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(asset_id) DO UPDATE SET
          asset_type=excluded.asset_type,
          original_file_name=excluded.original_file_name,
          original_path=excluded.original_path,
          visibility=excluded.visibility,
          is_archived=excluded.is_archived,
          width=excluded.width,
          height=excluded.height,
          duration=excluded.duration,
          file_created_at=excluded.file_created_at,
          updated_at=excluded.updated_at,
          exif_description=excluded.exif_description,
          metadata_json=excluded.metadata_json
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def set_kv(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO kv (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, value),
    )
    conn.commit()


def get_kv(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    return None if row is None else row["value"]


def unscored_assets(conn: sqlite3.Connection, limit: int) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM assets
        WHERE raw_score IS NULL
          AND score_failed_at IS NULL
        ORDER BY ifnull(score_attempts, 0) ASC, file_created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def all_scored_assets(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM assets
        WHERE raw_score IS NOT NULL
        """
    ).fetchall()


def persist_score(
    conn: sqlite3.Connection,
    asset_id: str,
    thumbnail_cache_path: str,
    raw_score: float,
    chinese_tags: List[str],
    scored_at: str,
    vision_labels: List[Dict[str, Any]],
    ocr_text: List[str],
    search_text: str,
) -> None:
    conn.execute(
        """
        UPDATE assets
        SET thumbnail_cache_path = ?,
            raw_score = ?,
            chinese_tags_json = ?,
            vision_labels_json = ?,
            ocr_text_json = ?,
            search_text = ?,
            scored_at = ?,
            score_last_error = NULL,
            score_failed_at = NULL
        WHERE asset_id = ?
        """,
        (
            thumbnail_cache_path,
            raw_score,
            json.dumps(chinese_tags, ensure_ascii=False),
            json.dumps(vision_labels, ensure_ascii=False),
            json.dumps(ocr_text, ensure_ascii=False),
            search_text,
            scored_at,
            asset_id,
        ),
    )
    conn.commit()


def record_score_attempt(conn: sqlite3.Connection, asset_id: str, attempted_at: str) -> None:
    conn.execute(
        """
        UPDATE assets
        SET score_attempts = ifnull(score_attempts, 0) + 1,
            score_last_attempt_at = ?
        WHERE asset_id = ?
        """,
        (attempted_at, asset_id),
    )
    conn.commit()


def mark_score_failure(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
    error_message: str,
    failed_at: str,
    permanent: bool,
) -> None:
    conn.execute(
        """
        UPDATE assets
        SET score_last_error = ?,
            score_last_attempt_at = ?,
            score_failed_at = ?
        WHERE asset_id = ?
        """,
        (
            error_message,
            failed_at,
            failed_at if permanent else None,
            asset_id,
        ),
    )
    conn.commit()


def normalize_scores(conn: sqlite3.Connection, rows: Iterable[sqlite3.Row]) -> None:
    rows = list(rows)
    row_map = {row["asset_id"]: row for row in rows}
    buckets: Dict[str, List[tuple[str, float]]] = {}
    for row in rows:
        buckets.setdefault(row["asset_type"], []).append((row["asset_id"], float(row["raw_score"])))

    updates: List[tuple[float, float, str, str, str]] = []
    for _, bucket in buckets.items():
        scores = np.array([score for _, score in bucket], dtype=np.float32)
        order = np.argsort(scores)
        percentiles = np.empty_like(scores)
        if len(scores) == 1:
            percentiles[0] = 1.0
        else:
            percentiles[order] = np.linspace(0.0, 1.0, len(scores), endpoint=True)
        for index, (asset_id, score) in enumerate(bucket):
            percentile = float(percentiles[index])
            grade = _grade_for_percentile(percentile)
            row = row_map[asset_id]
            tags = set(json.loads(row["chinese_tags_json"] or "[]"))
            force_archive = bool({"屏幕截图", "文档", "收据", "白板"} & tags)
            if row["asset_type"] == "VIDEO":
                grade, suggested_action = _video_decision(
                    row=row,
                    score=score,
                    percentile=percentile,
                    grade=grade,
                    force_archive=force_archive,
                    tags=tags,
                )
            else:
                if force_archive and grade in {"S", "A", "B"}:
                    grade = "D" if score < 0.25 else "C"
                suggested_action = (
                    "keep"
                    if percentile >= 0.8 and score >= 0.25 and not force_archive
                    else "archive"
                )
            updates.append(
                (score, percentile, grade, suggested_action, asset_id)
            )
    conn.executemany(
        """
        UPDATE assets
        SET normalized_score = ?,
            percentile = ?,
            grade = ?,
            suggested_action = ?
        WHERE asset_id = ?
        """,
        updates,
    )
    conn.commit()


def _grade_for_percentile(percentile: float) -> str:
    if percentile >= 0.95:
        return "S"
    if percentile >= 0.80:
        return "A"
    if percentile >= 0.60:
        return "B"
    if percentile >= 0.35:
        return "C"
    return "D"


def _video_decision(
    *,
    row: sqlite3.Row,
    score: float,
    percentile: float,
    grade: str,
    force_archive: bool,
    tags: set[str],
) -> tuple[str, str]:
    if force_archive:
        if grade in {"S", "A", "B"}:
            grade = "D" if score < 0.25 else "C"
        return grade, "archive"

    original_path = (row["original_path"] or "").lower()
    file_name = (row["original_file_name"] or "").lower()
    duration_seconds = _duration_to_seconds(row["duration"])
    has_story_signal = bool(
        tags & {"人物", "人像", "合照", "儿童", "宝宝", "宠物", "狗", "猫", "海边", "日落", "天空", "风景", "山", "花", "城市", "街道", "建筑", "旅行", "户外", "室内", "食物", "汽车"}
    )
    is_vlog = "/vlog/" in original_path or original_path.startswith("/usr/src/app/external/nas_photos/vlog/")
    is_trip_named = file_name.endswith(".mp4") and len(file_name) >= 8 and file_name[:8].isdigit()
    is_long_form = duration_seconds >= 180
    is_mid_form_memory = duration_seconds >= 45 and has_story_signal

    preserve_video = is_vlog or is_trip_named or is_long_form or is_mid_form_memory
    keep_by_score = percentile >= 0.8 and score >= 0.25

    if preserve_video:
        if grade in {"D", "C"}:
            grade = "B"
        return grade, "keep"
    if keep_by_score:
        return grade, "keep"
    return grade, "archive"


def _duration_to_seconds(duration: Optional[str]) -> float:
    if not duration:
        return 0.0
    parts = duration.split(":")
    if len(parts) != 3:
        return 0.0
    hours = float(parts[0] or 0)
    minutes = float(parts[1] or 0)
    seconds = float(parts[2] or 0)
    return (hours * 3600.0) + (minutes * 60.0) + seconds


def query_assets(
    conn: sqlite3.Connection,
    *,
    asset_type: Optional[str] = None,
    grade: Optional[str] = None,
    suggested_action: Optional[str] = None,
    limit: Optional[int] = 200,
) -> List[sqlite3.Row]:
    where = ["raw_score IS NOT NULL"]
    params: List[Any] = []
    if asset_type:
        where.append("asset_type = ?")
        params.append(asset_type)
    if grade:
        where.append("grade = ?")
        params.append(grade)
    if suggested_action:
        where.append("suggested_action = ?")
        params.append(suggested_action)
    sql = f"""
      SELECT * FROM assets
      WHERE {' AND '.join(where)}
      ORDER BY percentile DESC, raw_score DESC
    """
    if limit is not None:
        params.append(limit)
        sql += "\n      LIMIT ?"
    return conn.execute(sql, params).fetchall()


def mark_trial_synced(conn: sqlite3.Connection, asset_ids: Iterable[str], timestamp: str) -> None:
    conn.executemany(
        "UPDATE assets SET trial_synced_at = ? WHERE asset_id = ?",
        [(timestamp, asset_id) for asset_id in asset_ids],
    )
    conn.commit()


def summary(conn: sqlite3.Connection) -> Dict[str, Any]:
    total = conn.execute("SELECT COUNT(*) AS c FROM assets").fetchone()["c"]
    scored = conn.execute("SELECT COUNT(*) AS c FROM assets WHERE raw_score IS NOT NULL").fetchone()["c"]
    failed = conn.execute(
        "SELECT COUNT(*) AS c FROM assets WHERE raw_score IS NULL AND score_failed_at IS NOT NULL"
    ).fetchone()["c"]
    pending = conn.execute(
        "SELECT COUNT(*) AS c FROM assets WHERE raw_score IS NULL AND score_failed_at IS NULL"
    ).fetchone()["c"]
    images = conn.execute(
        "SELECT COUNT(*) AS c FROM assets WHERE asset_type = 'IMAGE'"
    ).fetchone()["c"]
    videos = conn.execute(
        "SELECT COUNT(*) AS c FROM assets WHERE asset_type = 'VIDEO'"
    ).fetchone()["c"]
    burst_picks = conn.execute(
        "SELECT COUNT(*) AS c FROM assets WHERE raw_score IS NOT NULL AND asset_type = 'IMAGE' AND ifnull(is_burst_pick, 1) = 1"
    ).fetchone()["c"]
    keep = conn.execute(
        "SELECT COUNT(*) AS c FROM assets WHERE raw_score IS NOT NULL AND suggested_action = 'keep'"
    ).fetchone()["c"]
    archive = conn.execute(
        "SELECT COUNT(*) AS c FROM assets WHERE raw_score IS NOT NULL AND suggested_action = 'archive'"
    ).fetchone()["c"]
    last_scored_at_row = conn.execute(
        "SELECT MAX(scored_at) AS scored_at FROM assets WHERE scored_at IS NOT NULL"
    ).fetchone()
    return {
        "total": total,
        "scored": scored,
        "unscored": pending,
        "failed": failed,
        "images": images,
        "videos": videos,
        "burstPicks": burst_picks,
        "keep": keep,
        "archive": archive,
        "scoreCoverage": (float(scored) / float(total)) if total else 0.0,
        "lastScoredAt": last_scored_at_row["scored_at"] if last_scored_at_row else None,
    }


def record_progress_snapshot(
    conn: sqlite3.Connection,
    snapshot: Dict[str, Any],
    *,
    min_interval_seconds: int = 15,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    last = conn.execute(
        """
        SELECT captured_at, total, scored, images, videos, burst_picks
        FROM progress_snapshots
        ORDER BY captured_at DESC
        LIMIT 1
        """
    ).fetchone()
    if last is not None:
        last_at = datetime.fromisoformat(last["captured_at"])
        values_unchanged = (
            int(last["total"]) == int(snapshot["total"])
            and int(last["scored"]) == int(snapshot["scored"])
            and int(last["images"]) == int(snapshot["images"])
            and int(last["videos"]) == int(snapshot["videos"])
            and int(last["burst_picks"]) == int(snapshot["burstPicks"])
        )
        if values_unchanged and (now - last_at).total_seconds() < min_interval_seconds:
            return {
                "capturedAt": last["captured_at"],
                "total": int(last["total"]),
                "scored": int(last["scored"]),
                "images": int(last["images"]),
                "videos": int(last["videos"]),
                "burstPicks": int(last["burst_picks"]),
            }

    captured_at = now.isoformat()
    try:
        conn.execute(
            """
            INSERT INTO progress_snapshots (
              captured_at, total, scored, images, videos, burst_picks
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                captured_at,
                int(snapshot["total"]),
                int(snapshot["scored"]),
                int(snapshot["images"]),
                int(snapshot["videos"]),
                int(snapshot["burstPicks"]),
            ),
        )
        conn.commit()
    except sqlite3.OperationalError as exc:
        if "locked" not in str(exc).lower():
            raise
        return {
            "capturedAt": captured_at,
            "total": int(snapshot["total"]),
            "scored": int(snapshot["scored"]),
            "images": int(snapshot["images"]),
            "videos": int(snapshot["videos"]),
            "burstPicks": int(snapshot["burstPicks"]),
        }
    return {
        "capturedAt": captured_at,
        "total": int(snapshot["total"]),
        "scored": int(snapshot["scored"]),
        "images": int(snapshot["images"]),
        "videos": int(snapshot["videos"]),
        "burstPicks": int(snapshot["burstPicks"]),
    }


def recent_progress_snapshots(
    conn: sqlite3.Connection,
    *,
    limit: int = 120,
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT captured_at, total, scored, images, videos, burst_picks
        FROM progress_snapshots
        ORDER BY captured_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    payload = [
        {
            "capturedAt": row["captured_at"],
            "total": int(row["total"]),
            "scored": int(row["scored"]),
            "images": int(row["images"]),
            "videos": int(row["videos"]),
            "burstPicks": int(row["burst_picks"]),
        }
        for row in reversed(rows)
    ]
    return payload
