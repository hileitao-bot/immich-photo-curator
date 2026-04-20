from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
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
        """
    )
    _ensure_column(conn, "assets", "vision_labels_json", "TEXT")
    _ensure_column(conn, "assets", "ocr_text_json", "TEXT")
    _ensure_column(conn, "assets", "search_text", "TEXT")
    _ensure_column(conn, "assets", "burst_group_id", "TEXT")
    _ensure_column(conn, "assets", "burst_group_size", "INTEGER")
    _ensure_column(conn, "assets", "burst_rank", "INTEGER")
    _ensure_column(conn, "assets", "is_burst_pick", "INTEGER NOT NULL DEFAULT 1")
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
        ORDER BY file_created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def all_scored_assets(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM assets
        WHERE raw_score IS NOT NULL
        ORDER BY normalized_score DESC, raw_score DESC
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
            scored_at = ?
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


def normalize_scores(conn: sqlite3.Connection, rows: Iterable[sqlite3.Row]) -> None:
    rows = list(rows)
    row_map = {row["asset_id"]: row for row in rows}
    buckets: Dict[str, List[tuple[str, float]]] = {}
    for row in rows:
        buckets.setdefault(row["asset_type"], []).append((row["asset_id"], float(row["raw_score"])))

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
            conn.execute(
                """
                UPDATE assets
                SET normalized_score = ?,
                    percentile = ?,
                    grade = ?,
                    suggested_action = ?
                WHERE asset_id = ?
                """,
                (score, percentile, grade, suggested_action, asset_id),
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
    images = conn.execute(
        "SELECT COUNT(*) AS c FROM assets WHERE asset_type = 'IMAGE'"
    ).fetchone()["c"]
    videos = conn.execute(
        "SELECT COUNT(*) AS c FROM assets WHERE asset_type = 'VIDEO'"
    ).fetchone()["c"]
    burst_picks = conn.execute(
        "SELECT COUNT(*) AS c FROM assets WHERE raw_score IS NOT NULL AND asset_type = 'IMAGE' AND ifnull(is_burst_pick, 1) = 1"
    ).fetchone()["c"]
    return {"total": total, "scored": scored, "images": images, "videos": videos, "burstPicks": burst_picks}
