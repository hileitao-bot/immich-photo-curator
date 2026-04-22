from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from . import db


TRIAL_ALBUMS = {
    "keep": "NASAI 试点-精选",
    "archive": "NASAI 试点-待归档",
    "video_keep": "NASAI 试点-视频高分",
}

HYBRID_ALBUMS = {
    "display": "NASAI 全量-精选展示",
    "video": "NASAI 全量-视频保留",
    "buffer_images": "NASAI 全量-系统缓冲图片",
    "buffer_videos": "NASAI 全量-系统缓冲视频",
}

HYBRID_ARCHIVE_ACTIONS = {"archive_duplicate", "archive_low", "archive_negative"}
HYBRID_TIMELINE_ACTIONS = {
    "display_keep",
    "display_keep_video",
    "protect_keep_video",
    "review_keep",
    "review_video",
}


def sync_trial_albums(conn, client, top_limit: int = 150, archive_limit: int = 150) -> Dict[str, int]:
    scored = db.all_scored_assets(conn)
    keep_assets = [row for row in scored if row["suggested_action"] == "keep"][:top_limit]
    archive_assets = [row for row in reversed(scored) if row["suggested_action"] == "archive"][:archive_limit]
    video_assets = [row for row in keep_assets if row["asset_type"] == "VIDEO"][:top_limit]

    created_albums = _reset_trial_albums(client)
    if keep_assets:
        client.add_assets_to_album(created_albums[TRIAL_ALBUMS["keep"]], [row["asset_id"] for row in keep_assets])
    if archive_assets:
        client.add_assets_to_album(
            created_albums[TRIAL_ALBUMS["archive"]],
            [row["asset_id"] for row in archive_assets],
        )
    if video_assets:
        client.add_assets_to_album(
            created_albums[TRIAL_ALBUMS["video_keep"]],
            [row["asset_id"] for row in video_assets],
        )

    now = datetime.now(timezone.utc).isoformat()
    db.mark_trial_synced(
        conn,
        [row["asset_id"] for row in keep_assets + archive_assets + video_assets],
        now,
    )
    return {
        "keep": len(keep_assets),
        "archive": len(archive_assets),
        "video_keep": len(video_assets),
    }


def sync_trial_tags(conn, client, *, limit: int = 300) -> Dict[str, int]:
    rows = db.query_assets(conn, limit=limit)
    tags = {tag["name"]: tag["id"] for tag in client.list_tags()}
    assignments: Dict[str, List[str]] = {}

    for row in rows:
        grade = row["grade"]
        if grade:
            assignments.setdefault(f"nasai/grade/{grade}", []).append(row["asset_id"])
        for chinese_tag in json.loads(row["chinese_tags_json"] or "[]"):
            assignments.setdefault(f"nasai/cn/{chinese_tag}", []).append(row["asset_id"])

    created = 0
    tagged_assets = 0
    for tag_name, asset_ids in assignments.items():
        tag_id = tags.get(tag_name)
        if tag_id is None:
            tag = client.create_tag(tag_name)
            tag_id = tag["id"]
            tags[tag_name] = tag_id
            created += 1
        client.tag_assets(tag_id, asset_ids)
        tagged_assets += len(asset_ids)
    return {"created_tags": created, "asset_tag_links": tagged_assets}


def apply_archive(conn, client, *, threshold: float = 0.8, limit: int | None = None) -> Dict[str, int]:
    rows = db.all_scored_assets(conn)
    if limit is not None:
        rows = rows[:limit]
    keep_ids = [row["asset_id"] for row in rows if float(row["percentile"] or 0.0) >= threshold]
    archive_ids = [row["asset_id"] for row in rows if float(row["percentile"] or 0.0) < threshold]
    if keep_ids:
        client.update_assets(keep_ids, visibility="timeline")
    if archive_ids:
        client.update_assets(archive_ids, visibility="archive")
    return {"timeline": len(keep_ids), "archive": len(archive_ids)}


def sync_hybrid_writeback(
    client,
    *,
    actions_path: Path,
    with_buffer_albums: bool = False,
    batch_size: int = 500,
) -> Dict[str, int]:
    actions = json.loads(actions_path.read_text())

    archive_ids: List[str] = []
    timeline_ids: List[str] = []
    display_ids: List[str] = []
    video_ids: List[str] = []
    buffer_image_ids: List[str] = []
    buffer_video_ids: List[str] = []

    for item in actions:
        asset_id = item["assetId"]
        action = item["action"]
        if action in HYBRID_ARCHIVE_ACTIONS:
            archive_ids.append(asset_id)
        elif action in HYBRID_TIMELINE_ACTIONS:
            timeline_ids.append(asset_id)

        if action == "display_keep":
            display_ids.append(asset_id)
        elif action in {"display_keep_video", "protect_keep_video"}:
            video_ids.append(asset_id)
        elif action == "review_keep":
            buffer_image_ids.append(asset_id)
        elif action == "review_video":
            buffer_video_ids.append(asset_id)

    if archive_ids:
        for batch in _batched(archive_ids, batch_size):
            client.update_assets(batch, visibility="archive")
    if timeline_ids:
        for batch in _batched(timeline_ids, batch_size):
            client.update_assets(batch, visibility="timeline")

    album_plan = {
        HYBRID_ALBUMS["display"]: display_ids,
        HYBRID_ALBUMS["video"]: video_ids,
    }
    if with_buffer_albums:
        album_plan[HYBRID_ALBUMS["buffer_images"]] = buffer_image_ids
        album_plan[HYBRID_ALBUMS["buffer_videos"]] = buffer_video_ids

    created_albums = _reset_named_albums(client, list(album_plan))
    linked_assets = 0
    for album_name, asset_ids in album_plan.items():
        if not asset_ids:
            continue
        album_id = created_albums[album_name]
        for batch in _batched(asset_ids, batch_size):
            client.add_assets_to_album(album_id, batch)
            linked_assets += len(batch)

    return {
        "timeline": len(timeline_ids),
        "archive": len(archive_ids),
        "display_album": len(display_ids),
        "video_album": len(video_ids),
        "buffer_image_album": len(buffer_image_ids) if with_buffer_albums else 0,
        "buffer_video_album": len(buffer_video_ids) if with_buffer_albums else 0,
        "album_asset_links": linked_assets,
    }


def _reset_trial_albums(client) -> Dict[str, str]:
    return _reset_named_albums(client, list(TRIAL_ALBUMS.values()))


def _reset_named_albums(client, album_names: List[str]) -> Dict[str, str]:
    existing = client.list_albums()
    target_names = set(album_names)
    for album in existing:
        if album.get("albumName") in target_names:
            client.delete_album(album["id"])
    album_ids: Dict[str, str] = {}
    for album_name in album_names:
        created = client.create_album(album_name)
        album_ids[album_name] = created["id"]
    return album_ids


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]
