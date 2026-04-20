from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Iterable, List

from . import db


TRIAL_ALBUMS = {
    "keep": "NASAI 试点-精选",
    "archive": "NASAI 试点-待归档",
    "video_keep": "NASAI 试点-视频高分",
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


def _reset_trial_albums(client) -> Dict[str, str]:
    existing = client.list_albums()
    for album in existing:
        if album.get("albumName") in TRIAL_ALBUMS.values():
            client.delete_album(album["id"])
    album_ids: Dict[str, str] = {}
    for album_name in TRIAL_ALBUMS.values():
        created = client.create_album(album_name)
        album_ids[album_name] = created["id"]
    return album_ids

