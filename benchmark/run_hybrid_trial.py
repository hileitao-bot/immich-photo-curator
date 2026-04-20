from __future__ import annotations

import json
import math
import re
import shutil
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from jinja2 import Template
from nasai.scoring import VisionScorer, _image_signature, _image_similarity
from PIL import Image


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DB_PATH = PROJECT_ROOT / "nasai.db"
RESULTS_DIR = ROOT / "results" / "hybrid"
REPORT_PATH = RESULTS_DIR / "index.html"
FILTERED_REPORT_PATH = RESULTS_DIR / "filtered.html"
ACTION_PATH = RESULTS_DIR / "actions.json"
AESTHETIC_CACHE_PATH = RESULTS_DIR / "aesthetic_scores.json"
FACE_ANALYSIS_CACHE_PATH = RESULTS_DIR / "face_analysis.json"
IMAGES_DIR = RESULTS_DIR / "images"
VIDEOS_DIR = RESULTS_DIR / "videos"
DEDUPE_DIR = RESULTS_DIR / "dedupe"
SYSTEM_BUFFER_DIR = RESULTS_DIR / "system_buffer"
SYSTEM_BUFFER_IMAGES_DIR = SYSTEM_BUFFER_DIR / "images"
SYSTEM_BUFFER_VIDEOS_DIR = SYSTEM_BUFFER_DIR / "videos"
SYSTEM_BUFFER_MANIFEST_PATH = SYSTEM_BUFFER_DIR / "manifest.json"
SYSTEM_BUFFER_README_PATH = SYSTEM_BUFFER_DIR / "README.txt"
TEMPLATE_PATH = ROOT / "templates" / "hybrid_report.html"
FILTERED_TEMPLATE_PATH = ROOT / "templates" / "hybrid_filtered_report.html"
IMMICH_BASE_URL = "http://192.168.1.18:2283"
NEGATIVE_TAGS = {"文档", "屏幕截图", "收据", "白板", "表格", "演示文稿", "聊天记录"}
TOP_PICK_PREVIEW_LIMIT = 240
DUPLICATE_GROUP_PREVIEW_LIMIT = 30
NEGATIVE_PREVIEW_LIMIT = 72
REVIEW_PREVIEW_LIMIT = 96
LOW_PRIORITY_PREVIEW_LIMIT = 96
VIDEO_PREVIEW_LIMIT = 80
GLOBAL_EXACT_GRAY_THRESHOLD = 0.995
GLOBAL_EXACT_HIST_THRESHOLD = 0.995


def main() -> None:
    prepare_results_dirs()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM assets
            WHERE raw_score IS NOT NULL
            ORDER BY file_created_at DESC
            """
        ).fetchall()
        image_rows = [row for row in rows if row["asset_type"] == "IMAGE" and has_thumb(row)]
        video_rows = [row for row in rows if row["asset_type"] == "VIDEO" and has_thumb(row)]

        aesthetic_scores = run_aesthetic(image_rows)
        duplicate_index = run_duplicate_detection(image_rows)
        face_scorer = build_face_scorer()
        face_analysis_cache = load_face_analysis_cache()
        payload = build_payload(
            image_rows,
            video_rows,
            aesthetic_scores,
            duplicate_index,
            face_scorer=face_scorer,
            face_analysis_cache=face_analysis_cache,
        )
        save_face_analysis_cache(face_analysis_cache)

        stage_sections(payload)
        export_system_buffer(payload)
        REPORT_PATH.write_text(Template(TEMPLATE_PATH.read_text()).render(**payload))
        FILTERED_REPORT_PATH.write_text(
            Template(FILTERED_TEMPLATE_PATH.read_text()).render(**payload)
        )
        ACTION_PATH.write_text(
            json.dumps(payload["actions"], ensure_ascii=False, indent=2)
        )
        print(f"Wrote hybrid report to {REPORT_PATH}")
    finally:
        conn.close()


def prepare_results_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for directory in [IMAGES_DIR, VIDEOS_DIR, DEDUPE_DIR, SYSTEM_BUFFER_DIR]:
        if directory.exists():
            shutil.rmtree(directory)
    for directory in [
        IMAGES_DIR,
        VIDEOS_DIR,
        DEDUPE_DIR,
        SYSTEM_BUFFER_IMAGES_DIR,
        SYSTEM_BUFFER_VIDEOS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def has_thumb(row: sqlite3.Row) -> bool:
    thumb = row["thumbnail_cache_path"]
    return bool(thumb) and Path(thumb).exists()


def run_aesthetic(rows: List[sqlite3.Row]) -> Dict[str, float]:
    cached = load_aesthetic_cache(rows)
    if cached is not None:
        return cached

    model, processor = convert_v2_5_from_siglip(local_files_only=True)
    model.eval()

    scores: Dict[str, float] = {}
    batch_rows: List[sqlite3.Row] = []

    def flush() -> None:
        if not batch_rows:
            return
        images = [Image.open(row["thumbnail_cache_path"]).convert("RGB") for row in batch_rows]
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits.squeeze(-1).tolist()
        if isinstance(logits, float):
            logits = [logits]
        for row, score in zip(batch_rows, logits):
            scores[row["asset_id"]] = float(score)
        batch_rows.clear()

    for row in rows:
        batch_rows.append(row)
        if len(batch_rows) >= 16:
            flush()
    flush()
    AESTHETIC_CACHE_PATH.write_text(json.dumps(scores, ensure_ascii=False))
    return scores


def load_aesthetic_cache(rows: List[sqlite3.Row]) -> Dict[str, float] | None:
    if not AESTHETIC_CACHE_PATH.exists():
        return None
    cached = json.loads(AESTHETIC_CACHE_PATH.read_text())
    asset_ids = {row["asset_id"] for row in rows}
    if asset_ids.issubset(cached.keys()):
        return {asset_id: float(cached[asset_id]) for asset_id in asset_ids}
    return None


def build_face_scorer() -> VisionScorer:
    return VisionScorer(
        PROJECT_ROOT / "cache",
        helper_source=PROJECT_ROOT / "tools" / "vision_probe.swift",
        helper_binary=PROJECT_ROOT / "preview" / "vision_probe",
    )


def load_face_analysis_cache() -> Dict[str, Dict[str, float]]:
    if not FACE_ANALYSIS_CACHE_PATH.exists():
        return {}
    cached = json.loads(FACE_ANALYSIS_CACHE_PATH.read_text())
    return {
        asset_id: {
            "faceCount": int(values.get("faceCount") or 0),
            "bestFaceCaptureQuality": float(values.get("bestFaceCaptureQuality") or -1.0),
            "largestFaceArea": float(values.get("largestFaceArea") or 0.0),
        }
        for asset_id, values in cached.items()
    }


def save_face_analysis_cache(cache: Dict[str, Dict[str, float]]) -> None:
    FACE_ANALYSIS_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def run_duplicate_detection(rows: List[sqlite3.Row]) -> Dict[str, Dict[str, Any]]:
    candidate_rows = [row for row in rows if row["file_created_at"]]
    if not candidate_rows:
        return {}

    items: List[Dict[str, Any]] = []
    duplicate_id_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        metadata = json.loads(row["metadata_json"] or "{}")
        item = {
            "assetId": row["asset_id"],
            "row": row,
            "path": Path(row["thumbnail_cache_path"]),
            "timestamp": timestamp_seconds(row["file_created_at"]),
            "burstGroupId": row["burst_group_id"] or None,
            "duplicateId": metadata.get("duplicateId") or None,
            "namedPeople": named_people_name_set_from_metadata(metadata),
            "hasPeopleSignal": has_people_signal(row),
            "width": int(row["width"] or 0),
            "height": int(row["height"] or 0),
        }
        items.append(item)
        if item["duplicateId"]:
            duplicate_id_groups[str(item["duplicateId"])].append(item)

    items.sort(key=lambda item: (item["timestamp"], item["assetId"]))
    parent = {item["assetId"]: item["assetId"] for item in items}

    def find(asset_id: str) -> str:
        while parent[asset_id] != asset_id:
            parent[asset_id] = parent[parent[asset_id]]
            asset_id = parent[asset_id]
        return asset_id

    def union(left_asset_id: str, right_asset_id: str) -> None:
        root_left = find(left_asset_id)
        root_right = find(right_asset_id)
        if root_left != root_right:
            parent[root_right] = root_left

    for group in duplicate_id_groups.values():
        if len(group) < 2:
            continue
        ordered_group = sorted(group, key=lambda item: (item["timestamp"], item["assetId"]))
        winner = ordered_group[0]["assetId"]
        for item in ordered_group[1:]:
            union(winner, item["assetId"])

    feature_cache: Dict[str, Any] = {}
    union_global_exact_duplicates(
        items,
        union=union,
        feature_cache=feature_cache,
    )
    max_window_seconds = 1800.0
    for left_index, left_item in enumerate(items):
        for right_item in items[left_index + 1 :]:
            delta_seconds = right_item["timestamp"] - left_item["timestamp"]
            if delta_seconds > max_window_seconds:
                break
            if not should_compare_duplicate_items(left_item, right_item, delta_seconds):
                continue
            gray_similarity, hist_similarity = _image_similarity(
                left_item["path"],
                right_item["path"],
                feature_cache,
            )
            if not is_visual_duplicate_pair(
                left_item,
                right_item,
                delta_seconds=delta_seconds,
                gray_similarity=gray_similarity,
                hist_similarity=hist_similarity,
            ):
                continue
            if not allow_duplicate_union(
                left_item["row"],
                right_item["row"],
                delta_seconds=delta_seconds,
            ):
                continue
            union(left_item["assetId"], right_item["assetId"])

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[find(item["assetId"])].append(item)

    duplicate_groups = [
        members
        for members in grouped.values()
        if len(members) > 1
    ]
    duplicate_groups.sort(
        key=lambda members: (
            -len(members),
            min(member["assetId"] for member in members),
        )
    )

    index: Dict[str, Dict[str, Any]] = {}
    for cluster_no, members in enumerate(duplicate_groups, start=1):
        cluster_id = f"dup-{cluster_no}"
        burst_group_id = next(
            (member["burstGroupId"] for member in members if member["burstGroupId"]),
            None,
        )
        for member in members:
            index[member["assetId"]] = {
                "clusterId": cluster_id,
                "clusterSize": len(members),
                "burstGroupId": burst_group_id,
            }
    return index


def build_payload(
    image_rows: List[sqlite3.Row],
    video_rows: List[sqlite3.Row],
    aesthetic_scores: Dict[str, float],
    duplicate_index: Dict[str, Dict[str, Any]],
    *,
    face_scorer: VisionScorer,
    face_analysis_cache: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    named_people_cache: Dict[str, Dict[str, Any]] = {}
    duplicate_clusters: Dict[str, List[sqlite3.Row]] = defaultdict(list)
    for row in image_rows:
        cluster = duplicate_index.get(row["asset_id"])
        if cluster and int(cluster["clusterSize"]) > 1:
            duplicate_clusters[cluster["clusterId"]].append(row)

    duplicate_winners: Dict[str, str] = {}
    duplicate_demotions: set[str] = set()
    duplicate_groups_payload: List[Dict[str, Any]] = []
    for cluster_id, rows in duplicate_clusters.items():
        if all(is_negative_like_image(row) for row in rows):
            continue
        face_aware_group = is_people_duplicate_group(
            rows,
            face_scorer=face_scorer,
            face_analysis_cache=face_analysis_cache,
        )
        ordered = sorted(
            rows,
            key=lambda row: duplicate_sort_key(
                row,
                aesthetic_scores,
                face_scorer=face_scorer,
                face_analysis_cache=face_analysis_cache,
                named_people_cache=named_people_cache,
                face_aware_group=face_aware_group,
            ),
            reverse=True,
        )
        winner = ordered[0]
        duplicate_winners[cluster_id] = winner["asset_id"]
        for row in ordered[1:]:
            duplicate_demotions.add(row["asset_id"])
        duplicate_groups_payload.append(
            {
                "clusterId": cluster_id,
                "size": len(ordered),
                "winner": build_image_card(
                    winner,
                    aesthetic_scores=esthetic(aesthetic_scores, winner["asset_id"]),
                    proposed_action="保留最佳",
                    reason="同一连拍候选组内的近重复最佳图",
                    duplicate_cluster=cluster_id,
                    named_people_cache=named_people_cache,
                    is_winner=True,
                ),
                "alternates": [
                    build_image_card(
                        row,
                        aesthetic_scores=esthetic(aesthetic_scores, row["asset_id"]),
                        proposed_action="归并到重复",
                        reason=f"并入 {winner['original_file_name']}",
                        duplicate_cluster=cluster_id,
                        named_people_cache=named_people_cache,
                        is_winner=False,
                    )
                    for row in ordered[1:]
                ],
            }
        )
    duplicate_groups_payload.sort(
        key=lambda group: (-group["size"], -group["winner"]["aestheticScore"])
    )

    negatives: List[sqlite3.Row] = []
    unique_candidates: List[sqlite3.Row] = []
    actions: List[Dict[str, Any]] = []
    for row in image_rows:
        if is_negative_like_image(row):
            negatives.append(row)
            actions.append(
                build_action(
                    row,
                    action="archive_negative",
                    reason="document_or_screenshot",
                    aesthetic_score=esthetic(aesthetic_scores, row["asset_id"]),
                    named_people_cache=named_people_cache,
                )
            )
        elif row["asset_id"] in duplicate_demotions:
            cluster_id = duplicate_index[row["asset_id"]]["clusterId"]
            winner_id = duplicate_winners[cluster_id]
            actions.append(
                build_action(
                    row,
                    action="archive_duplicate",
                    reason=f"duplicate_of:{winner_id}",
                    aesthetic_score=esthetic(aesthetic_scores, row["asset_id"]),
                    duplicate_cluster=cluster_id,
                    named_people_cache=named_people_cache,
                )
            )
        else:
            unique_candidates.append(row)

    unique_candidates.sort(
        key=lambda row: unique_image_sort_key(
            row,
            aesthetic_scores,
            named_people_cache=named_people_cache,
        ),
        reverse=True,
    )
    display_keep_target = max(120, math.ceil(len(unique_candidates) * 0.30))
    review_keep_target = math.ceil(len(unique_candidates) * 0.40)

    top_picks: List[Dict[str, Any]] = []
    review_keep_payload_all: List[Dict[str, Any]] = []
    low_priority_payload_all: List[Dict[str, Any]] = []
    selected_display_items: List[Dict[str, Any]] = []
    display_feature_cache: Dict[str, Any] = {}
    display_keep_count = 0
    review_keep_count = 0
    archive_low_count = 0

    for index, row in enumerate(unique_candidates):
        aesthetic_score = esthetic(aesthetic_scores, row["asset_id"])
        similar_anchor = None
        if display_keep_count < display_keep_target:
            similar_anchor = similar_display_anchor(
                row,
                selected_items=selected_display_items,
                feature_cache=display_feature_cache,
            )

        if display_keep_count < display_keep_target and similar_anchor is None:
            action = "display_keep"
            reason = "top_30_percent_unique_images"
            label = "精选展示"
            display_keep_count += 1
            selected_display_items.append(display_similarity_item_for_row(row))
        elif similar_anchor is not None:
            action = "review_keep"
            reason = f"similar_scene_to_display:{similar_anchor['asset_id']}"
            label = "场景候补"
            review_keep_count += 1
        elif review_keep_count < review_keep_target:
            action = "review_keep"
            reason = "mid_band_unique_images"
            label = "系统缓冲"
            review_keep_count += 1
        else:
            action = "archive_low"
            reason = "low_band_unique_images"
            label = "低优先级"
            archive_low_count += 1
        actions.append(
                build_action(
                    row,
                    action=action,
                    reason=reason,
                    aesthetic_score=aesthetic_score,
                    named_people_cache=named_people_cache,
                )
            )
        if action == "display_keep" and len(top_picks) < TOP_PICK_PREVIEW_LIMIT:
            top_picks.append(
                build_image_card(
                    row,
                    aesthetic_scores=aesthetic_score,
                    proposed_action=label,
                    reason=reason_label(reason),
                    duplicate_cluster=duplicate_index.get(row["asset_id"], {}).get("clusterId"),
                    named_people_cache=named_people_cache,
                    is_winner=True,
                )
            )
        elif action == "review_keep":
            review_keep_payload_all.append(
                build_image_card(
                    row,
                    aesthetic_scores=aesthetic_score,
                    proposed_action=label,
                    reason=reason_label(reason),
                    duplicate_cluster=duplicate_index.get(row["asset_id"], {}).get("clusterId"),
                    named_people_cache=named_people_cache,
                    is_winner=False,
                )
            )
        elif action == "archive_low":
            low_priority_payload_all.append(
                build_image_card(
                    row,
                    aesthetic_scores=aesthetic_score,
                    proposed_action=label,
                    reason=reason_label(reason),
                    duplicate_cluster=duplicate_index.get(row["asset_id"], {}).get("clusterId"),
                    named_people_cache=named_people_cache,
                    is_winner=False,
                )
            )

    negatives.sort(
        key=lambda row: (
            "屏幕截图" not in json.loads(row["chinese_tags_json"] or "[]"),
            "文档" not in json.loads(row["chinese_tags_json"] or "[]"),
            row["file_created_at"] or "",
        )
    )
    negative_payload_all = [
        build_image_card(
            row,
            aesthetic_scores=esthetic(aesthetic_scores, row["asset_id"]),
            proposed_action="归档样本",
            reason="文档/截图优先下沉",
            duplicate_cluster=duplicate_index.get(row["asset_id"], {}).get("clusterId"),
            named_people_cache=named_people_cache,
            is_winner=False,
        )
        for row in negatives
    ]

    protected_videos: List[sqlite3.Row] = []
    other_videos: List[sqlite3.Row] = []
    review_videos: List[sqlite3.Row] = []
    for row in video_rows:
        if is_protected_video(
            row,
            named_people_cache=named_people_cache,
            face_scorer=face_scorer,
            face_analysis_cache=face_analysis_cache,
        ):
            protected_videos.append(row)
            actions.append(
                build_action(
                    row,
                    action="protect_keep_video",
                    reason="named_people_or_story_video",
                    aesthetic_score=None,
                    named_people_cache=named_people_cache,
                )
            )
        elif float(row["percentile"] or 0.0) >= 0.8:
            other_videos.append(row)
            actions.append(
                build_action(
                    row,
                    action="display_keep_video",
                    reason="high_percentile_video",
                    aesthetic_score=None,
                    named_people_cache=named_people_cache,
                )
            )
        else:
            review_videos.append(row)
            actions.append(
                build_action(
                    row,
                    action="review_video",
                    reason="video_not_auto_archived_in_preview",
                    aesthetic_score=None,
                    named_people_cache=named_people_cache,
                )
            )
    protected_videos.sort(
        key=lambda row: (
            -named_people_info_for_row(row, named_people_cache=named_people_cache)["count"],
            "/vlog/" not in (row["original_path"] or "").lower(),
            -(float(row["percentile"] or 0.0)),
        )
    )
    other_videos.sort(
        key=lambda row: (
            -named_people_info_for_row(row, named_people_cache=named_people_cache)["count"],
            -(float(row["percentile"] or 0.0)),
            -duration_to_seconds(row["duration"]),
            -(float(row["raw_score"] or 0.0)),
        )
    )
    review_videos.sort(
        key=lambda row: (
            -named_people_info_for_row(row, named_people_cache=named_people_cache)["count"],
            -(float(row["percentile"] or 0.0)),
            -duration_to_seconds(row["duration"]),
            -(float(row["raw_score"] or 0.0)),
        )
    )
    video_payload = [
        build_video_card(
            row,
            proposed_action="视频保护",
            reason="已命名人物、检测到人脸且命中合照/儿童/宝宝、vlog、旅行命名视频、长视频默认先保留",
            named_people_cache=named_people_cache,
        )
        for row in protected_videos[:VIDEO_PREVIEW_LIMIT]
    ]
    video_display_payload_all = [
        build_video_card(
            row,
            proposed_action="高分未保护视频",
            reason="分位前 20%，但未命中人脸家庭短视频、vlog 或长视频保护规则",
            named_people_cache=named_people_cache,
        )
        for row in other_videos
    ]
    video_review_payload_all = [
        build_video_card(
            row,
            proposed_action="系统缓冲视频",
            reason="暂不精选，也不自动归档，作为系统缓冲层保留",
            named_people_cache=named_people_cache,
        )
        for row in review_videos
    ]

    summary = {
        "scoredTotal": len(image_rows) + len(video_rows),
        "imageCount": len(image_rows),
        "videoCount": len(video_rows),
        "negativeImages": len(negatives),
        "duplicateGroups": len(duplicate_groups_payload),
        "duplicateDemotions": len(duplicate_demotions),
        "uniqueCandidates": len(unique_candidates),
        "displayKeeps": display_keep_count,
        "reviewKeeps": review_keep_count,
        "archiveLows": archive_low_count,
        "protectedVideos": len(protected_videos),
        "displayKeepVideos": len(other_videos),
        "reviewVideos": len(review_videos),
    }
    return {
        "summary": summary,
        "filteredPageName": FILTERED_REPORT_PATH.name,
        "topPicks": top_picks,
        "reviewKeeps": review_keep_payload_all[:REVIEW_PREVIEW_LIMIT],
        "reviewKeepsAll": review_keep_payload_all,
        "lowPrioritySamples": low_priority_payload_all[:LOW_PRIORITY_PREVIEW_LIMIT],
        "lowPrioritySamplesAll": low_priority_payload_all,
        "duplicateGroups": duplicate_groups_payload[:DUPLICATE_GROUP_PREVIEW_LIMIT],
        "duplicateGroupsAll": duplicate_groups_payload,
        "negativeSamples": negative_payload_all[:NEGATIVE_PREVIEW_LIMIT],
        "negativeSamplesAll": negative_payload_all,
        "videoProtected": video_payload,
        "videoDisplayAll": video_display_payload_all,
        "videoReviewAll": video_review_payload_all,
        "actions": sorted(actions, key=lambda item: (item["action"], item["assetId"])),
    }


def stage_sections(payload: Dict[str, Any]) -> None:
    seen_images: set[str] = set()
    seen_videos: set[str] = set()

    for card in payload["topPicks"]:
        stage_card(card, seen_images, seen_videos)
    for card in payload["reviewKeepsAll"]:
        stage_card(card, seen_images, seen_videos)
    for card in payload["lowPrioritySamplesAll"]:
        stage_card(card, seen_images, seen_videos)
    for group in payload["duplicateGroupsAll"]:
        stage_card(group["winner"], seen_images, seen_videos)
        for card in group["alternates"]:
            stage_card(card, seen_images, seen_videos)
    for card in payload["negativeSamplesAll"]:
        stage_card(card, seen_images, seen_videos)
    for card in payload["videoDisplayAll"]:
        stage_card(card, seen_images, seen_videos)
    for card in payload["videoReviewAll"]:
        stage_card(card, seen_images, seen_videos)
    for card in payload["videoProtected"]:
        stage_card(card, seen_images, seen_videos)


def stage_card(card: Dict[str, Any], seen_images: set[str], seen_videos: set[str]) -> None:
    source = Path(card["sourceThumbPath"])
    if card["type"] == "VIDEO":
        if card["assetId"] in seen_videos:
            return
        seen_videos.add(card["assetId"])
        target = VIDEOS_DIR / Path(card["thumbPath"]).name
    else:
        if card["assetId"] in seen_images:
            return
        seen_images.add(card["assetId"])
        target = IMAGES_DIR / Path(card["thumbPath"]).name
    if source.exists() and not target.exists():
        shutil.copy2(source, target)


def export_system_buffer(payload: Dict[str, Any]) -> None:
    images = export_system_buffer_cards(
        payload["reviewKeepsAll"],
        target_dir=SYSTEM_BUFFER_IMAGES_DIR,
        relative_prefix="images",
    )
    videos = export_system_buffer_cards(
        payload["videoReviewAll"],
        target_dir=SYSTEM_BUFFER_VIDEOS_DIR,
        relative_prefix="videos",
    )
    manifest = {
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(SYSTEM_BUFFER_DIR),
        "imageCount": len(images),
        "videoCount": len(videos),
        "images": images,
        "videos": videos,
    }
    SYSTEM_BUFFER_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    SYSTEM_BUFFER_README_PATH.write_text(build_system_buffer_readme(manifest))


def export_system_buffer_cards(
    cards: List[Dict[str, Any]],
    *,
    target_dir: Path,
    relative_prefix: str,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    seen_asset_ids: set[str] = set()
    for card in cards:
        asset_id = str(card["assetId"])
        if asset_id in seen_asset_ids:
            continue
        seen_asset_ids.add(asset_id)
        source = Path(card["sourceThumbPath"])
        exported_name = system_buffer_export_name(card)
        target_path = target_dir / exported_name
        if source.exists():
            shutil.copy2(source, target_path)
        entries.append(
            {
                "assetId": asset_id,
                "fileName": card["fileName"],
                "type": card["type"],
                "proposedAction": card["proposedAction"],
                "reason": card["reason"],
                "currentAction": card["currentAction"],
                "grade": card["grade"],
                "rawScore": card["rawScore"],
                "aestheticScore": card.get("aestheticScore"),
                "namedPeople": card.get("namedPeople") or [],
                "tags": card.get("tags") or [],
                "duration": card.get("duration"),
                "openUrl": card["openUrl"],
                "path": card["path"],
                "sourceThumbPath": str(source),
                "exportedThumbPath": (
                    f"{relative_prefix}/{exported_name}" if source.exists() else None
                ),
            }
        )
    return entries


def system_buffer_export_name(card: Dict[str, Any]) -> str:
    source_suffix = Path(card["sourceThumbPath"]).suffix or Path(card["thumbPath"]).suffix or ".webp"
    base_name = Path(card["fileName"] or card["assetId"]).name
    safe_base_name = sanitize_export_name(base_name) or str(card["assetId"])
    return f"{safe_base_name}__{card['assetId']}{source_suffix}"


def sanitize_export_name(value: str) -> str:
    cleaned = re.sub(r"[^\w.\- ]+", "_", value, flags=re.UNICODE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" .")
    return cleaned[:160] or "asset"


def build_system_buffer_readme(manifest: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "系统缓冲导出说明",
            "",
            "此目录只包含系统缓冲层的本地预览缩略图，不包含 NAS 原始文件。",
            "images/ 对应系统缓冲图片，videos/ 对应系统缓冲视频封面。",
            "manifest.json 记录了每个条目的 Immich 打开链接、原始路径、标签、人物和当前判定原因。",
            "",
            f"生成时间: {manifest['generatedAt']}",
            f"系统缓冲图片: {manifest['imageCount']}",
            f"系统缓冲视频: {manifest['videoCount']}",
            "",
            "建议筛选方式：",
            "1. 先按目录快速看缩略图。",
            "2. 需要追溯时，在 manifest.json 里搜索文件名或 assetId。",
            "3. 用 openUrl 直接回到 Immich 查看原资源。",
            "",
        ]
    )


def build_image_card(
    row: sqlite3.Row,
    *,
    aesthetic_scores: float,
    proposed_action: str,
    reason: str,
    duplicate_cluster: str | None,
    named_people_cache: Dict[str, Dict[str, Any]],
    is_winner: bool,
) -> Dict[str, Any]:
    named_people = named_people_info_for_row(row, named_people_cache=named_people_cache)
    return {
        "assetId": row["asset_id"],
        "fileName": row["original_file_name"],
        "thumbPath": f"images/{media_name(row)}",
        "sourceThumbPath": row["thumbnail_cache_path"],
        "type": "IMAGE",
        "grade": row["grade"],
        "currentAction": row["suggested_action"],
        "rawScore": round(float(row["raw_score"] or 0.0), 3),
        "aestheticScore": round(aesthetic_scores, 3),
        "proposedAction": proposed_action,
        "reason": reason,
        "duplicateCluster": duplicate_cluster,
        "isWinner": is_winner,
        "namedPeople": named_people["names"],
        "namedPeopleCount": named_people["count"],
        "tags": json.loads(row["chinese_tags_json"] or "[]"),
        "openUrl": open_url(row["asset_id"]),
        "path": row["original_path"],
    }


def build_video_card(
    row: sqlite3.Row,
    *,
    proposed_action: str,
    reason: str,
    named_people_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    named_people = named_people_info_for_row(row, named_people_cache=named_people_cache)
    return {
        "assetId": row["asset_id"],
        "fileName": row["original_file_name"],
        "thumbPath": f"videos/{media_name(row)}",
        "sourceThumbPath": row["thumbnail_cache_path"],
        "type": "VIDEO",
        "grade": row["grade"],
        "currentAction": row["suggested_action"],
        "rawScore": round(float(row["raw_score"] or 0.0), 3),
        "aestheticScore": None,
        "proposedAction": proposed_action,
        "reason": reason,
        "duplicateCluster": None,
        "isWinner": True,
        "namedPeople": named_people["names"],
        "namedPeopleCount": named_people["count"],
        "tags": json.loads(row["chinese_tags_json"] or "[]"),
        "duration": row["duration"],
        "openUrl": open_url(row["asset_id"]),
        "path": row["original_path"],
    }


def build_action(
    row: sqlite3.Row,
    *,
    action: str,
    reason: str,
    aesthetic_score: float | None,
    named_people_cache: Dict[str, Dict[str, Any]],
    duplicate_cluster: str | None = None,
) -> Dict[str, Any]:
    named_people = named_people_info_for_row(row, named_people_cache=named_people_cache)
    return {
        "assetId": row["asset_id"],
        "type": row["asset_type"],
        "fileName": row["original_file_name"],
        "path": row["original_path"],
        "action": action,
        "reason": reason,
        "currentAction": row["suggested_action"],
        "grade": row["grade"],
        "rawScore": round(float(row["raw_score"] or 0.0), 3),
        "aestheticScore": None if aesthetic_score is None else round(float(aesthetic_score), 3),
        "namedPeople": named_people["names"],
        "duplicateCluster": duplicate_cluster,
    }


def image_area(row: sqlite3.Row) -> int:
    return int(row["width"] or 0) * int(row["height"] or 0)


def is_negative_like_image(row: sqlite3.Row) -> bool:
    tags = set(json.loads(row["chinese_tags_json"] or "[]"))
    if tags & NEGATIVE_TAGS:
        return True

    labels = {
        str(label.get("identifier", "")).lower()
        for label in json.loads(row["vision_labels_json"] or "[]")
    }
    if labels & {"document", "printed_page", "receipt", "screenshot", "handwriting"}:
        return True

    raw_score = float(row["raw_score"] or 0.0)
    if raw_score < 0 and labels & {"chart", "diagram", "sign", "whiteboard", "book"}:
        return True
    return False


def named_people_info_for_row(
    row: sqlite3.Row,
    *,
    named_people_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    asset_id = row["asset_id"]
    cached = named_people_cache.get(asset_id)
    if cached is not None:
        return cached

    names: List[str] = []
    seen: set[str] = set()
    favorite_count = 0
    largest_face_area = 0.0
    metadata = json.loads(row["metadata_json"] or "{}")
    for person in metadata.get("people") or []:
        if person.get("isHidden"):
            continue
        name = str(person.get("name") or "").strip()
        if not name:
            continue
        if name not in seen:
            seen.add(name)
            names.append(name)
        if person.get("isFavorite"):
            favorite_count += 1
        for face in person.get("faces") or []:
            width = float(face.get("imageWidth") or 0.0)
            height = float(face.get("imageHeight") or 0.0)
            if width <= 0 or height <= 0:
                continue
            x1 = float(face.get("boundingBoxX1") or 0.0)
            y1 = float(face.get("boundingBoxY1") or 0.0)
            x2 = float(face.get("boundingBoxX2") or 0.0)
            y2 = float(face.get("boundingBoxY2") or 0.0)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)
            largest_face_area = max(largest_face_area, area)

    result = {
        "names": names,
        "count": len(names),
        "favoriteCount": favorite_count,
        "largestFaceArea": largest_face_area,
    }
    named_people_cache[asset_id] = result
    return result


def unique_image_sort_key(
    row: sqlite3.Row,
    aesthetic_scores: Dict[str, float],
    *,
    named_people_cache: Dict[str, Dict[str, Any]],
) -> tuple:
    named_people = named_people_info_for_row(row, named_people_cache=named_people_cache)
    return (
        int(named_people["count"] > 0),
        named_people["count"],
        named_people["favoriteCount"],
        named_people["largestFaceArea"],
        aesthetic_scores.get(row["asset_id"], -999.0),
        float(row["raw_score"] or 0.0),
        image_area(row),
    )


def is_people_duplicate_group(
    rows: List[sqlite3.Row],
    *,
    face_scorer: VisionScorer,
    face_analysis_cache: Dict[str, Dict[str, float]],
) -> bool:
    return any(
        has_people_signal(row)
        or face_info_for_row(
            row,
            face_scorer=face_scorer,
            face_analysis_cache=face_analysis_cache,
        )["faceCount"]
        > 0
        for row in rows
    )


def duplicate_sort_key(
    row: sqlite3.Row,
    aesthetic_scores: Dict[str, float],
    *,
    face_scorer: VisionScorer,
    face_analysis_cache: Dict[str, Dict[str, float]],
    named_people_cache: Dict[str, Dict[str, Any]],
    face_aware_group: bool,
) -> tuple:
    named_people = named_people_info_for_row(row, named_people_cache=named_people_cache)
    if face_aware_group:
        face_info = face_info_for_row(
            row,
            face_scorer=face_scorer,
            face_analysis_cache=face_analysis_cache,
        )
        return (
            int(named_people["count"] > 0),
            named_people["count"],
            named_people["favoriteCount"],
            named_people["largestFaceArea"],
            float(face_info["bestFaceCaptureQuality"]),
            float(face_info["largestFaceArea"]),
            int(face_info["faceCount"]),
            -int(row["burst_rank"] or 10_000),
            aesthetic_scores.get(row["asset_id"], -999.0),
            float(row["raw_score"] or 0.0),
            image_area(row),
        )
    return (
        int(named_people["count"] > 0),
        named_people["count"],
        named_people["favoriteCount"],
        named_people["largestFaceArea"],
        aesthetic_scores.get(row["asset_id"], -999.0),
        float(row["raw_score"] or 0.0),
        image_area(row),
    )


def face_info_for_row(
    row: sqlite3.Row,
    *,
    face_scorer: VisionScorer,
    face_analysis_cache: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    asset_id = row["asset_id"]
    cached = face_analysis_cache.get(asset_id)
    if cached is not None:
        return cached

    result = {
        "faceCount": 0,
        "bestFaceCaptureQuality": -1.0,
        "largestFaceArea": 0.0,
    }
    thumb_path = row["thumbnail_cache_path"]
    if thumb_path:
        try:
            analysis = face_scorer.analyze_path(Path(thumb_path))
            result = {
                "faceCount": int(analysis.get("faceCount") or 0),
                "bestFaceCaptureQuality": float(analysis.get("bestFaceCaptureQuality") or -1.0),
                "largestFaceArea": float(analysis.get("largestFaceArea") or 0.0),
            }
        except Exception:
            result = result
    face_analysis_cache[asset_id] = result
    return result


def allow_duplicate_union(
    left_row: sqlite3.Row,
    right_row: sqlite3.Row,
    *,
    delta_seconds: float | None = None,
    exact_match: bool = False,
) -> bool:
    if exact_match:
        return True

    delta = delta_seconds
    if delta is None:
        delta = abs(
            timestamp_seconds(left_row["file_created_at"])
            - timestamp_seconds(right_row["file_created_at"])
        )

    left_names = row_named_people_name_set(left_row)
    right_names = row_named_people_name_set(right_row)
    if left_names and right_names and not (left_names & right_names):
        return False

    if not (has_people_signal(left_row) or has_people_signal(right_row)):
        return True

    if left_names & right_names:
        return delta <= 1800.0

    if left_row["burst_group_id"] and left_row["burst_group_id"] == right_row["burst_group_id"]:
        return True

    return delta <= 20.0


def should_compare_duplicate_items(
    left_item: Dict[str, Any],
    right_item: Dict[str, Any],
    delta_seconds: float,
) -> bool:
    if not compatible_dimensions(left_item, right_item):
        return False

    if left_item["duplicateId"] and left_item["duplicateId"] == right_item["duplicateId"]:
        return True

    if left_item["burstGroupId"] and left_item["burstGroupId"] == right_item["burstGroupId"]:
        return True

    if left_item["namedPeople"] & right_item["namedPeople"]:
        return delta_seconds <= 1800.0

    if left_item["hasPeopleSignal"] or right_item["hasPeopleSignal"]:
        return delta_seconds <= 20.0

    return delta_seconds <= 180.0


def is_visual_duplicate_pair(
    left_item: Dict[str, Any],
    right_item: Dict[str, Any],
    *,
    delta_seconds: float,
    gray_similarity: float,
    hist_similarity: float,
) -> bool:
    named_overlap = left_item["namedPeople"] & right_item["namedPeople"]
    same_burst = (
        left_item["burstGroupId"]
        and left_item["burstGroupId"] == right_item["burstGroupId"]
    )
    has_people = left_item["hasPeopleSignal"] or right_item["hasPeopleSignal"]

    if named_overlap:
        if delta_seconds <= 5.0 and hist_similarity >= 0.94 and gray_similarity >= 0.95:
            return True
        if delta_seconds <= 15.0 and hist_similarity >= 0.97 and gray_similarity >= 0.86:
            return True
        if delta_seconds <= 180.0 and hist_similarity >= 0.96 and gray_similarity >= 0.90:
            return True
        if delta_seconds <= 1800.0 and hist_similarity >= 0.992 and gray_similarity >= 0.965:
            return True

    if same_burst:
        if delta_seconds <= 25.0 and hist_similarity >= 0.95 and gray_similarity >= 0.90:
            return True
        if hist_similarity >= 0.965 and gray_similarity >= 0.92:
            return True

    if has_people:
        return delta_seconds <= 8.0 and hist_similarity >= 0.985 and gray_similarity >= 0.94

    if delta_seconds <= 25.0 and hist_similarity >= 0.95 and gray_similarity >= 0.90:
        return True
    return hist_similarity >= 0.965 and gray_similarity >= 0.92


def compatible_dimensions(left_item: Dict[str, Any], right_item: Dict[str, Any]) -> bool:
    left_width = int(left_item["width"] or 0)
    left_height = int(left_item["height"] or 0)
    right_width = int(right_item["width"] or 0)
    right_height = int(right_item["height"] or 0)
    if not left_width or not left_height or not right_width or not right_height:
        return True

    left_orientation = left_width >= left_height
    right_orientation = right_width >= right_height
    if left_orientation != right_orientation:
        return False

    left_ratio = max(left_width, left_height) / max(1, min(left_width, left_height))
    right_ratio = max(right_width, right_height) / max(1, min(right_width, right_height))
    return abs(left_ratio - right_ratio) <= 0.08


def orientation_matches(left_item: Dict[str, Any], right_item: Dict[str, Any]) -> bool:
    left_width = int(left_item["width"] or 0)
    left_height = int(left_item["height"] or 0)
    right_width = int(right_item["width"] or 0)
    right_height = int(right_item["height"] or 0)
    if not left_width or not left_height or not right_width or not right_height:
        return True
    return (left_width >= left_height) == (right_width >= right_height)


def aspect_ratio_value(item: Dict[str, Any]) -> float:
    width = int(item["width"] or 0)
    height = int(item["height"] or 0)
    if not width or not height:
        return 0.0
    return max(width, height) / max(1, min(width, height))


def compatible_display_dimensions(
    left_item: Dict[str, Any],
    right_item: Dict[str, Any],
) -> bool:
    if compatible_dimensions(left_item, right_item):
        return True

    left_ratio = aspect_ratio_value(left_item)
    right_ratio = aspect_ratio_value(right_item)
    if not left_ratio or not right_ratio or abs(left_ratio - right_ratio) > 0.02:
        return False

    delta_seconds = abs(left_item["timestamp"] - right_item["timestamp"])
    named_overlap = left_item["namedPeople"] & right_item["namedPeople"]
    same_burst = (
        left_item["burstGroupId"]
        and left_item["burstGroupId"] == right_item["burstGroupId"]
    )
    if same_burst:
        return delta_seconds <= 180.0
    if named_overlap:
        return delta_seconds <= 60.0
    return delta_seconds <= 8.0


def row_named_people_name_set(row: sqlite3.Row) -> set[str]:
    metadata = json.loads(row["metadata_json"] or "{}")
    return named_people_name_set_from_metadata(metadata)


def named_people_name_set_from_metadata(metadata: Dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for person in metadata.get("people") or []:
        if person.get("isHidden"):
            continue
        name = str(person.get("name") or "").strip()
        if name:
            names.add(name)
    return names


def union_global_exact_duplicates(
    items: List[Dict[str, Any]],
    *,
    union,
    feature_cache: Dict[str, Any],
) -> None:
    if len(items) < 2:
        return

    gray_vectors = []
    hist_vectors = []
    for item in items:
        gray_vector, hist_vector = _image_signature(item["path"], feature_cache)
        gray_vectors.append(gray_vector)
        hist_vectors.append(hist_vector)

    gray_matrix = np.stack(gray_vectors)
    hist_matrix = np.stack(hist_vectors)
    gray_scores = gray_matrix @ gray_matrix.T
    hist_scores = hist_matrix @ hist_matrix.T

    for left_index, left_item in enumerate(items):
        candidate_offsets = np.where(
            (gray_scores[left_index, left_index + 1 :] >= GLOBAL_EXACT_GRAY_THRESHOLD)
            & (hist_scores[left_index, left_index + 1 :] >= GLOBAL_EXACT_HIST_THRESHOLD)
        )[0]
        for offset in candidate_offsets:
            right_index = left_index + 1 + int(offset)
            right_item = items[right_index]
            if not compatible_dimensions(left_item, right_item):
                continue
            if not allow_duplicate_union(
                left_item["row"],
                right_item["row"],
                exact_match=True,
            ):
                continue
            union(left_item["assetId"], right_item["assetId"])


def display_similarity_item_for_row(row: sqlite3.Row) -> Dict[str, Any]:
    metadata = json.loads(row["metadata_json"] or "{}")
    return {
        "assetId": row["asset_id"],
        "row": row,
        "path": Path(row["thumbnail_cache_path"]),
        "timestamp": timestamp_seconds(row["file_created_at"]),
        "burstGroupId": row["burst_group_id"] or None,
        "namedPeople": named_people_name_set_from_metadata(metadata),
        "hasPeopleSignal": has_people_signal(row),
        "width": int(row["width"] or 0),
        "height": int(row["height"] or 0),
    }


def similar_display_anchor(
    row: sqlite3.Row,
    *,
    selected_items: List[Dict[str, Any]],
    feature_cache: Dict[str, Any],
) -> sqlite3.Row | None:
    candidate = display_similarity_item_for_row(row)
    for selected in selected_items:
        if not should_compare_display_items(candidate, selected):
            continue
        gray_similarity, hist_similarity = _image_similarity(
            candidate["path"],
            selected["path"],
            feature_cache,
        )
        if is_display_sibling_pair(
            candidate,
            selected,
            gray_similarity=gray_similarity,
            hist_similarity=hist_similarity,
        ):
            return selected["row"]
    return None


def should_compare_display_items(
    left_item: Dict[str, Any],
    right_item: Dict[str, Any],
) -> bool:
    if not compatible_display_dimensions(left_item, right_item):
        return False

    delta_seconds = abs(left_item["timestamp"] - right_item["timestamp"])
    if left_item["namedPeople"] & right_item["namedPeople"]:
        return delta_seconds <= 3600.0
    if left_item["burstGroupId"] and left_item["burstGroupId"] == right_item["burstGroupId"]:
        return True
    return delta_seconds <= 15.0


def is_display_sibling_pair(
    left_item: Dict[str, Any],
    right_item: Dict[str, Any],
    *,
    gray_similarity: float,
    hist_similarity: float,
) -> bool:
    delta_seconds = abs(left_item["timestamp"] - right_item["timestamp"])
    named_overlap = left_item["namedPeople"] & right_item["namedPeople"]
    same_burst = (
        left_item["burstGroupId"]
        and left_item["burstGroupId"] == right_item["burstGroupId"]
    )
    same_orientation = orientation_matches(left_item, right_item)

    if gray_similarity >= GLOBAL_EXACT_GRAY_THRESHOLD and hist_similarity >= GLOBAL_EXACT_HIST_THRESHOLD:
        return True

    if not same_orientation:
        if same_burst and delta_seconds <= 30.0 and gray_similarity >= 0.92 and hist_similarity >= 0.95:
            return True
        if same_burst and delta_seconds <= 180.0 and gray_similarity >= 0.90 and hist_similarity >= 0.97:
            return True
        if named_overlap and delta_seconds <= 60.0 and gray_similarity >= 0.94 and hist_similarity >= 0.94:
            return True
        if delta_seconds <= 8.0 and gray_similarity >= 0.96 and hist_similarity >= 0.96:
            return True

    if named_overlap and delta_seconds <= 12.0 and gray_similarity >= 0.78 and hist_similarity >= 0.97:
        return True

    if named_overlap and delta_seconds <= 60.0 and gray_similarity >= 0.86 and hist_similarity >= 0.985:
        return True

    if named_overlap and delta_seconds <= 3600.0 and gray_similarity >= 0.91 and hist_similarity >= 0.985:
        return True

    if same_burst and delta_seconds <= 30.0 and gray_similarity >= 0.91 and hist_similarity >= 0.89:
        return True

    if same_burst and delta_seconds <= 180.0 and gray_similarity >= 0.93 and hist_similarity >= 0.96:
        return True

    return False


def has_people_signal(row: sqlite3.Row) -> bool:
    tags = set(json.loads(row["chinese_tags_json"] or "[]"))
    if tags & {"合照", "儿童", "宝宝", "人像"}:
        return True
    labels = json.loads(row["vision_labels_json"] or "[]")
    people_labels = {"people", "person", "adult", "child", "baby"}
    return any(label.get("identifier") in people_labels for label in labels)


def timestamp_seconds(value: str | None) -> float:
    if not value:
        return 0.0
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def media_name(row: sqlite3.Row) -> str:
    suffix = Path(row["thumbnail_cache_path"]).suffix or ".webp"
    return f"{row['asset_id']}{suffix}"


def dedupe_name(row: sqlite3.Row) -> str:
    return media_name(row)


def open_url(asset_id: str) -> str:
    return f"{IMMICH_BASE_URL}/photos/{asset_id}"


def esthetic(scores: Dict[str, float], asset_id: str) -> float:
    return float(scores.get(asset_id, 0.0))


def is_protected_video(
    row: sqlite3.Row,
    *,
    named_people_cache: Dict[str, Dict[str, Any]],
    face_scorer: VisionScorer,
    face_analysis_cache: Dict[str, Dict[str, float]],
) -> bool:
    path = (row["original_path"] or "").lower()
    file_name = (row["original_file_name"] or "").lower()
    if is_screen_recording_video(row):
        return False
    if named_people_info_for_row(row, named_people_cache=named_people_cache)["count"] > 0:
        return True
    if has_family_people_tag(row) and has_detected_or_scored_face(
        row,
        face_scorer=face_scorer,
        face_analysis_cache=face_analysis_cache,
    ):
        return True
    duration_seconds = duration_to_seconds(row["duration"])
    return (
        "/vlog/" in path
        or (file_name.endswith(".mp4") and len(file_name) >= 8 and file_name[:8].isdigit())
        or duration_seconds >= 180
        or (duration_seconds >= 45 and has_story_tag(row))
    )


def is_screen_recording_video(row: sqlite3.Row) -> bool:
    file_name = (row["original_file_name"] or "").lower()
    path = (row["original_path"] or "").lower()
    if file_name.startswith("screenrecording_") or "/screenrecording_" in path:
        return True
    tags = set(json.loads(row["chinese_tags_json"] or "[]"))
    return bool("屏幕截图" in tags and "screenrecording" in file_name)


def has_story_tag(row: sqlite3.Row) -> bool:
    tags = set(json.loads(row["chinese_tags_json"] or "[]"))
    return bool(
        tags
        & {
            "合照",
            "儿童",
            "宝宝",
            "宠物",
            "猫",
            "狗",
            "海边",
            "天空",
            "户外",
            "街道",
            "建筑",
            "花",
            "食物",
        }
    )


def has_family_people_tag(row: sqlite3.Row) -> bool:
    tags = set(json.loads(row["chinese_tags_json"] or "[]"))
    return bool(tags & {"合照", "儿童", "宝宝"})


def has_detected_face_metadata(row: sqlite3.Row) -> bool:
    metadata = json.loads(row["metadata_json"] or "{}")
    for person in metadata.get("people") or []:
        if person.get("isHidden"):
            continue
        if person.get("faces"):
            return True
    return bool(metadata.get("unassignedFaces"))


def has_detected_or_scored_face(
    row: sqlite3.Row,
    *,
    face_scorer: VisionScorer,
    face_analysis_cache: Dict[str, Dict[str, float]],
) -> bool:
    if has_detected_face_metadata(row):
        return True
    return (
        face_info_for_row(
            row,
            face_scorer=face_scorer,
            face_analysis_cache=face_analysis_cache,
        )["faceCount"]
        > 0
    )


def duration_to_seconds(duration: str | None) -> float:
    if not duration:
        return 0.0
    parts = duration.split(":")
    if len(parts) != 3:
        return 0.0
    return (float(parts[0]) * 3600.0) + (float(parts[1]) * 60.0) + float(parts[2])


def reason_label(reason: str) -> str:
    if reason.startswith("similar_scene_to_display:"):
        return "与已入选精选图场景过近，展示只保留一张"
    labels = {
        "top_30_percent_unique_images": "去重后唯一图中的前 30%",
        "mid_band_unique_images": "去重后唯一图中的中段内容",
        "low_band_unique_images": "去重后唯一图中的低段内容",
    }
    return labels.get(reason, reason)


if __name__ == "__main__":
    main()
