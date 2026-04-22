from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
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
PROGRESS_PATH = RESULTS_DIR / "progress.json"
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
IMMICH_BASE_URL = os.environ.get("IMMICH_BASE_URL", "http://127.0.0.1:2283").rstrip("/")
NEGATIVE_TAGS = {"文档", "屏幕截图", "收据", "白板", "表格", "演示文稿", "聊天记录"}
TOP_PICK_PREVIEW_LIMIT = 240
DUPLICATE_GROUP_PREVIEW_LIMIT = 30
NEGATIVE_PREVIEW_LIMIT = 72
REVIEW_PREVIEW_LIMIT = 96
LOW_PRIORITY_PREVIEW_LIMIT = 96
VIDEO_PREVIEW_LIMIT = 80
GLOBAL_EXACT_GRAY_THRESHOLD = 0.995
GLOBAL_EXACT_HIST_THRESHOLD = 0.995
AESTHETIC_BATCH_SIZE = 16
AESTHETIC_CACHE_FLUSH_EVERY = 256
FACE_CACHE_FLUSH_EVERY = 128
DISPLAY_TIME_BUCKET_SECONDS = 60
DISPLAY_SHORT_WINDOW_SECONDS = 15
DISPLAY_NAMED_WINDOW_SECONDS = 3600
DEFAULT_AESTHETIC_EXTRA_CANDIDATES = 12_000
DEFAULT_AESTHETIC_MIN_CANDIDATES = 12_000
DEFAULT_AESTHETIC_MAX_CANDIDATES = 0


@dataclass(slots=True)
class HybridConfig:
    image_limit: int | None = None
    video_limit: int | None = None
    aesthetic_extra_candidates: int = DEFAULT_AESTHETIC_EXTRA_CANDIDATES
    aesthetic_min_candidates: int = DEFAULT_AESTHETIC_MIN_CANDIDATES
    aesthetic_max_candidates: int = DEFAULT_AESTHETIC_MAX_CANDIDATES
    stage_media: bool = True
    export_system_buffer: bool = True


def parse_args() -> HybridConfig:
    parser = argparse.ArgumentParser(
        description="Generate the hybrid Immich preview report from nasai.db.",
    )
    parser.add_argument("--image-limit", type=int, default=None)
    parser.add_argument("--video-limit", type=int, default=None)
    parser.add_argument(
        "--aesthetic-extra-candidates",
        type=int,
        default=DEFAULT_AESTHETIC_EXTRA_CANDIDATES,
    )
    parser.add_argument(
        "--aesthetic-min-candidates",
        type=int,
        default=DEFAULT_AESTHETIC_MIN_CANDIDATES,
    )
    parser.add_argument(
        "--aesthetic-max-candidates",
        type=int,
        default=DEFAULT_AESTHETIC_MAX_CANDIDATES,
        help="Maximum number of images to run the aesthetic model on. Use 0 for no cap.",
    )
    parser.add_argument(
        "--no-stage-media",
        dest="stage_media",
        action="store_false",
        help="Skip staging preview thumbnails into benchmark/results/hybrid/images and videos.",
    )
    parser.add_argument(
        "--no-export-system-buffer",
        dest="export_system_buffer",
        action="store_false",
        help="Skip exporting the full system buffer thumbnail directories.",
    )
    args = parser.parse_args()
    return HybridConfig(
        image_limit=args.image_limit,
        video_limit=args.video_limit,
        aesthetic_extra_candidates=args.aesthetic_extra_candidates,
        aesthetic_min_candidates=args.aesthetic_min_candidates,
        aesthetic_max_candidates=args.aesthetic_max_candidates,
        stage_media=args.stage_media,
        export_system_buffer=args.export_system_buffer,
    )


def main(config: HybridConfig | None = None) -> None:
    config = config or parse_args()
    prepare_results_dirs(export_system_buffer=config.export_system_buffer)
    write_progress(
        status="running",
        stage="loading_assets",
        message="正在从 nasai.db 读取已评分资产。",
        config=progress_config_dict(config),
    )

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
        if config.image_limit is not None:
            image_rows = image_rows[: config.image_limit]
        if config.video_limit is not None:
            video_rows = video_rows[: config.video_limit]

        named_people_cache: Dict[str, Dict[str, Any]] = {}
        duplicate_index = build_duplicate_index(image_rows)
        aesthetic_candidate_rows, aesthetic_candidate_summary = select_aesthetic_candidate_rows(
            image_rows,
            duplicate_index,
            named_people_cache=named_people_cache,
            config=config,
        )
        write_progress(
            status="running",
            stage="aesthetic",
            message="正在补算高价值候选图的审美分。",
            counts={
                "images": len(image_rows),
                "videos": len(video_rows),
                "aestheticCandidates": len(aesthetic_candidate_rows),
                **aesthetic_candidate_summary,
            },
            config=progress_config_dict(config),
        )

        aesthetic_scores, aesthetic_stats = run_aesthetic(
            aesthetic_candidate_rows,
            progress_context={
                "images": len(image_rows),
                "videos": len(video_rows),
                "aestheticCandidates": len(aesthetic_candidate_rows),
                **aesthetic_candidate_summary,
            },
        )
        face_scorer = build_face_scorer()
        face_analysis_cache = load_face_analysis_cache()
        payload = build_payload(
            image_rows,
            video_rows,
            aesthetic_scores,
            duplicate_index,
            face_scorer=face_scorer,
            face_analysis_cache=face_analysis_cache,
            named_people_cache=named_people_cache,
            aesthetic_stats=aesthetic_stats,
            config=config,
        )
        save_face_analysis_cache(face_analysis_cache)

        write_progress(
            status="running",
            stage="rendering",
            message="正在生成网页和动作清单。",
            counts=payload["summary"],
            config=progress_config_dict(config),
        )
        if config.stage_media:
            stage_sections(payload)
        if config.export_system_buffer:
            export_system_buffer(payload)
        REPORT_PATH.write_text(Template(TEMPLATE_PATH.read_text()).render(**payload))
        FILTERED_REPORT_PATH.write_text(
            Template(FILTERED_TEMPLATE_PATH.read_text()).render(**payload)
        )
        ACTION_PATH.write_text(
            json.dumps(payload["actions"], ensure_ascii=False, indent=2)
        )
        write_progress(
            status="complete",
            stage="done",
            message=f"Hybrid 报告已生成: {REPORT_PATH}",
            counts=payload["summary"],
            config=progress_config_dict(config),
        )
        print(f"Wrote hybrid report to {REPORT_PATH}")
    finally:
        conn.close()


def prepare_results_dirs(*, export_system_buffer: bool) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for file_path in [REPORT_PATH, FILTERED_REPORT_PATH, ACTION_PATH]:
        if file_path.exists():
            file_path.unlink()
    cleanup_dirs = [IMAGES_DIR, VIDEOS_DIR, DEDUPE_DIR]
    if export_system_buffer:
        cleanup_dirs.append(SYSTEM_BUFFER_DIR)
    for directory in cleanup_dirs:
        if directory.exists():
            shutil.rmtree(directory)
    create_dirs = [IMAGES_DIR, VIDEOS_DIR, DEDUPE_DIR]
    if export_system_buffer:
        create_dirs.extend([SYSTEM_BUFFER_IMAGES_DIR, SYSTEM_BUFFER_VIDEOS_DIR])
    for directory in create_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    write_in_progress_placeholder()


def write_in_progress_placeholder() -> None:
    html = """<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>NASAI Hybrid Running</title>
    <style>
      body {
        margin: 0;
        font-family: "PingFang SC", "Noto Sans SC", sans-serif;
        background: #f4f1ea;
        color: #171717;
      }
      main {
        max-width: 880px;
        margin: 48px auto;
        padding: 0 20px;
      }
      .panel {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(23, 23, 23, 0.08);
        border-radius: 28px;
        padding: 28px;
        box-shadow: 0 20px 60px rgba(30, 24, 17, 0.08);
      }
      a { color: #0d5b4d; text-decoration: none; }
      code {
        background: rgba(13, 91, 77, 0.08);
        border-radius: 10px;
        padding: 2px 8px;
      }
    </style>
  </head>
  <body>
    <main>
      <section class="panel">
        <h1>Hybrid 正在生成</h1>
        <p>全量报告尚未完成。请查看 <code>benchmark/results/hybrid/progress.json</code> 获取实时阶段和数量，完成后这里会被正式结果页覆盖。</p>
      </section>
    </main>
  </body>
</html>
"""
    REPORT_PATH.write_text(html)
    FILTERED_REPORT_PATH.write_text(html)


def has_thumb(row: sqlite3.Row) -> bool:
    thumb = row["thumbnail_cache_path"]
    return bool(thumb) and Path(thumb).exists()


def run_aesthetic(
    rows: List[sqlite3.Row],
    *,
    progress_context: Dict[str, Any] | None = None,
) -> tuple[Dict[str, float], Dict[str, int]]:
    cached = load_json_dict(AESTHETIC_CACHE_PATH)
    asset_ids = {row["asset_id"] for row in rows}
    scores: Dict[str, float] = {
        asset_id: float(score)
        for asset_id, score in cached.items()
        if asset_id in asset_ids
    }
    cached_hits = len(scores)
    pending_rows = [row for row in rows if row["asset_id"] not in scores]
    if not pending_rows:
        return scores, {
            "aestheticCandidates": len(rows),
            "aestheticCached": cached_hits,
            "aestheticComputed": 0,
            "aestheticCacheSize": len(cached),
        }

    model, processor = convert_v2_5_from_siglip(local_files_only=True)
    model.eval()

    batch_rows: List[sqlite3.Row] = []
    new_scores = 0

    def flush() -> None:
        nonlocal new_scores
        if not batch_rows:
            return
        images = [Image.open(row["thumbnail_cache_path"]).convert("RGB") for row in batch_rows]
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits.squeeze(-1).tolist()
        if isinstance(logits, float):
            logits = [logits]
        for row, score in zip(batch_rows, logits):
            asset_id = row["asset_id"]
            value = float(score)
            scores[asset_id] = value
            cached[asset_id] = value
            new_scores += 1
        if new_scores and (
            new_scores % AESTHETIC_CACHE_FLUSH_EVERY == 0
            or new_scores == len(pending_rows)
        ):
            save_json_dict(AESTHETIC_CACHE_PATH, cached)
            write_progress(
                status="running",
                stage="aesthetic",
                message="正在补算高价值候选图的审美分。",
                counts={
                    **(progress_context or {}),
                    "aestheticCandidates": len(rows),
                    "aestheticCached": cached_hits,
                    "aestheticComputed": new_scores,
                    "aestheticRemaining": len(pending_rows) - new_scores,
                },
            )
        batch_rows.clear()

    for row in pending_rows:
        batch_rows.append(row)
        if len(batch_rows) >= AESTHETIC_BATCH_SIZE:
            flush()
    flush()
    save_json_dict(AESTHETIC_CACHE_PATH, cached)
    return scores, {
        "aestheticCandidates": len(rows),
        "aestheticCached": cached_hits,
        "aestheticComputed": new_scores,
        "aestheticCacheSize": len(cached),
    }

def load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_json_dict(path: Path, data: Dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp_path.replace(path)


def progress_config_dict(config: HybridConfig) -> Dict[str, Any]:
    return {
        "imageLimit": config.image_limit,
        "videoLimit": config.video_limit,
        "aestheticExtraCandidates": config.aesthetic_extra_candidates,
        "aestheticMinCandidates": config.aesthetic_min_candidates,
        "aestheticMaxCandidates": config.aesthetic_max_candidates,
        "stageMedia": config.stage_media,
        "exportSystemBuffer": config.export_system_buffer,
    }


def write_progress(
    *,
    status: str,
    stage: str,
    message: str,
    counts: Dict[str, Any] | None = None,
    config: Dict[str, Any] | None = None,
) -> None:
    payload = {
        "status": status,
        "stage": stage,
        "message": message,
        "updatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "counts": counts or {},
        "config": config or {},
        "reportPath": str(REPORT_PATH),
        "filteredReportPath": str(FILTERED_REPORT_PATH),
        "actionsPath": str(ACTION_PATH),
        "systemBufferPath": str(SYSTEM_BUFFER_DIR),
    }
    save_json_dict(PROGRESS_PATH, payload)


def build_face_scorer() -> VisionScorer:
    return VisionScorer(
        PROJECT_ROOT / "cache",
        helper_source=PROJECT_ROOT / "tools" / "vision_probe.swift",
        helper_binary=PROJECT_ROOT / "preview" / "vision_probe",
    )


def load_face_analysis_cache() -> Dict[str, Dict[str, float]]:
    cached = load_json_dict(FACE_ANALYSIS_CACHE_PATH)
    return {
        asset_id: {
            "faceCount": int(values.get("faceCount") or 0),
            "bestFaceCaptureQuality": float(values.get("bestFaceCaptureQuality") or -1.0),
            "largestFaceArea": float(values.get("largestFaceArea") or 0.0),
        }
        for asset_id, values in cached.items()
    }


def save_face_analysis_cache(cache: Dict[str, Dict[str, float]]) -> None:
    save_json_dict(FACE_ANALYSIS_CACHE_PATH, cache)


def build_duplicate_index(rows: List[sqlite3.Row]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        burst_group_id = row["burst_group_id"]
        if burst_group_id:
            grouped[str(burst_group_id)].append(row)

    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), min(member["asset_id"] for member in item[1])),
    )
    index: Dict[str, Dict[str, Any]] = {}
    for cluster_no, (burst_group_id, members) in enumerate(ordered_groups, start=1):
        cluster_id = f"dup-{cluster_no}"
        for member in members:
            index[member["asset_id"]] = {
                "clusterId": cluster_id,
                "clusterSize": len(members),
                "burstGroupId": burst_group_id,
                "burstRank": int(member["burst_rank"] or 0),
                "isBurstPick": bool(
                    member["is_burst_pick"] if member["is_burst_pick"] is not None else 1
                ),
            }
    return index


def select_aesthetic_candidate_rows(
    rows: List[sqlite3.Row],
    duplicate_index: Dict[str, Dict[str, Any]],
    *,
    named_people_cache: Dict[str, Dict[str, Any]],
    config: HybridConfig,
) -> tuple[List[sqlite3.Row], Dict[str, int]]:
    duplicate_demotions = {
        row["asset_id"]
        for row in rows
        if (cluster := duplicate_index.get(row["asset_id"])) and not cluster["isBurstPick"]
    }
    unique_rows = [
        row
        for row in rows
        if row["asset_id"] not in duplicate_demotions and not is_negative_like_image(row)
    ]
    if not unique_rows:
        return [], {
            "eligibleUniqueImages": 0,
            "namedPeopleCandidates": 0,
            "roughCandidates": 0,
        }

    rough_sorted = sorted(
        unique_rows,
        key=lambda row: rough_unique_sort_key(
            row,
            named_people_cache=named_people_cache,
        ),
        reverse=True,
    )
    display_keep_target = max(120, math.ceil(len(unique_rows) * 0.30))
    named_people_candidates = 0
    selected: Dict[str, sqlite3.Row] = {}
    for row in unique_rows:
        if named_people_info_for_row(row, named_people_cache=named_people_cache)["count"] <= 0:
            continue
        selected[row["asset_id"]] = row
        named_people_candidates += 1
    target_floor = max(
        named_people_candidates + config.aesthetic_extra_candidates,
        display_keep_target + config.aesthetic_extra_candidates,
        config.aesthetic_min_candidates,
    )
    if config.aesthetic_max_candidates and config.aesthetic_max_candidates > 0:
        rough_target = min(target_floor, config.aesthetic_max_candidates, len(unique_rows))
    else:
        rough_target = len(unique_rows)
    for row in rough_sorted:
        if len(selected) >= rough_target:
            break
        selected.setdefault(row["asset_id"], row)

    ordered_rows = sorted(
        selected.values(),
        key=lambda row: rough_unique_sort_key(
            row,
            named_people_cache=named_people_cache,
        ),
        reverse=True,
    )
    return ordered_rows, {
        "eligibleUniqueImages": len(unique_rows),
        "namedPeopleCandidates": named_people_candidates,
        "roughCandidates": len(ordered_rows),
    }


def build_payload(
    image_rows: List[sqlite3.Row],
    video_rows: List[sqlite3.Row],
    aesthetic_scores: Dict[str, float],
    duplicate_index: Dict[str, Dict[str, Any]],
    *,
    face_scorer: VisionScorer,
    face_analysis_cache: Dict[str, Dict[str, float]],
    named_people_cache: Dict[str, Dict[str, Any]],
    aesthetic_stats: Dict[str, int],
    config: HybridConfig,
) -> Dict[str, Any]:
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
        ordered = sorted(
            rows,
            key=lambda row: burst_duplicate_sort_key(
                row,
                duplicate_index=duplicate_index,
                aesthetic_scores=aesthetic_scores,
                named_people_cache=named_people_cache,
            ),
        )
        winner = ordered[0]
        duplicate_winners[cluster_id] = winner["asset_id"]
        for row in ordered[1:]:
            duplicate_demotions.add(row["asset_id"])
        if len(duplicate_groups_payload) < DUPLICATE_GROUP_PREVIEW_LIMIT:
            duplicate_groups_payload.append(
                {
                    "clusterId": cluster_id,
                    "size": len(ordered),
                    "winner": build_image_card(
                        winner,
                        aesthetic_scores=aesthetic_value_for_row(winner, aesthetic_scores),
                        proposed_action="保留最佳",
                        reason="沿用全量 burst 去重保留图",
                        duplicate_cluster=cluster_id,
                        named_people_cache=named_people_cache,
                        is_winner=True,
                        aesthetic_is_estimated=winner["asset_id"] not in aesthetic_scores,
                    ),
                    "alternates": [
                        build_image_card(
                            row,
                            aesthetic_scores=aesthetic_value_for_row(row, aesthetic_scores),
                            proposed_action="归并到重复",
                            reason=f"并入 {winner['original_file_name']}",
                            duplicate_cluster=cluster_id,
                            named_people_cache=named_people_cache,
                            is_winner=False,
                            aesthetic_is_estimated=row["asset_id"] not in aesthetic_scores,
                        )
                        for row in ordered[1:]
                    ],
                }
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
                    aesthetic_score=aesthetic_value_for_row(row, aesthetic_scores),
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
                    aesthetic_score=aesthetic_value_for_row(row, aesthetic_scores),
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
    review_keep_preview: List[Dict[str, Any]] = []
    review_keep_payload_all: List[Dict[str, Any]] = []
    low_priority_preview: List[Dict[str, Any]] = []
    selected_display_index = DisplaySimilarityIndex()
    display_feature_cache: Dict[str, Any] = {}
    display_keep_count = 0
    review_keep_count = 0
    archive_low_count = 0

    for row in unique_candidates:
        aesthetic_score = aesthetic_value_for_row(row, aesthetic_scores)
        aesthetic_is_estimated = row["asset_id"] not in aesthetic_scores
        similar_anchor = None
        if display_keep_count < display_keep_target:
            similar_anchor = similar_display_anchor(
                row,
                selected_index=selected_display_index,
                feature_cache=display_feature_cache,
            )

        if display_keep_count < display_keep_target and similar_anchor is None:
            action = "display_keep"
            reason = "top_30_percent_unique_images"
            label = "精选展示"
            display_keep_count += 1
            selected_display_index.add(display_similarity_item_for_row(row))
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
                    aesthetic_is_estimated=aesthetic_is_estimated,
                )
            )
        elif action == "review_keep":
            card = build_image_card(
                row,
                aesthetic_scores=aesthetic_score,
                proposed_action=label,
                reason=reason_label(reason),
                duplicate_cluster=duplicate_index.get(row["asset_id"], {}).get("clusterId"),
                named_people_cache=named_people_cache,
                is_winner=False,
                aesthetic_is_estimated=aesthetic_is_estimated,
            )
            review_keep_payload_all.append(card)
            if len(review_keep_preview) < REVIEW_PREVIEW_LIMIT:
                review_keep_preview.append(card)
        elif action == "archive_low" and len(low_priority_preview) < LOW_PRIORITY_PREVIEW_LIMIT:
            low_priority_preview.append(
                build_image_card(
                    row,
                    aesthetic_scores=aesthetic_score,
                    proposed_action=label,
                    reason=reason_label(reason),
                    duplicate_cluster=duplicate_index.get(row["asset_id"], {}).get("clusterId"),
                    named_people_cache=named_people_cache,
                    is_winner=False,
                    aesthetic_is_estimated=aesthetic_is_estimated,
                )
            )

    negatives.sort(
        key=lambda row: (
            "屏幕截图" not in json.loads(row["chinese_tags_json"] or "[]"),
            "文档" not in json.loads(row["chinese_tags_json"] or "[]"),
            row["file_created_at"] or "",
        )
    )
    negative_preview = [
        build_image_card(
            row,
            aesthetic_scores=aesthetic_value_for_row(row, aesthetic_scores),
            proposed_action="归档样本",
            reason="文档/截图优先下沉",
            duplicate_cluster=duplicate_index.get(row["asset_id"], {}).get("clusterId"),
            named_people_cache=named_people_cache,
            is_winner=False,
            aesthetic_is_estimated=row["asset_id"] not in aesthetic_scores,
        )
        for row in negatives[:NEGATIVE_PREVIEW_LIMIT]
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
        for row in other_videos[:VIDEO_PREVIEW_LIMIT]
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
        "duplicateGroups": len(duplicate_clusters),
        "duplicateDemotions": len(duplicate_demotions),
        "uniqueCandidates": len(unique_candidates),
        "displayKeeps": display_keep_count,
        "reviewKeeps": review_keep_count,
        "archiveLows": archive_low_count,
        "protectedVideos": len(protected_videos),
        "displayKeepVideos": len(other_videos),
        "reviewVideos": len(review_videos),
        "aestheticCandidates": aesthetic_stats["aestheticCandidates"],
        "aestheticCached": aesthetic_stats["aestheticCached"],
        "aestheticComputed": aesthetic_stats["aestheticComputed"],
    }
    return {
        "summary": summary,
        "filteredPageName": FILTERED_REPORT_PATH.name,
        "systemBufferReadmeName": "system_buffer/README.txt",
        "systemBufferManifestName": "system_buffer/manifest.json",
        "topPicks": top_picks,
        "reviewKeeps": review_keep_preview,
        "reviewKeepsAll": review_keep_payload_all,
        "lowPrioritySamples": low_priority_preview,
        "duplicateGroups": duplicate_groups_payload,
        "negativeSamples": negative_preview,
        "videoProtected": video_payload,
        "videoDisplay": video_display_payload_all,
        "videoReview": video_review_payload_all[:VIDEO_PREVIEW_LIMIT],
        "videoReviewAll": video_review_payload_all,
        "config": progress_config_dict(config),
        "actions": sorted(actions, key=lambda item: (item["action"], item["assetId"])),
    }


def stage_sections(payload: Dict[str, Any]) -> None:
    seen_images: set[str] = set()
    seen_videos: set[str] = set()

    for card in payload["topPicks"]:
        stage_card(card, seen_images, seen_videos)
    for card in payload["reviewKeeps"]:
        stage_card(card, seen_images, seen_videos)
    for card in payload["lowPrioritySamples"]:
        stage_card(card, seen_images, seen_videos)
    for group in payload["duplicateGroups"]:
        stage_card(group["winner"], seen_images, seen_videos)
        for card in group["alternates"]:
            stage_card(card, seen_images, seen_videos)
    for card in payload["negativeSamples"]:
        stage_card(card, seen_images, seen_videos)
    for card in payload["videoDisplay"]:
        stage_card(card, seen_images, seen_videos)
    for card in payload["videoReview"]:
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
        stage_media_file(source, target)


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
            stage_media_file(source, target_path)
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
                "aestheticDisplay": card.get("aestheticDisplay"),
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


def stage_media_file(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


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
    aesthetic_is_estimated: bool,
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
        "aestheticDisplay": (
            f"估算 {aesthetic_scores:.3f}" if aesthetic_is_estimated else f"{aesthetic_scores:.3f}"
        ),
        "aestheticEstimated": aesthetic_is_estimated,
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
        "aestheticDisplay": "-",
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


def aesthetic_value_for_row(
    row: sqlite3.Row,
    aesthetic_scores: Dict[str, float],
) -> float:
    cached = aesthetic_scores.get(row["asset_id"])
    if cached is not None:
        return float(cached)
    return 1.0 + (4.0 * float(row["percentile"] or 0.0))


def rough_unique_sort_key(
    row: sqlite3.Row,
    *,
    named_people_cache: Dict[str, Dict[str, Any]],
) -> tuple:
    named_people = named_people_info_for_row(row, named_people_cache=named_people_cache)
    return (
        int(named_people["count"] > 0),
        named_people["count"],
        named_people["favoriteCount"],
        named_people["largestFaceArea"],
        float(row["percentile"] or 0.0),
        float(row["raw_score"] or 0.0),
        image_area(row),
    )


def burst_duplicate_sort_key(
    row: sqlite3.Row,
    *,
    duplicate_index: Dict[str, Dict[str, Any]],
    aesthetic_scores: Dict[str, float],
    named_people_cache: Dict[str, Dict[str, Any]],
) -> tuple:
    cluster = duplicate_index.get(row["asset_id"]) or {}
    named_people = named_people_info_for_row(row, named_people_cache=named_people_cache)
    return (
        0 if cluster.get("isBurstPick") else 1,
        int(cluster.get("burstRank") or 10_000),
        -named_people["count"],
        -named_people["favoriteCount"],
        -named_people["largestFaceArea"],
        -aesthetic_value_for_row(row, aesthetic_scores),
        -float(row["raw_score"] or 0.0),
        -image_area(row),
        row["asset_id"],
    )


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
        aesthetic_value_for_row(row, aesthetic_scores),
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
    if len(face_analysis_cache) % FACE_CACHE_FLUSH_EVERY == 0:
        save_face_analysis_cache(face_analysis_cache)
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


class DisplaySimilarityIndex:
    def __init__(self) -> None:
        self._by_burst: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._by_bucket: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self._by_named_bucket: Dict[tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)

    def add(self, item: Dict[str, Any]) -> None:
        bucket = time_bucket(item["timestamp"])
        self._by_bucket[bucket].append(item)
        burst_group_id = item["burstGroupId"]
        if burst_group_id:
            self._by_burst[str(burst_group_id)].append(item)
        for name in item["namedPeople"]:
            self._by_named_bucket[(name, bucket)].append(item)

    def candidates_for(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates: Dict[str, Dict[str, Any]] = {}
        bucket = time_bucket(item["timestamp"])

        burst_group_id = item["burstGroupId"]
        if burst_group_id:
            for selected in self._by_burst.get(str(burst_group_id), []):
                candidates[selected["assetId"]] = selected

        short_window_buckets = max(
            1,
            math.ceil(DISPLAY_SHORT_WINDOW_SECONDS / DISPLAY_TIME_BUCKET_SECONDS),
        )
        for current_bucket in range(bucket - short_window_buckets, bucket + short_window_buckets + 1):
            for selected in self._by_bucket.get(current_bucket, []):
                candidates[selected["assetId"]] = selected

        named_window_buckets = max(
            1,
            math.ceil(DISPLAY_NAMED_WINDOW_SECONDS / DISPLAY_TIME_BUCKET_SECONDS),
        )
        for name in item["namedPeople"]:
            for current_bucket in range(
                bucket - named_window_buckets,
                bucket + named_window_buckets + 1,
            ):
                for selected in self._by_named_bucket.get((name, current_bucket), []):
                    candidates[selected["assetId"]] = selected
        return list(candidates.values())


def similar_display_anchor(
    row: sqlite3.Row,
    *,
    selected_index: DisplaySimilarityIndex,
    feature_cache: Dict[str, Any],
) -> sqlite3.Row | None:
    candidate = display_similarity_item_for_row(row)
    for selected in selected_index.candidates_for(candidate):
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
        return delta_seconds <= DISPLAY_NAMED_WINDOW_SECONDS
    if left_item["burstGroupId"] and left_item["burstGroupId"] == right_item["burstGroupId"]:
        return True
    return delta_seconds <= DISPLAY_SHORT_WINDOW_SECONDS


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


def time_bucket(timestamp: float) -> int:
    return int(timestamp // DISPLAY_TIME_BUCKET_SECONDS)


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
