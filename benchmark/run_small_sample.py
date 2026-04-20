from __future__ import annotations

import json
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from imagededup.methods import CNN, PHash
from jinja2 import Template
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import ChineseCLIPModel, ChineseCLIPProcessor


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DB_PATH = PROJECT_ROOT / "nasai.db"
RESULTS_DIR = ROOT / "results"
IMAGES_DIR = RESULTS_DIR / "images"
VIDEOS_DIR = RESULTS_DIR / "videos"
TEMPLATE_PATH = ROOT / "templates" / "report.html"
REPORT_PATH = RESULTS_DIR / "report.html"
SEARCH_TOP_K = 6
ML_CLIP_IMAGE_MODEL = "clip-ViT-B-32"
ML_CLIP_TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
CHINESE_CLIP_MODEL = "OFA-Sys/chinese-clip-vit-base-patch16"
IMMICH_BASE_URL = "http://192.168.1.18:2283"
SEARCH_QUERIES = [
    {
        "query": "小朋友",
        "kind": "concept",
        "note": "语义测试: 对应儿童/宝宝，不要求图库里真的出现“小朋友”这几个字。",
        "relevance": ["儿童", "宝宝", "child"],
    },
    {
        "query": "多人合影",
        "kind": "concept",
        "note": "语义测试: 对应合照类内容。",
        "relevance": ["合照"],
    },
    {
        "query": "鲜花",
        "kind": "concept",
        "note": "语义测试: 对应花/flower/blossom。",
        "relevance": ["花", "flower", "blossom"],
    },
    {
        "query": "海滩",
        "kind": "concept",
        "note": "语义测试: 对应海边/海滩类内容。",
        "relevance": ["海边", "beach", "ocean", "water_body"],
    },
    {
        "query": "街景",
        "kind": "concept",
        "note": "语义测试: 对应街道/road/street/building。",
        "relevance": ["街道", "street", "road", "building"],
    },
    {
        "query": "饭菜",
        "kind": "concept",
        "note": "语义测试: 对应食物/food。",
        "relevance": ["食物", "food"],
    },
    {
        "query": "票据截图",
        "kind": "concept",
        "note": "语义测试: 对应收据/文档/屏幕截图。",
        "relevance": ["收据", "屏幕截图", "receipt", "screenshot", "document"],
    },
    {
        "query": "小猫",
        "kind": "concept",
        "note": "语义测试: 对应猫/cat。",
        "relevance": ["猫", "cat"],
    },
    {
        "query": "西湖",
        "kind": "named",
        "note": "地名测试: 更适合文件名/路径/描述类检索。",
        "relevance": ["西湖"],
    },
    {
        "query": "云南旅行",
        "kind": "named",
        "note": "地点测试: 更适合路径和文件名里的中文地名。",
        "relevance": ["云南"],
    },
]
SCHEME_LABELS = {
    "keyword": "关键词直搜",
    "multilingual_clip": "多语言 CLIP",
    "chinese_clip": "Chinese-CLIP",
}


def main() -> None:
    prepare_results_dirs()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        duplicate_groups = select_duplicate_groups(conn)
        standalone_images = select_standalone_images(conn)
        videos = select_videos(conn)
        search_rows = select_search_assets(conn)

        image_assets = flatten_groups(duplicate_groups) + standalone_images
        stage_assets(image_assets, asset_type="IMAGE")
        stage_assets(videos, asset_type="VIDEO")

        p_hash_clusters = run_phash(image_assets)
        cnn_clusters = run_cnn(image_assets)
        aesthetic_scores = run_aesthetic(image_assets)

        search_payload, search_summary = build_search_payload(search_rows)
        stage_search_results(search_payload)

        groups_payload = build_group_payload(
            duplicate_groups,
            p_hash_clusters=p_hash_clusters,
            cnn_clusters=cnn_clusters,
            aesthetic_scores=aesthetic_scores,
        )
        standalone_payload = build_asset_payload(
            standalone_images,
            p_hash_clusters=p_hash_clusters,
            cnn_clusters=cnn_clusters,
            aesthetic_scores=aesthetic_scores,
        )
        videos_payload = build_video_payload(videos)

        summary = {
            "duplicateGroupCount": len(groups_payload),
            "duplicateAssetCount": len(flatten_groups(duplicate_groups)),
            "standaloneCount": len(standalone_payload),
            "videoCount": len(videos_payload),
            "phashClusters": count_multi_asset_clusters(p_hash_clusters),
            "cnnClusters": count_multi_asset_clusters(cnn_clusters),
            "searchAssetCount": len(search_rows),
            "searchQueryCount": len(search_payload),
        }

        render_report(
            summary=summary,
            groups=groups_payload,
            standalone=standalone_payload,
            videos=videos_payload,
            search=search_payload,
            search_summary=search_summary,
        )
        print(f"Wrote benchmark report to {REPORT_PATH}")
    finally:
        conn.close()


def prepare_results_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for directory in [IMAGES_DIR, VIDEOS_DIR]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def select_duplicate_groups(conn: sqlite3.Connection) -> List[List[sqlite3.Row]]:
    groups = conn.execute(
        """
        WITH candidate_groups AS (
          SELECT
            burst_group_id,
            MAX(burst_group_size) AS group_size,
            MAX(file_created_at) AS last_at,
            SUM(
              CASE
                WHEN chinese_tags_json LIKE '%文档%'
                  OR chinese_tags_json LIKE '%屏幕截图%'
                  OR chinese_tags_json LIKE '%白板%'
                  OR chinese_tags_json LIKE '%收据%'
                THEN 1 ELSE 0
              END
            ) AS negative_count
          FROM assets
          WHERE asset_type = 'IMAGE'
            AND IFNULL(burst_group_size, 1) > 1
          GROUP BY burst_group_id
        )
        SELECT burst_group_id
        FROM candidate_groups
        WHERE negative_count < group_size
        ORDER BY group_size DESC, last_at DESC
        LIMIT 10
        """
    ).fetchall()

    payload: List[List[sqlite3.Row]] = []
    for group in groups:
        rows = conn.execute(
            """
            SELECT *
            FROM assets
            WHERE burst_group_id = ?
            ORDER BY file_created_at
            """,
            (group["burst_group_id"],),
        ).fetchall()
        payload.append(rows)
    return payload


def select_standalone_images(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    keep_rows = conn.execute(
        """
        SELECT *
        FROM assets
        WHERE asset_type = 'IMAGE'
          AND IFNULL(burst_group_size, 1) = 1
          AND suggested_action = 'keep'
        ORDER BY file_created_at DESC
        LIMIT 8
        """
    ).fetchall()
    archive_rows = conn.execute(
        """
        SELECT *
        FROM assets
        WHERE asset_type = 'IMAGE'
          AND IFNULL(burst_group_size, 1) = 1
          AND suggested_action = 'archive'
          AND chinese_tags_json NOT LIKE '%文档%'
          AND chinese_tags_json NOT LIKE '%屏幕截图%'
        ORDER BY file_created_at DESC
        LIMIT 8
        """
    ).fetchall()
    return list(keep_rows) + list(archive_rows)


def select_videos(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    vlog_rows = conn.execute(
        """
        SELECT *
        FROM assets
        WHERE asset_type = 'VIDEO'
          AND original_path LIKE '%/vlog/%'
        ORDER BY file_created_at DESC
        LIMIT 10
        """
    ).fetchall()
    return list(vlog_rows)


def select_search_assets(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT *
        FROM assets
        WHERE thumbnail_cache_path IS NOT NULL
          AND thumbnail_cache_path <> ''
        ORDER BY file_created_at DESC
        LIMIT 500
        """
    ).fetchall()
    return [row for row in rows if Path(row["thumbnail_cache_path"]).exists()]


def flatten_groups(groups: Iterable[List[sqlite3.Row]]) -> List[sqlite3.Row]:
    rows: List[sqlite3.Row] = []
    for group in groups:
        rows.extend(group)
    return rows


def stage_assets(rows: Iterable[sqlite3.Row], *, asset_type: str) -> None:
    for row in rows:
        stage_asset(row, asset_type=asset_type)


def stage_asset(row: sqlite3.Row, *, asset_type: str) -> None:
    source = Path(row["thumbnail_cache_path"])
    target = staged_media_path(row, asset_type=asset_type)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists() and not target.exists():
        shutil.copy2(source, target)


def staged_media_path(row: sqlite3.Row, *, asset_type: str) -> Path:
    suffix = Path(row["thumbnail_cache_path"]).suffix or ".webp"
    root = IMAGES_DIR if asset_type == "IMAGE" else VIDEOS_DIR
    return root / f"{row['asset_id']}{suffix}"


def staged_image_path(row: sqlite3.Row) -> Path:
    return staged_media_path(row, asset_type="IMAGE")


def staged_image_name(row: sqlite3.Row) -> str:
    return staged_image_path(row).name


def run_phash(rows: List[sqlite3.Row]) -> Dict[str, Dict[str, Any]]:
    image_dir = str(IMAGES_DIR)
    phash = PHash()
    encodings = phash.encode_images(image_dir=image_dir)
    duplicates = phash.find_duplicates(encoding_map=encodings, max_distance_threshold=8)
    return build_cluster_index(encodings.keys(), duplicates, prefix="phash")


def run_cnn(rows: List[sqlite3.Row]) -> Dict[str, Dict[str, Any]]:
    image_dir = str(IMAGES_DIR)
    cnn = CNN(verbose=False)
    encodings = cnn.encode_images(image_dir=image_dir)
    duplicates = cnn.find_duplicates(encoding_map=encodings, min_similarity_threshold=0.92)
    return build_cluster_index(encodings.keys(), duplicates, prefix="cnn")


def build_cluster_index(
    image_names: Iterable[str],
    duplicates: Dict[str, List[Any]],
    *,
    prefix: str,
) -> Dict[str, Dict[str, Any]]:
    image_names = list(image_names)
    parent = {name: name for name in image_names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, values in duplicates.items():
        for value in values:
            right = value[0] if isinstance(value, (tuple, list)) else value
            if right in parent:
                union(left, right)

    grouped: Dict[str, List[str]] = defaultdict(list)
    for name in image_names:
        grouped[find(name)].append(name)

    normalized: Dict[str, Dict[str, Any]] = {}
    ordered_groups = sorted(grouped.values(), key=lambda item: (-len(item), sorted(item)[0]))
    for index, names in enumerate(ordered_groups, start=1):
        cluster_id = f"{prefix}-{index}"
        for name in names:
            normalized[name] = {"clusterId": cluster_id, "clusterSize": len(names)}
    return normalized


def run_aesthetic(rows: List[sqlite3.Row]) -> Dict[str, float]:
    model, processor = convert_v2_5_from_siglip(local_files_only=True)
    model.eval()

    scores: Dict[str, float] = {}
    batch_paths: List[Path] = []
    batch_names: List[str] = []

    def flush() -> None:
        if not batch_paths:
            return
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits.squeeze(-1).tolist()
        if isinstance(logits, float):
            logits = [logits]
        for name, score in zip(batch_names, logits):
            scores[name] = float(score)
        batch_paths.clear()
        batch_names.clear()

    for row in rows:
        batch_paths.append(staged_image_path(row))
        batch_names.append(staged_image_name(row))
        if len(batch_paths) >= 8:
            flush()
    flush()
    return scores


def build_group_payload(
    groups: List[List[sqlite3.Row]],
    *,
    p_hash_clusters: Dict[str, Dict[str, Any]],
    cnn_clusters: Dict[str, Dict[str, Any]],
    aesthetic_scores: Dict[str, float],
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        assets = build_asset_payload(
            group,
            p_hash_clusters=p_hash_clusters,
            cnn_clusters=cnn_clusters,
            aesthetic_scores=aesthetic_scores,
        )
        aesthetic_pick = max(assets, key=lambda asset: asset["aestheticScore"])
        current_pick = next((asset for asset in assets if asset["burstRank"] == 1), assets[0])
        payload.append(
            {
                "title": f"连拍组 {index}",
                "size": len(assets),
                "currentPick": current_pick["fileName"],
                "aestheticPick": aesthetic_pick["fileName"],
                "phashClusterCount": len({asset["phashClusterId"] for asset in assets}),
                "cnnClusterCount": len({asset["cnnClusterId"] for asset in assets}),
                "assets": assets,
            }
        )
    return payload


def build_asset_payload(
    rows: List[sqlite3.Row],
    *,
    p_hash_clusters: Dict[str, Dict[str, Any]],
    cnn_clusters: Dict[str, Dict[str, Any]],
    aesthetic_scores: Dict[str, float],
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for row in rows:
        image_name = staged_image_name(row)
        p_hash = p_hash_clusters.get(image_name, {"clusterId": "-", "clusterSize": 1})
        cnn = cnn_clusters.get(image_name, {"clusterId": "-", "clusterSize": 1})
        payload.append(
            {
                "assetId": row["asset_id"],
                "fileName": row["original_file_name"],
                "thumbPath": f"images/{image_name}",
                "grade": row["grade"],
                "action": row["suggested_action"],
                "rawScore": float(row["raw_score"] or 0.0),
                "aestheticScore": round(float(aesthetic_scores.get(image_name, 0.0)), 3),
                "burstRank": int(row["burst_rank"] or 0),
                "burstGroupSize": int(row["burst_group_size"] or 1),
                "phashClusterId": p_hash["clusterId"],
                "phashClusterSize": int(p_hash["clusterSize"]),
                "cnnClusterId": cnn["clusterId"],
                "cnnClusterSize": int(cnn["clusterSize"]),
                "tags": json.loads(row["chinese_tags_json"] or "[]"),
                "openUrl": open_url(row),
            }
        )
    return payload


def build_video_payload(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for row in rows:
        image_name = staged_image_name(row)
        payload.append(
            {
                "assetId": row["asset_id"],
                "fileName": row["original_file_name"],
                "thumbPath": f"videos/{image_name}",
                "duration": row["duration"],
                "grade": row["grade"],
                "action": row["suggested_action"],
                "path": row["original_path"],
                "openUrl": open_url(row),
            }
        )
    return payload


def count_multi_asset_clusters(cluster_index: Dict[str, Dict[str, Any]]) -> int:
    return len({item["clusterId"] for item in cluster_index.values() if item["clusterSize"] > 1})


def build_search_payload(rows: List[sqlite3.Row]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    keyword_rankings = run_keyword_rankings(rows)
    multilingual_rankings = run_multilingual_clip_rankings(rows)
    chinese_clip_rankings = run_chinese_clip_rankings(rows)

    payload: List[Dict[str, Any]] = []
    scheme_hits: Dict[str, List[float]] = defaultdict(list)
    concept_hits: Dict[str, List[float]] = defaultdict(list)
    named_hits: Dict[str, List[float]] = defaultdict(list)

    for spec in SEARCH_QUERIES:
        relevance = search_relevance_terms(spec)
        relevant_rows = [row for row in rows if is_relevant(row, relevance)]
        schemes = []
        for scheme_key, rankings in [
            ("keyword", keyword_rankings),
            ("multilingual_clip", multilingual_rankings),
            ("chinese_clip", chinese_clip_rankings),
        ]:
            entries = rankings[spec["query"]][:SEARCH_TOP_K]
            hit_count = sum(1 for entry in entries if entry["relevant"])
            hit_rate = hit_count / SEARCH_TOP_K
            scheme_hits[scheme_key].append(hit_rate)
            if spec["kind"] == "concept":
                concept_hits[scheme_key].append(hit_rate)
            else:
                named_hits[scheme_key].append(hit_rate)
            schemes.append(
                {
                    "key": scheme_key,
                    "label": SCHEME_LABELS[scheme_key],
                    "hitCount": hit_count,
                    "hitRate": round(hit_rate, 3),
                    "results": [build_search_result_asset(entry) for entry in entries],
                }
            )
        payload.append(
            {
                "query": spec["query"],
                "kind": spec["kind"],
                "note": spec["note"],
                "relevantCount": len(relevant_rows),
                "schemes": schemes,
            }
        )

    summary = []
    for scheme_key, label in SCHEME_LABELS.items():
        summary.append(
            {
                "label": label,
                "overallHitRate": round(mean_or_zero(scheme_hits[scheme_key]), 3),
                "conceptHitRate": round(mean_or_zero(concept_hits[scheme_key]), 3),
                "namedHitRate": round(mean_or_zero(named_hits[scheme_key]), 3),
            }
        )
    return payload, summary


def run_keyword_rankings(rows: List[sqlite3.Row]) -> Dict[str, List[Dict[str, Any]]]:
    rankings: Dict[str, List[Dict[str, Any]]] = {}
    for spec in SEARCH_QUERIES:
        query = spec["query"].lower()
        scored = []
        for row in rows:
            corpus = row_corpus(row)
            score = 0.0
            if query in corpus:
                score += 100.0
            score += corpus.count(query) * 5.0
            if (row["original_file_name"] or "").lower().find(query) >= 0:
                score += 12.0
            if (row["original_path"] or "").lower().find(query) >= 0:
                score += 8.0
            if score > 0:
                scored.append(
                    {
                        "row": row,
                        "score": score,
                        "relevant": is_relevant(row, search_relevance_terms(spec)),
                    }
                )
        scored.sort(key=lambda item: item["score"], reverse=True)
        rankings[spec["query"]] = scored
    return rankings


def run_multilingual_clip_rankings(rows: List[sqlite3.Row]) -> Dict[str, List[Dict[str, Any]]]:
    device = torch_device()
    image_model = SentenceTransformer(ML_CLIP_IMAGE_MODEL, device=device)
    text_model = SentenceTransformer(ML_CLIP_TEXT_MODEL, device=device)

    images = load_search_images(rows)
    image_embeddings = image_model.encode(
        images,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    text_embeddings = text_model.encode(
        [spec["query"] for spec in SEARCH_QUERIES],
        batch_size=16,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    similarity = util.cos_sim(text_embeddings, image_embeddings).cpu().tolist()
    return build_similarity_rankings(rows, similarity)


def run_chinese_clip_rankings(rows: List[sqlite3.Row]) -> Dict[str, List[Dict[str, Any]]]:
    preferred_device = torch_device()
    processor = ChineseCLIPProcessor.from_pretrained(CHINESE_CLIP_MODEL)
    model = ChineseCLIPModel.from_pretrained(CHINESE_CLIP_MODEL)
    device = move_model(model, preferred_device)
    model.eval()

    image_features = []
    for start in range(0, len(rows), 16):
        batch_rows = rows[start:start + 16]
        batch_images = load_search_images(batch_rows)
        inputs = processor(images=batch_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            features = model.get_image_features(pixel_values=pixel_values)
        image_features.append(torch.nn.functional.normalize(features, dim=-1).cpu())
    image_matrix = torch.cat(image_features, dim=0)

    text_inputs = processor(
        text=[spec["query"] for spec in SEARCH_QUERIES],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    with torch.inference_mode():
        text_outputs = model.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            token_type_ids=text_inputs.get("token_type_ids"),
        )
        pooled = text_outputs.last_hidden_state[:, 0, :]
        text_features = model.text_projection(pooled)
    text_matrix = torch.nn.functional.normalize(text_features, dim=-1).cpu()
    similarity = (text_matrix @ image_matrix.T).tolist()
    return build_similarity_rankings(rows, similarity)


def build_similarity_rankings(
    rows: List[sqlite3.Row],
    similarity: List[List[float]],
) -> Dict[str, List[Dict[str, Any]]]:
    rankings: Dict[str, List[Dict[str, Any]]] = {}
    for query_index, spec in enumerate(SEARCH_QUERIES):
        scored = []
        for row, score in zip(rows, similarity[query_index]):
            scored.append(
                {
                    "row": row,
                    "score": float(score),
                    "relevant": is_relevant(row, search_relevance_terms(spec)),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        rankings[spec["query"]] = scored
    return rankings


def stage_search_results(search_payload: List[Dict[str, Any]]) -> None:
    seen: set[str] = set()
    for query in search_payload:
        for scheme in query["schemes"]:
            for result in scheme["results"]:
                asset_id = result["assetId"]
                if asset_id in seen:
                    continue
                seen.add(asset_id)
                source = Path(result["sourceThumbPath"])
                target = IMAGES_DIR / Path(result["thumbPath"]).name
                if result["type"] == "VIDEO":
                    target = VIDEOS_DIR / Path(result["thumbPath"]).name
                if source.exists() and not target.exists():
                    shutil.copy2(source, target)


def build_search_result_asset(entry: Dict[str, Any]) -> Dict[str, Any]:
    row = entry["row"]
    asset_type = row["asset_type"]
    source_thumb = Path(row["thumbnail_cache_path"])
    target_name = f"{row['asset_id']}{source_thumb.suffix or '.webp'}"
    thumb_root = "images" if asset_type == "IMAGE" else "videos"
    return {
        "assetId": row["asset_id"],
        "fileName": row["original_file_name"],
        "type": asset_type,
        "thumbPath": f"{thumb_root}/{target_name}",
        "sourceThumbPath": str(source_thumb),
        "grade": row["grade"],
        "action": row["suggested_action"],
        "score": round(float(entry["score"]), 4),
        "relevant": bool(entry["relevant"]),
        "tags": json.loads(row["chinese_tags_json"] or "[]"),
        "openUrl": open_url(row),
        "path": row["original_path"],
    }


def load_search_images(rows: List[sqlite3.Row]) -> List[Image.Image]:
    return [Image.open(row["thumbnail_cache_path"]).convert("RGB") for row in rows]


def row_corpus(row: sqlite3.Row) -> str:
    return " ".join(
        [
            row["search_text"] or "",
            row["original_file_name"] or "",
            row["original_path"] or "",
            row["exif_description"] or "",
        ]
    ).lower()


def search_relevance_terms(spec: Dict[str, Any]) -> List[str]:
    return [spec["query"], *spec["relevance"]]


def is_relevant(row: sqlite3.Row, keywords: List[str]) -> bool:
    corpus = row_corpus(row)
    return any(keyword.lower() in corpus for keyword in keywords)


def mean_or_zero(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def torch_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def move_model(model: torch.nn.Module, preferred_device: str) -> str:
    try:
        model.to(preferred_device)
        return preferred_device
    except Exception:
        model.to("cpu")
        return "cpu"


def open_url(row: sqlite3.Row) -> str:
    return f"{IMMICH_BASE_URL}/photos/{row['asset_id']}"


def render_report(
    *,
    summary: Dict[str, Any],
    groups: List[Dict[str, Any]],
    standalone: List[Dict[str, Any]],
    videos: List[Dict[str, Any]],
    search: List[Dict[str, Any]],
    search_summary: List[Dict[str, Any]],
) -> None:
    template = Template(TEMPLATE_PATH.read_text())
    html = template.render(
        summary=summary,
        groups=groups,
        standalone=standalone,
        videos=videos,
        search=search,
        search_summary=search_summary,
    )
    REPORT_PATH.write_text(html)


if __name__ == "__main__":
    main()
