from __future__ import annotations

import json
import subprocess
import threading
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import httpx
import numpy as np
from PIL import Image

from . import db
from .taxonomy import VISION_LABEL_MAP


class VisionScorer:
    def __init__(self, cache_dir: Path, *, helper_source: Path, helper_binary: Path) -> None:
        self.cache_dir = cache_dir
        self.helper_source = helper_source
        self.helper_binary = helper_binary
        self._worker: Optional[subprocess.Popen[str]] = None
        self._worker_lock = threading.Lock()
        self._ensure_helper()

    def _ensure_helper(self) -> None:
        self.helper_binary.parent.mkdir(parents=True, exist_ok=True)
        if self.helper_binary.exists() and self.helper_binary.stat().st_mtime >= self.helper_source.stat().st_mtime:
            return
        subprocess.run(
            ["swiftc", str(self.helper_source), "-o", str(self.helper_binary)],
            check=True,
            capture_output=True,
            text=True,
        )

    def close(self) -> None:
        with self._worker_lock:
            self._stop_worker_locked()

    def _stop_worker_locked(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        try:
            if worker.stdin:
                worker.stdin.close()
        except Exception:
            pass
        try:
            worker.terminate()
            worker.wait(timeout=1.0)
        except Exception:
            try:
                worker.kill()
                worker.wait(timeout=1.0)
            except Exception:
                pass

    def _worker_process(self) -> subprocess.Popen[str]:
        with self._worker_lock:
            if self._worker is not None and self._worker.poll() is None:
                return self._worker
            self._stop_worker_locked()
            self._worker = subprocess.Popen(
                [str(self.helper_binary), "--stdio"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            return self._worker

    def analyze_path(self, image_path: Path) -> Dict:
        for attempt in range(2):
            worker = self._worker_process()
            try:
                assert worker.stdin is not None
                assert worker.stdout is not None
                worker.stdin.write(f"{image_path}\n")
                worker.stdin.flush()
                response = worker.stdout.readline()
            except (BrokenPipeError, OSError, ValueError) as exc:
                with self._worker_lock:
                    self._stop_worker_locked()
                if attempt == 0:
                    continue
                raise RuntimeError(f"vision_probe worker IO failed: {exc}") from exc

            if not response:
                stderr_text = ""
                if worker.stderr is not None:
                    stderr_text = worker.stderr.read().strip()
                with self._worker_lock:
                    self._stop_worker_locked()
                if attempt == 0:
                    continue
                raise RuntimeError(
                    "vision_probe worker exited unexpectedly"
                    + (f": {stderr_text}" if stderr_text else "")
                )

            payload = json.loads(response)
            if payload.get("ok"):
                return payload["result"]
            raise RuntimeError(payload.get("error") or "vision_probe worker returned an empty error")

        raise RuntimeError("vision_probe worker failed after retry")

    def score_asset(
        self,
        asset: Dict,
        thumbnail_path: Path,
        thumbnail_bytes: bytes,
    ) -> Tuple[float, List[str], List[Dict], List[str], str]:
        if thumbnail_path.exists():
            image = Image.open(thumbnail_path)
        else:
            image = Image.open(BytesIO(thumbnail_bytes))
        image = image.convert("RGB")

        analysis = self.analyze_path(thumbnail_path)
        labels = analysis.get("labels", [])
        ocr_text = analysis.get("text", [])
        visual_score = self._visual_score(image)
        label_bonus = self._label_bonus(labels)
        label_penalty = self._label_penalty(labels, ocr_text, asset)
        raw_score = visual_score + label_bonus - label_penalty
        chinese_tags = self._derive_tags(labels, ocr_text, asset)
        search_text = self._build_search_text(asset, chinese_tags, labels, ocr_text)
        return float(raw_score), chinese_tags, labels, ocr_text, search_text

    def _visual_score(self, image: Image.Image) -> float:
        arr = np.asarray(image, dtype=np.float32) / 255.0
        gray = arr.mean(axis=2)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        sharpness = float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())
        saturation = float((arr.max(axis=2) - arr.min(axis=2)).mean())
        exposure = 1.0 - abs(brightness - 0.52)
        return (sharpness * 2.2) + (contrast * 1.0) + (saturation * 0.8) + exposure

    def _label_bonus(self, labels: List[Dict]) -> float:
        score = 0.0
        for label in labels:
            identifier = label["identifier"].lower()
            confidence = float(label["confidence"])
            if identifier in {"portrait", "person", "people", "child", "baby", "dog", "cat"}:
                score += confidence * 0.45
            if identifier in {"beach", "sunset", "sunrise", "mountain", "flower", "food"}:
                score += confidence * 0.35
        return score

    def _label_penalty(self, labels: List[Dict], ocr_text: List[str], asset: Dict) -> float:
        score = 0.0
        for label in labels:
            identifier = label["identifier"].lower()
            confidence = float(label["confidence"])
            if identifier in {
                "document",
                "printed_page",
                "receipt",
                "screenshot",
                "handwriting",
                "whiteboard",
                "menu",
            }:
                score += confidence * 1.4
        if ocr_text and sum(len(line) for line in ocr_text) > 18:
            score += 0.4
        exif_description = ((asset.get("exifInfo") or {}).get("description") or "").lower()
        if "screenshot" in exif_description:
            score += 0.6
        if (asset.get("type") or "").upper() == "VIDEO":
            score += 0.05
        return score

    def _derive_tags(self, labels: List[Dict], ocr_text: List[str], asset: Dict) -> List[str]:
        tags: List[str] = []
        for label in labels[:8]:
            translated = VISION_LABEL_MAP.get(label["identifier"].lower())
            if translated and translated not in tags:
                tags.append(translated)
        if ocr_text:
            if any("ppt" in line.lower() for line in ocr_text):
                tags.append("演示文稿")
            if any("wechat" in line.lower() or "微信" in line for line in ocr_text):
                tags.append("聊天记录")
            if any(char.isdigit() for char in "".join(ocr_text)):
                tags.append("文档")
        exif_description = ((asset.get("exifInfo") or {}).get("description") or "").lower()
        if "screenshot" in exif_description and "屏幕截图" not in tags:
            tags.insert(0, "屏幕截图")
        if (asset.get("type") or "").upper() == "VIDEO" and "视频封面" not in tags:
            tags.append("视频封面")
        return tags[:6]

    def _build_search_text(
        self,
        asset: Dict,
        chinese_tags: List[str],
        labels: List[Dict],
        ocr_text: List[str],
    ) -> str:
        english_labels = " ".join(label["identifier"] for label in labels[:10])
        chinese = " ".join(chinese_tags)
        ocr = " ".join(ocr_text)
        file_name = asset.get("originalFileName") or ""
        path = asset.get("originalPath") or ""
        exif_description = ((asset.get("exifInfo") or {}).get("description") or "")
        return " ".join([chinese, english_labels, ocr, file_name, path, exif_description]).strip()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def run_scoring(
    conn,
    scorer: VisionScorer,
    client,
    rows: Iterable,
    *,
    finalize: bool = True,
) -> Dict[str, int]:
    stats = {
        "processed": 0,
        "permanentFailures": 0,
        "transientFailures": 0,
    }
    for row in rows:
        asset_id = row["asset_id"]
        cache_path = scorer.cache_dir / f"{asset_id}.webp"
        attempted_at = datetime.now(timezone.utc).isoformat()
        db.record_score_attempt(conn, asset_id, attempted_at)
        try:
            if cache_path.exists():
                thumb_bytes = cache_path.read_bytes()
            else:
                thumb_bytes = client.thumbnail(asset_id)
                cache_path.write_bytes(thumb_bytes)
            asset_payload = json.loads(row["metadata_json"])
            raw_score, tags, labels, ocr_text, search_text = scorer.score_asset(
                asset_payload, cache_path, thumb_bytes
            )
            db.persist_score(
                conn,
                asset_id=asset_id,
                thumbnail_cache_path=str(cache_path),
                raw_score=raw_score,
                chinese_tags=tags,
                scored_at=attempted_at,
                vision_labels=labels,
                ocr_text=ocr_text,
                search_text=search_text,
            )
            stats["processed"] += 1
        except Exception as exc:
            permanent = _is_permanent_scoring_error(exc)
            db.mark_score_failure(
                conn,
                asset_id=asset_id,
                error_message=_error_message(exc),
                failed_at=attempted_at,
                permanent=permanent,
            )
            key = "permanentFailures" if permanent else "transientFailures"
            stats[key] += 1
            continue
    if finalize:
        finalize_scores(conn, scorer=scorer)
    return stats


def _is_permanent_scoring_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {404, 410}:
        return True
    message = str(exc).lower()
    return "too small in at least one dimension" in message


def _error_message(exc: Exception, limit: int = 500) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return message[:limit]


def finalize_scores(
    conn,
    *,
    scorer: "VisionScorer | None" = None,
    apply_dedupe: bool = True,
) -> Dict[str, int]:
    rows = db.all_scored_assets(conn)
    db.normalize_scores(conn, rows)
    if not apply_dedupe:
        return {"scored_assets": len(rows), "groups": 0, "demoted": 0}
    dedupe_stats = apply_burst_dedup(conn, rows, scorer=scorer)
    dedupe_stats["scored_assets"] = len(rows)
    return dedupe_stats


def apply_burst_dedup(
    conn,
    rows: Iterable,
    *,
    scorer: "VisionScorer | None" = None,
    window_seconds: int = 180,
    quick_window_seconds: int = 25,
    hist_threshold: float = 0.965,
    quick_hist_threshold: float = 0.95,
    gray_threshold: float = 0.92,
    quick_gray_threshold: float = 0.90,
) -> Dict[str, int]:
    rows = [
        row
        for row in rows
        if row["asset_type"] == "IMAGE"
        and row["thumbnail_cache_path"]
        and Path(row["thumbnail_cache_path"]).exists()
        and row["file_created_at"]
    ]
    conn.execute(
        """
        UPDATE assets
        SET burst_group_id = NULL,
            burst_group_size = NULL,
            burst_rank = NULL,
            is_burst_pick = 1
        WHERE raw_score IS NOT NULL
        """
    )
    if not rows:
        conn.commit()
        return {"groups": 0, "demoted": 0}

    items = []
    for row in rows:
        items.append(
            {
                "asset_id": row["asset_id"],
                "timestamp": _timestamp_seconds(row["file_created_at"]),
                "raw_score": float(row["raw_score"] or 0.0),
                "area": int(row["width"] or 0) * int(row["height"] or 0),
                "path": Path(row["thumbnail_cache_path"]),
            }
        )
    items.sort(key=lambda item: (item["timestamp"], item["asset_id"]))

    parent = list(range(len(items)))
    feature_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            delta = items[right]["timestamp"] - items[left]["timestamp"]
            if delta > window_seconds:
                break
            gray_similarity, hist_similarity = _image_similarity(
                items[left]["path"],
                items[right]["path"],
                feature_cache,
            )
            same_burst = (
                delta <= quick_window_seconds
                and hist_similarity >= quick_hist_threshold
                and gray_similarity >= quick_gray_threshold
            ) or (
                hist_similarity >= hist_threshold and gray_similarity >= gray_threshold
            )
            if same_burst:
                union(left, right)

    groups: Dict[int, List[dict]] = {}
    for index, item in enumerate(items):
        groups.setdefault(find(index), []).append(item)

    demoted = 0
    duplicate_groups = 0
    face_cache: Dict[str, Dict] = {}

    for members in groups.values():
        if len(members) == 1:
            continue
        duplicate_groups += 1
        member_rows = [
            conn.execute(
                "SELECT chinese_tags_json, vision_labels_json FROM assets WHERE asset_id = ?",
                (item["asset_id"],),
            ).fetchone()
            for item in members
        ]
        face_aware_group = any(
            _should_use_face_priority(row, item, scorer, face_cache)
            for row, item in zip(member_rows, members)
        )
        group_has_keep = any(
            conn.execute(
                "SELECT suggested_action FROM assets WHERE asset_id = ?",
                (item["asset_id"],),
            ).fetchone()["suggested_action"] == "keep"
            for item in members
        )
        group_should_preserve = group_has_keep or _should_preserve_scene(
            member_rows,
            members,
            scorer=scorer,
            face_cache=face_cache,
        )
        members.sort(
            key=lambda item: _burst_sort_key(
                item,
                scorer=scorer,
                face_aware_group=face_aware_group,
                face_cache=face_cache,
            )
        )
        group_id = f"burst:{members[0]['asset_id']}"
        for rank, item in enumerate(members, start=1):
            conn.execute(
                """
                UPDATE assets
                SET burst_group_id = ?,
                    burst_group_size = ?,
                    burst_rank = ?,
                    is_burst_pick = ?
                WHERE asset_id = ?
                """,
                (group_id, len(members), rank, 1 if rank == 1 else 0, item["asset_id"]),
            )
            if rank == 1 and group_should_preserve:
                conn.execute(
                    """
                    UPDATE assets
                    SET suggested_action = 'keep'
                    WHERE asset_id = ?
                    """,
                    (item["asset_id"],),
                )
            elif rank > 1:
                cursor = conn.execute(
                    """
                    UPDATE assets
                    SET suggested_action = 'archive'
                    WHERE asset_id = ? AND suggested_action = 'keep'
                    """,
                    (item["asset_id"],),
                )
                demoted += int(cursor.rowcount or 0)

    conn.commit()
    return {"groups": duplicate_groups, "demoted": demoted}


def _should_use_face_priority(row, item: dict, scorer: "VisionScorer | None", face_cache: Dict[str, Dict]) -> bool:
    tags = set(json.loads((row["chinese_tags_json"] if row else "[]") or "[]"))
    labels = {
        label.get("identifier", "").lower()
        for label in json.loads((row["vision_labels_json"] if row else "[]") or "[]")
    }
    if tags & {"人物", "人像", "合照", "儿童", "宝宝", "自拍"}:
        return True
    if labels & {"person", "people", "portrait", "child", "baby", "selfie"}:
        return True
    face_info = _face_info_for_item(item, scorer, face_cache)
    return face_info["face_count"] > 0


def _should_preserve_scene(
    member_rows,
    members: List[dict],
    *,
    scorer: "VisionScorer | None",
    face_cache: Dict[str, Dict],
) -> bool:
    negative_tags = {"屏幕截图", "文档", "收据", "白板", "演示文稿", "聊天记录", "表格"}
    positive_tags = {
        "人物",
        "人像",
        "合照",
        "儿童",
        "宝宝",
        "自拍",
        "宠物",
        "狗",
        "猫",
        "海边",
        "日落",
        "天空",
        "风景",
        "山",
        "树",
        "花",
        "雪",
        "夜景",
        "城市",
        "街道",
        "建筑",
        "旅行",
        "室内",
        "户外",
        "食物",
        "蛋糕",
        "汽车",
        "婚礼",
        "学校",
        "节日",
        "聚会",
        "玩具",
        "运动",
        "游泳",
        "海浪",
        "云",
    }

    candidates = []
    for row, item in zip(member_rows, members):
        tags = set(json.loads((row["chinese_tags_json"] if row else "[]") or "[]"))
        face_info = _face_info_for_item(item, scorer, face_cache)
        candidates.append(
            {
                "tags": tags,
                "raw_score": item["raw_score"],
                "face_count": face_info["face_count"],
            }
        )

    if not candidates:
        return False

    if all(candidate["tags"] and candidate["tags"] <= negative_tags for candidate in candidates):
        return False

    for candidate in candidates:
        if candidate["tags"] & positive_tags:
            return True
        if candidate["face_count"] > 0:
            return True
        if candidate["raw_score"] >= 1.75 and not (candidate["tags"] & negative_tags):
            return True

    return False


def _burst_sort_key(
    item: dict,
    *,
    scorer: "VisionScorer | None",
    face_aware_group: bool,
    face_cache: Dict[str, Dict],
) -> tuple:
    face_info = _face_info_for_item(item, scorer, face_cache) if face_aware_group else None
    if face_aware_group:
        return (
            -(face_info["best_face_quality"] if face_info else -1.0),
            -(face_info["largest_face_area"] if face_info else 0.0),
            -(face_info["face_count"] if face_info else 0),
            -item["raw_score"],
            -item["area"],
            item["timestamp"],
            item["asset_id"],
        )
    return (
        -item["raw_score"],
        -item["area"],
        item["timestamp"],
        item["asset_id"],
    )


def _face_info_for_item(item: dict, scorer: "VisionScorer | None", face_cache: Dict[str, Dict]) -> Dict[str, float]:
    key = item["asset_id"]
    cached = face_cache.get(key)
    if cached is not None:
        return cached

    result = {"face_count": 0, "best_face_quality": -1.0, "largest_face_area": 0.0}
    if scorer is not None:
        try:
            analysis = scorer.analyze_path(item["path"])
            result = {
                "face_count": int(analysis.get("faceCount") or 0),
                "best_face_quality": float(analysis.get("bestFaceCaptureQuality") or -1.0),
                "largest_face_area": float(analysis.get("largestFaceArea") or 0.0),
            }
        except Exception:
            result = result
    face_cache[key] = result
    return result


def _timestamp_seconds(timestamp: str) -> float:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()


def _image_similarity(
    left_path: Path,
    right_path: Path,
    cache: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, float]:
    left_gray, left_hist = _image_signature(left_path, cache)
    right_gray, right_hist = _image_signature(right_path, cache)
    gray_similarity = float(np.dot(left_gray, right_gray))
    hist_similarity = float(np.dot(left_hist, right_hist))
    return gray_similarity, hist_similarity


def _image_signature(path: Path, cache: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    key = str(path)
    if key in cache:
        return cache[key]

    image = Image.open(path).convert("RGB").resize((32, 32))
    array = np.asarray(image, dtype=np.float32) / 255.0

    gray = array.mean(axis=2).reshape(-1)
    gray /= np.linalg.norm(gray) + 1e-8

    hist_parts = []
    for channel_index in range(3):
        hist, _ = np.histogram(array[:, :, channel_index], bins=8, range=(0, 1), density=True)
        hist_parts.append(hist.astype(np.float32))
    hist_vector = np.concatenate(hist_parts)
    hist_vector /= np.linalg.norm(hist_vector) + 1e-8

    cache[key] = (gray, hist_vector)
    return cache[key]
