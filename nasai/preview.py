from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from . import db


class PreviewApp:
    def __init__(self, settings, client, conn) -> None:
        self.settings = settings
        self.client = client
        self.conn = conn
        self.templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    def build(self) -> FastAPI:
        app = FastAPI(title="NASAI Preview")

        @app.get("/", response_class=HTMLResponse)
        def index(request: Request):
            return self.templates.TemplateResponse(
                request=request,
                name="index.html",
                context={"summary": db.summary(self.conn)},
            )

        @app.get("/progress", response_class=HTMLResponse)
        def progress_page(request: Request):
            return self.templates.TemplateResponse(
                request=request,
                name="progress.html",
                context={"summary": db.summary(self.conn)},
            )

        @app.get("/api/summary")
        def summary():
            return db.summary(self.conn)

        @app.get("/api/progress")
        def progress():
            summary = db.summary(self.conn)
            db.record_progress_snapshot(self.conn, summary)
            snapshots = db.recent_progress_snapshots(self.conn, limit=180)
            return {
                "summary": summary,
                "snapshots": snapshots[-48:],
                "metrics": self._progress_metrics(summary, snapshots),
                "processes": self._pipeline_processes(),
                "storage": self._storage_stats(),
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            }

        @app.get("/api/assets")
        def assets(
            query: Optional[str] = None,
            asset_type: Optional[str] = Query(default=None, alias="type"),
            grade: Optional[str] = None,
            action: Optional[str] = None,
            limit: int = 200,
        ):
            rows = db.query_assets(
                self.conn,
                asset_type=asset_type,
                grade=grade,
                suggested_action=action,
                limit=None,
            )
            if query:
                rows = self._semantic_rank(rows, query)
            total = len(rows)
            display_rows = rows[:limit]
            return JSONResponse(
                content=[self._serialize_asset(row) for row in display_rows],
                headers={
                    "X-Total-Count": str(total),
                    "X-Displayed-Count": str(len(display_rows)),
                    "X-Result-Limit": str(limit),
                },
            )

        @app.get("/api/thumb/{asset_id}")
        def thumb(asset_id: str):
            row = self.conn.execute(
                "SELECT thumbnail_cache_path FROM assets WHERE asset_id = ?",
                (asset_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="asset not found")
            cache_path = row["thumbnail_cache_path"]
            if cache_path and Path(cache_path).exists():
                data = Path(cache_path).read_bytes()
            else:
                data = self.client.thumbnail(asset_id)
            return Response(content=data, media_type="image/webp")

        return app

    def _semantic_rank(self, rows, query: str) -> List:
        scored = []
        tokens = [token.lower() for token in query.replace("，", " ").replace(",", " ").split() if token.strip()]
        if not tokens:
            tokens = [query.lower()]
        for row in rows:
            corpus = " ".join(
                [
                    row["search_text"] or "",
                    row["chinese_tags_json"] or "",
                    row["original_file_name"] or "",
                    row["original_path"] or "",
                ]
            ).lower()
            similarity = 0.0
            for token in tokens:
                if token in corpus:
                    similarity += 10.0 + (len(token) / 10.0)
            scored.append((similarity, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for score, row in scored if score > 0]

    def _serialize_asset(self, row) -> Dict[str, Any]:
        tags = json.loads(row["chinese_tags_json"] or "[]")
        return {
            "assetId": row["asset_id"],
            "fileName": row["original_file_name"],
            "type": row["asset_type"],
            "visibility": row["visibility"],
            "score": row["raw_score"],
            "percentile": row["percentile"],
            "grade": row["grade"],
            "suggestedAction": row["suggested_action"],
            "tags": tags,
            "ocrText": json.loads(row["ocr_text_json"] or "[]"),
            "originalPath": row["original_path"],
            "openUrl": self.client.open_asset_url(row["asset_id"]),
            "burstGroupSize": row["burst_group_size"] or 1,
            "isBurstPick": bool(row["is_burst_pick"] if row["is_burst_pick"] is not None else 1),
        }

    def _pipeline_processes(self) -> Dict[str, Dict[str, Any]]:
        rows = self._process_rows()
        return {
            "discover": self._match_process(rows, "nasai discover"),
            "scoreQueue": self._match_process(rows, "nasai score-queue"),
            "preview": self._match_process(rows, "nasai preview"),
        }

    def _process_rows(self) -> List[Dict[str, Any]]:
        result = subprocess.run(
            ["ps", "-axo", "pid,%cpu,%mem,etime,command"],
            check=True,
            capture_output=True,
            text=True,
        )
        rows: List[Dict[str, Any]] = []
        for raw_line in result.stdout.splitlines()[1:]:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            pid, cpu, mem, elapsed, command = parts
            rows.append(
                {
                    "pid": int(pid),
                    "cpu": float(cpu),
                    "memory": float(mem),
                    "elapsed": elapsed,
                    "command": command,
                }
            )
        return rows

    def _match_process(self, rows: List[Dict[str, Any]], needle: str) -> Dict[str, Any]:
        candidates = [
            row
            for row in rows
            if needle in row["command"]
            and "ps -axo" not in row["command"]
            and "rg " not in row["command"]
        ]
        preferred = [row for row in candidates if "uv run " not in row["command"]]
        row = preferred[0] if preferred else (candidates[0] if candidates else None)
        if row is None:
            return {"running": False}
        return {
            "running": True,
            "pid": row["pid"],
            "cpu": row["cpu"],
            "memory": row["memory"],
            "elapsed": row["elapsed"],
            "command": row["command"],
        }

    def _storage_stats(self) -> Dict[str, Any]:
        db_files = [
            self.settings.db_path,
            self.settings.db_path.with_name(f"{self.settings.db_path.name}-wal"),
            self.settings.db_path.with_name(f"{self.settings.db_path.name}-shm"),
        ]
        db_size_bytes = sum(path.stat().st_size for path in db_files if path.exists())
        cached_thumb_rows = self.conn.execute(
            """
            SELECT COUNT(*) AS c
            FROM assets
            WHERE thumbnail_cache_path IS NOT NULL AND thumbnail_cache_path <> ''
            """
        ).fetchone()["c"]
        return {
            "dbSizeBytes": db_size_bytes,
            "cachedThumbRows": cached_thumb_rows,
            "dbPath": str(self.settings.db_path),
            "cacheDir": str(self.settings.cache_dir),
            "previewDir": str(self.settings.preview_dir),
        }

    def _progress_metrics(
        self,
        summary: Dict[str, Any],
        snapshots: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        indexed_per_min = 0.0
        scored_per_min = 0.0
        baseline = None
        latest = snapshots[-1] if snapshots else None
        if len(snapshots) >= 2 and latest is not None:
            latest_at = datetime.fromisoformat(latest["capturedAt"])
            for candidate in reversed(snapshots[:-1]):
                candidate_at = datetime.fromisoformat(candidate["capturedAt"])
                if (latest_at - candidate_at).total_seconds() >= 60:
                    baseline = candidate
                    break
            if baseline is None:
                baseline = snapshots[0]
            baseline_at = datetime.fromisoformat(baseline["capturedAt"])
            elapsed_seconds = max((latest_at - baseline_at).total_seconds(), 1.0)
            indexed_per_min = max(
                0.0,
                (float(latest["total"]) - float(baseline["total"])) / elapsed_seconds * 60.0,
            )
            scored_per_min = max(
                0.0,
                (float(latest["scored"]) - float(baseline["scored"])) / elapsed_seconds * 60.0,
            )
        unscored = int(summary["unscored"])
        eta_minutes = None
        if scored_per_min > 0:
            eta_minutes = unscored / scored_per_min
        return {
            "indexedPerMin": indexed_per_min,
            "scoredPerMin": scored_per_min,
            "etaMinutesForIndexedBacklog": eta_minutes,
            "snapshotCount": len(snapshots),
        }
