from __future__ import annotations

import json
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

        @app.get("/api/summary")
        def summary():
            return db.summary(self.conn)

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
