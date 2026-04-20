from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import typer
import uvicorn

from . import db
from .config import Settings
from .immich import ImmichClient
from .preview import PreviewApp
from .scoring import VisionScorer, apply_burst_dedup, run_scoring
from .sync import apply_archive, sync_trial_albums, sync_trial_tags

app = typer.Typer(help="Local AI scoring and preview workflow for Immich.")


@contextmanager
def app_context():
    settings = Settings.load()
    conn = db.connect(settings.db_path)
    db.init_db(conn)
    client = ImmichClient(settings.immich_base_url, settings.immich_api_key)
    try:
        yield settings, conn, client
    finally:
        client.close()
        conn.close()


@app.command()
def discover(limit: int = typer.Option(1000, min=1), page_size: int = typer.Option(500, min=1, max=1000)):
    """Pull asset metadata from Immich into the local SQLite index."""
    with app_context() as (_, conn, client):
        imported = 0
        page = None
        while imported < limit:
            remaining = min(page_size, limit - imported)
            payload = client.search_metadata(page=page, size=remaining, with_exif=True)
            items = payload["assets"]["items"]
            if not items:
                break
            db.upsert_assets(conn, items)
            imported += len(items)
            page = payload["assets"].get("nextPage")
            if not page:
                break
        typer.echo(f"Imported {imported} assets into {conn.execute('SELECT COUNT(*) FROM assets').fetchone()[0]} local rows.")


@app.command()
def score(limit: int = typer.Option(1000, min=1)):
    """Download thumbnails, score them locally, and persist embeddings and tags."""
    with app_context() as (settings, conn, client):
        rows = db.unscored_assets(conn, limit)
        if not rows:
            typer.echo("No unscored assets found.")
            return
        project_root = settings.db_path.parent
        scorer = VisionScorer(
            settings.cache_dir,
            helper_source=project_root / "tools" / "vision_probe.swift",
            helper_binary=settings.preview_dir / "vision_probe",
        )
        processed = run_scoring(conn, scorer, client, rows)
        typer.echo(f"Scored {processed} assets.")


@app.command("sync-trial")
def sync_trial():
    """Create trial albums inside Immich for previewing before archive writeback."""
    with app_context() as (_, conn, client):
        stats = sync_trial_albums(conn, client)
        typer.echo(f"Synced trial albums: {stats}")


@app.command("dedupe")
def dedupe():
    """Collapse burst-like duplicate photos so only the best image remains kept."""
    with app_context() as (settings, conn, _client):
        project_root = settings.db_path.parent
        scorer = VisionScorer(
            settings.cache_dir,
            helper_source=project_root / "tools" / "vision_probe.swift",
            helper_binary=settings.preview_dir / "vision_probe",
        )
        db.normalize_scores(conn, db.all_scored_assets(conn))
        stats = apply_burst_dedup(conn, db.all_scored_assets(conn), scorer=scorer)
        typer.echo(f"Applied burst dedupe: {stats}")


@app.command("sync-tags")
def sync_tags(limit: int = typer.Option(300, min=1)):
    """Write grade tags and Chinese tags back into Immich for the trial set."""
    with app_context() as (_, conn, client):
        stats = sync_trial_tags(conn, client, limit=limit)
        typer.echo(f"Synced tags: {stats}")


@app.command("apply-archive")
def apply_archive_command(
    threshold: float = typer.Option(0.8, min=0.0, max=1.0),
    limit: Optional[int] = typer.Option(default=None, min=1),
):
    """Apply archive/timeline visibility by percentile threshold."""
    with app_context() as (_, conn, client):
        stats = apply_archive(conn, client, threshold=threshold, limit=limit)
        typer.echo(f"Applied archive visibility: {stats}")


@app.command()
def preview(host: str = "127.0.0.1", port: int = 8765):
    """Launch the local preview UI."""
    with app_context() as (settings, conn, client):
        preview_app = PreviewApp(settings, client, conn).build()
        uvicorn.run(preview_app, host=host, port=port, log_level="info")
