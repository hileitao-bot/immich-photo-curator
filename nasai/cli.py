from __future__ import annotations

import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Any, Optional

import typer
import uvicorn

from . import db
from .config import Settings
from .immich import ImmichClient
from .preview import PreviewApp
from .scoring import VisionScorer, finalize_scores, run_scoring
from .sync import apply_archive, sync_hybrid_writeback, sync_trial_albums, sync_trial_tags

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


def helper_binary_path(settings: Settings):
    return settings.preview_dir / "vision_probe_persistent"


def build_scorer(settings: Settings) -> VisionScorer:
    project_root = settings.db_path.parent
    return VisionScorer(
        settings.cache_dir,
        helper_source=project_root / "tools" / "vision_probe.swift",
        helper_binary=helper_binary_path(settings),
    )


def drain_score_queue_once(
    settings: Settings,
    conn,
    client,
    *,
    batch_size: int,
    finalize_every: int,
    dedupe_when_drained: bool,
) -> dict[str, Any]:
    scorer = build_scorer(settings)
    try:
        batches_since_finalize = 0
        total_processed = 0
        total_permanent_failures = 0
        total_transient_failures = 0

        while True:
            rows = db.unscored_assets(conn, batch_size)
            if not rows:
                if batches_since_finalize:
                    finalize_scores(conn, apply_dedupe=False)
                break

            stats = run_scoring(conn, scorer, client, rows, finalize=False)
            total_processed += stats["processed"]
            total_permanent_failures += stats["permanentFailures"]
            total_transient_failures += stats["transientFailures"]
            batches_since_finalize += 1

            typer.echo(
                "Incremental score "
                f"batch={stats['processed']} "
                f"permanent_failures={stats['permanentFailures']} "
                f"transient_failures={stats['transientFailures']} "
                f"total_processed={total_processed}"
            )

            if batches_since_finalize >= finalize_every:
                finalize_scores(conn, apply_dedupe=False)
                batches_since_finalize = 0

        finalize_stats = finalize_scores(
            conn,
            scorer=scorer if dedupe_when_drained else None,
            apply_dedupe=dedupe_when_drained,
        )
        return {
            "processed": total_processed,
            "permanentFailures": total_permanent_failures,
            "transientFailures": total_transient_failures,
            "finalize": finalize_stats,
        }
    finally:
        scorer.close()


def discover_recent_assets(
    conn,
    client,
    *,
    page_size: int,
    max_pages: int,
    stop_after_known_pages: int,
) -> dict[str, int | str | None]:
    imported = 0
    new_assets = 0
    pages = 0
    known_streak = 0
    page = None

    while pages < max_pages:
        payload = client.search_metadata(page=page, size=page_size, with_exif=True)
        items = payload["assets"]["items"]
        if not items:
            break

        asset_ids = [item["id"] for item in items]
        placeholders = ",".join("?" for _ in asset_ids)
        existing_ids = {
            row["asset_id"]
            for row in conn.execute(
                f"SELECT asset_id FROM assets WHERE asset_id IN ({placeholders})",
                asset_ids,
            ).fetchall()
        }
        page_new_assets = sum(1 for asset_id in asset_ids if asset_id not in existing_ids)
        db.upsert_assets(conn, items)

        imported += len(items)
        new_assets += page_new_assets
        pages += 1
        known_streak = known_streak + 1 if page_new_assets == 0 else 0
        page = payload["assets"].get("nextPage")

        typer.echo(
            "Incremental discover "
            f"page={pages} imported={len(items)} new={page_new_assets} total_new={new_assets}"
        )

        if not page or known_streak >= stop_after_known_pages:
            break

    db.set_kv(conn, "last_incremental_discover_at", datetime.now(timezone.utc).isoformat())
    return {
        "pages": pages,
        "imported": imported,
        "newAssets": new_assets,
        "nextPage": page,
    }


def run_hybrid_refresh(
    project_root: Path,
    *,
    stage_media: bool,
    export_system_buffer: bool,
) -> None:
    command = [
        "uv",
        "run",
        "--extra",
        "benchmark",
        "python",
        "benchmark/run_hybrid_trial.py",
    ]
    if not stage_media:
        command.append("--no-stage-media")
    if not export_system_buffer:
        command.append("--no-export-system-buffer")
    subprocess.run(command, cwd=project_root, check=True)


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
        scorer = build_scorer(settings)
        try:
            stats = run_scoring(conn, scorer, client, rows)
            typer.echo(
                "Scored "
                f"{stats['processed']} assets "
                f"(permanent_failures={stats['permanentFailures']} transient_failures={stats['transientFailures']})."
            )
        finally:
            scorer.close()


@app.command("score-queue")
def score_queue(
    batch_size: int = typer.Option(1000, min=1),
    finalize_every: int = typer.Option(20, min=1),
    dedupe_when_drained: bool = typer.Option(
        True,
        "--dedupe-when-drained/--no-dedupe-when-drained",
        help="Only run burst dedupe after the queue drains, not during periodic finalize passes.",
    ),
    idle_sleep: int = typer.Option(30, min=1),
    once: bool = typer.Option(False, help="Exit after current backlog drains."),
):
    """Continuously score assets in batches and periodically recompute grades/dedupe."""
    with app_context() as (settings, conn, client):
        scorer = build_scorer(settings)
        try:
            batches_since_finalize = 0
            total_processed = 0
            needs_drain_dedupe = False
            while True:
                rows = db.unscored_assets(conn, batch_size)
                if not rows:
                    if batches_since_finalize:
                        stats = finalize_scores(conn, apply_dedupe=False)
                        typer.echo(f"Finalize complete (scores only): {stats}")
                        batches_since_finalize = 0
                        needs_drain_dedupe = dedupe_when_drained
                    if needs_drain_dedupe:
                        stats = finalize_scores(conn, scorer=scorer, apply_dedupe=True)
                        typer.echo(f"Drain dedupe complete: {stats}")
                        needs_drain_dedupe = False
                    if once:
                        typer.echo(f"Score queue drained after processing {total_processed} assets.")
                        return
                    typer.echo(f"No unscored assets found. Sleeping {idle_sleep}s.")
                    sleep(idle_sleep)
                    continue

                stats = run_scoring(conn, scorer, client, rows, finalize=False)
                processed = stats["processed"]
                total_processed += processed
                batches_since_finalize += 1
                needs_drain_dedupe = dedupe_when_drained
                total, scored, failed = conn.execute(
                    """
                    SELECT COUNT(*),
                           SUM(raw_score IS NOT NULL),
                           SUM(raw_score IS NULL AND score_failed_at IS NOT NULL)
                    FROM assets
                    """
                ).fetchone()
                typer.echo(
                    "Scored "
                    f"batch={processed} "
                    f"permanent_failures={stats['permanentFailures']} "
                    f"transient_failures={stats['transientFailures']} "
                    f"total_processed={total_processed} total={total} scored={scored} failed={failed}"
                )
                if batches_since_finalize >= finalize_every:
                    stats = finalize_scores(conn, apply_dedupe=False)
                    typer.echo(f"Finalize complete (scores only): {stats}")
                    batches_since_finalize = 0
        finally:
            scorer.close()


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
        scorer = build_scorer(settings)
        try:
            stats = finalize_scores(conn, scorer=scorer)
            typer.echo(f"Applied burst dedupe: {stats}")
        finally:
            scorer.close()


@app.command()
def finalize(dedupe: bool = typer.Option(True, "--dedupe/--no-dedupe")):
    """Recompute normalized scores and optionally refresh burst dedupe for all scored assets."""
    with app_context() as (settings, conn, _client):
        scorer = None
        if dedupe:
            scorer = build_scorer(settings)
        try:
            stats = finalize_scores(conn, scorer=scorer, apply_dedupe=dedupe)
            typer.echo(f"Finalize complete: {stats}")
        finally:
            if scorer is not None:
                scorer.close()


@app.command("incremental")
def incremental(
    discover_page_size: int = typer.Option(500, min=1, max=1000),
    discover_max_pages: int = typer.Option(20, min=1),
    discover_stop_after_known_pages: int = typer.Option(3, min=1),
    score_batch_size: int = typer.Option(1000, min=1),
    score_finalize_every: int = typer.Option(20, min=1),
    dedupe_when_drained: bool = typer.Option(
        True,
        "--dedupe-when-drained/--no-dedupe-when-drained",
    ),
    refresh_hybrid: bool = typer.Option(
        True,
        "--refresh-hybrid/--no-refresh-hybrid",
        help="Rebuild the full hybrid actions/report after incremental scoring.",
    ),
    sync_to_immich: bool = typer.Option(
        True,
        "--sync-to-immich/--no-sync-to-immich",
        help="Write the refreshed hybrid result back to Immich visibility and albums.",
    ),
    with_buffer_albums: bool = typer.Option(
        False,
        "--with-buffer-albums/--no-buffer-albums",
        help="Also create Immich albums for system buffer images and videos.",
    ),
    hybrid_stage_media: bool = typer.Option(
        True,
        "--hybrid-stage-media/--no-hybrid-stage-media",
        help="Refresh staged preview thumbnails while rebuilding the hybrid report.",
    ),
    hybrid_export_system_buffer: bool = typer.Option(
        True,
        "--hybrid-export-system-buffer/--no-hybrid-export-system-buffer",
        help="Refresh the local system buffer export while rebuilding the hybrid report.",
    ),
    sync_batch_size: int = typer.Option(1000, min=1, max=1000),
):
    """Run the daily incremental pipeline for newly added Immich assets."""
    with app_context() as (settings, conn, client):
        project_root = settings.db_path.parent
        discover_stats = discover_recent_assets(
            conn,
            client,
            page_size=discover_page_size,
            max_pages=discover_max_pages,
            stop_after_known_pages=discover_stop_after_known_pages,
        )

        pending_after_discover = db.summary(conn)["unscored"]
        if not discover_stats["newAssets"] and pending_after_discover == 0:
            typer.echo("Incremental pipeline found no new assets and no pending scores. Skipping.")
            return

        score_stats = drain_score_queue_once(
            settings,
            conn,
            client,
            batch_size=score_batch_size,
            finalize_every=score_finalize_every,
            dedupe_when_drained=dedupe_when_drained,
        )
        typer.echo(f"Incremental score drain complete: {score_stats}")

        actions_path = project_root / "benchmark" / "results" / "hybrid" / "actions.json"
        if refresh_hybrid:
            typer.echo("Refreshing hybrid report and action manifest.")
            run_hybrid_refresh(
                project_root,
                stage_media=hybrid_stage_media,
                export_system_buffer=hybrid_export_system_buffer,
            )
        elif sync_to_immich and not actions_path.exists():
            raise typer.BadParameter(
                "actions.json is missing; enable --refresh-hybrid or generate the hybrid report first."
            )

        if sync_to_immich:
            typer.echo("Writing refreshed hybrid actions back to Immich.")
            sync_stats = sync_hybrid_writeback(
                client,
                actions_path=actions_path,
                with_buffer_albums=with_buffer_albums,
                batch_size=sync_batch_size,
            )
            typer.echo(f"Incremental Immich sync complete: {sync_stats}")


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


@app.command("sync-hybrid")
def sync_hybrid(
    actions_path: Path = typer.Option(
        Path("benchmark/results/hybrid/actions.json"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Hybrid action manifest generated from benchmark/run_hybrid_trial.py.",
    ),
    with_buffer_albums: bool = typer.Option(
        False,
        "--with-buffer-albums/--no-buffer-albums",
        help="Also create Immich albums for system buffer images and videos.",
    ),
    batch_size: int = typer.Option(500, min=1, max=1000),
):
    """Safely write hybrid results back to Immich using visibility and albums only."""
    with app_context() as (_, _conn, client):
        stats = sync_hybrid_writeback(
            client,
            actions_path=actions_path,
            with_buffer_albums=with_buffer_albums,
            batch_size=batch_size,
        )
        typer.echo(f"Synced hybrid writeback: {stats}")


@app.command()
def preview(host: str = "127.0.0.1", port: int = 8765):
    """Launch the local preview UI."""
    with app_context() as (settings, conn, client):
        preview_app = PreviewApp(settings, client, conn).build()
        uvicorn.run(preview_app, host=host, port=port, log_level="info")
