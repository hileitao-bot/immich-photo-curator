from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    immich_base_url: str
    immich_api_key: str
    db_path: Path
    cache_dir: Path
    preview_dir: Path

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        base_url = os.environ.get("IMMICH_BASE_URL", "").rstrip("/")
        api_key = os.environ.get("IMMICH_API_KEY", "")
        db_path = Path(os.environ.get("NASAI_DB_PATH", Path.cwd() / "nasai.db")).expanduser()
        cache_dir = Path(os.environ.get("NASAI_CACHE_DIR", Path.cwd() / "cache")).expanduser()
        preview_dir = Path(
            os.environ.get("NASAI_PREVIEW_DIR", Path.cwd() / "preview")
        ).expanduser()
        if not base_url:
            raise RuntimeError("IMMICH_BASE_URL is required")
        if not api_key:
            raise RuntimeError("IMMICH_API_KEY is required")
        cache_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(
            immich_base_url=base_url,
            immich_api_key=api_key,
            db_path=db_path,
            cache_dir=cache_dir,
            preview_dir=preview_dir,
        )
