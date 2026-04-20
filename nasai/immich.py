from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx


@dataclass
class ImmichClient:
    base_url: str
    api_key: str

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"x-api-key": self.api_key},
            timeout=httpx.Timeout(60.0),
        )

    def close(self) -> None:
        self._client.close()

    def current_user(self) -> Dict[str, Any]:
        response = self._client.get("/api/users/me")
        response.raise_for_status()
        return response.json()

    def search_metadata(
        self,
        *,
        page: Optional[str] = None,
        size: int = 1000,
        with_exif: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "size": size,
            "withDeleted": False,
            "withExif": with_exif,
            "withPeople": True,
            "withStacked": True,
            "order": "desc",
        }
        if page:
            payload["page"] = page
        response = self._client.post("/api/search/metadata", json=payload)
        response.raise_for_status()
        return response.json()

    def asset(self, asset_id: str) -> Dict[str, Any]:
        response = self._client.get(f"/api/assets/{asset_id}")
        response.raise_for_status()
        return response.json()

    def thumbnail(self, asset_id: str) -> bytes:
        response = self._client.get(f"/api/assets/{asset_id}/thumbnail")
        response.raise_for_status()
        return response.content

    def thumbnail_to_file(self, asset_id: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = self.thumbnail(asset_id)
        dest.write_bytes(data)
        return dest

    def list_albums(self) -> List[Dict[str, Any]]:
        response = self._client.get("/api/albums")
        response.raise_for_status()
        return response.json()

    def create_album(self, album_name: str) -> Dict[str, Any]:
        response = self._client.post("/api/albums", json={"albumName": album_name})
        response.raise_for_status()
        return response.json()

    def add_assets_to_album(self, album_id: str, asset_ids: Iterable[str]) -> List[Dict[str, Any]]:
        response = self._client.put(
            f"/api/albums/{album_id}/assets", json={"ids": list(asset_ids)}
        )
        response.raise_for_status()
        return response.json()

    def delete_album(self, album_id: str) -> None:
        response = self._client.delete(f"/api/albums/{album_id}")
        response.raise_for_status()

    def list_tags(self) -> List[Dict[str, Any]]:
        response = self._client.get("/api/tags")
        response.raise_for_status()
        return response.json()

    def create_tag(self, name: str) -> Dict[str, Any]:
        response = self._client.post("/api/tags", json={"name": name})
        response.raise_for_status()
        return response.json()

    def tag_assets(self, tag_id: str, asset_ids: Iterable[str]) -> List[Dict[str, Any]]:
        response = self._client.put(f"/api/tags/{tag_id}/assets", json={"ids": list(asset_ids)})
        response.raise_for_status()
        return response.json()

    def update_assets(
        self,
        asset_ids: Iterable[str],
        *,
        visibility: Optional[str] = None,
        rating: Optional[int] = None,
        description: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {"ids": list(asset_ids)}
        if visibility is not None:
            payload["visibility"] = visibility
        if rating is not None:
            payload["rating"] = rating
        if description is not None:
            payload["description"] = description
        response = self._client.put("/api/assets", json=payload)
        response.raise_for_status()

    def open_asset_url(self, asset_id: str) -> str:
        return f"{self.base_url}/photos/{asset_id}"

    def authenticated_thumbnail_request(
        self, asset_id: str, request_headers: Optional[Dict[str, str]] = None
    ) -> httpx.Response:
        headers = {"x-api-key": self.api_key}
        if request_headers:
            headers.update(request_headers)
        return self._client.get(f"/api/assets/{asset_id}/thumbnail", headers=headers)
