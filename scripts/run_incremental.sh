#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCKDIR="$WORKDIR/.incremental.lock"
PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

mkdir -p "$WORKDIR/logs"

if [ -z "$UV_BIN" ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] nasai incremental failed: uv not found in PATH"
  exit 1
fi

if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] nasai incremental skipped: lock exists"
  exit 0
fi

cleanup() {
  rmdir "$LOCKDIR" 2>/dev/null || true
}

trap cleanup EXIT

cd "$WORKDIR"

if [ -f "$WORKDIR/.env" ]; then
  set -a
  source "$WORKDIR/.env"
  set +a
fi

if [ -z "${IMMICH_BASE_URL:-}" ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] nasai incremental failed: IMMICH_BASE_URL missing"
  exit 1
fi

if ! /usr/bin/python3 - <<'PY'
import os
import socket
import sys
from urllib.parse import urlparse

url = urlparse(os.environ["IMMICH_BASE_URL"])
host = url.hostname
port = url.port or (443 if url.scheme == "https" else 80)
try:
    with socket.create_connection((host, port), timeout=5):
        pass
except OSError:
    sys.exit(1)
PY
then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] nasai incremental skipped: Immich host unreachable"
  exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] nasai incremental start"
"$UV_BIN" run nasai incremental
echo "[$(date '+%Y-%m-%d %H:%M:%S')] nasai incremental done"
