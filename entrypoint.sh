#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] wait Qdrant ${QDRANT_URL:-http://qdrant:6333} ..."
for i in {1..120}; do
  curl -fsS "${QDRANT_URL:-http://qdrant:6333}" >/dev/null 2>&1 && break
  sleep 2
  [ "$i" -eq 120 ] && echo "Qdrant not reachable" >&2 && exit 1
done

STATE_DIR="/app/.state"; mkdir -p "$STATE_DIR"
HASH_FILE="$STATE_DIR/src.hash"; FLAG="$STATE_DIR/ingested.flag"

# ざっくり差分判定（Gitあり/なし両対応）
if command -v git >/dev/null 2>&1 && [ -d "/app/src_repos/.git" ]; then
  CUR_HASH=$(git -C /app/src_repos rev-parse HEAD || echo "nogit")
else
  CUR_HASH=$(find /app/src_repos -type f -not -path "*/.git/*" -printf "%P %T@\n" | sort | sha256sum | cut -d' ' -f1 || echo "nofiles")
fi
PREV_HASH=$(cat "$HASH_FILE" 2>/dev/null || echo "none")

if [ "${FORCE_REINDEX:-0}" = "1" ] || [ ! -f "$FLAG" ] || [ "$CUR_HASH" != "$PREV_HASH" ]; then
  echo "[entrypoint] run ingest.py ..."
  python /app/ingest.py
  echo "$CUR_HASH" > "$HASH_FILE"
  date -u +"%FT%TZ" > "$FLAG"
else
  echo "[entrypoint] skip ingest (no change)."
fi

# （任意）Ollamaの疎通確認
if [ -n "${OLLAMA_BASE_URL:-}" ]; then
  echo "[entrypoint] check Ollama ${OLLAMA_BASE_URL}"
  for _ in {1..60}; do curl -fsS "${OLLAMA_BASE_URL}/api/tags" >/dev/null 2>&1 && break; sleep 2; done
fi

echo "[entrypoint] start Chainlit + FastAPI(/code_search)"
exec chainlit run /app/app.py --host 0.0.0.0 --port 8000
