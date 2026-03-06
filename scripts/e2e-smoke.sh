#!/bin/bash
# E2E smoke test for MemForge MCP server.
# Starts the full stack via Docker Compose, verifies health,
# runs remember + recall + forget, then tears down.
# Exit code 0 = pass, non-zero = fail.

set -euo pipefail

MCP_URL="http://localhost:3100/mcp"
ACCEPT="Accept: application/json, text/event-stream"
CONTENT="Content-Type: application/json"

cleanup() {
  echo "[e2e] Tearing down..."
  docker compose down -v 2>/dev/null || true
}
trap cleanup EXIT

# Helper: call an MCP tool and return parsed JSON from SSE
mcp_call() {
  local id="$1"
  local tool="$2"
  local args="$3"
  local raw
  raw=$(curl -sf -X POST "$MCP_URL" \
    -H "$CONTENT" \
    -H "$ACCEPT" \
    -d "{\"jsonrpc\":\"2.0\",\"id\":$id,\"method\":\"tools/call\",\"params\":{\"name\":\"$tool\",\"arguments\":$args}}" \
    | grep '^data: ' | sed 's/^data: //')
  echo "$raw"
}

echo "[e2e] Starting services..."
docker compose up -d --build --wait 2>&1 | tail -3

# --- Health check ---
echo "[e2e] Health check..."
curl -sf http://localhost:3100/health | jq -e '.status == "ok"' > /dev/null \
  || { echo "FAIL: health check"; exit 1; }
echo "[e2e] Health: OK"

# --- Remember ---
echo "[e2e] Testing remember..."
RES=$(mcp_call 1 "remember" '{"content":"E2E test memory","user_id":"e2e"}')
echo "$RES" | jq -e '.result' > /dev/null \
  || { echo "FAIL: remember returned no result"; exit 1; }
MEMORY_ID=$(echo "$RES" | jq -r '.result.content[0].text' | jq -r '.memory_id')
[ "$MEMORY_ID" != "null" ] && [ -n "$MEMORY_ID" ] \
  || { echo "FAIL: remember returned no memory_id"; exit 1; }
echo "[e2e] Remember: OK (id=${MEMORY_ID:0:8}...)"

# --- Recall ---
echo "[e2e] Testing recall..."
RES=$(mcp_call 2 "recall" '{"query":"test","user_id":"e2e"}')
echo "$RES" | jq -e '.result' > /dev/null \
  || { echo "FAIL: recall returned no result"; exit 1; }
CONTEXT=$(echo "$RES" | jq -r '.result.content[0].text')
echo "$CONTEXT" | grep -q "E2E test memory" \
  || { echo "FAIL: recall did not return stored memory"; exit 1; }
echo "[e2e] Recall: OK"

# --- Forget ---
echo "[e2e] Testing forget..."
RES=$(mcp_call 3 "forget" "{\"memory_id\":\"$MEMORY_ID\"}")
DELETED=$(echo "$RES" | jq -r '.result.content[0].text' | jq -r '.deleted')
[ "$DELETED" = "true" ] \
  || { echo "FAIL: forget did not delete memory"; exit 1; }
echo "[e2e] Forget: OK"

echo ""
echo "E2E PASSED"
