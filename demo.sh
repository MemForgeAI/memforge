#!/bin/bash
set -e

# MemForge Demo — Nuclear Memory for AI Agents
# This script demonstrates the full memory lifecycle:
#   remember -> recall -> reflect -> forget

MCP_URL="http://localhost:3100/mcp"
ACCEPT="Accept: application/json, text/event-stream"
CONTENT="Content-Type: application/json"

# Helper: call an MCP tool and parse SSE response
mcp_call() {
  local id="$1"
  local tool="$2"
  local args="$3"
  curl -s -X POST "$MCP_URL" \
    -H "$CONTENT" \
    -H "$ACCEPT" \
    -d "{\"jsonrpc\":\"2.0\",\"id\":$id,\"method\":\"tools/call\",\"params\":{\"name\":\"$tool\",\"arguments\":$args}}" \
    | grep '^data: ' | sed 's/^data: //'
}

echo ""
echo "====================================="
echo "  MEMFORGE — Nuclear Memory for AI Agents"
echo "====================================="
echo ""

# --- Start services ---
echo "[1/6] Starting services..."
POSTGRES_HOST_PORT="${POSTGRES_HOST_PORT:-5432}" docker compose up -d --build --wait 2>&1 | tail -1
echo "       Services healthy."
echo ""

# --- Remember: store three memories ---
echo "[2/6] Storing memories..."
echo ""

echo "  > \"User prefers direct flights and hates layovers\""
RES=$(mcp_call 1 "remember" '{"content":"User prefers direct flights and hates layovers","user_id":"demo"}')
echo "    $(echo "$RES" | jq -r '.result.content[0].text' | jq -r '"type=\(.memory_type) id=\(.memory_id[0:8])..."')"
echo ""

echo "  > \"User is allergic to shellfish\" (importance=1.0)"
RES=$(mcp_call 2 "remember" '{"content":"User is allergic to shellfish","user_id":"demo","importance":1.0}')
echo "    $(echo "$RES" | jq -r '.result.content[0].text' | jq -r '"type=\(.memory_type) id=\(.memory_id[0:8])..."')"
echo ""

echo "  > \"User booked JAL to Tokyo last month, loved the service\""
RES=$(mcp_call 3 "remember" '{"content":"User booked JAL to Tokyo last month, loved the service","user_id":"demo"}')
MEMORY_ID=$(echo "$RES" | jq -r '.result.content[0].text' | jq -r '.memory_id')
echo "    $(echo "$RES" | jq -r '.result.content[0].text' | jq -r '"type=\(.memory_type) id=\(.memory_id[0:8])... entities=\(.entities_created | length)"')"
echo ""

# --- Recall: ask a question ---
echo "[3/6] Recalling context for: \"planning a flight to Japan for dinner reservations\""
echo ""
RES=$(mcp_call 4 "recall" '{"query":"planning a flight to Japan for dinner reservations","user_id":"demo","token_budget":500}')
CONTEXT=$(echo "$RES" | jq -r '.result.content[0].text')
echo "  --- Context Document ---"
echo "$CONTEXT" | sed 's/^/  /'
echo "  ------------------------"
echo ""

# --- Reflect: consolidate insights ---
echo "[4/6] Reflecting on patterns..."
RES=$(mcp_call 5 "reflect" '{"user_id":"demo","lookback_hours":1}')
INSIGHTS=$(echo "$RES" | jq -r '.result.content[0].text' | jq -r '.insights_created | length')
echo "       Insights created: $INSIGHTS"
echo ""

# --- Dedup: store a duplicate ---
echo "[5/6] Testing dedup — storing same flight preference again..."
RES=$(mcp_call 6 "remember" '{"content":"User always wants direct flights, no layovers","user_id":"demo"}')
DEDUP=$(echo "$RES" | jq -r '.result.content[0].text' | jq -r '.duplicate')
echo "       Duplicate detected: $DEDUP"
echo ""

# --- Forget: remove a memory ---
echo "[6/6] Forgetting memory: ${MEMORY_ID:0:8}..."
RES=$(mcp_call 7 "forget" "{\"memory_id\":\"$MEMORY_ID\"}")
DELETED=$(echo "$RES" | jq -r '.result.content[0].text' | jq -r '.deleted')
echo "       Deleted: $DELETED"
echo ""

echo "====================================="
echo "  Demo complete."
echo "  Run 'docker compose down -v' to clean up."
echo "====================================="
echo ""
