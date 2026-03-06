"""MemForge adapter — talks to the MemForge MCP server via HTTP (SSE transport)."""

from __future__ import annotations

import json
import time

import httpx

from benchmark.adapters.base import MemoryAdapter
from benchmark.models import RecallResult, StoreResult

# MCP StreamableHTTP returns Server-Sent Events. We need to parse them.
MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _parse_sse_response(text: str) -> dict:
    """Extract the JSON-RPC result from an SSE response body.

    SSE format:
        event: message
        data: {"result": {...}, "jsonrpc": "2.0", "id": 1}
    """
    for line in text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    # Fallback: try parsing the whole thing as JSON
    return json.loads(text)


class MemforgeAdapter(MemoryAdapter):
    name = "memforge"

    def __init__(self, base_url: str = "http://localhost:3100") -> None:
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._request_id: int = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def setup(self) -> float:
        start = time.perf_counter()
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)

        # Health check
        resp = await self._client.get("/health")
        resp.raise_for_status()

        # MCP initialize
        resp = await self._client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "memforge-benchmark", "version": "0.1.0"},
                },
            },
            headers=MCP_HEADERS,
        )
        resp.raise_for_status()
        self._session_id = resp.headers.get("mcp-session-id")

        # Send initialized notification (required by MCP spec)
        notify_headers = {**MCP_HEADERS}
        if self._session_id:
            notify_headers["mcp-session-id"] = self._session_id
        await self._client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            },
            headers=notify_headers,
        )

        return time.perf_counter() - start

    async def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        assert self._client is not None
        headers = {**MCP_HEADERS}
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        resp = await self._client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
            headers=headers,
        )
        resp.raise_for_status()
        return _parse_sse_response(resp.text)

    async def store(
        self,
        content: str,
        *,
        user_id: str = "alex",
        agent_id: str = "coding-assistant",
        importance: float | None = None,
        shared: bool = False,
        task_id: str | None = None,
    ) -> StoreResult:
        args: dict = {"content": content, "user_id": user_id, "agent_id": agent_id}
        if importance is not None:
            args["importance"] = importance
        if shared:
            args["shared"] = True
        if task_id:
            args["task_id"] = task_id

        start = time.perf_counter()
        result = await self._call_tool("remember", args)
        elapsed = (time.perf_counter() - start) * 1000

        content_blocks = result.get("result", {}).get("content", [])
        text = content_blocks[0]["text"] if content_blocks else "{}"
        parsed = json.loads(text)

        return StoreResult(
            memory_id=parsed.get("memory_id"),
            api_calls=0,
            latency_ms=elapsed,
        )

    async def recall(
        self,
        query: str,
        *,
        user_id: str = "alex",
        agent_id: str = "coding-assistant",
        token_budget: int = 2000,
        include_shared: bool = False,
        task_id: str | None = None,
    ) -> RecallResult:
        args: dict = {
            "query": query,
            "user_id": user_id,
            "agent_id": agent_id,
            "token_budget": token_budget,
        }
        if include_shared:
            args["include_shared"] = True
        if task_id:
            args["task_id"] = task_id

        start = time.perf_counter()
        result = await self._call_tool("recall", args)
        elapsed = (time.perf_counter() - start) * 1000

        content_blocks = result.get("result", {}).get("content", [])
        text = content_blocks[0]["text"] if content_blocks else ""

        # Recall returns raw formatted context text, NOT JSON.
        # The server returns result.context directly as the text content.
        context = text
        # Approximate token count: words * 1.3
        total_tokens = int(len(context.split()) * 1.3) if context else 0

        # Extract individual memories from the formatted context
        memories = []
        if context and context != "No relevant memories found.":
            for line in context.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    memories.append(line[2:])

        return RecallResult(
            memories=memories,
            formatted_context=context if context else None,
            total_tokens=total_tokens,
            latency_ms=elapsed,
            api_calls=0,
        )

    async def batch_store(
        self,
        items: list[dict],
        *,
        concurrency: int = 5,
    ) -> list[StoreResult]:
        """Bulk store via REST batch endpoint (bypasses MCP, much faster)."""
        assert self._client is not None
        start = time.perf_counter()

        resp = await self._client.post(
            "/api/batch-remember",
            json={"items": items, "concurrency": concurrency},
            timeout=600.0,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = (time.perf_counter() - start) * 1000

        results = []
        for r in data.get("results", []):
            results.append(StoreResult(
                memory_id=r.get("memory_id"),
                api_calls=0,
                latency_ms=elapsed / len(items) if items else 0,
            ))
        return results

    async def reset(self) -> None:
        pass

    async def teardown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
