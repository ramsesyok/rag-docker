# mcp/server.py
import os
import httpx
from mcp.server.fastmcp import FastMCP

RAG_BASE = os.environ.get("RAG_BASE", "http://rag:8001")
MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "5173"))

# ★ host/port はコンストラクタで指定する（run()に渡さない）
mcp = FastMCP(
    "code-rag",
    host=MCP_HOST,
    port=MCP_PORT,
    # お好みで：
    # stateless_http=True,            # セッションを持たない軽量運用
    # streamable_http_path="/mcp",    # 既定は "/mcp"
)

@mcp.tool()
async def code_search(
    query: str,
    kind: str = "regex",
    lang: list[str] | None = None,
    path_globs: list[str] | None = None,
    case_sensitive: bool = False,
    max_results: int = 20,
    before_after_lines: int = 2,
) -> str:
    payload = {
        "query": query,
        "kind": kind,
        "lang": lang or [],
        "path_globs": path_globs or [],
        "case_sensitive": case_sensitive,
        "max_results": max_results,
        "before_after_lines": before_after_lines,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{RAG_BASE}/code_search", json=payload)
        r.raise_for_status()
        data = r.json()

    hits = data.get("hits", [])
    if not hits:
        return "No matches."
    blocks = []
    for h in hits[:max_results]:
        header = f"{h.get('repo','')}/{h.get('path','')}:{h.get('start_line','?')}-{h.get('end_line','?')}"
        preview = (h.get("preview") or "").rstrip()
        blocks.append(f"### {header}\n```\n{preview}\n```")
    return "\n\n".join(blocks)

@mcp.tool()
async def ask_code(question: str, top_k: int = 6) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{RAG_BASE}/query", json={"question": question, "top_k": top_k})
        r.raise_for_status()
        data = r.json()

    answer = data.get("answer", "(no answer)")
    cits = data.get("citations", [])
    if cits:
        refs = "\n".join(
            f"- {c.get('repo','')}/{c.get('path','')}:{c.get('startLine','?')}-{c.get('endLine','?')}"
            for c in cits
        )
        return f"{answer}\n\nReferences:\n{refs}"
    return answer

if __name__ == "__main__":
    # ★ ここでは host/port を渡さない
    mcp.run(transport="streamable-http")
