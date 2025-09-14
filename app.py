#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py
- Chainlit のチャットUIと、/code_search・/query を提供する FastAPI を同一プロセスで起動。
- LLM は Ollama（別PCでも可、環境変数 OLLAMA_BASE_URL）。
- 埋め込みは HuggingFaceEmbedding（OpenAIキー不要）。ingest.py と同じ EMBED_MODEL を使うこと。

【超ざっくり流れ（初心者向け）】
1) Qdrant（ベクタDB）に ingest.py で投入済みのベクトルを使って、質問文を意味検索（RAGの取得）
2) 取得したスニペットを再ランク（CrossEncoder）して、より関連の高い上位だけ残す
3) それらを根拠として LLM（Ollama）に回答文の生成を依頼（プロンプトに根拠を詰め込む）
4) Chat UI（Chainlit）に回答＋引用（ファイル/行番号）を表示

【必要な環境変数（.env 推奨）】
  OLLAMA_BASE_URL=http://host.docker.internal:11434   # または別PCのIP:11434（Docker内からホストのOllamaへ）
  OLLAMA_MODEL=llama3:13b
  EMBED_MODEL=BAAI/bge-m3
  QDRANT_URL=http://qdrant:6333
  QDRANT_COLLECTION=codebase
  API_PORT=8001
  LOG_LEVEL=INFO    # DEBUG/INFO/WARN/ERROR

ヒント: まず ingest.py を実行して Qdrant にデータを投入→ 次に app.py を起動すると質問できるようになります。
"""

import os
import re
import fnmatch
import shutil
import json
import subprocess
import threading
import logging
from pathlib import Path
from typing import List, Optional, Literal, Tuple

# -------- ログ設定 --------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("app")

# -------- Chainlit --------
import chainlit as cl

# -------- LlamaIndex / Qdrant 初期化 --------
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# 埋め込み（OpenAI既定を避けるために必ず明示設定）
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 再ランク（CPUでも可）
from sentence_transformers import CrossEncoder

# -------- FastAPI（/code_search, /query） --------
from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# -------- 設定値 --------
SRC_ROOT = Path(__file__).parent.resolve() / "src_repos"
API_PORT = int(os.environ.get("API_PORT", "8001"))

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "codebase")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# LLM (Ollama)
# ポイント: OpenAI API は使わずローカル/別PCの Ollama を利用します。
#           base_url は Docker から見た Ollama の場所（host.docker.internal など）に合わせます。
Settings.llm = Ollama(
    model=os.environ.get("OLLAMA_MODEL", "llama3:13b"),
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    request_timeout=120.0
)

# Embedding（OpenAIの既定を回避）
# ポイント: ingest.py と同じ埋め込みモデル名を必ず指定してください（検索側と登録側で揃える）。
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
log.info(f"Embedding model = {EMBED_MODEL}")

# Qdrant 接続＆Index
# ポイント: 既に Qdrant に入っているベクトルを検索するためのインデックスを作成します。
qclient = QdrantClient(url=QDRANT_URL)
vstore = QdrantVectorStore(client=qclient, collection_name=COLLECTION)
storage = StorageContext.from_defaults(vector_store=vstore)
index = VectorStoreIndex.from_vector_store(vstore)

# 再ランカー（最初のロードは少し時間がかかります）
# ポイント: CrossEncoder は「質問×スニペット」をペアで評価して関連度を再計算します。
log.info(f"Loading reranker: {RERANK_MODEL} ...")
reranker = CrossEncoder(RERANK_MODEL)
log.info("Reranker ready.")

# ======== 共有ユーティリティ ========
def rerank_nodes(query: str, nodes: List[Any], top_k: int = 6) -> List[Any]:
    """意味検索の初期結果を CrossEncoder で並べ替え、上位 top_k を返す。

    初期の `nodes` はベクトル類似度で近い順ですが、
    CrossEncoder（再ランク）で「本当に質問に答えるのに良いスニペットか」を再評価します。
    """
    if not nodes:
        return []
    texts, unwrapped = [], []
    for n in nodes:
        node = getattr(n, "node", n)
        unwrapped.append(node)
        texts.append(node.get_text() if hasattr(node, "get_text") else getattr(node, "text", ""))

    pairs = [[query, t] for t in texts]
    try:
        # 環境変数でバッチサイズ/ソフトマックスの有無を調整可能に
        batch = int(os.environ.get("RERANK_BATCH", "16"))
        apply_softmax = os.environ.get("RERANK_SOFTMAX", "0") == "1"

        scores = reranker.predict(
            pairs,
            batch_size=batch,
            show_progress_bar=False,
            apply_softmax=apply_softmax,
        )
    except Exception as e:
        log.warning(f"Rerank failed, fallback original order: {e}")
        return unwrapped[:top_k]

    ranked = sorted(zip(unwrapped, scores), key=lambda x: float(x[1]), reverse=True)
    return [n for n, _ in ranked[:top_k]]

def build_prompt(question: str, nodes: List[Any]) -> str:
    """LLM に渡す最終プロンプトを構築する。

    - 根拠スニペットを [repo/path Lstart-Lend] + 本文 で列挙
    - 回答は「根拠のみで」行うようシステムメッセージで強制
    """
    context_blocks = []
    for n in nodes:
        m = getattr(n, "metadata", {}) or {}
        header = f"[{m.get('repo','')}/{m.get('path','')} L{m.get('start_line','?')}-{m.get('end_line','?')}]"
        text = n.get_text() if hasattr(n, "get_text") else getattr(n, "text", "")
        context_blocks.append(f"{header}\n{text}")

    sysinst = (
        "あなたはコード解析アシスタントです。以下の根拠スニペットのみを用いて回答してください。\n"
        "根拠が不十分なら『不明』と答えてください。推測や外部知識は禁止です。"
    )
    prompt = f"{sysinst}\n\n【質問】\n{question}\n\n【根拠スニペット】\n" + "\n\n".join(context_blocks)
    return prompt

def run_query_pipeline(question: str, top_k_vec: int = 30, top_k_final: int = 6) -> Tuple[str, List[dict]]:
    """質問に答えるための一連の処理（RAG パイプライン）。

    1) ベクトル検索（similarity_top_k=top_k_vec）で粗く候補を拾う
    2) CrossEncoder（再ランク）で top_k_final まで厳選
    3) 根拠付きプロンプトを組み立てて LLM へ
    4) 引用情報（ファイル/行番号）も組み立てて返す

    戻り値: (LLM回答テキスト, 引用メタデータの配列)
    """
    retriever = index.as_retriever(similarity_top_k=top_k_vec)
    try:
        nodes = retriever.retrieve(question)
    except Exception as e:
        log.error(f"Retriever error: {e}")
        nodes = []

    # NodeWithScore -> 再ランク
    top_nodes = rerank_nodes(question, nodes, top_k=top_k_final)

    # LLM へ
    prompt = build_prompt(question, top_nodes)
    try:
        resp = Settings.llm.complete(prompt)
        answer = resp.text
    except Exception as e:
        log.error(f"LLM error: {e}")
        answer = f"[LLM error] {e}"

    citations = gather_citations(top_nodes)
    return answer, citations

# ======== FastAPI: /code_search & /query ========

EXT_MAP = {
  "c": [".c", ".h"],
  "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"],
  "java": [".java"],
  "rust": [".rs"],
  "javascript": [".js", ".mjs", ".cjs", ".jsx"],
  "typescript": [".ts", ".tsx", ".mts", ".cts", ".d.ts"],
  "vue": [".vue"],
}

class CodeSearchReq(BaseModel):
    # /code_search 用の入力。literal（完全一致）か regex（正規表現）を選べます。
    query: str
    kind: Literal["literal", "regex"] = "literal"
    lang: Optional[List[str]] = None
    path_globs: Optional[List[str]] = None
    case_sensitive: bool = True
    max_results: int = 200
    before_after_lines: int = 3

class CodeSearchHit(BaseModel):
    # /code_search の1件分のヒット結果（プレビューは前後行付き）。
    repo: str
    path: str
    lang: str
    start_line: int
    end_line: int
    preview: str

class QueryReq(BaseModel):
    # /query 用の入力。top_k は最終的に何件の根拠を使うか。
    question: str
    top_k: int = 6

app = FastAPI(title="RAG API (/code_search, /query)")

def _lang_of(path: Path) -> str:
    ext = path.suffix.lower()
    for k, arr in EXT_MAP.items():
        if ext in arr:
            return k
    return "unknown"

def _iter_files(root: Path, langs: Optional[List[str]], globs: Optional[List[str]]):
    """対象ファイルを列挙。

    - `langs` 指定時は拡張子でフィルタ
    - `.git` や `node_modules` などは除外
    - `globs` 指定時はパスのワイルドカードでさらに絞り込み
    """
    allow_ext = None
    if langs:
        allow_ext = set(e for L in langs for e in EXT_MAP.get(L, []))
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in {".git","node_modules","build","target"} for part in rel.parts):
            continue
        if allow_ext and p.suffix.lower() not in allow_ext:
            continue
        if globs and not any(fnmatch.fnmatch(str(rel).replace("\\","/"), g) for g in globs):
            continue
        yield p

def _search_with_ripgrep(req: CodeSearchReq) -> List[CodeSearchHit]:
    """rg（ripgrep）が使える場合はこちらを優先して高速検索。

    JSON 出力をパースして前後行つきのプレビューを整形します。
    インストールされていない環境では空リストを返し、Python 実装にフォールバック。
    """
    hits: List[CodeSearchHit] = []
    if shutil.which("rg") is None:
        return hits
    cmd = ["rg", "--json", "-n", f"-C{req.before_after_lines}"]
    if req.kind == "literal":
        cmd.append("-F")
    if not req.case_sensitive:
        cmd.append("-i")
    if req.path_globs:
        for g in req.path_globs:
            cmd += ["-g", g]
    cmd.append(req.query)
    cmd.append(str(SRC_ROOT))

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        path_cache = {}
        count = 0
        for line in proc.stdout:  # type: ignore
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") != "match":
                continue
            data = obj.get("data", {})
            path_text = data.get("path", {}).get("text")
            if not path_text:
                continue
            abs_path = Path(path_text)
            if SRC_ROOT not in abs_path.resolve().parents and abs_path.resolve() != SRC_ROOT.resolve():
                continue
            rel = abs_path.relative_to(SRC_ROOT)
            repo = rel.parts[0] if len(rel.parts) > 1 else ""
            lang = _lang_of(abs_path)
            try:
                if abs_path not in path_cache:
                    path_cache[abs_path] = abs_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                lines = path_cache[abs_path]
                lnum = data.get("line_number", 1)
            except Exception:
                continue
            start = max(1, lnum - req.before_after_lines)
            end = min(len(lines), lnum + req.before_after_lines)
            snippet = "\n".join(f"{i:>5}: {lines[i-1]}" for i in range(start, end+1))
            hits.append(CodeSearchHit(
                repo=repo, path=str(rel), lang=lang,
                start_line=start, end_line=end, preview=snippet
            ))
            count += 1
            if count >= req.max_results:
                break
        proc.terminate()
    except Exception as e:
        log.warning(f"ripgrep failed: {e}")
    return hits

def _search_with_python(req: CodeSearchReq) -> List[CodeSearchHit]:
    """rg がないときの純Python版検索（遅いが依存なし）。"""
    hits: List[CodeSearchHit] = []
    flags = 0 if req.case_sensitive else re.IGNORECASE
    rx = re.compile(req.query if req.kind == "regex" else re.escape(req.query), flags)
    count = 0
    for p in _iter_files(SRC_ROOT, req.lang, req.path_globs):
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, start=1):
            if rx.search(line):
                start = max(1, i - req.before_after_lines)
                end = min(len(lines), i + req.before_after_lines)
                snippet = "\n".join(f"{j:>5}: {lines[j-1]}" for j in range(start, end+1))
                rel = p.relative_to(SRC_ROOT)
                repo = rel.parts[0] if len(rel.parts) > 1 else ""
                hits.append(CodeSearchHit(
                    repo=repo, path=str(rel), lang=_lang_of(p),
                    start_line=start, end_line=end, preview=snippet
                ))
                count += 1
                if count >= req.max_results:
                    return hits
    return hits

@app.post("/code_search")
def code_search(req: CodeSearchReq):
    """コード全文検索API。rg が無ければ Python で代替。"""
    hits = _search_with_ripgrep(req)
    if not hits:
        hits = _search_with_python(req)
    return {"hits": [h.model_dump() for h in hits]}

@app.post("/query")
def query(req: QueryReq):
    """RAG + 再ランク + LLMで回答を生成するAPI。"""
    answer, cits = run_query_pipeline(req.question, top_k_vec=30, top_k_final=req.top_k)
    return {"answer": answer, "citations": cits}

def start_api_server():
    log.info(f"Starting FastAPI on 0.0.0.0:{API_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="info")

# ======== Chainlit ハンドラ ========

WELCOME = """ようこそ 👋

このボットは「既存コードの分析・調査」を高速化するためのアシスタントです。
- 🧠 RAG（意味検索）で関連コードを集め、回答は必ず根拠スニペット付き
- 🔎 厳密検索が必要なときは `/search` コマンド（例: `/search regex:"\\bstrcpy\\(" lang:c path:"src/**"`）
"""

@cl.on_chat_start
async def on_start():
    await cl.Message(content=WELCOME).send()

@cl.on_message
async def on_message(message):
    # ユーザーからの1メッセージに対して RAG パイプラインを回して回答を返します。
    # 重い処理はスレッドに逃がし、Chainlit メッセージは途中で「検索中…」を表示します。
    # message を文字列に
    if isinstance(message, str):
        text = message.strip()
    else:
        text = (getattr(message, "content", "") or "").strip()

    # プレースホルダ
    msg = cl.Message(content="検索中…（RAG + re-rank 実行中）")
    await msg.send()

    # 重い処理は別スレッドで
    answer, cits = await cl.make_async(run_query_pipeline)(text, top_k_vec=30, top_k_final=6)

    refs = "\n".join(
        f"- `{c['repo']}/{c['path']}` L{c['startLine']}-{c['endLine']}" for c in cits
    ) or "(no references)"
    md = f"{answer}\n\n**References**\n{refs}"

    # Chainlitのバージョン差に対応（update の呼び方が異なるため両対応）
    try:
        # 新しめのAPI: contentをプロパティに代入してからupdate()
        msg.content = md
        await msg.update()
    except TypeError:
        # 旧API: update(content=...) が使える
        await msg.update(content=md)

# 追加：app.py に貼り付け（run_query_pipeline より上に置く）
from typing import Any

def gather_citations(nodes: list[Any]) -> list[dict]:
    """
    NodeWithScore / TextNode から、参照用メタデータを抽出して返す。
    - 表示用に repo/path と行番号を取り出して、Chainlit に渡せる形に整形します。
    run_query_pipeline() の戻り値にそのまま載せます。
    """
    citations: list[dict] = []
    for n in nodes:
        node = getattr(n, "node", n)  # NodeWithScore なら .node、そうでなければそのまま
        m = getattr(node, "metadata", {}) or {}
        citations.append({
            "repo": m.get("repo", ""),
            "path": m.get("path", ""),
            "lang": m.get("lang", ""),
            "startLine": m.get("start_line", m.get("startLine", "?")),
            "endLine": m.get("end_line", m.get("endLine", "?")),
        })
    return citations

# ======== エントリポイント：FastAPI を別スレッドで起動 ========
if os.environ.get("DISABLE_CODE_SEARCH_API", "0") != "1":
    th = threading.Thread(target=start_api_server, daemon=True)
    th.start()
