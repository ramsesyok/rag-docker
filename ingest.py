#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest.py (robust logging + Vue/JSX/TSX + fallback)
- 言語: C/C++/Java/Rust/JavaScript/TypeScript/Go/Vue(.vue)
- 目的: 既存コードを LlamaIndex の CodeSplitter（AST/構造）でチャンク化し、
        HuggingFace の埋め込みでベクトル化して Qdrant に投入。
- 特徴:
  * 詳細ログ（INFO でも要所、DEBUG で細部）
  * CodeSplitter 失敗・空チャンク時は SentenceSplitter にフォールバック（既定ON）
  * .vue は SFC を script/template に分解して処理
  * 処理内訳（言語別・失敗理由・フォールバック使用数）と Qdrant 件数を出力

環境変数（.env 推奨）
- SRC_ROOT=./src_repos
- QDRANT_URL=http://qdrant:6333
- QDRANT_COLLECTION=codebase
- EMBED_MODEL=BAAI/bge-m3
- LOG_LEVEL=INFO                   # DEBUG/INFO/WARN/ERROR
- INCLUDE_GLOBS=                   # 例: "src/**,packages/**"
- EXCLUDE_DIRS=.git,node_modules,build,target,dist,out,vendor,bin,.next,.turbo
- MAX_FILE_MB=2                    # MB 超はスキップ（既定 2MB）
- LANGS=cpp,java,rust,c,javascript,typescript,go,vue
- FALLBACK_SENTENCE_SPLITTER=1     # 0で無効
- SENTENCE_CHUNK=400
- SENTENCE_OVERLAP=40
"""

import os
import sys
import re
import time
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

# -------- ログ設定 --------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("ingest")

# -------- 設定 --------
BASE = Path(__file__).parent.resolve()
SRC_ROOT = Path(os.environ.get("SRC_ROOT", str(BASE / "src_repos"))).resolve()
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "codebase")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")

INCLUDE_GLOBS = [g.strip() for g in os.environ.get("INCLUDE_GLOBS", "").split(",") if g.strip()]
EXCLUDE_DIRS = {
    d.strip() for d in os.environ.get(
        "EXCLUDE_DIRS",
        ".git,node_modules,build,target,dist,out,vendor,bin,.next,.turbo"
    ).split(",")
}
MAX_FILE_MB = float(os.environ.get("MAX_FILE_MB", "2"))
LANGS = {s.strip().lower() for s in os.environ.get(
    "LANGS",
    "cpp,java,rust,c,javascript,typescript,go,vue"
).split(",")}

FALLBACK_SENTENCE = os.environ.get("FALLBACK_SENTENCE_SPLITTER", "1") != "0"
SENTENCE_CHUNK = int(os.environ.get("SENTENCE_CHUNK", "400"))
SENTENCE_OVERLAP = int(os.environ.get("SENTENCE_OVERLAP", "40"))

# 言語→拡張子
LANG_EXTS: Dict[str, Tuple[str, ...]] = {
    "c": (".c", ".h"),
    "cpp": (".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"),
    "java": (".java",),
    "rust": (".rs",),
    "javascript": (".js", ".mjs", ".cjs", ".jsx"),
    "typescript": (".ts", ".tsx", ".mts", ".cts", ".d.ts"),
    "go": (".go",),
    "vue": (".vue",),
}
# 逆引き（拡張子→言語；対象 LANGS のみ）
EXT_TO_LANG: Dict[str, str] = {}
for lang, exts in LANG_EXTS.items():
    if lang in LANGS:
        for e in exts:
            EXT_TO_LANG[e] = lang

# -------- LlamaIndex / Qdrant --------
try:
    from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
    from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
    from llama_index.vector_stores.qdrant import QdrantVectorStore
except Exception:
    log.exception("LlamaIndex の import に失敗しました。requirements.txt を確認してください。")
    sys.exit(2)

try:
    from qdrant_client import QdrantClient
except Exception:
    log.exception("qdrant-client の import に失敗しました。requirements.txt を確認してください。")
    sys.exit(2)

# Embedding: HuggingFace ラッパー（OpenAI不要）
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    log.info(f"Embedding: HuggingFaceEmbedding({EMBED_MODEL_NAME}) を使用します。")
except Exception:
    log.exception("HuggingFaceEmbedding 初期化に失敗。`pip install llama-index-embeddings-huggingface` が必要です。")
    sys.exit(2)

# -------- Util --------
def file_iter(root: Path) -> Iterable[Path]:
    """対象ファイルを列挙（拡張子・除外ディレクトリ・include_globs を考慮）"""
    from fnmatch import fnmatch
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if INCLUDE_GLOBS:
            unix_rel = str(rel).replace("\\", "/")
            if not any(fnmatch(unix_rel, g) for g in INCLUDE_GLOBS):
                continue
        if p.suffix.lower() not in EXT_TO_LANG:
            continue
        yield p

def read_text_safely(path: Path) -> Optional[str]:
    """サイズ上限や読み込み例外に配慮した読み取り"""
    try:
        size = path.stat().st_size
    except Exception:
        return None
    if size > MAX_FILE_MB * 1024 * 1024:
        stats["skipped_large"] += 1
        if LOG_LEVEL == "DEBUG":
            log.debug(f"skip (size>{MAX_FILE_MB}MB): {path}")
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        stats["read_error"] += 1
        if LOG_LEVEL == "DEBUG":
            log.debug(f"skip (read error): {path}", exc_info=True)
        return None

def compute_line_span_by_streaming(text: str, chunk: str, start_pos: int) -> Tuple[int, int, int]:
    """チャンクの原文上の開始/終了行を概算（前回終端から検索して誤マッチを抑制）"""
    idx = text.find(chunk, start_pos)
    if idx == -1:
        idx = text.find(chunk)
        if idx == -1:
            return (1, max(1, chunk.count("\n") + 1), start_pos)
    start_line = text.count("\n", 0, idx) + 1
    end_line = start_line + chunk.count("\n")
    new_cursor = idx + len(chunk)
    return (start_line, end_line, new_cursor)

def split_vue_sfc(full_text: str):
    """
    .vue を script/template に分割。
    返り値: リスト[{kind:'script'|'template', lang:'javascript'|'typescript'|None, text:str, start_idx:int}]
    """
    blocks = []
    # <script>（lang="ts" の場合は TypeScript とみなす）
    for m in re.finditer(r"<script(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)</script>",
                         full_text, flags=re.IGNORECASE):
        attrs = m.group("attrs") or ""
        body = m.group("body") or ""
        is_ts = re.search(r'lang\s*=\s*["\']ts["\']', attrs, re.IGNORECASE) is not None
        lang = "typescript" if is_ts else "javascript"
        blocks.append({"kind": "script", "lang": lang, "text": body, "start_idx": m.start("body")})
    # <template>
    for m in re.finditer(r"<template[^>]*>(?P<body>[\s\S]*?)</template>",
                         full_text, flags=re.IGNORECASE):
        body = m.group("body") or ""
        blocks.append({"kind": "template", "lang": None, "text": body, "start_idx": m.start("body")})
    blocks.sort(key=lambda b: b["start_idx"])
    return blocks

# -------- Stats --------
stats: Dict[str, int] = {
    "files_seen": 0,
    "by_lang_c": 0, "by_lang_cpp": 0, "by_lang_java": 0,
    "by_lang_rust": 0, "by_lang_js": 0, "by_lang_ts": 0, "by_lang_go": 0, "by_lang_vue": 0,
    "skipped_large": 0,
    "read_error": 0,
    "splitter_init_fail": 0,
    "split_fail": 0,
    "empty_after_code_split": 0,
    "fallback_used": 0,
    "docs_appended": 0,
    "vue_script_chunks": 0,
    "vue_template_chunks": 0,
}

def inc_lang(lang: str):
    if lang == "c": stats["by_lang_c"] += 1
    elif lang == "cpp": stats["by_lang_cpp"] += 1
    elif lang == "java": stats["by_lang_java"] += 1
    elif lang == "rust": stats["by_lang_rust"] += 1
    elif lang == "javascript": stats["by_lang_js"] += 1
    elif lang == "typescript": stats["by_lang_ts"] += 1
    elif lang == "go": stats["by_lang_go"] += 1
    elif lang == "vue": stats["by_lang_vue"] += 1

# -------- Main --------
def main() -> None:
    t0 = time.time()
    log.info("=== Ingest start ===")
    log.info(f"SRC_ROOT            : {SRC_ROOT}")
    log.info(f"QDRANT_URL          : {QDRANT_URL}")
    log.info(f"QDRANT_COLLECTION   : {COLLECTION}")
    log.info(f"EMBED_MODEL         : {EMBED_MODEL_NAME}")
    log.info(f"LANGS               : {sorted(LANGS)}")
    log.info(f"INCLUDE_GLOBS       : {INCLUDE_GLOBS or '(none)'}")
    log.info(f"EXCLUDE_DIRS        : {sorted(EXCLUDE_DIRS)}")
    log.info(f"MAX_FILE_MB         : {MAX_FILE_MB}")
    log.info(f"FALLBACK_SENTENCE   : {int(FALLBACK_SENTENCE)}  (chunk={SENTENCE_CHUNK}, overlap={SENTENCE_OVERLAP})")
    log.info(f"LOG_LEVEL           : {LOG_LEVEL}")

    if not SRC_ROOT.exists():
        log.error(f"SRC_ROOT が見つかりません: {SRC_ROOT}")
        sys.exit(1)

    # Qdrant 疎通
    try:
        qclient = QdrantClient(url=QDRANT_URL)
        _ = qclient.get_collections()
        log.info("Qdrant への疎通 OK")
    except Exception:
        log.exception("Qdrant へ接続できません。QDRANT_URL を確認してください。")
        sys.exit(2)

    vstore = QdrantVectorStore(client=qclient, collection_name=COLLECTION)
    storage = StorageContext.from_defaults(vector_store=vstore)

    # 対象リポ
    repos = [d for d in SRC_ROOT.iterdir() if d.is_dir()]
    if not repos:
        log.warning("解析対象のリポジトリが見つかりません（SRC_ROOT 直下にディレクトリが必要）")

    total_files = 0
    for r in repos:
        n = sum(1 for _ in file_iter(r))
        total_files += n
        log.info(f"[scan] repo={r.name:<20s} files={n}")

    if total_files == 0:
        log.warning("対象ファイルが 0 件です（拡張子/除外設定を確認）")

    # 収集・分割
    docs: List[Document] = []
    files_processed = 0
    chunks_total = 0

    for repo_dir in repos:
        repo_name = repo_dir.name
        log.info(f"[repo] {repo_name} の処理を開始します")
        for f in file_iter(repo_dir):
            stats["files_seen"] += 1
            files_processed += 1
            if files_processed % 200 == 0:
                log.info(f"[progress] files={files_processed}/{total_files} chunks={chunks_total} "
                         f"(fallback={stats['fallback_used']})")

            lang = EXT_TO_LANG.get(f.suffix.lower())
            if not lang:
                continue
            # vue 以外の通常カウントはここで
            if lang != "vue":
                inc_lang(lang)

            text = read_text_safely(f)
            if text is None or text.strip() == "":
                continue

            # ---- Vue (.vue) 特別処理 ----
            if lang == "vue":
                inc_lang("vue")
                blocks = split_vue_sfc(text)
                if not blocks:
                    # 何も抽出できない場合は通常のフォールバックに任せる（下の共通処理へ落とす）
                    pass
                else:
                    for b in blocks:
                        sub_text = b["text"]
                        start_line_base = text.count("\n", 0, b["start_idx"]) + 1

                        if b["kind"] == "script":
                            # script は JS/TS として CodeSplitter → 失敗時 SentenceSplitter
                            sub_chunks: List[str] = []
                            try:
                                sub_splitter = CodeSplitter(language=b["lang"], chunk_lines=0,
                                                            max_chars=1200, chunk_lines_overlap=0)
                                sub_chunks = sub_splitter.split_text(sub_text)
                            except Exception:
                                sub_chunks = []
                            if not sub_chunks and FALLBACK_SENTENCE:
                                try:
                                    ssplit = SentenceSplitter(chunk_size=SENTENCE_CHUNK,
                                                              chunk_overlap=SENTENCE_OVERLAP)
                                    sub_chunks = ssplit.split_text(sub_text)
                                except Exception:
                                    sub_chunks = []

                            cursor = 0
                            for ch in sub_chunks:
                                idx = sub_text.find(ch, cursor)
                                if idx < 0:
                                    idx = sub_text.find(ch)
                                local_start = sub_text.count("\n", 0, idx) + 1
                                start_line = start_line_base + (local_start - 1)
                                end_line = start_line + ch.count("\n")
                                docs.append(Document(
                                    text=ch,
                                    metadata={
                                        "repo": repo_name,
                                        "path": str(f.relative_to(SRC_ROOT)),
                                        "lang": b["lang"],            # javascript / typescript
                                        "section": "vue_script",
                                        "start_line": start_line,
                                        "end_line": end_line,
                                    }
                                ))
                            chunks_total += len(sub_chunks)
                            stats["docs_appended"] += len(sub_chunks)
                            stats["vue_script_chunks"] += len(sub_chunks)

                        elif b["kind"] == "template":
                            # template は文分割で十分
                            sub_chunks: List[str] = []
                            try:
                                ssplit = SentenceSplitter(chunk_size=SENTENCE_CHUNK,
                                                          chunk_overlap=SENTENCE_OVERLAP)
                                sub_chunks = ssplit.split_text(sub_text)
                            except Exception:
                                sub_chunks = []

                            cursor = 0
                            for ch in sub_chunks:
                                idx = sub_text.find(ch, cursor)
                                if idx < 0:
                                    idx = sub_text.find(ch)
                                local_start = sub_text.count("\n", 0, idx) + 1
                                start_line = start_line_base + (local_start - 1)
                                end_line = start_line + ch.count("\n")
                                docs.append(Document(
                                    text=ch,
                                    metadata={
                                        "repo": repo_name,
                                        "path": str(f.relative_to(SRC_ROOT)),
                                        "lang": "vue-template",
                                        "section": "vue_template",
                                        "start_line": start_line,
                                        "end_line": end_line,
                                    }
                                ))
                            chunks_total += len(sub_chunks)
                            stats["docs_appended"] += len(sub_chunks)
                            stats["vue_template_chunks"] += len(sub_chunks)

                # .vue のときはここで次のファイルへ
                continue
            # ---- /Vue 特別処理 ----

            # 通常: CodeSplitter → 失敗/空ならフォールバック
            splitter = None
            try:
                splitter = CodeSplitter(language=lang, chunk_lines=0,
                                        max_chars=1200, chunk_lines_overlap=0)
            except Exception:
                stats["splitter_init_fail"] += 1
                if LOG_LEVEL == "DEBUG":
                    log.debug(f"CodeSplitter 初期化失敗（{lang}）: {f}", exc_info=True)

            chunks: List[str] = []
            if splitter is not None:
                try:
                    chunks = splitter.split_text(text)
                except Exception:
                    stats["split_fail"] += 1
                    if LOG_LEVEL == "DEBUG":
                        log.debug(f"CodeSplitter split 失敗: {f}", exc_info=True)
                    chunks = []

            if (not chunks) and FALLBACK_SENTENCE:
                stats["empty_after_code_split"] += 1
                try:
                    ssplit = SentenceSplitter(chunk_size=SENTENCE_CHUNK,
                                              chunk_overlap=SENTENCE_OVERLAP)
                    chunks = ssplit.split_text(text)
                    stats["fallback_used"] += 1
                except Exception:
                    chunks = []

            if chunks:
                cursor = 0
                for ch in chunks:
                    start_line, end_line, cursor = compute_line_span_by_streaming(text, ch, cursor)
                    docs.append(Document(
                        text=ch,
                        metadata={
                            "repo": repo_name,
                            "path": str(f.relative_to(SRC_ROOT)),
                            "lang": lang,
                            "start_line": start_line,
                            "end_line": end_line,
                        }
                    ))
                chunks_total += len(chunks)
                stats["docs_appended"] += len(chunks)

    t_collect = time.time()
    log.info(f"[collect] files={files_processed} chunks={chunks_total} time={t_collect - t0:.1f}s")
    log.info(
        "[detail] by_lang c={by_lang_c} cpp={by_lang_cpp} java={by_lang_java} "
        "rust={by_lang_rust} js={by_lang_js} ts={by_lang_ts} go={by_lang_go} vue={by_lang_vue} | "
        "skipped_large={skipped_large} read_err={read_error} "
        "splitter_init_fail={splitter_init_fail} split_fail={split_fail} "
        "empty_after_code_split={empty_after_code_split} fallback_used={fallback_used} "
        "docs={docs_appended} | vue_script_chunks={vue_script_chunks} vue_template_chunks={vue_template_chunks}"
        .format(**stats)
    )

    # Qdrant へ投入
    if not docs:
        log.warning("投入する Document が 0 件です。終了します。")
        return

    log.info(f"[index] Qdrant への投入を開始します（{len(docs)} docs）")
    try:
        _ = VectorStoreIndex.from_documents(docs, storage_context=storage)
    except Exception:
        log.exception("VectorStoreIndex.from_documents でエラー")
        sys.exit(2)

    # 最終ポイント件数（利用可能な場合）
    try:
        cnt = qclient.count(collection_name=COLLECTION, exact=True)  # type: ignore
        count_val = getattr(cnt, "count", None)
        if count_val is not None:
            log.info(f"[qdrant] points_count={count_val}")
    except Exception:
        # バージョン差異で失敗することがあるので致命にしない
        pass

    t1 = time.time()
    log.info(f"=== Ingest done === docs={len(docs)} elapsed={t1 - t0:.1f}s")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.error("Interrupted")
        sys.exit(130)
