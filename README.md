# RAG Docker

## 概要
- Docker 上で動く RAG チャット環境。Chainlit のチャット UI と検索 API（`/code_search`・`/query`）を提供します。
- ベクタ DB は Qdrant。リポジトリ内のソースコードをインデックス（ingest）できます。
- LLM は Ollama を利用（ローカルの Ollama に接続）。

## 必要条件
- Docker Desktop（または互換環境）
- Ollama のインストールとモデル取得（例: `ollama pull llama3:13b`）

## クイックスタート
- `.env.example` をコピーして `.env` を作成し、必要に応じて編集:
  - `OLLAMA_BASE_URL=http://host.docker.internal:11434`（Mac/Windows）
    - Linux はホストの IP に変更（例: `http://192.168.x.x:11434`）
  - `OLLAMA_MODEL=llama3:13b`
  - `EMBED_MODEL=BAAI/bge-m3`
  - それ以外は既定値のままで OK（例: `QDRANT_URL=http://qdrant:6333`）
- 対象のソースコードを `src_repos/` に配置（プロジェクト単位のフォルダで OK）
  - 例: `src_repos/myapp/` に配置
- 起動: `docker compose up -d`
- 初回はインデックス作成が走ります（多少時間がかかります）
- チャット UI: `http://localhost:8000`
- API: `http://localhost:8001`

## よく使う操作
- インデックスを強制再作成:
  - `.env` の `FORCE_REINDEX=1` を有効化 → `docker compose restart rag`
- 通常の再起動:
  - ソース更新後 `docker compose restart rag`
- ログ確認:
  - `docker compose logs -f rag`
- Qdrant のデータ永続化:
  - Docker ボリューム `qdrant_data` に保存（`docker compose down -v` で削除）

## API
- `/code_search`（POST）: コード検索
  - URL: `http://localhost:8001/code_search`
  - 例(JSON):
    - `{ "query": "TODO", "kind": "literal", "case_sensitive": true, "max_results": 50 }`
- `/query`（POST）: RAG + コンテキスト + LLM による応答
  - URL: `http://localhost:8001/query`
  - 例(JSON):
    - `{ "question": "設定ファイルの読み込み場所はどこ？", "top_k": 6 }`

## トラブルシューティング
- チャット UI が起動しない / 応答しない:
  - `docker compose logs -f rag` でインデックス完了やエラーを確認
- Ollama に接続できない:
  - `OLLAMA_BASE_URL` を確認（Mac/Windows は `host.docker.internal`、Linux はホスト IP）
  - 必要モデルを取得（例: `ollama pull llama3:13b`）
- 検索でヒットしない:
  - `src_repos/` 配下に対象コードがあるか、対象が正しいか確認

## データのリセット
- Qdrant のデータを含めて完全削除:
  - `docker compose down -v`
  - 必要に応じて `docker compose up -d` で再構築


# チャットで使う「コピペ用テンプレ」

## 1) バグ原因候補の列挙（根拠必須）

```
【目的】<例：NPEの原因候補をしぼる>
【症状/手掛かり】<例外文・ログ行・関数名・再現条件>
【範囲】<repo名/フォルダ/言語（任意）>
【出力フォーマット】
- 観測：<何が起きているか1文>
- 候補1：<概要>（<file>:Lx–Ly）
  根拠：<抜粋1〜2行>
- 候補2：…
※ 根拠が足りない場合は「不明」と明記
```

## 2) 影響範囲（改修計画の前段）

```
【目的】<型/メソッドの変更に伴う影響の把握>
【変更案】<例：UserIdをUUIDへ>
【範囲】<repoやディレクトリ>
【出力】依存のある箇所をカテゴリ別に：
- 型定義/エンティティ：<file:line>
- シリアライズ/DB：<file:line>
- API/コントローラ：<file:line>
- テスト：<file:line>
最後に「影響大の順に上位3」を要約
```

## 3) APIドキュメント（ドラフト）

```
【目的】<例：PaymentClientのAPIドキュメント下書き>
【対象】<クラス名/パッケージ/モジュール>
【出力】Markdown
- 役割（1段落）
- メソッド一覧（シグネチャ / パラメータ / 例外 / 戻り値）
- 使用例（最小コード）
- 注意点（スレッド安全性/エラー時の動作）
各項目に根拠ファイルと行番号を括弧で併記
```

## 4) 設定/ビルド起因の不具合の切り分け

```
【目的】<ビルド/設定の問題らしいバグの切り分け>
【症状】<例：releaseビルドのみ失敗>
【環境＋フラグ】<例：-O2、feature X ON>
【出力】仮説→検証順に箇条書き：
1) コンパイルフラグに敏感な箇所（<file:line>）…根拠抜粋
2) feature X で条件分岐している箇所（…）
3) 非初期化/未定義動作の懸念（…）
```


## 備考
- UI は Chainlit（ポート `8000`）、API は FastAPI（ポート `8001`）。
- 埋め込みモデル（`EMBED_MODEL`）はインデックス（ingest）と推論（app）で一致させる必要があります。
- Continue.dev 連携用 MCP サーバー（ポート `5173`）が含まれます。未使用の場合は無視してかまいません。
