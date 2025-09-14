
## tree
```
rag-docker/
├─ docker-compose.yml
├─ Dockerfile
├─ entrypoint.sh
├─ requirements.txt
├─ .env.example          # ← コピーして .env を作成
├─ ingest.py             # 既存のAST分割→Qdrant投入スクリプト
├─ app.py                # 既存のChainlit。/code_searchパッチ適用済み版
├─ mcp/     
|    ├─ Dockerfile
|    ├─ requirements.txt
|    └─ server.py        #
└─ src_repos/            # 解析したいリポジトリをここに置く
```

## Continue.devの設定
```json
{
  "mcpServers": {
    "rag-mcp": {
      "transport": "sse",
      "url": "http://localhost:5173/sse"
    }
  }
}
```
