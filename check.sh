# コンテナからの詳細ログ
docker compose exec rag bash -lc 'set -x; echo $OLLAMA_BASE_URL; curl -v ${OLLAMA_BASE_URL}/api/tags | head -n1'

# Windows 側のポート/プロセス確認
netstat -aon | grep 11434

# docker-compose.yml の rag サービスの extra_hosts と environment の該当箇所

# プロキシ環境変数の有無（コンテナ内）
docker compose exec rag env | egrep -i "http_proxy|https_proxy|no_proxy"

