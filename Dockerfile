FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential ca-certificates wget ripgrep \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ingest.py app.py entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000 8001
ENTRYPOINT ["/app/entrypoint.sh"]
