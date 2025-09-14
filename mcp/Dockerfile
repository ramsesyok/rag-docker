# 余計なもの不要。GPUも不要。
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir "mcp[cli]>=1.14.0" httpx

COPY server.py /app/server.py

ENV RAG_BASE=http://rag:8001
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=5173

EXPOSE 5173
CMD ["python", "/app/server.py"]
