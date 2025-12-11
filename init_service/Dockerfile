FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy shared modules and init service code
COPY common ./common
COPY init_service ./init_service

ENV PYTHONPATH=/app

RUN pip install --no-cache-dir \
    "chromadb" \
    "pydantic<2.0.0"

CMD ["python", "-m", "init_service.init_embeddings"]
