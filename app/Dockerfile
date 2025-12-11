FROM python:3.11-slim

WORKDIR /app

# Install basic OS dependencies (curl for health checks / debugging)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy shared code and application code
COPY common ./common
COPY app ./app

ENV PYTHONPATH=/app

# Install Python dependencies
RUN pip install --no-cache-dir \
    "fastapi" \
    "uvicorn[standard]" \
    "chromadb" \
    "pydantic<2.0.0"

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
