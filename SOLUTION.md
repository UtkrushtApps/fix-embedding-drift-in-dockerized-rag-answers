# Solution Steps

1. Create a shared configuration module so all services use the same settings.
- Add `common/config.py` with a `Settings` class based on `pydantic.BaseSettings`.
- Include Chroma connection info (`CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_COLLECTION`).
- Include embedding settings (`EMBEDDING_MODEL_NAME`, `EMBEDDING_CONFIG_VERSION`) and a `log_level`.
- Add a convenience `chroma_url` property for logging/diagnostics.

2. Introduce a reusable embedding configuration abstraction.
- Add `common/embedding.py`.
- Implement `normalise_model_name(raw_model_name)` that returns a canonical name (`"default"` when unset).
- Create an `EmbeddingConfig` dataclass holding `model_name`, `implementation` and `version`.
- Implement `EmbeddingConfig.config_id` that hashes a JSON payload of these fields into a short, stable identifier.
- Implement `build_embedding_function(raw_model_name)`:
  - If model is empty or `"default"`, return `DefaultEmbeddingFunction()`.
  - Otherwise return `SentenceTransformerEmbeddingFunction(model_name=model)`.

3. Add shared exception types for clearer error handling.
- Create `common/exceptions.py` with two custom exceptions:
  - `EmbeddingConfigMismatchError` for detected embedding drift.
  - `ChromaConnectionError` for connectivity problems with ChromaDB.
- Use these instead of generic exceptions so the API can respond with appropriate HTTP status codes.

4. Create a shared ingestion utility that can be used by both the init service and any future admin rebuild flows.
- Add `common/ingest.py`.
- Define a `Document` dataclass with `id`, `content`, and `metadata`.
- Implement `load_documents_from_path(root)` that:
  - Walks the directory recursively.
  - Loads `.md` and `.txt` files as UTF-8.
  - Skips non-text or empty files.
  - Uses the path relative to `root` as the document id and stores basic metadata (file path, file name).
- Implement `_delete_collection_if_exists(client, name)` to delete a Chroma collection safely if it exists.
- Implement `rebuild_collection(...)` that:
  - Deletes any existing collection with the given name.
  - Creates a new collection with metadata `embedding_config_id` and `embedding_model_name`.
  - Ingests all documents in batches using `collection.add`.

5. Set up a shared logging configuration for consistent observability across services.
- Add `common/logging_config.py` with `configure_logging(level, service_name)`.
- Configure a console handler with a formatter that logs time, level, logger name, service name, and message.
- Use `dictConfig` to ensure predictable logging in containers.
- Call `configure_logging` from both the FastAPI app and the init service.

6. Implement the initialization / ingestion service with embedding-drift aware rebuild logic.
- Create `init_service/init_embeddings.py`.
- Load settings via `Settings()` and configure logging with service name `"init-service"`.
- Build `EmbeddingConfig` from the normalised model name and version, and instantiate the embedding function via `build_embedding_function`.
- Create a Chroma `HttpClient` using `CHROMA_HOST` and `CHROMA_PORT`.
- Read `REBUILD_ON_EMBEDDING_MISMATCH` from the environment (default `true`).
- Try to `get_collection(name=CHROMA_COLLECTION)`:
  - If found, inspect `collection.metadata.get("embedding_config_id")`.
    - If it matches the current `EmbeddingConfig.config_id`, log and exit successfully (no rebuild needed).
    - If it differs or is missing:
      - If `REBUILD_ON_EMBEDDING_MISMATCH` is false, log an error and exit with status 1.
      - If true, log that you're rebuilding due to embedding drift.
  - If `NotFoundError` is raised, log that the collection does not yet exist and will be created.
- Load documents from `/app/data/docs` using `load_documents_from_path`.
- Call `rebuild_collection` with the client, collection name, embedding function, `embedding_config_id`, and documents.
- Exit with status 0 on success, 1 on error.

7. Containerize the initialization service so it can run as a one-shot job.
- Create `init_service/Dockerfile` based on `python:3.11-slim`.
- Install required packages (`chromadb`, `pydantic<2`).
- Copy `common/` and `init_service/` into the image and set `PYTHONPATH=/app`.
- Configure the default command as `python -m init_service.init_embeddings`.
- This Docker image will be used by docker-compose as a one-time init / rebuild task.

8. Implement the RAG service core with explicit embedding drift detection.
- Create `app/rag_service.py` and define a `RAGService` class.
- In `__init__`:
  - Store `settings`.
  - Build an `EmbeddingConfig` from the normalized embedding model and version.
  - Create the embedding function with `build_embedding_function`.
  - Instantiate a `chromadb.HttpClient` with the configured host/port.
  - Initialize internal state: `_collection`, `_collection_embedding_config_id`, `_embedding_drift_detected`, `_chroma_connected`.
- Implement `refresh_state()`:
  - Attempt to `get_collection(name=CHROMA_COLLECTION, embedding_function=...)`.
    - If `NotFoundError`, mark Chroma as connected, but with no collection and no drift (queries will just return empty results), and log a warning.
    - If the collection is found:
      - Read `metadata["embedding_config_id"]`.
      - If missing, mark `_embedding_drift_detected=True`, store the collection, and log an error indicating that a rebuild is required.
      - If present but different from `EmbeddingConfig.config_id`, mark `_embedding_drift_detected=True` and log details of the mismatch.
      - If equal, mark `_embedding_drift_detected=False` and log that embeddings are aligned.
  - On any unexpected error, mark `_chroma_connected=False`, clear the collection, log the exception, and raise `ChromaConnectionError`.
- Implement simple getters: `chroma_connected`, `embedding_drift_detected`, `is_ready`, and `get_embedding_status()` (returning a dict with collection name, current model, current config id, stored config id, and drift flag).
- Add `check_chroma_health()` which calls `client.heartbeat()` when available or falls back to `list_collections()`, returning a boolean.
- Implement `query(question, k)`:
  - If `is_ready` is false, raise `EmbeddingConfigMismatchError`.
  - If `_collection` is `None`, log and return an empty result set.
  - Otherwise, call `collection.query(query_texts=[question], n_results=k, include=["metadatas", "documents", "distances"])` and normalise the result into a simple dict.

9. Implement the FastAPI application that wraps the RAG service and exposes health and query endpoints.
- Create `app/schemas.py` with Pydantic models:
  - `QueryRequest` (fields: `question`, `k`).
  - `RetrievedDocument` (id, score, metadata, content).
  - `QueryResponse` (question, list of `RetrievedDocument`).
  - `HealthStatus` (status, chroma_connected, embedding_drift_detected, details dict).
  - `EmbeddingStatus` (collection_name, current_embedding_model, current_embedding_config_id, collection_embedding_config_id, drift_detected).
- Create `app/main.py`:
  - Load global `Settings()` and call `configure_logging(settings.log_level, service_name="rag-app")`.
  - Define dependency functions `get_settings()` and `get_rag_service(settings)` that construct a `RAGService`.
  - Instantiate `FastAPI` with a suitable title.
  - In a `startup` event handler, create a `RAGService` and call `refresh_state()`, logging but not crashing if Chroma is unavailable.
  - Implement `GET /health`:
    - Call `rag.refresh_state()`.
    - Call `rag.check_chroma_health()`.
    - Build a `HealthStatus` with `status="ok"` when Chroma is reachable and no drift is detected, otherwise `status="degraded"`.
  - Implement `GET /chroma-health` returning `{ "chroma_healthy": bool }` via `rag.check_chroma_health()`.
  - Implement `GET /admin/embedding-status` returning an `EmbeddingStatus` built from `rag.get_embedding_status()`.
  - Implement `POST /query` (body: `QueryRequest`, response: `QueryResponse`):
    - Call `rag.query(question, k)`.
    - Catch `EmbeddingConfigMismatchError` and respond with HTTP 503 and a message instructing operators to run the init service to rebuild.
    - Catch `ChromaConnectionError` and respond with HTTP 503 and a generic Chroma unavailable message.
    - Convert Chroma distances to a simple similarity score (e.g. `1 / (1 + distance)`) and map the raw results into `RetrievedDocument` models.

10. Containerize the FastAPI RAG application and wire everything together with docker-compose.
- Create `app/Dockerfile` based on `python:3.11-slim`:
  - Install `curl` for diagnostics.
  - Copy `common/` and `app/` into `/app` and set `PYTHONPATH=/app`.
  - Install `fastapi`, `uvicorn[standard]`, `chromadb`, and `pydantic<2`.
  - Expose port 8080 and set the command to `uvicorn app.main:app --host 0.0.0.0 --port 8080`.
- Create `docker-compose.yml` in the repo root:
  - Define the `chromadb` service using the official `chromadb/chroma` image.
    - Mount a named volume `chroma-data` to `/data` for persistence.
    - Set `CHROMA_DB_IMPL=duckdb+parquet` and `PERSIST_DIRECTORY=/data`.
    - Map host port 8001 to container port 8000.
    - Add a `curl`-based healthcheck hitting `http://localhost:8000/api/v1/collections`.
  - Define the `app` service:
    - Build from context `.` with `app/Dockerfile`.
    - Set environment variables for Chroma host (`chromadb`), port (`8000`), collection name, embedding model (blank or `default` by default), embedding config version, and log level.
    - Depend on `chromadb` with `condition: service_healthy` and expose port 8080.
  - Define the `init-service` one-shot ingestion container:
    - Build from context `.` with `init_service/Dockerfile`.
    - Mount the documentation directory `./data/docs` into `/app/data/docs:ro`.
    - Use the same Chroma and embedding-related environment variables as `app`.
    - Use `REBUILD_ON_EMBEDDING_MISMATCH=true` by default.
    - Depend on `chromadb` health and run the module `init_service.init_embeddings` once (no restart).

11. Validate the end-to-end behaviour and embedding drift protection.
- Bring up ChromaDB and the app: `docker compose up chromadb app`.
- Run the init service once to ingest docs: `docker compose run --rm init-service`.
- Call `GET http://localhost:8080/health` and verify `status=ok`, `embedding_drift_detected=false`.
- Call `POST http://localhost:8080/query` with a typical Utkrusht FAQ question and verify relevant snippets are returned.
- Simulate an embedding change by updating `EMBEDDING_MODEL_NAME` (and optionally `EMBEDDING_CONFIG_VERSION`) in the environment and restarting only the app (not the init-service).
  - On startup, `refresh_state()` will detect that `embedding_config_id` in the collection metadata does not match the new configuration and set `embedding_drift_detected=true`.
  - `GET /health` should now report `status=degraded` and `embedding_drift_detected=true`.
  - `POST /query` should return HTTP 503 with a message instructing you to rebuild.
- Run `docker compose run --rm init-service` again with the new embedding config.
  - The init service will detect the mismatch, delete and rebuild the collection with fresh vectors and updated metadata.
- Call `GET /health` again; it should now show `embedding_drift_detected=false`, and queries should work with embeddings that match the current configuration.

