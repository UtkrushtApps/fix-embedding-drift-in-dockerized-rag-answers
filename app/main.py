from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException, status

from common.config import Settings
from common.exceptions import ChromaConnectionError, EmbeddingConfigMismatchError
from common.logging_config import configure_logging
from app.rag_service import RAGService
from app.schemas import EmbeddingStatus, HealthStatus, QueryRequest, QueryResponse, RetrievedDocument


def get_settings() -> Settings:
    return Settings()


def get_rag_service(settings: Settings = Depends(get_settings)) -> RAGService:
    # In a real production app you'd likely manage this as a singleton or with
    # FastAPI's lifespan context. For this assessment, constructing it via a
    # dependency is sufficient and keeps the code easy to follow.
    return RAGService(settings)


settings = Settings()
configure_logging(settings.log_level, service_name="rag-app")
logger = logging.getLogger(__name__)

app = FastAPI(title="Utkrusht Documentation RAG Service", version="1.0.0")


@app.on_event("startup")
async def startup_event() -> None:
    # Warm up connection and detect any embedding drift as early as possible.
    rag = RAGService(settings)
    try:
        rag.refresh_state()
    except ChromaConnectionError:
        logger.warning(
            "ChromaDB is not reachable at startup. Health endpoint will report this until it recovers."
        )


@app.get("/health", response_model=HealthStatus)
async def health(rag: RAGService = Depends(get_rag_service)) -> HealthStatus:
    """Basic health endpoint including Chroma connectivity and embedding drift."""

    details = {}

    try:
        rag.refresh_state()
    except ChromaConnectionError as exc:
        logger.error("Health check: Chroma connection error: %s", exc)

    chroma_ok = rag.chroma_connected and rag.check_chroma_health()
    details.update(rag.get_embedding_status())

    status_str = "ok" if chroma_ok and not rag.embedding_drift_detected else "degraded"

    return HealthStatus(
        status=status_str,
        chroma_connected=chroma_ok,
        embedding_drift_detected=rag.embedding_drift_detected,
        details=details,
    )


@app.get("/chroma-health")
async def chroma_health(rag: RAGService = Depends(get_rag_service)) -> dict:
    """Check direct connectivity to ChromaDB over the Docker network."""

    ok = rag.check_chroma_health()
    return {"chroma_healthy": ok}


@app.get("/admin/embedding-status", response_model=EmbeddingStatus)
async def embedding_status(rag: RAGService = Depends(get_rag_service)) -> EmbeddingStatus:
    """Return current embedding configuration and drift status for observability."""

    rag.refresh_state()
    status_dict = rag.get_embedding_status()
    return EmbeddingStatus(**status_dict)


@app.post("/query", response_model=QueryResponse)
async def query_docs(
    request: QueryRequest,
    rag: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """Semantic search over Utkrusht documentation stored in ChromaDB.

    This endpoint only performs retrieval. It returns the most relevant
    documentation snippets; a separate generation layer can consume these.
    """

    try:
        result = rag.query(request.question, k=request.k)
    except EmbeddingConfigMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Embedding configuration mismatch detected. "
                "Please trigger a reindex via the initialization service before querying."
            ),
        ) from exc
    except ChromaConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ChromaDB is not reachable. Please try again later.",
        ) from exc

    ids = result["ids"]
    distances = result["distances"]
    metadatas = result["metadatas"]
    documents = result["documents"]

    # Convert Chroma distances (smaller is closer) into simple similarity
    # scores in (0, 1]. This is purely for client convenience.
    def distance_to_score(d: float) -> float:
        return 1.0 / (1.0 + float(d))

    retrieved = [
        RetrievedDocument(
            id=str(doc_id),
            score=distance_to_score(dist),
            metadata=meta or {},
            content=doc or "",
        )
        for doc_id, dist, meta, doc in zip(ids, distances, metadatas, documents)
    ]

    return QueryResponse(question=request.question, results=retrieved)
