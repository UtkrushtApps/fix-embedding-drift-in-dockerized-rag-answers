from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Query payload for the RAG endpoint."""

    question: str = Field(..., description="Natural language question")
    k: int = Field(3, ge=1, le=20, description="Number of retrieval results")


class RetrievedDocument(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]
    content: str


class QueryResponse(BaseModel):
    question: str
    results: List[RetrievedDocument]


class HealthStatus(BaseModel):
    status: str
    chroma_connected: bool
    embedding_drift_detected: bool
    details: Optional[Dict[str, Any]] = None


class EmbeddingStatus(BaseModel):
    collection_name: str
    current_embedding_model: str
    current_embedding_config_id: str
    collection_embedding_config_id: Optional[str]
    drift_detected: bool
