from __future__ import annotations

import logging
from typing import List, Optional

import chromadb
from chromadb.errors import NotFoundError

from common.config import Settings
from common.embedding import EmbeddingConfig, build_embedding_function, normalise_model_name
from common.exceptions import ChromaConnectionError, EmbeddingConfigMismatchError

logger = logging.getLogger(__name__)


class RAGService:
    """Encapsulates the retrieval pipeline and embedding drift checks."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embedding_config = EmbeddingConfig(
            model_name=normalise_model_name(settings.embedding_model_name),
            version=settings.embedding_config_version,
        )
        self._embedding_function = build_embedding_function(settings.embedding_model_name)

        self._client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )

        self._collection: Optional[chromadb.api.models.Collection.Collection] = None
        self._collection_embedding_config_id: Optional[str] = None
        self._embedding_drift_detected: bool = False
        self._chroma_connected: bool = False

    # ------------------------------------------------------------------
    # State management / health
    # ------------------------------------------------------------------
    def refresh_state(self) -> None:
        """Refresh connection status and embedding alignment from ChromaDB.

        This method is safe to call repeatedly (e.g. in health checks). It
        will not create or modify collections; it only inspects the current
        state and updates in-memory flags.
        """

        logger.debug("Refreshing RAG service state from ChromaDB")

        try:
            # Attach to the collection if it exists, with the correct
            # embedding function for queries.
            try:
                collection = self._client.get_collection(
                    name=self._settings.chroma_collection,
                    embedding_function=self._embedding_function,
                )
            except NotFoundError:
                logger.warning(
                    "Chroma collection '%s' not found. RAG queries will return no results until the init service ingests documents.",
                    self._settings.chroma_collection,
                )
                self._collection = None
                self._collection_embedding_config_id = None
                self._embedding_drift_detected = False
                self._chroma_connected = True
                return

            metadata = collection.metadata or {}
            stored_config_id = metadata.get("embedding_config_id")

            if stored_config_id is None:
                # Existing collection with no embedding metadata – treat as
                # potential drift and require a rebuild to be safe.
                logger.error(
                    "Chroma collection '%s' has no 'embedding_config_id' metadata. "
                    "Embedding drift cannot be ruled out; please rebuild the collection via the init service.",
                    self._settings.chroma_collection,
                )
                self._collection = collection
                self._collection_embedding_config_id = None
                self._embedding_drift_detected = True
            else:
                drift = stored_config_id != self._embedding_config.config_id
                if drift:
                    logger.error(
                        "Embedding configuration mismatch detected for collection '%s'. "
                        "stored_config_id=%s, current_config_id=%s. "
                        "Refuse to answer queries until the collection is rebuilt.",
                        self._settings.chroma_collection,
                        stored_config_id,
                        self._embedding_config.config_id,
                    )
                else:
                    logger.info(
                        "Embedding configuration is aligned for collection '%s' (config_id=%s)",
                        self._settings.chroma_collection,
                        stored_config_id,
                    )

                self._collection = collection
                self._collection_embedding_config_id = stored_config_id
                self._embedding_drift_detected = drift

            self._chroma_connected = True

        except Exception as exc:  # broad catch to mark connectivity issues
            self._collection = None
            self._collection_embedding_config_id = None
            self._embedding_drift_detected = False
            self._chroma_connected = False
            logger.exception("Failed to refresh state from ChromaDB: %s", exc)
            raise ChromaConnectionError("Unable to connect to ChromaDB") from exc

    @property
    def chroma_connected(self) -> bool:
        return self._chroma_connected

    @property
    def embedding_drift_detected(self) -> bool:
        return self._embedding_drift_detected

    @property
    def is_ready(self) -> bool:
        """Whether the service is safe to answer RAG queries."""

        return self._chroma_connected and not self._embedding_drift_detected

    def get_embedding_status(self) -> dict:
        """Return a structured view of embedding alignment state."""

        return {
            "collection_name": self._settings.chroma_collection,
            "current_embedding_model": normalise_model_name(
                self._settings.embedding_model_name
            ),
            "current_embedding_config_id": self._embedding_config.config_id,
            "collection_embedding_config_id": self._collection_embedding_config_id,
            "drift_detected": self._embedding_drift_detected,
        }

    def check_chroma_health(self) -> bool:
        """Perform a lightweight health check against ChromaDB.

        Uses the client's built-in `heartbeat` endpoint if available, or falls
        back to listing collections.
        """

        try:
            # Newer versions of chromadb expose heartbeat on HttpClient
            if hasattr(self._client, "heartbeat"):
                self._client.heartbeat()
            else:
                self._client.list_collections()
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Chroma health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------
    def query(self, question: str, k: int = 3):
        """Run a semantic search against the documentation collection.

        This method only performs retrieval; generation can be layered on top
        by other components if desired.
        """

        if not self.is_ready:
            raise EmbeddingConfigMismatchError(
                "RAG service is not ready: either Chroma is unavailable or embedding drift was detected."
            )

        if self._collection is None:
            # No collection yet (e.g. init service hasn't run). Return empty
            # results rather than error.
            logger.warning(
                "Query requested but collection '%s' does not exist or is empty.",
                self._settings.chroma_collection,
            )
            return {
                "question": question,
                "ids": [],
                "distances": [],
                "metadatas": [],
                "documents": [],
            }

        logger.info("Running semantic search for question: %s", question)

        results = self._collection.query(
            query_texts=[question],
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )

        # Chroma returns lists per query; we only send one query at a time.
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]

        return {
            "question": question,
            "ids": ids,
            "distances": distances,
            "metadatas": metadatas,
            "documents": documents,
        }
