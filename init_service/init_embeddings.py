from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError

from common.config import Settings
from common.embedding import EmbeddingConfig, build_embedding_function, normalise_model_name
from common.ingest import load_documents_from_path, rebuild_collection
from common.logging_config import configure_logging


logger = logging.getLogger(__name__)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def main() -> int:
    settings = Settings()
    configure_logging(settings.log_level, service_name="init-service")

    logger.info("Starting initialization service")

    embedding_model = normalise_model_name(settings.embedding_model_name)
    embedding_config = EmbeddingConfig(
        model_name=embedding_model,
        version=settings.embedding_config_version,
    )
    embedding_fn = build_embedding_function(settings.embedding_model_name)

    logger.info(
        "Using embedding configuration: model=%s, config_id=%s",
        embedding_model,
        embedding_config.config_id,
    )

    try:
        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to create Chroma HTTP client: %s", exc)
        return 1

    rebuild_on_mismatch = _bool_env("REBUILD_ON_EMBEDDING_MISMATCH", default=True)

    try:
        try:
            collection = client.get_collection(name=settings.chroma_collection)
            metadata = collection.metadata or {}
            stored_config_id = metadata.get("embedding_config_id")

            if stored_config_id == embedding_config.config_id:
                logger.info(
                    "Existing collection '%s' already has matching embeddings (config_id=%s). "
                    "Skipping rebuild.",
                    settings.chroma_collection,
                    stored_config_id,
                )
                return 0

            logger.warning(
                "Embedding config mismatch for collection '%s': stored_config_id=%s, current_config_id=%s",
                settings.chroma_collection,
                stored_config_id,
                embedding_config.config_id,
            )

            if not rebuild_on_mismatch:
                logger.error(
                    "REBUILD_ON_EMBEDDING_MISMATCH is false; refusing to delete the collection. "
                    "Please delete or rebuild the collection manually."
                )
                return 1

            logger.info(
                "Rebuilding collection '%s' due to embedding drift...",
                settings.chroma_collection,
            )

        except NotFoundError:
            logger.info(
                "Chroma collection '%s' does not exist. It will be created and populated.",
                settings.chroma_collection,
            )

        docs_root = Path("/app/data/docs")
        documents = load_documents_from_path(docs_root)

        if not documents:
            logger.warning(
                "No documents found under %s. Initialization will complete, but queries will return no results.",
                docs_root,
            )

        rebuild_collection(
            client=client,
            collection_name=settings.chroma_collection,
            embedding_function=embedding_fn,
            embedding_config_id=embedding_config.config_id,
            embedding_model_name=embedding_model,
            documents=documents,
        )

        logger.info("Initialization complete")
        return 0

    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Initialization failed: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
