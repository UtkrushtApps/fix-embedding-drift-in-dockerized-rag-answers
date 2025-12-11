from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import chromadb

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Simple representation of a document to be indexed in ChromaDB."""

    id: str
    content: str
    metadata: dict


def load_documents_from_path(root: str | Path) -> List[Document]:
    """Load .md and .txt documents from the given directory tree.

    Each file becomes one document whose id is the relative path from `root`.
    """

    root_path = Path(root)
    documents: list[Document] = []

    if not root_path.exists():
        logger.warning("Documents root directory does not exist: %s", root_path)
        return documents

    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("Skipping non-text file: %s", path)
            continue

        if not text.strip():
            logger.debug("Skipping empty document: %s", path)
            continue

        rel_id = str(path.relative_to(root_path))
        documents.append(
            Document(
                id=rel_id,
                content=text,
                metadata={
                    "source_path": str(path),
                    "file_name": path.name,
                },
            )
        )

    logger.info("Loaded %d documents from %s", len(documents), root_path)
    return documents


def _delete_collection_if_exists(client: chromadb.HttpClient, name: str) -> None:
    """Delete the collection if it already exists (idempotent)."""

    for collection in client.list_collections():
        # HTTP client returns simple objects with a `name` attribute
        if getattr(collection, "name", None) == name:
            logger.info("Deleting existing Chroma collection '%s'", name)
            client.delete_collection(name=name)
            return


def rebuild_collection(
    client: chromadb.HttpClient,
    collection_name: str,
    embedding_function,
    embedding_config_id: str,
    embedding_model_name: str,
    documents: Iterable[Document],
    batch_size: int = 32,
):
    """Drop and rebuild a Chroma collection with fresh embeddings.

    This is used both by the one-time initialization service and by any
    administrative maintenance flows that need to refresh embeddings.
    """

    docs = list(documents)
    if not docs:
        logger.warning("No documents to ingest. Collection '%s' will be empty.", collection_name)

    _delete_collection_if_exists(client, collection_name)

    logger.info(
        "Creating collection '%s' with embedding_config_id=%s, model=%s",
        collection_name,
        embedding_config_id,
        embedding_model_name,
    )

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "embedding_config_id": embedding_config_id,
            "embedding_model_name": embedding_model_name,
        },
        embedding_function=embedding_function,
    )

    # Ingest in batches for better memory behaviour
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        logger.debug(
            "Ingesting batch %d-%d into collection '%s'",
            start,
            start + len(batch) - 1,
            collection_name,
        )
        collection.add(
            ids=[d.id for d in batch],
            documents=[d.content for d in batch],
            metadatas=[d.metadata for d in batch],
        )

    logger.info(
        "Finished ingesting %d documents into collection '%s'",
        len(docs),
        collection_name,
    )
    return collection
