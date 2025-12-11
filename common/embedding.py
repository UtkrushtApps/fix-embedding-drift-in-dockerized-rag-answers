from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional

from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Logical description of the embedding configuration.

    This is converted into a stable `config_id` string which we store in the
    Chroma collection's metadata. If the configuration changes (for example,
    because the model name changes), the `config_id` will change, and we can
    detect embedding drift.
    """

    model_name: str  # "default" means DefaultEmbeddingFunction
    implementation: str = "sentence-transformers"
    version: str = "v1"  # optional, mainly for operators/observability

    @property
    def config_id(self) -> str:
        """Stable identifier for this embedding configuration.

        We hash a JSON representation so that any field changes lead to a
        different identifier.
        """

        payload = json.dumps(
            {
                "model_name": self.model_name,
                "implementation": self.implementation,
                "version": self.version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def normalise_model_name(raw_model_name: Optional[str]) -> str:
    """Normalise the model name from settings into a canonical form.

    If `None` or empty, we treat it as "default", which means
    `DefaultEmbeddingFunction` and its default sentence-transformers model.
    """

    if not raw_model_name:
        return "default"
    return raw_model_name.strip()


def build_embedding_function(raw_model_name: Optional[str]):
    """Instantiate the embedding function for the configured model.

    - If the model name is empty or "default", use Chroma's
      `DefaultEmbeddingFunction`.
    - Otherwise, use a specific `SentenceTransformerEmbeddingFunction`.
    """

    model_name = normalise_model_name(raw_model_name)

    if model_name == "default":
        return DefaultEmbeddingFunction()

    return SentenceTransformerEmbeddingFunction(model_name=model_name)
