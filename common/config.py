from __future__ import annotations

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Shared between the FastAPI app and the initialization service so that
    embedding configuration and Chroma connection details are consistent.
    """

    chroma_host: str = Field("chromadb", env="CHROMA_HOST")
    chroma_port: int = Field(8000, env="CHROMA_PORT")
    chroma_collection: str = Field("utkrusht_docs", env="CHROMA_COLLECTION")

    # If empty or "default", we use Chroma's DefaultEmbeddingFunction.
    # If set to a sentence-transformers model name, we use that model.
    embedding_model_name: str | None = Field(None, env="EMBEDDING_MODEL_NAME")

    # Optional human-readable version string operators can bump when they
    # intentionally change embedding behaviour.
    embedding_config_version: str = Field("v1", env="EMBEDDING_CONFIG_VERSION")

    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def chroma_url(self) -> str:
        return f"http://{self.chroma_host}:{self.chroma_port}"
