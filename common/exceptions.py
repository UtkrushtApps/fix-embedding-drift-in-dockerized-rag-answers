class EmbeddingConfigMismatchError(RuntimeError):
    """Raised when the configured embedding differs from the stored collection.

    This indicates that document vectors are out of sync with the embedding
    configuration used for queries.
    """

    pass


class ChromaConnectionError(RuntimeError):
    """Raised when the application cannot talk to the ChromaDB HTTP server."""

    pass
