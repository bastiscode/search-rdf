"""Type stubs for search_index module."""

from typing import Iterator, final

@final
class TextData:
    """Text data source backed by a TSV file."""

    @staticmethod
    def build(tsv_file: str, data_dir: str) -> None:
        """
        Build text data from a TSV file.

        Args:
            tsv_file: Path to input TSV file (first column is identifier)
            data_dir: Output directory for the text data
        """
        ...

    @staticmethod
    def load(data_dir: str) -> TextData:
        """
        Load text data from a directory.

        Args:
            data_dir: Directory containing the text data

        Returns:
            Loaded TextData instance
        """
        ...

    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...
    def __iter__(self) -> Iterator[list[str]]: ...
    def num_fields(self, id: int) -> int | None: ...
    def field(self, id: int, field: int) -> str | None: ...
    def fields(self, id: int) -> list[str] | None: ...
    def identifier(self, id: int) -> str | None: ...
    def id_from_identifier(self, identifier: str) -> int | None: ...

@final
class TextEmbeddings:
    """Text embeddings with associated text data."""

    @staticmethod
    def build(tsv_file: str, embeddings_file: str, data_dir: str) -> None:
        """
        Build text embeddings from a TSV file and embeddings file.

        Args:
            tsv_file: Path to input TSV file (first column is identifier)
            embeddings_file: Path to embeddings file (safetensors format)
            data_dir: Output directory for the text embeddings data
        """
        ...

    @staticmethod
    def load(data_dir: str) -> TextEmbeddings:
        """
        Load text embeddings from a directory.

        Args:
            data_dir: Directory containing the text embeddings data

        Returns:
            Loaded TextEmbeddings instance
        """
        ...

    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...
    def num_dimensions(self) -> int: ...
    def precision(self) -> str: ...
    def model(self) -> str: ...
    def fields_per_data_point(self, data_id: int) -> int | None: ...
    def identifier(self, id: int) -> str | None: ...
    def id_from_identifier(self, identifier: str) -> int | None: ...
    def text_data(self) -> TextData:
        """
        Get the underlying text data.

        Returns:
            TextData instance
        """
        ...

    def get_embedding(self, field_id: int) -> list[float] | bytes | None:
        """
        Get an embedding by field ID.

        Args:
            field_id: Field ID

        Returns:
            Embedding as list of floats (for float32) or bytes (for binary),
            or None if not found
        """
        ...

@final
class KeywordIndex:
    """Keyword search index for text data."""

    @staticmethod
    def build(data: TextData, index_dir: str) -> None:
        """
        Build a keyword search index.

        Args:
            data: TextData to index
            index_dir: Output directory for the index
        """
        ...

    @staticmethod
    def load(data: TextData, index_dir: str) -> KeywordIndex:
        """
        Load a keyword search index.

        Args:
            data: TextData associated with the index
            index_dir: Directory containing the index

        Returns:
            Loaded KeywordIndex instance
        """
        ...

    def search(
        self,
        query: str,
        k: int = 10,
        exact: bool = False,
        min_score: float | None = None,
    ) -> list[tuple[int, int, float]]:
        """
        Search the index with a query.

        Args:
            query: Search query string
            k: Number of results to return (default: 10)
            exact: Use exact search instead of approximate (default: False)
            min_score: Minimum score threshold (optional)

        Returns:
            List of (document_id, field_index, score) tuples
        """
        ...

    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...

@final
class TextEmbeddingIndex:
    """Embedding-based search index for text data."""

    @staticmethod
    def build(data: TextEmbeddings, index_dir: str, metric: str | None = None) -> None:
        """
        Build an embedding search index.

        Args:
            data: TextEmbeddings to index
            index_dir: Output directory for the index
            metric: Distance metric ("cosine", "inner_product", "ip", "l2", "hamming")
        """
        ...

    @staticmethod
    def load(data: TextEmbeddings, index_dir: str) -> TextEmbeddingIndex:
        """
        Load an embedding search index.

        Args:
            data: TextEmbeddings associated with the index
            index_dir: Directory containing the index

        Returns:
            Loaded TextEmbeddingIndex instance
        """
        ...

    def search_embedding(
        self,
        embedding: list[float] | bytes,
        k: int = 100,
        exact: bool = False,
        min_score: float | None = None,
    ) -> list[tuple[int, int, float]]:
        """
        Search the index with an embedding vector.

        Args:
            embedding: Query embedding vector
            k: Number of results to return (default: 100)
            exact: Use exact search instead of approximate (default: False)
            min_score: Minimum score threshold (optional)

        Returns:
            List of (document_id, field_index, score) tuples
        """
        ...

    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...

__all__ = [
    "KeywordIndex",
    "TextData",
    "TextEmbeddingIndex",
    "TextEmbeddings",
]
