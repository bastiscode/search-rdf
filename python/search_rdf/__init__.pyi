"""Type stubs for search_rdf module."""

from typing import Any, Iterator, final

@final
class Data:
    """Data source built from JSONL or SPARQL result file."""

    @staticmethod
    def build_from_items(items: list[dict[str, Any]], data_dir: str) -> None:
        """
        Build data from a list of items.

        Args:
            items: List of data items (dictionary with "identifier" and "fields")
            data_dir: Output directory for the data
        """
        ...

    @staticmethod
    def build_from_jsonl(file_path: str, data_dir: str) -> None:
        """
        Build data from a JSONL file.

        Args:
            file_path: Path to the JSONL file
            data_dir: Output directory for the data
        """
        ...

    @staticmethod
    def build_from_sparql_result(
        file_path: str,
        data_dir: str,
        format: str | None = None,
        default_field_type: str = "text",
    ) -> None:
        """
        Build data from a SPARQL result file.

        Args:
            file_path: Path to the SPARQL result file
            data_dir: Output directory for the data
            format: Format of the SPARQL result file ("json", "xml", or "tsv", or None for auto-detect)
            default_field_type: Default field type for literals ("text", "image", or "image-inline")
        """
        ...

    @staticmethod
    def load(data_dir: str) -> Data:
        """
        Load data from a directory.

        Args:
            data_dir: Directory containing the data

        Returns:
            Loaded Data instance
        """
        ...

    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...
    def __iter__(self) -> Iterator[tuple[str, list[str]]]: ...
    def num_fields(self, id: int) -> int | None: ...
    def field(self, id: int, field: int) -> str | None: ...
    def fields(self, id: int) -> list[str] | None: ...
    def main_field(self, id: int) -> str | None: ...
    def identifier(self, id: int) -> str | None: ...
    def id_from_identifier(self, identifier: str) -> int | None: ...

@final
class KeywordIndex:
    """Keyword search index for data."""

    @staticmethod
    def build(data: Data, index_dir: str) -> None:
        """
        Build a keyword search index.

        Args:
            data: Data to index
            index_dir: Output directory for the index
        """
        ...

    @staticmethod
    def load(data: Data, index_dir: str) -> KeywordIndex:
        """
        Load a keyword search index.

        Args:
            data: Data associated with the index
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
        allow_ids: set[int] | None = None,
    ) -> list[tuple[int, int, float]]:
        """
        Search the index with a query.

        Args:
            query: Search query string
            k: Number of results to return (default: 10)
            exact: Use exact search instead of approximate (default: False)
            min_score: Minimum score threshold (optional)
            allow_ids: Set of document IDs to filter results (optional)

        Returns:
            List of (document_id, field_index, score) tuples
        """
        ...

    def data(self) -> Data:
        """
        Get the data associated with the index.

        Returns:
            Data instance
        """
        ...

    @property
    def index_type(self) -> str:
        """
        Get the type of the index.

        Returns:
            Index type as a string
        """
        ...

@final
class EmbeddingIndex:
    """Embedding-based search index for data."""

    @staticmethod
    def build(
        data: Data,
        embedding_path: str,
        index_dir: str,
        metric: str | None = None,
        precision: str | None = None,
    ) -> None:
        """
        Build an embedding search index.

        Args:
            data: Data to index
            embedding_path: Path to the embeddings for the data
            index_dir: Output directory for the index
            metric: Distance metric ("cosine", "inner_product", "ip", "l2", "hamming")
            precision: Precision ("float32", "binary", "float16", "bfloat16", "int8")
        """
        ...

    @staticmethod
    def load(data: Data, embedding_path: str, index_dir: str) -> EmbeddingIndex:
        """
        Load an embedding search index.

        Args:
            data: Data associated with the index
            embedding_path: Path to the embeddings for the data
            index_dir: Directory containing the index

        Returns:
            Loaded EmbeddingIndex instance
        """
        ...

    def search(
        self,
        embedding: list[float],
        k: int = 10,
        exact: bool = False,
        min_score: float | None = None,
        rerank: float | None = None,
        allow_ids: set[int] | None = None,
    ) -> list[tuple[int, int, float]]:
        """
        Search the index with an embedding vector.

        Args:
            embedding: Query embedding vector
            k: Number of results to return (default: 100)
            exact: Use exact search instead of approximate (default: False)
            min_score: Minimum score threshold (optional)
            rerank: Reranking factor for oversampling (optional)
            allow_ids: Set of document IDs to filter results (optional)

        Returns:
            List of (document_id, field_index, score) tuples
        """
        ...

    @property
    def index_type(self) -> str:
        """
        Get the type of the index.

        Returns:
            Index type as a string
        """
        ...

    @property
    def model(self) -> str:
        """
        Get the model used for embeddings.

        Returns:
            Model name as a string
        """
        ...

    @property
    def num_dimensions(self) -> int:
        """
        Get the number of dimensions in the embeddings.

        Returns:
            Number of dimensions as an integer
        """
        ...

    def data(self) -> Data:
        """
        Get the data associated with the index.

        Returns:
            Data instance
        """
        ...

__all__ = [
    "Data",
    "EmbeddingIndex",
    "KeywordIndex",
]
