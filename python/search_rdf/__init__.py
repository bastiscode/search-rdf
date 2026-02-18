from importlib import metadata

from search_rdf._internal import (
    Data,
    EmbeddingIndex,
    KeywordIndex,
    embedding_index_hardware_acceleration,
)

try:
    __version__ = metadata.version("search_rdf")
except metadata.PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "Data",
    "EmbeddingIndex",
    "KeywordIndex",
    "embedding_index_hardware_acceleration",
]
