"""Text embedding using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import trange


class TextEmbeddingModel:
    """Text embedding model using sentence-transformers.

    Args:
        model: Name or path of the sentence-transformer model
        device: Device to use for inference ('cuda', 'cpu', or None for auto-detection)
    """

    def __init__(self, model: str, device: str | None = None):
        self.model = model
        self.encoder = SentenceTransformer(model, device=device)
        self.dim: int = self.encoder.get_sentence_embedding_dimension()  # type: ignore
        assert self.dim is not None, "unable to get embedding dimension"

    def embed(
        self,
        texts: list[str],
        embedding_dim: int | None = None,
        batch_size: int | None = None,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed
            precision: Either 'float32' or 'ubinary'
            embedding_dim: Target embedding dimension (must be <= model dimension)
            batch_size: Batch size for encoding (default: all texts in one batch)
            show_progress: Whether to show progress bar

        Returns:
            NumPy array of embeddings with shape (len(texts), embedding_dim)
        """
        if embedding_dim and embedding_dim < self.dim:
            dim = embedding_dim
        else:
            dim = self.dim

        if not texts:
            return np.empty((0, dim))

        if batch_size is None:
            batch_size = len(texts)

        # Sort texts by length to minimize padding
        indices = np.argsort([-len(text) for text in texts])
        sorted_texts = [texts[i] for i in indices]
        full_embeddings = []

        # Process in batches to avoid OOM for large datasets
        for i in trange(
            0,
            len(sorted_texts),
            batch_size,
            desc="Calculating embeddings",
            disable=not show_progress,
        ):
            batch = sorted_texts[i : i + batch_size]
            embeddings = self.encoder.encode(  # type: ignore
                batch,
                normalize_embeddings=normalize,
                batch_size=len(batch),
                show_progress_bar=False,
            )[:, :dim]
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            full_embeddings.extend(embeddings)

        embeddings = np.vstack(full_embeddings)
        inv_indices = np.argsort(indices)

        # Verify that inv_indices correctly restores the original order
        assert all(t == sorted_texts[i] for t, i in zip(texts, inv_indices))

        return embeddings[inv_indices]
