"""Embedding models for text, images, and multi-modal (OpenCLIP)."""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F
from tqdm import trange
from transformers import AutoImageProcessor, AutoModel


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
            embedding_dim: Target embedding dimension (must be <= model dimension)
            batch_size: Batch size for encoding (default: all texts in one batch)
            normalize: Whether to normalize embeddings to unit length
            show_progress: Whether to show progress bar

        Returns:
            NumPy array of embeddings with shape (len(texts), embedding_dim)
        """
        if embedding_dim and embedding_dim < self.dim:
            dim = embedding_dim
        else:
            dim = self.dim

        if not texts:
            return np.empty((0, dim), dtype=np.float32)

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


class ImageEmbeddingModel:
    """Image embedding model using HuggingFace transformers.

    Args:
        model: Name or path of the image embedding model
        device: Device to use for inference ('cuda', 'cpu', or None for auto-detection)
    """

    def __init__(self, model: str, device: str | None = None):
        self.model = model
        self.encoder = AutoModel.from_pretrained(model, dtype="auto", device_map=device)
        self.device = next(self.encoder.parameters()).device
        self.encoder.eval()
        self.dim: int = self.encoder.config.hidden_size
        self.processor = AutoImageProcessor.from_pretrained(model)

    def embed(
        self,
        images: list[np.ndarray],
        batch_size: int | None = None,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed a list of images.

        Args:
            images: List of images to embed
            normalize: Whether to normalize embeddings to unit length

        Returns:
            NumPy array of embeddings with shape (len(images), embedding_dim)
        """
        if not images:
            return np.empty((0, self.dim))

        full_embeddings = []
        if batch_size is None:
            batch_size = len(images)

        for i in trange(
            0,
            len(images),
            batch_size,
            desc="Calculating image embeddings",
            disable=not show_progress,
        ):
            inputs = self.processor(
                images=images[i : i + batch_size],
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.encoder(**inputs)

            # always take the embedding of the first token (typically [CLS])
            embeddings = outputs.last_hidden_state[:, 0, :]

            if normalize:
                embeddings = F.normalize(embeddings)

            full_embeddings.append(embeddings.cpu().numpy())

        embeddings = np.vstack(full_embeddings)

        return embeddings.astype(np.float32)


class OpenClipModel:
    """Multi-modal embedding model using OpenCLIP.

    Produces text and image embeddings in a shared vector space.

    Args:
        model: HF hub path (e.g. ``hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K``)
               or local path.
        device: Device for inference ('cuda', 'cpu', or None for auto).
    """

    def __init__(self, model: str, device: str | None = None):
        import open_clip

        self.model_name = model
        clip_model, self.preprocess = open_clip.create_model_from_pretrained(model)
        if device is not None:
            clip_model = clip_model.to(device)
        self.device = next(clip_model.parameters()).device
        clip_model.eval()
        self.clip_model = clip_model
        self.tokenizer = open_clip.get_tokenizer(model)

        # Determine embedding dimension via a dummy forward pass
        with torch.inference_mode():
            dummy = self.tokenizer(["hello"]).to(self.device)
            dummy_emb = self.clip_model.encode_text(dummy)
            self.dim: int = dummy_emb.shape[-1]

    def embed_text(
        self,
        texts: list[str],
        batch_size: int | None = None,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        if batch_size is None:
            batch_size = len(texts)

        all_embeddings = []
        for i in trange(
            0,
            len(texts),
            batch_size,
            desc="OpenCLIP text embeddings",
            disable=not show_progress,
        ):
            batch = texts[i : i + batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            with torch.inference_mode():
                emb = self.clip_model.encode_text(tokens)
            if normalize:
                emb = F.normalize(emb)
            all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_image(
        self,
        images: list[np.ndarray],
        batch_size: int | None = None,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        from PIL import Image

        if not images:
            return np.empty((0, self.dim), dtype=np.float32)

        if batch_size is None:
            batch_size = len(images)

        all_embeddings = []
        for i in trange(
            0,
            len(images),
            batch_size,
            desc="OpenCLIP image embeddings",
            disable=not show_progress,
        ):
            batch = images[i : i + batch_size]
            pil_images = [Image.fromarray(img) for img in batch]
            tensors = torch.stack([self.preprocess(img) for img in pil_images]).to(
                self.device
            )
            with torch.inference_mode():
                emb = self.clip_model.encode_image(tensors)
            if normalize:
                emb = F.normalize(emb)
            all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)
