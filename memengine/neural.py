"""Optional, local sentence-transformers backends."""

import os

import numpy as np

from memengine.reranker import checked_scores
from memengine.vector import normalize_vectors


class NeuralBackendUnavailableError(RuntimeError):
    """The optional runtime or requested local neural weights are unavailable."""


def _load(model, component, **options):
    try:
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer

        cls = SentenceTransformer if component == "embedding" else CrossEncoder
        return cls(model, device=os.getenv("LOCAL_NEURAL_DEVICE") or None, **options), torch
    except Exception as exc:
        raise NeuralBackendUnavailableError(
            f"Local neural {component} could not load {model!r}. "
            f"Install .[neural] and provide cached weights or a local model path. "
            f"{type(exc).__name__}: {exc}"
        ) from exc


class LocalNeuralEmbedder:
    def __init__(self, model=None, *, batch_size=32):
        self.model = model or os.getenv("LOCAL_EMBED_MODEL") or "intfloat/multilingual-e5-small"
        self._encoder, _ = _load(self.model, "embedding")
        self.batch_size = batch_size
        self._e5 = "e5" in self.model.lower()

    @property
    def model_name(self):
        return f"local-neural-v1:{self.model}"

    def _encode(self, texts, prefix):
        if not texts:
            return np.empty((0, self._encoder.get_sentence_embedding_dimension()), dtype="<f4")
        inputs = [prefix + text for text in texts] if self._e5 else texts
        try:
            vectors = self._encoder.encode(inputs, batch_size=self.batch_size, show_progress_bar=False,
                                           convert_to_numpy=True, normalize_embeddings=True)
            return normalize_vectors(vectors)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            raise NeuralBackendUnavailableError(f"Local neural embedding failed for {self.model!r}: {exc}") from exc

    def encode(self, texts):
        return self._encode(texts, "passage: ")

    def encode_queries(self, texts):
        return self._encode(texts, "query: ")


class LocalCrossEncoderReranker:
    is_calibrated = True

    def __init__(self, model=None, *, batch_size=16):
        self.model = model or os.getenv("LOCAL_RERANK_MODEL") or "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        self._encoder, torch = _load(self.model, "reranker", max_length=384)
        self._identity = torch.nn.Identity()
        self.batch_size = batch_size

    def score(self, query, candidates):
        if not candidates:
            return []
        pairs = [(query, fact["plain"]) for fact in candidates]
        logits = np.asarray(self._encoder.predict(pairs, batch_size=self.batch_size, activation_fn=self._identity,
                                                 show_progress_bar=False, convert_to_numpy=True))
        if logits.shape not in {(len(pairs),), (len(pairs), 1)} or not np.isfinite(logits).all():
            raise ValueError("local cross-encoder must return one finite relevance logit per pair")
        return checked_scores((1 / (1 + np.exp(-np.clip(logits.reshape(-1), -80, 80)))).tolist(), len(pairs))


def load_local_backends():
    """Prefer an entirely ONNX pair, then try the compatible Torch pair."""
    from memengine.neural_onnx import OnnxCrossEncoderReranker, OnnxNeuralEmbedder

    try:
        return OnnxNeuralEmbedder(), OnnxCrossEncoderReranker()
    except NeuralBackendUnavailableError as onnx_error:
        try:
            return LocalNeuralEmbedder(), LocalCrossEncoderReranker()
        except NeuralBackendUnavailableError as torch_error:
            raise NeuralBackendUnavailableError(f"{onnx_error}\n{torch_error}") from torch_error
