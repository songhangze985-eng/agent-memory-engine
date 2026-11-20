"""Optional FastEmbed models backed by ONNX Runtime, without Torch imports."""

import os

import numpy as np

from memengine.assembler import _excerpt
from memengine.neural import NeuralBackendUnavailableError
from memengine.reranker import checked_scores
from memengine.temporal import competition_terms
from memengine.textutil import strip_step_tags
from memengine.vector import normalize_vectors

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def _create_model(model, component):
    try:
        if component == "embed":
            from fastembed import TextEmbedding as factory
        else:
            from fastembed.rerank.cross_encoder import TextCrossEncoder as factory
        model_path = os.getenv(f"LOCAL_{component.upper()}_MODEL_PATH")
        options = {"specific_model_path": model_path} if model_path else {}
        if threads := os.getenv("LOCAL_NEURAL_THREADS"):
            options["threads"] = int(threads)
        if os.getenv("HF_HUB_OFFLINE", "").upper() in {"1", "YES", "TRUE", "ON"}:
            options["local_files_only"] = True
        return factory(model_name=model, **options)
    except Exception as exc:
        raise NeuralBackendUnavailableError(
            f"FastEmbed ONNX {component} could not load {model!r}. "
            f"Install .[onnx]; check HF_ENDPOINT and cached model files. {type(exc).__name__}: {exc}"
        ) from exc


class OnnxNeuralEmbedder:
    def __init__(self, model=None, *, batch_size=32):
        self.model = model or os.getenv("LOCAL_EMBED_MODEL") or EMBED_MODEL
        self._encoder = _create_model(self.model, "embed")
        self.batch_size = batch_size
        self._e5 = "multilingual-e5-" in self.model.casefold()

    @property
    def model_name(self):
        return f"onnx-fastembed-v1:{self.model}"

    def _encode(self, texts, prefix):
        inputs = [prefix + text for text in texts] if self._e5 else texts
        try:
            if not texts:
                return np.empty((0, self._encoder.embedding_size), dtype="<f4")
            return normalize_vectors(list(self._encoder.embed(inputs, batch_size=self.batch_size)))
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            raise NeuralBackendUnavailableError(f"FastEmbed vector inference failed for {self.model!r}: {exc}") from exc

    def encode(self, texts):
        return self._encode(texts, "passage: ")

    def encode_queries(self, texts):
        return self._encode(texts, "query: ")


class OnnxCrossEncoderReranker:
    is_calibrated = True

    def __init__(self, model=None, *, batch_size=16):
        self.model = model or os.getenv("LOCAL_RERANK_MODEL") or RERANK_MODEL
        self._encoder = _create_model(self.model, "rerank")
        self.batch_size = batch_size

    def score(self, query, candidates):
        terms = competition_terms(query)
        documents = [_excerpt(strip_step_tags(fact["plain"]), terms, 128) for fact in candidates]
        if not documents:
            return []
        logits = np.asarray(list(self._encoder.rerank(query, documents, batch_size=self.batch_size)), dtype=float)
        if logits.shape != (len(documents),) or not np.isfinite(logits).all():
            raise ValueError("ONNX cross-encoder must return one finite relevance logit per pair")
        return checked_scores((1 / (1 + np.exp(-np.clip(logits, -80, 80)))).tolist(), len(documents))
