"""One shared candidate trace for independent retrieval and context measurement."""

from dataclasses import dataclass, replace
from functools import lru_cache
import sys
from time import perf_counter
from unittest.mock import patch

from memengine.assembler import assemble_context, render_prefix
from memengine.engine import CognitiveMemoryEngine, _fit_context, baseline_rrf
from memengine.llm import MockLLMClient, build_llm_from_env
from memengine.neural import NeuralBackendUnavailableError, load_local_backends
from memengine.reranker import CalibratedLocalReranker
from memengine.textutil import estimate_tokens
from memengine.vector import HashedTfidfEmbedder

TOP_K = 10
TOKEN_BUDGET = 2048
LABELS = {"bipolar": "Full System", "rrf": "Pure RRF"}


def add_backend_argument(parser):
    parser.add_argument("--backend", choices=("hashed", "local-neural", "auto"), default="hashed")


@lru_cache(maxsize=1)
def _neural_models():
    vector, reranker = load_local_backends()
    print(f"Local neural backend: {type(vector).__name__} ({vector.model_name}), "
          f"{type(reranker).__name__} ({reranker.model})", file=sys.stderr)
    return vector, reranker


def resolve_backend(backend):
    if backend != "hashed":
        try:
            _neural_models()
        except NeuralBackendUnavailableError as exc:
            if backend != "auto":
                raise
            print(f"{exc} Falling back to backend=hashed.", file=sys.stderr)
            return "hashed"
        return "local-neural"
    return backend


def cli_backend(parser, backend):
    try:
        return resolve_backend(backend)
    except NeuralBackendUnavailableError as exc:
        parser.exit(2, f"{exc}\n")


def make_engine(mode, *, allow_casual=False, client=None, backend="hashed", timeline_buffer=True):
    if mode == "live":
        if client is None:
            client, resolved, _ = build_llm_from_env()
            if resolved != "live":
                raise ValueError("live mode requires LLM_API_KEY or OPENAI_API_KEY")
    elif client is None:
        client = MockLLMClient()
    vector, reranker = _neural_models() if backend == "local-neural" else (HashedTfidfEmbedder(), CalibratedLocalReranker())
    return CognitiveMemoryEngine(db_url=":memory:", llm=client, vector=vector, reranker=reranker,
                                 enable_casual_bypass=allow_casual, enable_timeline_buffer=timeline_buffer)


@dataclass
class Pair:
    retrieved: dict
    baseline: list
    candidates: list
    context: dict
    rrf_text: str
    full_ms: float
    rrf_ms: float
    preassembly_ms: float | None = None

    def efficiency(self, mode, previous_prefix=None):
        if mode == "rrf":
            tokens = estimate_tokens(self.rrf_text)
            return {"injected_tokens": tokens, "prefix_tokens": 0, "dynamic_tokens": tokens, "drilled_raw_tokens": 0,
                    "latency_ms": self.rrf_ms, "prefix_hit": None, "evidence_turn_ids": []}
        counts = self.context["token_breakdown"]
        version = self.context["prefix_version"] if self.context["system_prefix"] else None
        return {"injected_tokens": counts["total"], "prefix_tokens": counts["system_prefix"],
                "dynamic_tokens": counts["dynamic_memory"] - counts["drilldown"], "drilled_raw_tokens": counts["drilldown"],
                "latency_ms": self.full_ms, "prefix_hit": version == previous_prefix if version and previous_prefix else None,
                "evidence_turn_ids": self.context["evidence_turn_ids"], "prefix_version": version}


def _shared_retrieval(engine, query, stamp, *, full_budget=TOKEN_BUDGET):
    captures = []
    original = engine._recall
    start = perf_counter()

    def capture(*args, **kwargs):
        candidates = original(*args, **kwargs)
        captures.append((candidates, perf_counter() - start))
        return candidates

    with patch.object(engine, "_recall", capture):
        retrieved = engine.retrieve(query, timestamp=stamp, read_only=True, top_k=TOP_K, token_budget=full_budget)
    candidates, shared_latency = captures[0] if captures else ([], perf_counter() - start)
    assert len(captures) == (retrieved["route"] == "TASK_POISON_CIRCLE")
    assert len({fact["id"] for fact in candidates}) == len(candidates)
    assert all(left["rrf_score"] >= right["rrf_score"] for left, right in zip(candidates, candidates[1:]))
    return retrieved, candidates, shared_latency


def paired_retrieve(engine, query, stamp):
    """Legacy bare-fact ablation, with equal budget and no context assembly."""
    retrieved, candidates, _ = _shared_retrieval(engine, query, stamp)
    return retrieved, _fit_context(baseline_rrf(candidates), TOP_K, TOKEN_BUDGET), candidates


def evaluate_pair(engine, query, stamp):
    start = perf_counter()
    prefix = render_prefix(engine.store.get_core_snapshot())
    remaining = TOKEN_BUDGET - estimate_tokens(prefix)
    if remaining < 0:
        raise ValueError("core profile exceeds evaluation context budget")
    retrieved, candidates, shared_latency = _shared_retrieval(engine, query, stamp, full_budget=remaining)
    preassembly_ms = (perf_counter() - start) * 1000
    context = assemble_context(engine.store, retrieved, prefix=prefix, token_budget=TOKEN_BUDGET,
                               drilldown_top_n=2, drilldown_window=0, query=query)
    full_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    baseline = _fit_context(baseline_rrf(candidates), TOP_K, TOKEN_BUDGET)
    rrf_text = "\n".join(fact["text"] for fact in baseline)
    return Pair(retrieved, baseline, candidates, context, rrf_text, full_ms,
                (shared_latency + perf_counter() - start) * 1000, preassembly_ms)


def attach_consolidated_context(engine, pair, query):
    """Hold fact retrieval fixed while measuring the additional cached prefix and observations."""
    started = perf_counter()
    prefix = render_prefix(engine.store.get_core_snapshot())
    context = assemble_context(engine.store, pair.retrieved, prefix=prefix,
                               token_budget=TOKEN_BUDGET + estimate_tokens(prefix),
                               drilldown_top_n=2, drilldown_window=0, query=query)
    retrieval_ms = pair.preassembly_ms if pair.preassembly_ms is not None else pair.full_ms
    return replace(pair, context=context, full_ms=retrieval_ms + (perf_counter() - started) * 1000)
