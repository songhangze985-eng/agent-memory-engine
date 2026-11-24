"""C3 & G4: Zero-LLM millisecond AmbientPlanner and LLM call budget guard."""
from __future__ import annotations

import time
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any

from agentmem_core.core.assemble import _apply_stale_policy
from agentmem_core.core.profiles import MemoryProfile, load_profile
from agentmem_core.core.scopes import ScopeChain, apply_scope_precedence
from agentmem_core.core.text import stem_terms
from agentmem_core.identifiers import extract_identifiers
from agentmem_core.llm import MockLLMClient
from agentmem_core.textutil import estimate_tokens, strip_step_tags
from agentmem_core.versioning import extract_versions


class BudgetExceededError(RuntimeError):
    """Raised when an execution path attempts to exceed its G4 LLM call budget."""


class _GuardedLLMProxy:
    """Wraps an LLM client to strictly enforce ``max_calls`` (0 for ambient, <=2 for question)."""

    def __init__(self, target: Any, *, max_calls: int, path_name: str):
        self._target = target
        self.max_calls = int(max_calls)
        self.path_name = path_name
        self.calls = 0

    def _consume(self, method: str) -> None:
        if self.calls + 1 > self.max_calls:
            raise BudgetExceededError(
                f"G4 LLM budget exceeded on {self.path_name} path via {method}(): "
                f"attempted call #{self.calls + 1}, budget={self.max_calls}"
            )
        self.calls += 1

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        self._consume("complete")
        if self._target is None:
            raise RuntimeError("No underlying LLM client configured")
        return self._target.complete(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _GuardedMockLLMProxy(MockLLMClient):
    """Preserves ``isinstance(llm, MockLLMClient)`` while still enforcing G4 call budgets."""

    def __init__(self, target: MockLLMClient, *, max_calls: int, path_name: str):
        self._target = target
        self.max_calls = int(max_calls)
        self.path_name = path_name
        self.calls = 0

    def complete(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        if self.calls + 1 > self.max_calls:
            raise BudgetExceededError(
                f"G4 LLM budget exceeded on {self.path_name} path via complete(): "
                f"attempted call #{self.calls + 1}, budget={self.max_calls}"
            )
        self.calls += 1
        return self._target.complete(prompt, *args, **kwargs)


@contextmanager
def enforce_llm_budget(engine, *, max_calls: int, path_name: str):
    """Context manager enforcing G4 LLM call caps on ``engine._engine.llm``."""
    raw_engine = getattr(engine, "_engine", engine)
    original_llm = getattr(raw_engine, "llm", None)
    if isinstance(original_llm, MockLLMClient):
        proxy: Any = _GuardedMockLLMProxy(original_llm, max_calls=max_calls, path_name=path_name)
    else:
        proxy = _GuardedLLMProxy(original_llm, max_calls=max_calls, path_name=path_name)
    raw_engine.llm = proxy
    try:
        yield proxy
    finally:
        raw_engine.llm = original_llm


class AmbientPlanner:
    """Zero-LLM hot-path retrieval & context packing for ambient turn/tool/file-open injection.

    Guarantees (G4 & C3):
      - Strictly ZERO LLM calls (runtime-enforced via ``enforce_llm_budget(max_calls=0)``).
      - Strictly ZERO neural cross-encoder / vector encoding latency on the hot path.
      - Uses O(log N) code-anchor index + exact identifier/version channel + SQLite/PG FTS5 BM25.
      - Applies hierarchical ``ScopeChain`` precedence and profile ``stale_policy``.
      - Strict lexical relevance floor to prevent false injections on unrelated tasks.
    """

    def __init__(self, engine):
        self.engine = engine

    def plan_and_assemble(
        self,
        query: str = "",
        *,
        active_files: list[str] | tuple[str, ...] = (),
        active_symbols: list[str] | tuple[str, ...] = (),
        error_text: str = "",
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str | None = None,
        scope_chain: ScopeChain | list[str] | tuple[str, ...] | str | None = None,
        repo_id: str | None = None,
        project_id: str | None = None,
        profile: str | Path | MemoryProfile | None = None,
        token_budget: int | None = None,
        top_k: int | None = None,
        min_overlap_ratio: float = 0.25,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        spec = self.engine._resolve_profile(profile)
        eff_budget = int(token_budget if token_budget is not None else spec.ambient_token_budget)
        eff_top_k = int(top_k if top_k is not None else spec.ambient_top_k)
        eff_chain = self.engine._resolve_scope_chain(
            spec,
            scope_chain=scope_chain,
            user_id=user_id,
            session_id=session_id,
            repo_id=repo_id,
            project_id=project_id,
        )
        exclude_stale = spec.stale_policy == "exclude"
        combined_query = " ".join(p.strip() for p in (query, error_text) if p and p.strip()).strip()
        q_stems = stem_terms(combined_query)
        identifiers = tuple(extract_identifiers(combined_query))
        versions = tuple(extract_versions(combined_query))

        with enforce_llm_budget(self.engine, max_calls=0, path_name="ambient") as guard:
            # 1. Preload system prefix (declarative profile paths + always_active)
            prefix = self.engine.profile.render_system_prefix(
                user_id=user_id,
                agent_id=agent_id,
                preload_paths=spec.preload_paths,
                preload_always_active=spec.preload_always_active,
                scope_chain=eff_chain,
                exclude_stale=exclude_stale,
            )
            prefix_tokens = estimate_tokens(prefix)
            if prefix_tokens > eff_budget:
                prefix = ""
                prefix_tokens = 0

            candidates_by_id: dict[str, dict[str, Any]] = {}
            anchor_hit_ids: set[str] = set()

            # 2. Direct O(log N) Code Anchor lookup (when enabled by profile)
            if spec.enable_code_anchors and (active_files or active_symbols):
                anchor_rows = self.engine.anchors.find_by_context(
                    active_files=active_files,
                    active_symbols=active_symbols,
                    user_id=user_id,
                    agent_id=agent_id,
                    scope_chain=eff_chain,
                    include_stale=not exclude_stale,
                    limit=eff_top_k * 2,
                )
                for row in anchor_rows:
                    fid = row["id"]
                    anchor_hit_ids.add(fid)
                    candidates_by_id[fid] = {**row, "_ambient_score": 2.0, "ambient_source": "code_anchor"}

            # 3. Exact Identifier & Version lookup (zero-confusion precision path)
            if identifiers:
                exact_rows = self.engine.store.search_exact_identifiers(
                    list(identifiers),
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=None,
                    include_superseded=False,
                    top_n=eff_top_k,
                )
                for row in exact_rows:
                    fid = row["id"]
                    prev = candidates_by_id.get(fid)
                    score = max(1.8, prev["_ambient_score"]) if prev else 1.8
                    candidates_by_id[fid] = {**row, "_ambient_score": score, "exact_ident_hit": True, "ambient_source": "exact_identifier"}

            # 4. Fast SQLite/PG FTS5 lexical search on facts (sub-millisecond, 0 LLM, 0 neural)
            if combined_query:
                lex_rows = self.engine.store.search_lexical(
                    combined_query,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=None,
                    include_superseded=False,
                    top_n=max(eff_top_k * 3, 20),
                )
                for rank_idx, fact in enumerate(lex_rows):
                    if fact.get("status") != "active":
                        continue
                    fid = fact["id"]
                    bm25_raw = abs(float(fact.get("bm25_score") or 0.0))
                    f_text = strip_step_tags(fact.get("plain") or "")
                    f_stems = stem_terms(f_text)
                    overlap = len(q_stems & f_stems)
                    overlap_ratio = overlap / max(1, len(q_stems))
                    # Strict anti-false-injection floor: require meaningful lexical match unless exact/version hit
                    version_hit = bool(versions and any(str(v) in f_text for v in versions))
                    if not version_hit and (overlap < 2 and overlap_ratio < max(min_overlap_ratio, 0.50)):
                        continue
                    if not version_hit and overlap_ratio < min_overlap_ratio:
                        continue
                    score = (1.2 if version_hit else 0.0) + overlap_ratio + 0.15 / (rank_idx + 1) + min(0.25, bm25_raw * 0.02)
                    prev = candidates_by_id.get(fid)
                    if prev is None or score > prev["_ambient_score"]:
                        candidates_by_id[fid] = {
                            **fact,
                            "_ambient_score": score,
                            "lexical_overlap": overlap,
                            "overlap_ratio": round(overlap_ratio, 4),
                            "ambient_source": prev["ambient_source"] if prev and prev["ambient_source"] == "code_anchor" else "lexical_fts",
                        }

            # 5. Enrich unified metadata (scope_key, override_key, vfs_path, staleness_status)
            all_ids = list(candidates_by_id.keys())
            meta_by_id = self.engine.ledger.meta_map(all_ids) if all_ids else {}
            enriched: list[dict[str, Any]] = []
            for fid, item in candidates_by_id.items():
                meta = meta_by_id.get(fid)
                if meta:
                    merged = {
                        **item,
                        "kind": meta["kind"],
                        "scope_key": meta.get("scope_key", "global"),
                        "override_key": meta.get("override_key", ""),
                        "vfs_path": meta["vfs_path"],
                        "title": meta["title"],
                        "index_summary": meta["index_summary"],
                        "verification_status": meta["verification_status"],
                        "salience_tag": meta["salience_tag"],
                        "always_active": meta["always_active"],
                        "staleness_status": meta.get("staleness_status", "fresh"),
                        "stale_reason": meta.get("stale_reason", ""),
                    }
                else:
                    merged = {
                        **item,
                        "kind": item.get("kind", "world"),
                        "scope_key": item.get("scope_key", f"user:{user_id}" if user_id else "global"),
                        "override_key": item.get("override_key", ""),
                        "staleness_status": item.get("staleness_status", "fresh"),
                        "stale_reason": item.get("stale_reason", ""),
                    }
                # Enforce exact version discipline: never inject mismatched version facts for a version-pinned query
                if versions:
                    plain_text = merged.get("plain") or ""
                    fact_vers = tuple(extract_versions(plain_text))
                    if fact_vers and not (set(fact_vers) & set(versions)):
                        continue
                enriched.append(merged)

            # 6. Sort by score, apply ScopeChain precedence & stale_policy
            enriched.sort(key=lambda x: (-float(x.get("_ambient_score", 0.0)), x.get("id", "")))
            scoped, overridden = apply_scope_precedence(enriched, eff_chain, suppress_overridden=True)
            final_candidates, excluded_stale_ids = _apply_stale_policy(scoped, spec.stale_policy)
            selected = final_candidates[:eff_top_k]

            # 7. Bounded token packing
            remaining = max(0, eff_budget - prefix_tokens)
            used = 0
            lines: list[str] = []
            packed_items: list[dict[str, Any]] = []
            for item in selected:
                plain = strip_step_tags(item.get("plain") or "").strip()
                label = item.get("vfs_path") or item.get("kind", "fact")
                skey = item.get("scope_key", "global")
                line = f"- [{label}|{skey}] {plain}"
                cost = estimate_tokens(line) + 1
                if used + cost > remaining:
                    # Try compact index_summary fallback before dropping
                    summary = item.get("index_summary") or plain[:80]
                    compact = f"- [{label}|{skey}] {summary}"
                    c_cost = estimate_tokens(compact) + 1
                    if used + c_cost > remaining:
                        break
                    line = compact
                    cost = c_cost
                lines.append(line)
                used += cost
                packed_items.append({k: v for k, v in item.items() if k != "_ambient_score"})

            dynamic = "\n".join(lines)
            dyn_tokens = estimate_tokens(dynamic)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

            return {
                "path": "ambient",
                "profile": spec.name,
                "scope_chain": eff_chain.as_list() if eff_chain is not None else None,
                "system_prefix": prefix,
                "prefix_version": sha256(prefix.encode("utf-8")).hexdigest()[:12],
                "dynamic_memory": dynamic,
                "items": packed_items,
                "ranked": packed_items,
                "anchor_hits": [i["id"] for i in packed_items if i["id"] in anchor_hit_ids],
                "excluded_stale_ids": excluded_stale_ids,
                "scope_overridden": [
                    {"id": o["id"], "scope_key": o["scope_key"], "overridden_by_scope": o.get("overridden_by_scope")}
                    for o in overridden
                ],
                "empty_recall": not packed_items,
                "llm_calls": guard.calls,
                "ambient_ms": elapsed_ms,
                "token_breakdown": {
                    "system_prefix": prefix_tokens,
                    "dynamic_memory": dyn_tokens,
                    "total": prefix_tokens + dyn_tokens,
                    "budget": eff_budget,
                },
            }
