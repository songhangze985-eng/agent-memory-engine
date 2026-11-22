"""Candidate generation and weighted RRF shared by recall modes."""
from memengine.chrono import temporal_query
from memengine.identifiers import compares_versions, extract_identifiers
from memengine.storage.sqlite import key_type
from memengine.linguistics import stem_features
from memengine.retrieval.activation import bridge_feedback, discovered_windows, external_entities, spreading_activation
from memengine.retrieval.coverage import facet_search_query, supported_facet
from memengine.temporal import competition_terms
from memengine.versioning import extract_versions, score_version_match
from memengine.textutil import normalize_entity, strip_step_tags, tokenize_for_fts
from memengine.vector import score_quantiles, weighted_rrf


def recall(store, per_channel_k, query, parsed, temporal, scope, direct, *, vector_search, fuse=weighted_rrf,
           include_cold=False, filters=None, plan=None):
    user_id, agent_id, session_id = scope
    options = dict(user_id=user_id, agent_id=agent_id, session_id=session_id, ids_only=True,
                   top_n=per_channel_k, include_cold=include_cold)
    if filters is not None:
        options["filters"] = filters
    expansions = " ".join(stem_features(query)) if plan and plan.wide else ""
    query_index = f"{query} {expansions}" if expansions else query
    semantic_query = parsed["semantic_vector_query"]
    wanted = extract_identifiers(query)
    filtered_time = filters is not None and (filters.time_start is not None or filters.time_end is not None)
    temporal_constraint = (temporal.is_historical or temporal.start_ts is not None
                           or temporal.end_ts is not None or filtered_time)
    quotas = plan.channel_quotas if plan else frozenset()
    reweight = bool(plan and plan.semantic_weighting)
    hashed = bool((expansions or quotas or reweight) and store.embedder.model_name.startswith("hash-"))
    if quotas and plan.route_channel_quotas and (
        not (plan.wide and plan.graph and hashed)
        or wanted or plan.versions or plan.windows or temporal_constraint or temporal_query(query)
    ):
        quotas = frozenset()
    if expansions and hashed:
        semantic_query = f"{semantic_query} {expansions}"
    named = bool(plan and any(key.startswith("entity:") and normalize_entity(key) in query.casefold()
                              for key in direct))
    dense_limit = min(per_channel_k, 42 if hashed else 50) if "dense" in quotas else per_channel_k
    lexical_limit = min(per_channel_k, 36 if hashed or named else 24) if "lexical" in quotas else per_channel_k
    dense = vector_search(semantic_query, **{**options, "top_n": dense_limit})
    lexical = store.search_lexical(query_index, **{**options, "top_n": lexical_limit})
    entities = direct or parsed["entities"]
    use_entity_inverted = plan is None or plan.query_entity_inverted
    inverted = store.search_entity_inverted(entities, **options) if use_entity_inverted else []
    timed = []
    previous = (temporal.is_historical and temporal.order_hint == "latest"
                and temporal.start_ts is None and temporal.end_ts is None)
    indexed_window = plan and plan.windows and not temporal.rev_index and not temporal.order_hint and not filtered_time
    if not indexed_window and temporal_constraint:
        timed = store.search_temporal_range(start_ts=temporal.start_ts, end_ts=temporal.end_ts,
                                           order_hint=temporal.order_hint, rev_index=temporal.rev_index,
                                           entities={normalize_entity(key) for key in direct}, previous_only=previous,
                                           **{**options, "include_cold": include_cold or temporal.is_historical})
    exact = store.search_exact_identifiers(wanted, **options)
    use_anchor_channel = not exact and not any(item.ident_type == "version" for item in wanted)
    anchor_options = {key: value for key, value in options.items() if key != "ids_only"}
    if "anchors" in quotas:
        anchor_options["top_n"] = min(12, per_channel_k)
    anchor_lex, anchor_vec = (store.search_turn_anchors(query, parsed["semantic_vector_query"], **anchor_options)
                              if use_anchor_channel else ([], []))
    contextual = bool(anchor_lex)
    channels = (dense, lexical, inverted, timed, exact, anchor_lex, anchor_vec) if contextual or plan else (dense, lexical, inverted, timed, exact)
    graph_hits, followup, feedback, calendar_hits, precision_hits, bridges = [], [], [], [], [], []
    directed, gap = [], None
    if plan:
        lexical_weight, semantic_weight = (.60, .32) if named else (.46, .44)
        if reweight and not named and not hashed:
            lexical_weight, semantic_weight = .30, .60
        if plan.windows:
            calendar_hits = store.occurrence_candidates(plan.windows, scope=scope, filters=filters,
                                                        include_cold=include_cold, top_n=per_channel_k)
        if plan.versions:
            precision_hits = store.version_candidates(plan.versions, scope=scope, filters=filters, include_cold=include_cold)
        facets = parsed.get("facets", ())
        directed_plan = plan.graph and facets and not (wanted or plan.versions or plan.windows or temporal.is_historical)
        if directed_plan:
            seeds = fuse(dense, inverted, w_vec=semantic_weight, w_hash=.08, extra_channels=(
                (lexical_weight, lexical), (.14, timed), (.14, anchor_lex), (.10, anchor_vec)))[:10]
            entity_names = {normalize_entity(key) for key in direct}

            def scoped_facts(hits):
                rows = store.recall_facts_by_ids([hit["id"] for hit in hits], scope=scope)
                return {row["id"]: {**row, "entity_anchor": bool(entity_names.intersection(
                            normalize_entity(key) for key in row["entity_z_keys"]))} for row in rows}

            anchor_facts = list(scoped_facts(seeds).values())
            gap = next((facet for facet in facets
                        if not supported_facet([facet], anchor_facts, require_entity=bool(direct))), None)
            if gap:
                followup_query = facet_search_query(query, gap)
                expanded = " ".join(stem_features(followup_query))
                hits = store.search_lexical(f"{followup_query} {expanded}".strip(),
                                            **{**options, "top_n": min(12, per_channel_k)})
                scoped = scoped_facts(hits)
                directed = [hit for hit in hits if hit["id"] in scoped and supported_facet(
                    [gap], [scoped[hit["id"]]], require_entity=bool(direct))][:2]
        elif plan.graph and plan.passive_expansion:
            seeds = fuse(lexical, dense, w_vec=lexical_weight, w_hash=semantic_weight,
                         extra_channels=((.16, anchor_lex), (.07, inverted), (.22, calendar_hits)))[:16]
            hydrated = {row["id"]: row for row in store.recall_facts_by_ids([row["id"] for row in seeds], scope=scope)}
            focus = competition_terms(query) - {normalize_entity(key) for key in direct}
            seed_facts = [hydrated[row["id"]] for row in seeds if row["id"] in hydrated]
            aligned = [row for row in seed_facts if focus & competition_terms(row["plain"])]
            seeds_for_graph = aligned[:10] or seed_facts[:8]
            graph_hits = spreading_activation(store, seeds_for_graph, scope=scope, filters=filters,
                                             include_cold=include_cold, top_n=40)
            bridges = external_entities(query, seed_facts)
            if not plan.windows and (inferred := discovered_windows(query, aligned[:6])):
                calendar_hits = store.occurrence_candidates(inferred, scope=scope, filters=filters,
                                                            include_cold=include_cold, top_n=min(18, per_channel_k))
            elif bridges and not plan.windows:
                followup = store.search_entity_inverted(bridges, **options)
            expanded = bridge_feedback(query, seed_facts)
            if expanded:
                feedback = store.search_lexical(f"{query_index} {expanded}", **{**options, "top_n": min(25, per_channel_k)})
        channels = (*channels, graph_hits, followup, feedback, calendar_hits, precision_hits, directed)
    directed_ids = {row["id"] for row in directed}
    depth = {row["id"]: row for row in graph_hits}
    bridge_ids = {row["id"] for row in (*followup, *feedback)}
    ids = {hit["id"] for channel in channels for hit in channel}
    candidates = store.recall_facts_by_ids(ids, scope=scope, identifiers=True) if plan else store.facts_by_ids(ids, scope=scope)
    details = {fact["id"]: fact for fact in candidates}
    time_ids = {hit["id"] for hit in timed}
    inverted_ids = {hit["id"] for hit in inverted}
    dense_scores = {hit["id"]: hit["vector_score"] for hit in dense}
    entity_names = {normalize_entity(key) for key in direct}
    category_names = set(parsed["categories"]) if not entities else set()
    historical_query = parsed["query_type"] == "TIME_TRAVEL"
    query_terms = set(tokenize_for_fts(query).split())
    query_keys = {item.ident_norm for item in wanted}
    query_versions = {item.ident_norm for item in wanted if item.ident_type == "version"}
    compare = historical_query or compares_versions(query)
    indexed_identifiers = {} if plan else store.identifiers_for(details)
    for fact in details.values():
        anchor = bool(entity_names.intersection(normalize_entity(key) for key in fact["entity_z_keys"]))
        identifiers = fact["identifiers"] if plan else indexed_identifiers.get(fact["id"], {})
        versions = {key for key, kind in identifiers.items() if kind == "version"}
        version_match = bool(query_versions & versions)
        exact_hits = len(query_keys & identifiers.keys())
        fact.update(vector_score=dense_scores.get(fact["id"], 0.0), entity_anchor=anchor or fact["id"] in inverted_ids,
                    temporal_match=fact["id"] in time_ids,
                    category_match=bool(category_names.intersection(fact["category_z_keys"])),
                    lexical_overlap=bool(query_terms.intersection(tokenize_for_fts(fact["plain"]).split())),
                    identifiers=identifiers, exact_ident_hit=exact_hits, exact_hit_ratio=exact_hits / max(1, len(query_keys)),
                    version_match=version_match, version_mismatch=bool(query_versions and versions and not version_match and not compare))
        if fact["id"] in directed_ids:
            fact["facet_followup"] = gap
        if fact["id"] in bridge_ids:
            fact["bridge_keys"] = bridges
        if fact["id"] in depth:
            fact.update({key: value for key, value in depth[fact["id"]].items() if key != "id"})
        if plan and plan.versions and not compare:
            memory_versions = extract_versions(fact["plain"])
            grades = [(score_version_match(q, m), m) for q in plan.versions for m in memory_versions]
            best = max(grades, key=lambda item: (item[0].confidence, item[1].prerelease is None), default=None)
            fact.update(version_match=bool(best and best[0].confidence),
                        version_mismatch=bool(best and not best[0].confidence), exact_hit_ratio=0.0)
            if best:
                grade, version = best
                fact.update(version_grade=grade.match_type, version_boost=grade.boost, version_confidence=grade.confidence,
                            version_multiplier=grade.multiplier, version_literal=version.literal,
                            version_prerelease=version.prerelease is not None)
                if grade.match_type == "UNDERSPECIFIED":
                    fact["state_tag"] = f"[粒度不足: 记录仅精确到 {version.literal}]"
                    fact["text"] = fact["state_tag"] + " " + fact["text"]
                elif grade.confidence:
                    fact["exact_ident_hit"] = max(1, fact["exact_ident_hit"])
        if fact["status"] == "superseded":
            fact["state_tag"] = "[历史快照]" if historical_query else "[历史已变更]"
            fact["text"] = f"{fact['state_tag']} {fact['text']}"
        if time_ids and (previous or temporal.start_ts is not None or temporal.end_ts is not None or temporal.rev_index is not None):
            if not fact["temporal_match"]:
                fact["rrf_factor"] = 0.35

    def hydrate(channel):
        return [{**details[hit["id"]], **hit} for hit in channel]

    if plan:
        primary = hydrate(dense), hydrate(inverted)
        fusion_options = dict(w_vec=semantic_weight, w_hash=.08, extra_channels=(
            (lexical_weight, hydrate(lexical)), (.14, hydrate(timed)), (.14, hydrate(anchor_lex)), (.10, hydrate(anchor_vec)),
            (.7, hydrate(exact)), (.60, hydrate(precision_hits)), (.45 if plan.windows else .14, hydrate(calendar_hits)),
            (.19, hydrate(graph_hits)), (.09, hydrate(followup)), (.17, hydrate(feedback)), (.17, hydrate(directed))))

        def ranked_head(rows):
            if not directed_ids:
                return rows[:60]
            promoted = [row for row in rows[5:] if row["id"] in directed_ids]
            return [*rows[:5], *promoted, *(row for row in rows[5:] if row["id"] not in directed_ids)][:60]

        if not plan.score_quantiles or wanted or plan.versions or plan.windows:
            return ranked_head(fuse(*primary, **fusion_options))
        normalized = fuse(*primary, **fusion_options, normalized_channels=(
            score_quantiles(dense, "vector_score"),
            {},
            score_quantiles(lexical, "bm25_score", lower_is_better=True),
        ))
        if not plan.preserve_candidate_at_50:
            return ranked_head(normalized)
        baseline = ranked_head(fuse(*primary, **fusion_options))
        head = {fact["id"] for fact in baseline[:50]}
        by_id = {fact["id"]: fact for fact in normalized}
        return [fact for fact in normalized if fact["id"] in head] + [by_id[fact["id"]] for fact in baseline[50:]]

    baseline = fuse(hydrate(dense), hydrate(inverted), w_vec=0.40, w_hash=0.15,
                            extra_channels=((0.35, hydrate(lexical)), (0.25, hydrate(timed))))[:40]
    # Context windows help when the ordinary recall is itself dominated by short replies.
    peer_hits = sum(hit["peer"] for hit in anchor_lex)
    short_hits = sum(len(strip_step_tags(fact["plain"])) <= 220 for fact in baseline[:10])
    contextual = contextual and (not baseline or (len(baseline) < 6 and peer_hits > 0)
                                or (len(anchor_lex) >= 6 and max(peer_hits, short_hits) >= 6))
    if not wanted and not contextual:
        return baseline
    if contextual:
        blended = fuse(hydrate(dense), hydrate(inverted), w_vec=0.40, w_hash=0.15, extra_channels=(
            (0.35, hydrate(lexical)), (0.25, hydrate(timed)), (0.14, hydrate(anchor_lex)), (0.12, hydrate(anchor_vec))))
        anchor_turns = {hit["id"]: hit["context_anchor_turn_id"] for hit in anchor_lex}
        anchor_scores = {hit["id"]: hit["vector_score"] for hit in anchor_vec}
        old_scores = {fact["id"]: fact["rrf_score"] for fact in baseline}
        pool = {fact["id"]: fact for fact in baseline[:12]}
        blended_scores = {item["id"]: item["rrf_score"] for item in blended}
        for fact in blended:
            pool[fact["id"]] = fact
            if len(pool) >= 40:
                break
        for fact in pool.values():
            fact.update(contextual_query=True, baseline_rrf_score=old_scores.get(fact["id"], 0.0),
                        context_anchor_turn_id=anchor_turns.get(fact["id"]), anchor_vector_score=anchor_scores.get(fact["id"], 0.0),
                        rrf_score=blended_scores.get(fact["id"], 0.0))
        return sorted(pool.values(), key=lambda fact: (-fact["rrf_score"], fact["id"]))
    routed = fuse(hydrate(dense), hydrate(exact), w_vec=0.30, w_hash=0.44,
                          extra_channels=((0.26, hydrate(lexical)),))
    base_scores = {fact["id"]: fact["rrf_score"] for fact in baseline}
    pool = {fact["id"]: fact for fact in routed[:30]}
    for fact in baseline:
        if len(pool) == 40:
            break
        pool.setdefault(fact["id"], {**fact, "rrf_score": 0.0})
    for fact in routed[30:]:
        if len(pool) == 40:
            break
        pool[fact["id"]] = fact
    return sorted(({**fact, "baseline_rrf_score": base_scores.get(fact["id"], 0.0)} for fact in pool.values()),
                  key=lambda fact: (-fact["rrf_score"], fact["id"]))
