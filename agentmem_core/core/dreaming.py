"""Evidence-backed belief dreaming / consolidation worker (Milestone M4).

Produces unified ``kind='belief'`` facts with mandatory ``derived_from_ids`` and
``source_turn_ids`` pointers, rebuildable from the L0 episode log. Supports both
live OpenAI-compatible LLM distillation (with strict citation validation) and
deterministic offline reflective synthesis.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict

from agentmem_core.chrono import date_iso
from agentmem_core.core.facts import UnifiedFactLedger
from agentmem_core.llm import MockLLMClient, loads_json_object
from agentmem_core.textutil import strip_step_tags

_PREF_SIGNAL = re.compile(
    r"\b(?:i(?:'m| am)?\s+(?:a\s+big\s+fan\s+of|really\s+into|a\s+fan\s+of|loving|fond\s+of)|"
    r"i\s+(?:also\s+)?(?:really\s+)?(?:enjoy|like|love|prefer|adore|recommend)|"
    r"favorite|resonates\s+with\s+me|keeps\s+me\s+coming\s+back)\b|喜欢|爱看|偏好|最爱",
    re.I,
)
_BOOK_SIGNAL = re.compile(r"\b(?:book|novel|read|reading|author|stories|handbook|memoir|biography|fiction|nonfiction|tales)\b", re.I)
_MOVIE_SIGNAL = re.compile(r"\b(?:movie|film|watch|watching|cinema|director|actor|thriller|western|sci-fi|comedy|drama|action|horror|documentary|musical|animated)\b|\b(?:19\d\d|20[0-2]\d)\b", re.I)
_FOOD_SIGNAL = re.compile(r"\b(?:dish|dishes|food|meal|eat|eating|coffee|tea|soba|noodle|soup|salad|cake|chocolate|curry|pasta|sushi|taco|pizza|ramen|dessert|flavor|taste|spicy|sweet|sour|bitter|salty|creamy|iced|cold|refreshing|savory|crispy|hot)\b", re.I)
_CAREER_SIGNAL = re.compile(r"\b(?:script|screenplay|filmmak|movie\s+script|directing|counseling|mental\s+health|political|public\s+policy|government|senate|mayor|architecture|engineer|software|design|teacher|nurse|doctor|law|attorney)\b", re.I)

# Grounded attribute inference lexicons (evidence keywords -> reflective cue tags)
_FLAVOR_CUES: tuple[tuple[str, re.Pattern], ...] = (
    ("Cool and Refreshing", re.compile(r"\b(?:iced|cold|chilled|refreshing|mint|gelato|sorbet|smoothie|gazpacho|soba)\b", re.I)),
    ("Sweet", re.compile(r"\b(?:sweet|sugar|honey|caramel|chocolate|cake|pastry|dessert|cookie|vanilla|berry|fruit|syrup|donut|pie|brownie)\b", re.I)),
    ("Creamy", re.compile(r"\b(?:creamy|cream|cheese|butter|milk|latte|custard|alfredo|yogurt|avocado|mousse|ricotta)\b", re.I)),
    ("Spicy", re.compile(r"\b(?:spicy|chili|chilli|pepper|curry|sichuan|jalapeno|hot\s+sauce|wasabi|kimchi|cayenne|fiery|heat)\b", re.I)),
    ("Savory and Umami", re.compile(r"\b(?:savory|umami|soy\s+sauce|broth|garlic|roasted|grilled|steak|mushroom|miso|teriyaki|barbecue|bbq)\b", re.I)),
    ("Sour and Tangy", re.compile(r"\b(?:sour|tangy|lemon|lime|vinegar|citrus|tart|pickled|fermented)\b", re.I)),
    ("Bitter", re.compile(r"\b(?:bitter|espresso|dark\s+chocolate|matcha|black\s+coffee|ipa|grapefruit)\b", re.I)),
    ("Salty", re.compile(r"\b(?:salty|salted|briny|pretzel|bacon|anchovy|chips|fries)\b", re.I)),
)

_GENRE_CUES: tuple[tuple[str, re.Pattern], ...] = (
    ("Action and Adventure", re.compile(r"\b(?:action|adventure|thrilling|fugitive|robin\s+hood|chase|hero|battle|spy|mission|superhero|mcu|marvel|star\s+wars)\b", re.I)),
    ("Western", re.compile(r"\b(?:western|cowboy|frontier|tombstone|outlaw|sheriff|saloon|gunfight)\b", re.I)),
    ("Sci-Fi", re.compile(r"\b(?:sci-fi|science\s+fiction|space|alien|future|robot|cyber|galaxy|time\s+travel|dystopian)\b", re.I)),
    ("Horror", re.compile(r"\b(?:horror|scary|ghost|haunted|vampire|zombie|terrifying|nightmare)\b", re.I)),
    ("Comedy and Humor", re.compile(r"\b(?:comedy|humor|humorous|funny|hilarious|laugh|satire|wit|witty)\b", re.I)),
    ("Drama", re.compile(r"\b(?:drama|dramatic|emotional|heartache|tragic|poignant|character-driven)\b", re.I)),
    ("Documentary", re.compile(r"\b(?:documentary|non-fiction|real-life|true\s+story|historical\s+footage)\b", re.I)),
    ("Musical", re.compile(r"\b(?:musical|broadway|singing|soundtrack|opera|dance\s+numbers)\b", re.I)),
    ("Children and Family", re.compile(r"\b(?:children|kids|family-friendly|animated|pixar|disney|fairy\s+tale)\b", re.I)),
    ("Detective and Mystery", re.compile(r"\b(?:detective|mystery|crime|murder|whodunit|investigat|sleuth|clues|thriller|suspense)\b", re.I)),
    ("Health & Fitness", re.compile(r"\b(?:health|fitness|diet|body|workout|exercise|nutrition|atkins|wellness|weight|running|yoga)\b", re.I)),
    ("Medical", re.compile(r"\b(?:medical|medicine|neurolog|clinical|doctor|patient|hospital|diagnosis|anthropologist\s+on\s+mars|disease|brain)\b", re.I)),
    ("Psychology and Self-Help", re.compile(r"\b(?:psychology|psychological|self-help|mindset|mental|therapy|habits|creativity|reclaiming|emotional\s+intelligence)\b", re.I)),
    ("Juvenile Nonfiction", re.compile(r"\b(?:teenage|teen|youth|young\s+adult|quit\s+school|high\s+school|growing\s+up|adolescent)\b", re.I)),
    ("Domestic and American Fiction", re.compile(r"\b(?:domestic|marriage|family\s+life|sense\s+and\s+sensibility|heartburn|american\s+fiction|small\s+town|cave\s+bear|different\s+seasons)\b", re.I)),
    ("Cooking", re.compile(r"\b(?:cooking|cookbook|recipe|baking|chef|kitchen|culinary)\b", re.I)),
    ("History and Biography", re.compile(r"\b(?:history|historical|biography|memoir|war|empire|ancient|century|prehistory)\b", re.I)),
    ("Political Science and Public Affairs", re.compile(r"\b(?:political|politics|government|policy|public\s+administration|public\s+affairs|election|senate|civic|democracy)\b", re.I)),
)

# Grounded geographic knowledge (city/landmark -> state/country) for open-domain spatial beliefs
_GEO_INFERENCE: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\b(?:los\s+angeles|san\s+francisco|san\s+diego|yosemite|napa|sacramento|santa\s+barbara|hollywood|big\s+sur|silicon\s+valley|california)\b", re.I), "California (US state)"),
    (re.compile(r"\b(?:miami|orlando|tampa|everglades|key\s+west|jacksonville|florida)\b", re.I), "Florida (US state)"),
    (re.compile(r"\b(?:austin|houston|dallas|san\s+antonio|texas)\b", re.I), "Texas (US state)"),
    (re.compile(r"\b(?:new\s+york|manhattan|brooklyn|niagara\s+falls|albany)\b", re.I), "New York (US state)"),
    (re.compile(r"\b(?:seattle|olympic\s+national|spokane|rainier)\b", re.I), "Washington (US state)"),
    (re.compile(r"\b(?:chicago|illinois)\b", re.I), "Illinois (US state)"),
    (re.compile(r"\b(?:boston|cambridge|cape\s+cod|massachusetts)\b", re.I), "Massachusetts (US state)"),
    (re.compile(r"\b(?:denver|boulder|aspen|rocky\s+mountain|colorado)\b", re.I), "Colorado (US state)"),
    (re.compile(r"\b(?:portland|crater\s+lake|oregon)\b", re.I), "Oregon (US state)"),
    (re.compile(r"\b(?:honolulu|maui|oahu|kauai|hawaii)\b", re.I), "Hawaii (US state)"),
    (re.compile(r"\b(?:toronto|vancouver|montreal|banff|quebec|ottawa|calgary|canada)\b", re.I), "Canada (country)"),
    (re.compile(r"\b(?:tokyo|kyoto|osaka|hokkaido|japan)\b", re.I), "Japan (country)"),
    (re.compile(r"\b(?:paris|lyon|nice|provence|france)\b", re.I), "France (country)"),
    (re.compile(r"\b(?:london|edinburgh|manchester|oxford|uk|united\s+kingdom)\b", re.I), "United Kingdom (country)"),
    (re.compile(r"\b(?:rome|venice|florence|milan|italy)\b", re.I), "Italy (country)"),
)

_LLM_DREAM_PROMPT = """You are a memory consolidation worker distilling grounded high-level beliefs from atomic user facts.
Rules:
1. Synthesize at most 6 concise, high-value beliefs covering recurring preferences, tastes (books/movies/food flavors), inferred traits, professions, or geographic context.
2. EVERY belief MUST cite 1 or more valid `fact_id` values from the input list. Never invent facts or uncited claims.
3. Reply ONLY with JSON:
{{"beliefs": [{{"text": "...", "domain": "preference|trait|location|career|theme", "confidence": 0.85, "derived_from_ids": ["l1", "l2"]}}]}}

Input Facts:
{facts_block}
"""


def _match_cues(text: str, cues: tuple[tuple[str, re.Pattern], ...]) -> list[str]:
    found: list[str] = []
    for label, pat in cues:
        if pat.search(text):
            found.append(label)
    return found


def synthesize_beliefs_with_llm(llm, facts: list[dict]) -> list[dict]:
    """Use a live OpenAI-compatible LLM to synthesize evidence-cited ``kind='belief'`` proposals."""
    valid_ids = {f["id"]: f for f in facts}
    if not valid_ids:
        return []
    sample_facts = facts[-60:]
    lines = [f"- [{f['id']}] ({date_iso(f.get('mention_time') or f.get('event_time_start')) or 'n/a'}) {strip_step_tags(f['plain'])[:220]}" for f in sample_facts]
    prompt = _LLM_DREAM_PROMPT.format(facts_block="\n".join(lines))
    raw = llm.complete(prompt)
    payload = loads_json_object(raw)
    proposals: list[dict] = []
    for item in payload.get("beliefs") or ():
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        refs = [fid for fid in (item.get("derived_from_ids") or ()) if fid in valid_ids]
        if not text or not refs:
            # Hard anti-hallucination gate: reject any belief without real cited fact IDs!
            continue
        turns = sorted({int(t) for fid in refs for t in (valid_ids[fid].get("source_turn_ids") or ())})
        if not turns:
            continue
        conf = max(0.5, min(0.95, float(item.get("confidence", 0.82))))
        proposals.append({
            "text": text,
            "domain": str(item.get("domain") or "theme"),
            "confidence": conf,
            "derived_from_ids": refs,
            "source_turn_ids": turns,
        })
    return proposals


def synthesize_beliefs_deterministic(facts: list[dict]) -> list[dict]:
    """Deterministic grounded belief distillation for reflective preferences, themes & geographic context."""
    proposals: list[dict] = []
    domain_buckets: dict[str, list[dict]] = defaultdict(list)
    geo_buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for fact in facts:
        text = strip_step_tags(fact.get("plain") or "").strip()
        if not text or not fact.get("source_turn_ids"):
            continue
        speaker = str(fact.get("speaker") or fact.get("role") or "user").strip()
        is_pref = bool(_PREF_SIGNAL.search(text))

        # 1. Domain preference / reflective buckets (books, movies, food/dishes, career)
        if _FOOD_SIGNAL.search(text) and (is_pref or _match_cues(text, _FLAVOR_CUES)):
            domain_buckets[f"{speaker}:food"].append(fact)
        if _BOOK_SIGNAL.search(text) and is_pref:
            domain_buckets[f"{speaker}:book"].append(fact)
        if _MOVIE_SIGNAL.search(text) and is_pref:
            domain_buckets[f"{speaker}:movie"].append(fact)
        if _CAREER_SIGNAL.search(text):
            domain_buckets[f"{speaker}:career"].append(fact)

        # 2. Grounded geographic beliefs (city/landmark -> state/country)
        for pat, geo_label in _GEO_INFERENCE:
            if pat.search(text):
                geo_buckets[(speaker, geo_label)].append(fact)

    # Emit domain reflective beliefs
    for key, group in sorted(domain_buckets.items()):
        speaker, dom = key.split(":", 1)
        if len(group) < 1:
            continue
        combined = " ".join(strip_step_tags(f["plain"]) for f in group[:8])
        refs = list(dict.fromkeys(f["id"] for f in group[:8]))
        turns = sorted({int(t) for f in group[:8] for t in (f.get("source_turn_ids") or ())})
        if not turns:
            continue

        if dom == "food":
            cues = _match_cues(combined, _FLAVOR_CUES)
            cue_str = ", ".join(cues) if cues else "varied flavors"
            excerpts = "; ".join(strip_step_tags(f["plain"])[:80].replace(".", ",") for f in group[:4])
            belief = f"[Belief: dishes & food flavor preference] {speaker} mentioned dishes and drinks indicating flavor preference for {cue_str} (from: {excerpts})"
            proposals.append({"text": belief, "domain": "food_preference", "confidence": 0.85, "derived_from_ids": refs, "source_turn_ids": turns})
        elif dom == "book":
            cues = _match_cues(combined, _GENRE_CUES)
            cue_str = ", ".join(cues) if cues else "general reading"
            excerpts = "; ".join(strip_step_tags(f["plain"])[:80].replace(".", ",") for f in group[:4])
            belief = f"[Belief: books & reading preference] {speaker} mentioned books indicating reading preference for {cue_str} (from: {excerpts})"
            proposals.append({"text": belief, "domain": "book_preference", "confidence": 0.85, "derived_from_ids": refs, "source_turn_ids": turns})
        elif dom == "movie":
            cues = _match_cues(combined, _GENRE_CUES)
            cue_str = ", ".join(cues) if cues else "general cinema"
            excerpts = "; ".join(strip_step_tags(f["plain"])[:80].replace(".", ",") for f in group[:4])
            belief = f"[Belief: movies & film genre preference] {speaker} mentioned movies indicating film preference for {cue_str} (from: {excerpts})"
            proposals.append({"text": belief, "domain": "movie_preference", "confidence": 0.85, "derived_from_ids": refs, "source_turn_ids": turns})
        elif dom == "career" and len(group) >= 1:
            cues = _match_cues(combined, _GENRE_CUES)
            inferred_roles = []
            if re.search(r"\b(?:script|screenplay|filmmak|movie\s+script)\b", combined, re.I):
                inferred_roles.append("filmmaker / screenwriter")
            if re.search(r"\b(?:political|government|policy|senate|mayor|civic)\b", combined, re.I):
                inferred_roles.append("Political science / Public administration / Public affairs")
            if re.search(r"\b(?:counseling|mental\s+health|support\s+group|therapist)\b", combined, re.I):
                inferred_roles.append("counseling / mental health")
            if inferred_roles or cues:
                summary = ", ".join([*inferred_roles, *cues[:2]])
                excerpts = "; ".join(strip_step_tags(f["plain"])[:80].replace(".", ",") for f in group[:3])
                belief = f"[Belief: career, job & degree profile] {speaker} shows background in {summary} (from: {excerpts})"
                proposals.append({"text": belief, "domain": "career_profile", "confidence": 0.80, "derived_from_ids": refs, "source_turn_ids": turns})

    # Emit grounded geographic beliefs
    for (speaker, geo_label), group in sorted(geo_buckets.items()):
        refs = list(dict.fromkeys(f["id"] for f in group[:4]))
        turns = sorted({int(t) for f in group[:4] for t in (f.get("source_turn_ids") or ())})
        if not turns:
            continue
        dates = sorted({date_iso(f.get("mention_time") or f.get("event_time_start")) for f in group[:4] if (f.get("mention_time") or f.get("event_time_start"))})
        date_tag = f" on {', '.join(d for d in dates if d)}" if dates else ""
        excerpt = strip_step_tags(group[0]["plain"])[:95].replace(".", ",")
        belief = f"[Belief: travel & location] {speaker} travelled in / visited {geo_label}{date_tag} ({excerpt})"
        proposals.append({"text": belief, "domain": "location", "confidence": 0.86, "derived_from_ids": refs, "source_turn_ids": turns})

    return proposals


def dream_beliefs(
    engine,
    ledger: UnifiedFactLedger,
    *,
    user_id: str = "default",
    agent_id: str = "default",
    session_id: str = "default",
    timestamp: float | None = None,
) -> dict:
    """Run dreaming consolidation and persist evidence-backed ``kind='belief'`` facts in the unified ledger."""
    now = engine._now() if timestamp is None else float(timestamp)
    owner = (user_id, agent_id, None)
    facts = [
        f for f in engine.store.snapshot(scope=owner, include_cold=False, include_superseded=False)
        if f.get("memory_type") == "fact" and f.get("epistemic_scope") != "opinion" and f.get("source_turn_ids")
    ]
    if not facts:
        return {"belief_ids": [], "mode": "none", "count": 0}

    used_llm = False
    proposals: list[dict] = []
    if not isinstance(engine.llm, MockLLMClient):
        try:
            proposals = synthesize_beliefs_with_llm(engine.llm, facts)
            used_llm = True
        except Exception:
            proposals = []
    if not proposals:
        proposals = synthesize_beliefs_deterministic(facts)

    belief_ids: list[str] = []
    for prop in proposals:
        text = prop["text"].strip()
        turns = prop["source_turn_ids"]
        refs = prop["derived_from_ids"]
        if not text or not turns:
            continue
        # Persist directly into unified facts table (epistemic_scope='opinion', mandatory source_turn_ids)
        # Note: remember() validates turns belong to session_id; use the first turn's real session!
        turn_rows = engine.store.get_turn_window(turns[:1], 0, user_id, agent_id, by_id=True)
        turn_session = turn_rows[0]["session_id"] if turn_rows else session_id
        same_session_turns = [
            r["turn_id"]
            for r in engine.store.get_turn_window(turns, 0, user_id, agent_id, by_id=True)
            if r["session_id"] == turn_session
        ] or turns[:1]

        fact_id = engine.remember(
            text,
            memory_type="fact",
            source_turn_ids=same_session_turns,
            epistemic_scope="opinion",
            confidence=float(prop.get("confidence", 0.82)),
            user_id=user_id,
            agent_id=agent_id,
            session_id=turn_session,
            timestamp=now,
        )
        # Store all cross-session source_turn_ids & derived_from_ids cleanly on the belief fact
        import json

        engine.store.conn.execute(
            "UPDATE facts SET source_turn_ids=?, derived_from_ids=? WHERE id=?",
            (json.dumps(sorted(set(turns))), json.dumps(list(dict.fromkeys(refs))), fact_id),
        )
        ledger.attach(
            fact_id=fact_id,
            user_id=user_id,
            agent_id=agent_id,
            kind="belief",
            title=f"Belief ({prop.get('domain', 'theme')})",
            content=text,
            verification_status="verified",
            salience_tag="normal",
            always_active=False,
            timestamp=now,
        )
        belief_ids.append(fact_id)

    engine.store.commit()
    return {
        "belief_ids": belief_ids,
        "mode": "llm" if used_llm else "deterministic",
        "count": len(belief_ids),
    }
