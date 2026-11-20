"""Conservative support checks for statements and their cited evidence."""

import re
import unicodedata
from fractions import Fraction

from memengine.chrono import resolve_temporal_spans
from memengine.linguistics import entity_name
from memengine.textutil import strip_step_tags
from memengine.temporal import competition_terms
from memengine.versioning import extract_versions, version_order


_CLAUSE = re.compile(r"(?<=[。！？!?;；])\s*|(?<=\.)\s+|\n+")
_STATE_TAG = re.compile(r"^(?:\[(?:历史快照|历史已变更|较早|可能已过时/较早记录)\]\s*)+")
_DENIAL = re.compile(r"\b(?:not true|false|denies?|incorrect|does not|did not|never|no evidence)\b|并非|并不|不属实|否认", re.I)
_POLARITY = re.compile(r"\b(?:not|no|never|neither|without)\b|n['’]t\b|并非|并不|否认|不曾|从未", re.I)
_EQUIVALENT = {
    "purchase": "buy", "purchases": "buys", "purchased": "bought", "purchasing": "buying",
    "commence": "begin", "commences": "begins", "commenced": "began", "commencing": "beginning",
    "metre": "meter", "metres": "meters",
}
_UNITS = {
    "mm": ("length", 1), "cm": ("length", 10), "m": ("length", 1000), "km": ("length", 1000000),
    "mg": ("mass", 1), "g": ("mass", 1000), "kg": ("mass", 1000000),
    "second": ("duration", 1), "minute": ("duration", 60), "hour": ("duration", 3600),
}
_UNIT_NAME = {"millimeter": "mm", "centimeter": "cm", "meter": "m", "kilometer": "km",
              "milligram": "mg", "gram": "g", "kilogram": "kg"}
_QUANTITY = re.compile(r"(?<![\w.,+-])([+-]?\d+(?:\.\d+)?)\s*"
                       r"(mm|cm|km|m|mg|kg|g|(?:milli|centi|kilo)?met(?:er|re)s?|(?:milli|kilo)?grams?|"
                       r"seconds?|minutes?|hours?)\b", re.I)
_LENGTH_CONTEXT = re.compile(r"\b(?:length|width|height|depth|distance|walk|run|trip|wide|tall|long|away)\b", re.I)
_STOP = {"what", "where", "when", "which", "who", "why", "how", "the", "is", "was", "does", "did", "this", "that",
         "can", "could", "would", "please", "tell", "me", "about", "for", "it", "and", "has", "have", "any"}


def _tokens(text):
    text = unicodedata.normalize("NFKC", strip_step_tags(text)).casefold()
    return re.findall(r"(?<![\w.])[-+]\d+(?:\.\d+)?|[a-z0-9]+(?:[.:-][a-z0-9]+)*|[\u3400-\u9fff]", text)


def _quantity(match):
    unit = match[2].casefold().removesuffix("s").replace("metre", "meter")
    if unit == "m" and not _LENGTH_CONTEXT.search(match.string):
        return match[0]
    value = Fraction(match[1])
    dimension, scale = _UNITS[_UNIT_NAME.get(unit, unit)]
    value *= scale
    sign = "negative" if value < 0 else "positive"
    return f" quantity:{dimension}:{sign}:{abs(value.numerator)}:{value.denominator} "


def _comparable(text, spans, aliases, *, first_person=False):
    normalized = unicodedata.normalize("NFKC", strip_step_tags(text)).casefold()
    for span in spans:
        if span.start is not None or span.end is not None:
            normalized = normalized.replace(span.text.casefold(),
                                            f" calendar:{span.precision}:{span.start}:{span.end} ", 1)
    words = _tokens(_QUANTITY.sub(_quantity, normalized))
    for alias in sorted((_tokens(entity_name(name)) for name in aliases), key=len, reverse=True):
        if alias and words[:len(alias)] == alias:
            words = ["<canonical-subject>", *words[len(alias):]]
            break
    else:
        if aliases and first_person and words[:1] == ["i"]:
            words = ["<canonical-subject>", *words[1:]]
    return [_EQUIVALENT.get(word, word) for word in words]


def _contains(words, wanted):
    return bool(wanted) and any(words[i:i + len(wanted)] == wanted for i in range(len(words) - len(wanted) + 1))


def supports_claim(claim, evidence, *, equivalences=False, aliases=(), claim_time=None, evidence_time=None, speaker=None):
    """Require direct support in one clause, including intact slot values."""
    claim = _STATE_TAG.sub("", claim)
    wanted = _tokens(claim)
    if not wanted:
        return False
    versions = extract_versions(claim)
    spans = resolve_temporal_spans(claim, claim_time)
    for clause in _CLAUSE.split(evidence):
        if _DENIAL.search(clause) and not _DENIAL.search(claim):
            continue
        words = _tokens(clause)
        supported = extract_versions(clause)
        if any(not any(version_order(want) == version_order(have)
                       and want.qualifier_op == have.qualifier_op
                       and want.product_family in {have.product_family, "default_pkg"}
                       for have in supported) for want in versions):
            continue
        calendar = resolve_temporal_spans(clause, evidence_time)
        if any(not any((want.start, want.end, want.precision) == (have.start, have.end, have.precision)
                       for have in calendar) for want in spans):
            continue
        if _contains(words, wanted):
            return True
        if equivalences and bool(_POLARITY.search(clause)) == bool(_POLARITY.search(claim)):
            body = re.sub(r"^" + re.escape(speaker) + r":\s*", "", clause, count=1, flags=re.I) if speaker else clause
            if _contains(_comparable(body, calendar, aliases, first_person=bool(speaker)),
                         _comparable(claim, spans, aliases)):
                return True
    return False


def claims_from_text(text):
    return [part.strip() for part in _CLAUSE.split(strip_step_tags(text)) if _tokens(part)]


def evidence_matches_query(query, row):
    terms = competition_terms(query) - _STOP
    text = row.get("plain") or row.get("text", "")
    overlap = terms & competition_terms(text)
    return bool(overlap) and (len(terms) == 1 or len(overlap) >= 2 or row.get("entity_anchor"))


def fact_is_current(row, as_of, historical):
    return (not row.get("is_stale") and row["status"] != "deleted"
            and (row["status"] == "active" or historical)
            and (as_of is None or ((row.get("expires_at") is None or row["expires_at"] > as_of)
                                  and (row.get("valid_from") is None or row["valid_from"] <= as_of))))


def admissible_evidence(bundle):
    scope = bundle["scope"]
    as_of = bundle["validity_as_of"]
    historical = bundle["parsed_query"]["query_type"] == "TIME_TRAVEL"
    eligible = {}
    for ident, row in bundle["evidence_map"].items():
        if any(row.get(key) != scope[key] for key in ("user_id", "agent_id")):
            continue
        if scope["session_id"] is not None and row.get("session_id") != scope["session_id"]:
            continue
        if row["kind"] in {"fact", "observation"}:
            if not fact_is_current(row, as_of, historical):
                continue
        elif row["kind"] == "working_item":
            if row["verification_status"] not in {"tool_verified", "user_confirmed"}:
                continue
        elif row.get("role") not in {"user", "tool"}:
            continue
        eligible[ident] = row
    return eligible


def assess_claims(payload, eligible, *, alias_lookup=None, claim_time=None):
    explicit = payload.get("claims")
    claims = explicit if explicit is not None else [{"text": text, "evidence_ids": []}
                                                    for text in claims_from_text(payload["final_answer"])]
    approved, unverified = [], []

    def supports(text, row):
        body = row.get("plain") or row["text"]
        options = dict(equivalences=True, claim_time=claim_time, evidence_time=row.get("mention_time"))
        if supports_claim(text, body, **options):
            return True
        aliases = alias_lookup(row) if alias_lookup else ()
        return bool(aliases) and supports_claim(text, body, aliases=aliases, speaker=row.get("speaker"), **options)

    for claim in claims:
        text, requested = claim["text"], claim["evidence_ids"]
        supporting = [key for key in (requested if explicit is not None else eligible)
                      if key in eligible and supports(text, eligible[key])]
        if supporting and (explicit is None or all(key in eligible for key in requested)):
            approved.append({"text": text, "evidence_ids": supporting})
        else:
            unverified.append({"text": text, "evidence_ids": [key for key in requested if key in eligible]})
    return approved, unverified
