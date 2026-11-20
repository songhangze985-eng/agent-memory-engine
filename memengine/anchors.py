"""Index-only context for a short reply to the preceding conversational turn."""

import re

from memengine.textutil import estimate_tokens, strip_step_tags

_SPEAKER = re.compile(r"^([\w][\w .'-]{0,35}):\s+")
_REFERENCE = re.compile(r"^(?:yes|no|sure|it|that|this|he|she|they|there|what about|how about|"
                        r"that's|thanks|yeah|absolutely|i did|i do|i have|i am)\b|"
                        r"^(?:是的|对的|没错|这[个件]|那[个件]|[他她它]|后来呢)", re.I)


def turn_context(current, previous):
    """Return a bounded previous utterance for a dependent conversational reply."""
    if not previous or not 0 <= current["ingest_ts"] - previous["ingest_ts"] <= 1200:
        return ""
    reply, antecedent = (strip_step_tags(row["text"]) for row in (current, previous))
    speaker, prior_speaker = _SPEAKER.match(reply), _SPEAKER.match(antecedent)
    distinct = (current["role"] != previous["role"]
                or bool(speaker and prior_speaker and speaker[1].casefold() != prior_speaker[1].casefold()))
    body = reply[speaker.end():] if speaker else reply
    if not distinct or not body or (len(body) >= 80 and not _REFERENCE.search(body)):
        return ""
    words = antecedent.split()
    shortened = ""
    for word in words:
        snippet = f"{shortened} {word}" if shortened else word
        if estimate_tokens(snippet) > 40:
            break
        shortened = snippet
    return shortened or antecedent[:40]
