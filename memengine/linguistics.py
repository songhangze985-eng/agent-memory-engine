"""Small deterministic English inflection and first-person normalization."""
import re
import unicodedata

from memengine.textutil import strip_step_tags

_IRREGULAR = dict(zip("went gone bought taught ran made took taken saw seen did done had wrote written got gotten brought felt left met kept ate eaten drank drunk drove driven spoke spoken thought knew known gave given children people men women mice teeth feet better".split(),
                      "go go buy teach run make take take see see do do have write write get get bring feel leave meet keep eat eat drink drink drive drive speak speak think know know give give child person man woman mouse tooth foot good".split()))
_WORD = re.compile(r"(?<![\w./_@+-])[A-Za-z]{3,}(?![\w/_@+-]|\.[A-Za-z0-9])")
_PRONOUN = re.compile(r"\b(I(?:['’](?:m|ve|d|ll))?|my|me|mine|we|us|our|ours)\b|我们的|我们|我的|我", re.I)


def stem(word):
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith(("ing", "ed")):
        root = word[:-3] if word.endswith("ing") else word[:-2]
        if root.endswith(("at", "iz", "bl")):
            return root + "e"
        if len(root) > 3 and root[-1] == root[-2] and root[-1] in "bdgmnprt":
            return root[:-1]
        return root
    if len(word) > 4 and word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is", "ous")):
        return word[:-1]
    return word


def stem_features(text):
    terms = dict.fromkeys(match[0].lower() for match in _WORD.finditer(text))
    return [base for word in terms if (base := stem(word)) != word]


def stem_terms(text):
    return {stem(match[0].lower()) for match in _WORD.finditer(text)}


def entity_name(name):
    name = unicodedata.normalize("NFKC", name).strip()
    name = re.sub(r"^(?:(?:mr|mrs|ms|miss|dr|prof|sir|the)\.?\s+)+", "", name, flags=re.I)
    return " ".join(name.casefold().strip(" .").split())


def index_text(text, speaker, spans):
    body = strip_step_tags(text)
    if speaker:
        body = re.sub(r"^" + re.escape(speaker) + r":\s*", "", body, count=1, flags=re.I)

        def attribute(match):
            term = match[0].lower().replace("’", "'")
            if term in ("my", "mine", "我的"):
                return speaker + ("的" if term == "我的" else "'s")
            if term in ("we", "us", "our", "ours", "我们", "我们的"):
                if term.startswith("我"):
                    return speaker + ("等人的" if term == "我们的" else "等人")
                return f"{speaker} and others" + ("'s" if term in ("our", "ours") else "")
            expansion = {"i'm": " is", "i've": " has", "i'd": " would", "i'll": " will"}
            return speaker + expansion.get(term, "")
        body = f"{speaker}: " + _PRONOUN.sub(attribute, body)
    dates, calendar_keys = [], {}
    for span in spans:
        start, end = span.iso_start, span.iso_end
        if start is None or end is None:
            continue
        dates.append(f"[occurrence: {start} to {end}]" if start != end else f"[occurrence: {start}]")
        for width in (7, 4):
            if start[:width] == end[:width]:
                calendar_keys[start[:width]] = None
    return " ".join([body, *dates, *calendar_keys, *stem_features(body)]).strip()
