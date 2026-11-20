"""Conservative, model-independent extraction of exact technical literals."""

from dataclasses import dataclass
import re
import unicodedata

_ATOM = r"[A-Za-z0-9_./@#+:\-]"
_VERSION = re.compile(rf"(?<!{_ATOM})v?\d+(?:\.\d+)+(?:[-+][A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)?(?![A-Za-z0-9_/@#+:\-]|\.\w)", re.I)
_TECH = re.compile(r"[A-Za-z0-9]+(?:[_.\-/@#+:][A-Za-z0-9_./@#+:\-]+)+|[A-Za-z]\w*", re.ASCII)
_QUOTED = re.compile(r'`([^`\n]+)`|"([^"\n]+)"|“([^”\n]+)”|《([^》\n]+)》')
_ERROR = re.compile(r"(?:[A-Z][A-Z0-9]*_)+(?:[A-Z0-9]+)$")
_CAMEL = re.compile(r"[a-z0-9][A-Z]")
_COMPARE = re.compile(r"升级|变更|对比|比较|之前|曾经|\b(?:upgrad\w*|migrat\w*|changelog|compar\w*|versus|vs|previous\w*|before)\b", re.I)


@dataclass(frozen=True)
class Identifier:
    ident: str
    ident_norm: str
    ident_type: str
    major_minor_family: str | None = None


def extract_identifiers(text: str) -> list[Identifier]:
    text = unicodedata.normalize("NFKC", text or "")
    found = {}
    version_spans = []
    for match in _VERSION.finditer(text):
        literal = match[0]
        norm = "v" + literal.casefold().removeprefix("v")
        family = "v" + ".".join(re.split(r"[-+]", norm[1:], maxsplit=1)[0].split(".")[:2])
        found[norm] = Identifier(literal, norm, "version", family)
        version_spans.append(match.span())
    for match in _TECH.finditer(text):
        if any(match.start() < end and start < match.end() for start, end in version_spans):
            continue
        literal = match[0].rstrip(".:")
        if not any(char.isalpha() for char in literal):
            continue
        # URLs and common prose apostrophes are handled by normal lexical recall.
        if "://" in literal or literal.lower() in {"https", "http"}:
            continue
        if _ERROR.fullmatch(literal):
            kind = "error_code"
        elif "_" in literal or "/" in literal or "." in literal or _CAMEL.search(literal):
            kind = "code_symbol"
        elif "-" in literal and len(literal) >= 5:
            kind = "project_token"
        elif literal.isupper() and len(literal) >= 3:
            kind = "code_symbol"
        else:
            continue
        norm = literal.casefold()
        found.setdefault(norm, Identifier(literal, norm, kind))
    for match in _QUOTED.finditer(text):
        literal = " ".join(next(group for group in match.groups() if group is not None).split())
        if 1 <= len(literal) <= 100 and any(char.isalnum() for char in literal) and not _VERSION.fullmatch(literal):
            norm = literal.casefold()
            found.setdefault(norm, Identifier(literal, norm, "quoted_name"))
    return list(found.values())


def compares_versions(query):
    return bool(_COMPARE.search(query))


def eligible_identifiers(candidates, query, historical, allow_empty=True):
    """Enforce explicit literal scope after candidate fusion."""
    ids = extract_identifiers(query)
    if any(item.ident_type == "version" for item in ids) and not historical and not compares_versions(query):
        matches = [fact for fact in candidates if not fact["version_mismatch"]]
        if allow_empty and not any(fact["version_match"] for fact in matches):
            return [], "version"
        return matches, "version"
    if (len(ids) == 1 and ids[0].ident_type not in {"quoted_name", "version"}
            and query.strip().strip("`") == ids[0].ident):
        if allow_empty and not any(fact["exact_ident_hit"] for fact in candidates):
            return [], "symbol"
        return candidates, "symbol"
    return candidates, None
