"""Asymmetric comparison of concrete, partial and ranged software versions."""
from __future__ import annotations

from dataclasses import dataclass, replace
import operator
import re

_DEFAULT = "default_pkg"
_VERSION = re.compile(
    r"(?<![\w./:-])(?P<op>>=|<=|==|~=|~|\^|>|<)?\s*(?P<v>v)?"
    r"(?P<nums>\d+(?:\.\d+){0,3})(?P<wild>\.[xX*])?"
    r"(?:-(?P<pre>[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*))?(?P<plus>\+(?:[A-Za-z0-9][A-Za-z0-9.-]*)?)?"
    r"(?![\w/-]|\.\w)", re.I)
_CLAUSE = re.compile(r"[,;.!?\n]|(?:\b(?:and|or|versus|vs)\b)", re.I)
_FAMILY = re.compile(r"([A-Za-z][A-Za-z0-9_-]{1,48})\s*(?:(?:version|release|ver(?:sion)?\.)\s*)?[:=]?\s*$", re.I)
_STOP = set("to from on at with and or in since before after was is use uses used runs running supports support requires require upgrade upgraded latest current versions version release ver between than thanonly build installed updated moved now earlier previously configuration config bug issue answer for compatible only no than package pkg runtime".split())
_KNOWN = {"python", "cmake", "cuda", "ubuntu", "api", "node", "nodejs", "postgresql", "sqlite", "java", "rust"}
_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}


@dataclass(frozen=True)
class Version:
    product_family: str
    numeric_tuple: tuple[int, ...]
    precision_depth: int
    prerelease: str | None = None
    qualifier_op: str = "=="
    bounds: tuple[Version, ...] = ()

    @property
    def literal(self):
        core = "v" + ".".join(map(str, self.numeric_tuple))
        return core + (f"-{self.prerelease}" if self.prerelease else "")


@dataclass(frozen=True)
class VersionMatch:
    match_type: str
    boost: float
    confidence: float
    multiplier: float = 1.0


MATCHES = {
    "EXACT": VersionMatch("EXACT", .45, 1),
    "CONTAINED_SPECIFIC": VersionMatch("CONTAINED_SPECIFIC", .32, .9),
    "RANGE_SATISFIED": VersionMatch("RANGE_SATISFIED", .30, .85),
    "UNDERSPECIFIED": VersionMatch("UNDERSPECIFIED", 0, .5, .65),
    "CONFLICT": VersionMatch("CONFLICT", 0, 0, .12),
}


def extract_versions(text):
    found = []
    for match in _VERSION.finditer(text):
        prefix = text[max(0, match.start()-80):match.start()].rstrip()
        family_match = _FAMILY.search(_CLAUSE.split(prefix)[-1])
        family = family_match[1].casefold() if family_match else _DEFAULT
        if family in _STOP:
            family = _DEFAULT
        nums = tuple(int(x) for x in match["nums"].split("."))
        if len(nums) == 1 and not (match["v"] or match["op"] or match["wild"] or family in _KNOWN):
            continue
        if 1900 <= nums[0] <= 2099 and not match["v"] and family not in _KNOWN:
            continue
        op = match["op"] or (">=" if match["plus"] == "+" else "==")
        if op == "~=" and len(nums) < 2:
            continue
        if match["wild"] and op == "==":
            op = "prefix"
        found.append(Version(family, nums, len(nums), match["pre"], op))
    if len(found) > 1 and all(item.qualifier_op != "==" for item in found):
        family = found[0].product_family
        if all(item.product_family in (_DEFAULT, family) for item in found):
            return [replace(found[0], bounds=tuple(found))]
    return found


def parse_version(text):
    versions = extract_versions(text)
    return versions[0] if versions else None


def version_order(version):
    digits = (*version.numeric_tuple, *(0 for _ in range(4 - version.precision_depth)))
    pre = re.sub(r"^(alpha|beta|rc)(\d+)$", r"\1.\2", version.prerelease or "")
    ids = tuple((0, int(item)) if item.isdigit() else (1, item) for item in pre.split(".") if item)
    return digits, (1,) if not version.prerelease else (0, ids)


def _satisfies(bound, concrete):
    if bound.qualifier_op == "prefix":
        return concrete.numeric_tuple[:bound.precision_depth] == bound.numeric_tuple
    if bound.qualifier_op == "~=":
        depth = bound.precision_depth - 1
        return (concrete.numeric_tuple[:depth] == bound.numeric_tuple[:depth]
                and version_order(concrete) >= version_order(bound))
    if bound.qualifier_op == "~":
        depth = min(2, bound.precision_depth)
        return (concrete.numeric_tuple[:depth] == bound.numeric_tuple[:depth]
                and version_order(concrete) >= version_order(bound))
    if bound.qualifier_op == "^":
        idx = next((i for i, n in enumerate(bound.numeric_tuple) if n != 0), bound.precision_depth - 1)
        return (version_order(concrete) >= version_order(bound)
                and concrete.numeric_tuple[:idx+1] == bound.numeric_tuple[:idx+1])
    return _OPS[bound.qualifier_op](version_order(concrete), version_order(bound))


def score_version_match(query_ver, mem_ver):
    q, m = query_ver, mem_ver
    if q.product_family != m.product_family and _DEFAULT not in (q.product_family, m.product_family):
        return MATCHES["CONFLICT"]
    if q.qualifier_op != "==" or m.qualifier_op != "==":
        if q.qualifier_op != "==" and m.qualifier_op != "==":
            kind = "RANGE_SATISFIED" if q == m else "UNDERSPECIFIED"
        else:
            limits, concrete = (q, m) if q.qualifier_op != "==" else (m, q)
            kind = "RANGE_SATISFIED" if all(_satisfies(bound, concrete) for bound in (limits.bounds or (limits,))) else "CONFLICT"
        return MATCHES[kind]
    a, b = q.numeric_tuple, m.numeric_tuple
    if a == b and version_order(q) == version_order(m):
        return MATCHES["EXACT"]
    if len(a) < len(b) and b[:len(a)] == a and not q.prerelease:
        return MATCHES["CONTAINED_SPECIFIC"]
    if len(b) < len(a) and a[:len(b)] == b and not m.prerelease:
        return MATCHES["UNDERSPECIFIED"]
    return MATCHES["CONFLICT"]
