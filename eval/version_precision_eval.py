"""Eighty graded version queries over shared scope and realistic distractors."""
import argparse
from contextlib import closing
from pathlib import Path

from eval.paired import make_engine
from eval.static_report import table, write_result
from agentmem_core.versioning import parse_version, score_version_match, version_order

PRODUCTS = ("Python", "CMake", "CUDA", "Ubuntu", "API", "NodeJS", "PostgreSQL", "SQLite", "Java", "Rust")


def evaluate():
    rows = []
    engine = make_engine("mock")
    with closing(engine.store):
        for i, product in enumerate(PRODUCTS):
            other = PRODUCTS[(i+1) % len(PRODUCTS)]
            major = i+3
            b, s, p = f"{major}.11", f"{major}.11.2", f"{major}.11.4"
            cases = (
                ("exact", f"{product} v{s}", f"{product} v{s} supports the optional cache.", f"{product} v{p}", "EXACT"),
                ("containment", f"{product} v{b}", f"{product} v{p} enables audit mode.", f"{product} v{p}-rc.1", "CONTAINED_SPECIFIC"),
                ("coarse", f"{product} v{s}", f"{product} v{b} supports read replicas.", f"{product} v{p}", "UNDERSPECIFIED"),
                ("sibling", f"{product} v{s}", f"{product} v{s} changes the retry policy.", f"{product} v{major}.11.1", "EXACT"),
                ("numeric_order", f"{product} v{major}.10", f"{product} v{major}.10 accepts these flags.",
                 f"{product} v{major}.2", "EXACT"),
                ("prerelease", f"{product} v{s}", f"{product} v{s} release is stable for staging.", f"{product} v{s}-rc1", "EXACT"),
                ("range", f"{product} v{b}", f"Supports {product} >= {major}.10, < {major}.13.", f"{product} v{major}.13", "RANGE_SATISFIED"),
                ("cross_product", f"{product} v{b}", f"{product} v{b} supports local profiles.", f"{other} v{b}", "EXACT"),
            )
            for category, query, good, bad, expected in cases:
                scope = {"session_id": f"{i}-{category}"}
                target = engine.retain(good, mention_time="2024-03-01", speaker="Releases", **scope)
                rival = engine.retain(bad, mention_time="2024-03-02", speaker="Releases", **scope)
                expected_id = target["operations"][0]["line_id"]
                rival_id = rival["operations"][0]["line_id"]
                ranked = engine.recall(query, token_budget=384, reference_time="2024-04-01", read_only=True, **scope)["ranked"]
                top = ranked[0] if ranked else {}
                grade = score_version_match(parse_version(query), parse_version(good)).match_type
                if category in {"numeric_order", "prerelease"}:
                    grade_ok = grade == expected and version_order(parse_version(bad)) < version_order(parse_version(good))
                else:
                    grade_ok = grade == expected
                if category == "coarse":
                    grade_ok = grade_ok and "粒度不足" in top.get("text", "")
                rows.append({"category": category, "query": query, "expected": expected, "expected_id": expected_id,
                             "top_id": top.get("id"), "actual": top.get("version_grade"), "rival_id": rival_id,
                             "precision": top.get("id") == expected_id,
                             "grade_correct": grade_ok and top.get("version_grade") == expected,
                             "confusion": category in {"cross_product", "sibling", "prerelease"} and
                                          rival_id in {row["id"] for row in ranked}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = evaluate()
    precision = sum(row["precision"] for row in results) / len(results)
    accuracy = sum(row["grade_correct"] for row in results) / len(results)
    confusion = sum(row["confusion"] for row in results)
    write_result({"benchmark": "Version Precision 80", "precision_at_1": precision, "classification_accuracy": accuracy,
                  "version_confusion": confusion, "items": results}, args.out,
                 table(("类别", "题数", "首位正确", "分级正确", "串台"), [(name, len(part),
                       sum(row["precision"] for row in part), sum(row["grade_correct"] for row in part),
                       sum(row["confusion"] for row in part)) for name in dict.fromkeys(row["category"] for row in results)
                       for part in [[row for row in results if row["category"] == name]]])
                 + f"\n\nPrecision@1 {precision:.2%} · 分级准确率 {accuracy:.2%} · 串台 {confusion}")
    for row in results:
        if not row["precision"] or not row["grade_correct"] or row["confusion"]:
            print(f"FAIL: {row}")
    return int(precision < .95 or accuracy < .95 or confusion != 0)


if __name__ == "__main__":
    raise SystemExit(main())
