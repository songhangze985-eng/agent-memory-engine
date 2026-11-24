"""Sixty deterministic calendar assertions, including live ledger and gap checks."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from eval.paired import make_engine
from eval.static_report import table, write_result
from agentmem_core.chrono import days_between, interval_score, parse_session_clock, resolve_temporal_spans

ANCHOR = "2024-03-11T13:00:00Z"
WINDOWS = [
    ("Yesterday", "2024-03-10", "2024-03-10"), ("Today", "2024-03-11", "2024-03-11"),
    ("Tomorrow", "2024-03-12", "2024-03-12"), ("last night", "2024-03-10", "2024-03-11"),
    ("last week", "2024-03-04", "2024-03-10"), ("this week", "2024-03-11", "2024-03-17"),
    ("next week", "2024-03-18", "2024-03-24"), ("last month", "2024-02-01", "2024-02-29"),
    ("this month", "2024-03-01", "2024-03-31"), ("next month", "2024-04-01", "2024-04-30"),
    ("last year", "2023-01-01", "2023-12-31"), ("this year", "2024-01-01", "2024-12-31"),
    ("next year", "2025-01-01", "2025-12-31"), ("last Monday", "2024-03-04", "2024-03-04"),
    ("last Sunday", "2024-03-10", "2024-03-10"), ("3 weeks ago", "2024-02-19", "2024-02-19"),
    ("two months ago", "2024-01-11", "2024-01-11"), ("one year ago", "2023-03-11", "2023-03-11"),
    ("in 2 weeks", "2024-03-25", "2024-03-25"), ("last May", "2023-05-01", "2023-05-31"),
    ("in May 2023", "2023-05-01", "2023-05-31"), ("in 2022", "2022-01-01", "2022-12-31"),
    ("February 29, 2024", "2024-02-29", "2024-02-29"), ("8 May", "2024-05-08", "2024-05-08"),
    ("last summer", "2023-06-01", "2023-08-31"), ("next fall", "2024-09-01", "2024-11-30"),
    ("this spring", "2024-03-01", "2024-05-31"), ("from May 2023 to June 2023", "2023-05-01", "2023-06-30"),
    ("昨天", "2024-03-10", "2024-03-10"), ("前天", "2024-03-09", "2024-03-09"),
    ("明天", "2024-03-12", "2024-03-12"), ("上周", "2024-03-04", "2024-03-10"),
    ("这周", "2024-03-11", "2024-03-17"), ("下周", "2024-03-18", "2024-03-24"),
    ("上个月", "2024-02-01", "2024-02-29"), ("这个月", "2024-03-01", "2024-03-31"),
    ("去年", "2023-01-01", "2023-12-31"), ("今年", "2024-01-01", "2024-12-31"),
    ("三天前", "2024-03-08", "2024-03-08"), ("两周前", "2024-02-26", "2024-02-26"),
    ("三个月前", "2023-12-11", "2023-12-11"), ("两年前", "2022-03-11", "2022-03-11"),
    ("2023年5月", "2023-05-01", "2023-05-31"), ("5月8日", "2024-05-08", "2024-05-08"),
    ("从2023年12月到2024年2月", "2023-12-01", "2024-02-29"),
    ("since 2021", "2021-01-01", "2024-03-11"), ("before 2022", None, "2021-12-31"),
    ("after May 2023", "2023-06-01", None),
]
CLOCKS = [
    ("2023-05-08T13:56:00Z", "2023-05-08T13:56:00+00:00"),
    ("1:56 pm on 8 May, 2023", "2023-05-08T13:56:00+00:00"),
    ("2023/05/20 (Sat) 14:30", "2023-05-20T14:30:00+00:00"),
    ("2026年11月20日", "2026-11-20T00:00:00+00:00"),
    ("2024-01-01T03:00:00+05:30", "2023-12-31T21:30:00+00:00"),
    ("2024/02/29", "2024-02-29T00:00:00+00:00"),
]


def evaluate():
    rows = []
    def check(group, prompt, got, expected):
        rows.append({"group": group, "prompt": prompt, "expected": expected, "actual": got, "pass": got == expected})
    for text, expected in CLOCKS:
        check("session", text, datetime.fromtimestamp(parse_session_clock(text), timezone.utc).isoformat(), expected)
    for text, first, last in WINDOWS:
        span = resolve_temporal_spans(text, ANCHOR)[0]
        check("span", text, (span.iso_start, span.iso_end), (first, last))
    def span(text):
        return resolve_temporal_spans(text, ANCHOR)[0]
    check("interval", "May overlaps May–June", interval_score(span("May 2023").start, span("May 2023").end,
          span("from May 2023 to June 2023")), 1.0)
    check("interval", "April outside May–June", interval_score(span("April 2023").start, span("April 2023").end,
          span("from May 2023 to June 2023")), 0.0)
    check("distance", "leap year boundary", days_between(span("2024-02-28"), span("2024-03-02")), 3)
    check("distance", "跨年", days_between(span("2023-12-31"), span("2024-01-02")), 2)
    engine = make_engine("mock")
    with closing(engine.store):
        engine.retain("2025年1月提到2024年6月结婚。", mention_time="2025-01-20", speaker="李明")
        engine.retain("李明在2024年8月搬家。", mention_time="2025-01-21", speaker="李明")
        hits = engine.recall("李明在2024年6月做了什么？", reference_time="2025-01-22", read_only=True, token_budget=256)["ranked"]
        event = hits[0]
        check("ledger", "actual occurrence vs mention", (event["occurrence_start"], event["mention_time"]),
              (parse_session_clock("2024-06-01"), parse_session_clock("2025-01-20")))
        check("ledger", "indexed relative anchor", "occurrence: 2024-06-01" in event["normalized_text"], True)
    assert len(rows) == 60
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = evaluate()
    rate = sum(row["pass"] for row in rows) / len(rows)
    write_result({"benchmark": "Chrono 60", "accuracy": rate, "items": rows}, args.out,
                 table(("分类", "题数", "正确"), [(name, sum(x["group"] == name for x in rows),
                       sum(x["group"] == name and x["pass"] for x in rows)) for name in dict.fromkeys(x["group"] for x in rows)])
                 + f"\n\n日历区间命中率 {rate:.2%}")
    for item in rows:
        if not item["pass"]:
            print(f"FAIL: {item}")
    return int(rate < .95)


if __name__ == "__main__":
    raise SystemExit(main())
