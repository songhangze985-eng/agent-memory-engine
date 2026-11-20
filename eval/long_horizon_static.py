"""Deterministic long-horizon, version-chain and archive integration benchmark."""
import argparse
import random
from contextlib import closing
from pathlib import Path

from eval.metrics import mean
from eval.paired import make_engine
from eval.static_report import table, write_result
from memengine.assembler import assemble_context

DAY = 86400
SCALES = (500, 1500, 3500)


def horizon_case(turn_count, seed):
    rng = random.Random(seed)
    engine = make_engine("mock")
    with closing(engine.store):
        store = engine.store
        start = 1_700_000_000
        key = ("实体:北辰预算",)
        dining_key = ("实体:北辰餐饮",)
        version_ids, fragments = [], {}
        routine = None
        # The fixture writes explicit facts through the production SQLite ledger.
        with store.transaction():
            block = store.create_block("北辰项目", start)
            for index in range(turn_count):
                stamp = start + index if index < turn_count * .8 else start + 110 * DAY + index
                update_index = int(turn_count * .8)
                if index == update_index:
                    stamp = start + 80 * DAY + index
                if index == 0:
                    body = "北辰餐饮禁忌：林澈对苯甲酸钠过敏，禁止使用含苯甲酸钠的添加剂。"
                    turn = store.append_turn(body, stamp)
                    taboo = store.insert_fact(block_id=block, plain=body, formatted=body, ingest_time=stamp, source_turn_ids=[turn],
                                              isolated=True, is_constraint=True, zkeys=dining_key)
                elif index <= 5 or index == update_index:
                    body = f"北辰预算的状态是 {20000 + (len(version_ids) + 1) * 1500} 元。"
                    turn = store.append_turn(body, stamp)
                    args = dict(plain=body, formatted=body, attribute="budget", value=str(20000 + (len(version_ids) + 1) * 1500),
                                ingest_time=stamp, zkeys=key)
                    fact = (store.insert_fact(block_id=block, source_turn_ids=[turn], **args) if not version_ids
                            else store.supersede_fact(version_ids[-1], current_turn_id=turn, **args))
                    version_ids.append(fact)
                elif index < 54:
                    group = (index - 6) // 2
                    body = f"北辰餐饮第{group}号例会确认了普通餐饮清点流程、座位安排与送餐回执。"
                    turn = store.append_turn(body, stamp)
                    if group in fragments:
                        store.merge_line_mentions(fragments[group], [], turn_id=turn)
                    else:
                        fragments[group] = store.insert_fact(block_id=block, plain=body, formatted=body, ingest_time=stamp,
                                                             source_turn_ids=[turn], zkeys=dining_key)
                elif index >= turn_count - 8:
                    body = "北辰餐饮每周例会按原餐饮菜单与常规订餐安排执行，后勤按时领取并归档回执。"
                    turn = store.append_turn(body, stamp)
                    if routine:
                        store.merge_line_mentions(routine, [], turn_id=turn)
                    else:
                        routine = store.insert_fact(block_id=block, plain=body, formatted=body, ingest_time=stamp,
                                                    source_turn_ids=[turn], zkeys=dining_key)
                else:
                    body = f"北辰第{index}号例行记录：后勤清点{rng.randrange(100, 999)}件，资料归档并发送例行确认回执。"
                    turn = store.append_turn(body, stamp)
                    store.insert_fact(block_id=block, plain=body, formatted=body, ingest_time=stamp, source_turn_ids=[turn])
        now = start + 120 * DAY
        queries = [("taboo", "北辰餐饮有什么过敏禁忌？", taboo), ("current", "北辰预算目前是多少？", version_ids[-1])]
        queries.extend(("history", f"北辰预算第{revision + 1}次修改之前是多少？", fact)
                       for revision, fact in enumerate(version_ids[:-1]))
        before_hot = store.conn.execute("SELECT count(*) FROM facts WHERE tier='hot'").fetchone()[0]
        before = [engine.retrieve(question, timestamp=now, read_only=True) for _, question, _ in queries]
        archived = set(engine.decay_and_archive(timestamp=now)["archived_ids"])
        after_hot = store.conn.execute("SELECT count(*) FROM facts WHERE tier='hot'").fetchone()[0]
        after = [engine.retrieve(question, timestamp=now, read_only=True) for _, question, _ in queries]

        def rank(result, gold):
            return next((position for position, fact in enumerate(result["ranked"], 1) if fact["id"] == gold), None)

        probe = [{"kind": kind, "query": q, "target": fact, "before_rank": rank(a, fact), "after_rank": rank(b, fact)}
                 for (kind, q, fact), a, b in zip(queries, before, after)]
        taboo_record = store.get_line(taboo)
        routine_record = store.get_line(routine)
        taboo_hit = next((fact for fact in after[0]["ranked"] if fact["id"] == taboo), None)
        tokens = [assemble_context(store, result, prefix="", token_budget=2048, drilldown_top_n=2,
                                   drilldown_window=0, query=question)["token_breakdown"]["total"]
                  for (_, question, _), result in zip(queries, after)]
        return {"turns": turn_count, "history_tokens": store.history_tokens(), "versions": len(version_ids),
                "hot_before": before_hot, "hot_after": after_hot, "hot_compression": 1 - after_hot / before_hot,
                "fruitless_fragments_cold": len(archived.intersection(fragments.values())),
                "routine_mentions": routine_record["mention_count"], "routine_hot": routine_record["tier"] == "hot",
                "taboo_hot": taboo_record["tier"] == "hot", "taboo_decay": taboo_hit["decay_factor"] if taboo_hit else None,
                "taboo_recall_at_1": int(probe[0]["after_rank"] == 1),
                "taboo_mrr": 1 / probe[0]["after_rank"] if probe[0]["after_rank"] else 0.0,
                "current_rank": probe[1]["after_rank"], "cold_wake_rate": mean(row["target"] in archived and row["after_rank"] is not None for row in probe[2:]),
                "cold_top1": mean(row["after_rank"] == 1 for row in probe[2:]),
                "average_tokens": mean(tokens), "items": probe}


def render(data):
    return table(("轮数", "历史 Token", "禁忌 R@1", "MRR@10", "当前排名", "热层事实数（前→后）", "热层缩减", "禁忌留热/衰减",
                  "历史唤醒", "历史 Top-1", "注入 Token"),
                 ((r["turns"], r["history_tokens"], r["taboo_recall_at_1"], f"{r['taboo_mrr']:.4f}",
                   r["current_rank"], f"{r['hot_before']} → {r['hot_after']}", f"{r['hot_compression']:.2%}",
                   f"{r['taboo_hot']} / {r['taboo_decay']}",
                   f"{r['cold_wake_rate']:.2%}", f"{r['cold_top1']:.2%}", f"{r['average_tokens']:.1f}")
                  for r in data["scales"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"benchmark": "Long Horizon Static", "seed": args.seed, "scales": [horizon_case(n, args.seed) for n in SCALES]}
    write_result(report, args.out, render(report))
    return int(any(not row["taboo_recall_at_1"] or not row["taboo_hot"] or row["taboo_decay"] != 1 or row["cold_wake_rate"] != 1
                   or row["fruitless_fragments_cold"] != 24 or not row["routine_hot"] or row["routine_mentions"] != 8
                   or row["current_rank"] != 1 or row["cold_top1"] != 1 or row["average_tokens"] >= 1500 for row in report["scales"]))


if __name__ == "__main__":
    raise SystemExit(main())
