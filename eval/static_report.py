"""Render static evaluation reports by dataset identity and named metric."""
import argparse
from datetime import date
import json
import math
from pathlib import Path

SIDES = {"bipolar": "Full", "rrf": "Pure RRF"}
FROZEN = (("03-membench.json", "MemBench", "paired_retrieval"), ("05-locomo.json", "LoCoMo", "overall"),
          ("06-longmemeval.json", "LongMemEval-S", "overall"), ("04-stress-precision.json", "Stress + Precision", "overall"))


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", r"\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(cell, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"expected {len(headers)} cells, got {len(row)}")
        lines.append("| " + " | ".join(map(cell, row)) + " |")
    return "\n".join(lines)


def number(value, fmt=".4f"):
    if value is None:
        return "—"
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"non-numeric metric: {value!r}")
    return format(value, fmt)


def read_record(path, benchmark):
    data = json.loads(path.read_text())
    if data["benchmark"] != benchmark:
        raise ValueError(f"{path}: expected {benchmark}, got {data['benchmark']}")
    return data


def scores(row):
    return tuple(number(row[key], fmt) for key, fmt in (
        ("recall_at_5", ".2%"), ("recall_at_10", ".2%"), ("mrr_at_10", ".4f")))


def summary_table(directory):
    rows = []
    for file, benchmark, key in FROZEN:
        data = read_record(directory / file, benchmark)
        full, rrf = (data[key][side] for side in SIDES)
        rows.append((benchmark, len(data["items"]), " / ".join(scores(full)), " / ".join(scores(rrf)),
                     number(full["avg_injected_tokens"], ".1f")))
    return table(("冻结基准", "题数", "Full R@5 / R@10 / MRR", "RRF R@5 / R@10 / MRR", "Full Token"), rows)


def official_table(report):
    rows = []
    for group, modes in {"Overall": report["overall"], **report["by_category"]}.items():
        for side, label in SIDES.items():
            row = modes[side]
            recall, tokens = row["recall_at_10"], row["avg_injected_tokens"]
            density = 1000 * recall / tokens if recall is not None and tokens else None
            rows.append((group, label, row["n"], *scores(row), number(row.get("session", {}).get("recall_at_10"), ".2%"),
                         number(row["update_dominance"], ".2%"), row["update_n"], number(tokens, ".1f"),
                         number(density, ".3f"), f"{row['false_empty']}/{row['answerable_n']}"))
    return table(("题型", "配置", "题数", "R@5", "R@10", "MRR@10", "Session R@10", "更新支配",
                  "更新分母", "平均 Token", "R@10 / 千 Token", "误空"), rows) + "\n"


def write_result(data, path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    path.with_suffix(".md").write_text(body + "\n")
    print(body)


def compact_row(label, before, after, cap):
    fields = ("recall_at_5", "recall_at_10", "mrr_at_10")
    if any(after[field] < before[field] for field in fields) or cap > 360:
        raise ValueError(f"{label}: compact policy breached evidence or prefix limits")
    token_fields = ("avg_prefix_tokens", "avg_dynamic_tokens", "avg_drilled_raw_tokens")
    usage = " → ".join(" / ".join(number(row[key], ".1f") for key in token_fields) for row in (before, after))
    comparisons = after["prefix_comparisons"]
    hits = f"{round(after['prefix_hit_rate'] * comparisons)}/{comparisons}" if comparisons else "—"
    return (label, " / ".join(scores(after)), cap, usage, hits,
            f"{after['observation_only_gold_queries']}/{after['n']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("eval/results/static_benchmark_report.md"))
    args = parser.parse_args()
    build_report(args.results, args.out)


def build_report(directory, out):
    from eval.ablation_matrix import render as matrix_table
    from eval.engineering_bench import render as engineering_table
    from eval.exact_ident_eval import static_table
    from eval.long_horizon_static import render as horizon_table

    locomo = read_record(directory / "full-locomo.json", "LoCoMo")
    longmem = read_record(directory / "full-long.json", "LongMemEval-S")
    mem = read_record(directory / "full-mem.json", "MemBench")
    horizon = read_record(directory / "long-horizon.json", "Long Horizon Static")
    exact = read_record(directory / "exact-static.json", "Exact Ident Static")
    engineering = read_record(directory / "engineering.json", "Engineering Static")
    matrix = read_record(directory / "ablation.json", "Ablation Matrix")
    frozen_long = read_record(directory / "06-longmemeval.json", "LongMemEval-S")
    update_n = frozen_long["overall"]["bipolar"]["update_n"]
    update_wins = sum(row["bipolar"].get("update_dominates", False) for row in frozen_long["items"])

    compact_rows = []
    for file, name, label in (("compact-locomo.json", "LoCoMo", "LoCoMo 100"),
                              ("compact-long.json", "LongMemEval-S", "LongMemEval-S 100")):
        compact = read_record(directory / file, name)
        frozen_path = "05-locomo.json" if name == "LoCoMo" else "06-longmemeval.json"
        original = read_record(directory / frozen_path, name)
        before = {row["id"]: row["evidence"] for row in original["items"]}
        after = {row["id"]: row["evidence"] for row in compact["items"]}
        if before != after:
            raise ValueError(f"{label}: consolidation cohorts differ")
        metrics, off = compact["overall"]["bipolar"], original["overall"]["bipolar"]
        max_prefix = max(row["bipolar"]["prefix_tokens"] for row in compact["items"])
        compact_rows.append(compact_row(label, off, metrics, max_prefix))
    mem_off, mem_on = (mem[key]["overall"]["bipolar"] for key in ("default", "consolidated"))
    max_prefix = max(row["consolidated_pair"]["bipolar"]["prefix_tokens"] for row in mem["items"])
    compact_rows.append(compact_row("MemBench 400", mem_off, mem_on, max_prefix))

    sections = [
        "# 零 API 静态评测报告",
        f"报告生成于 {date.today().isoformat()}。以下为固定种子 {locomo['seed']} 的哈希静态检索，冻结 MemBench 99 题使用 42–46。"
        "Top-10，默认容量 2048 估算 Token，显式蒸馏的静态前缀单独计量。延迟受本机负载影响。",
        "## 冻结四基准总表",
        summary_table(directory),
        "## 全量 LoCoMo",
        f"{len(locomo['items'])} 题，`{locomo['dataset_status']}`，缓存 SHA-256 `{locomo['sha256']}`。"
        f" [源数据]({locomo['dataset_source']})",
        official_table(locomo),
        "## 全量 LongMemEval-S",
        f"{len(longmem['items'])} 题，`{longmem['dataset_status']}`，缓存 SHA-256 `{longmem['sha256']}`。"
        f"六个官方主类，带 _abs 标记的拒答题另列。 [源数据]({longmem['dataset_source']})",
        official_table(longmem),
        "## MemBench 400",
        "从本机已缓存子类型中，按四个大类各采样 100 题。缺失文件为 " + "、".join(mem["uncached_subtypes"]) + "。"
        " [源数据](https://github.com/import-myself/Membench/tree/main/MemData)",
        "### 默认",
        official_table(mem["default"]),
        "### 紧凑蒸馏",
        official_table(mem["consolidated"]),
        "## 紧凑前缀核验",
        table(("基准", "开启后 R@5 / R@10 / MRR", "最大前缀", "平均前缀 / 动态 / 原话 Token（关→开）",
               "指纹复用", "Observation 额外覆盖金标题数"), compact_rows),
        "三组事实 Recall 与 MRR 均持平。此实验固定事实检索并单列前缀预算。指纹复用表示推理端可复用静态前缀的机会。",
        "## 更新题复核",
        f"冻结样本更新支配为 {update_wins}/{update_n}。上轮未胜出的六题中，五题问累计数量或时长，一题含明确的历史锚点。"
        "数量抽取会忽略查询已明确的数字，如设备型号数字，以比较证据中的实际测量值。保留原相关性门槛和历史语义。"
        "其余问题还涉及候选池缺失、基础相关性落差和旧事件时间与新断言时间的不同。",
        "## 超长程状态与冷归档",
        horizon_table(horizon),
        "写入六个预算版本、单次禁忌、八次同主题例行记录和 24 条各重复两次的夹层事实。"
        "热层缩减按可搜索事实数计算，旧事实仍由 SQLite 保留。注入量包含实际装配的事实、标签及原话。"
        "五次历史探测要求首位命中，而且目标版本确实位于冷层。",
        "## 精确标识符 120 题",
        static_table(exact),
        "## 隔离、只读与规模性能",
        engineering_table(engineering),
        "10 个租户交替写入。延迟为定向符号查询，写入吞吐使用每轮一个 SQLite 事务调用显式 remember。",
        "## 逐层组件消融",
        matrix_table(matrix),
        "## 信息密度与轻量优势对比",
        "上表的信息密度定义为 Evidence Recall@10 ÷ 每千个注入 Token。它以本地估算 Token 和金标证据为单位。"
        "Hindsight 的 AMB 流程包含检索、模型作答和裁判，EverOS 公布了 Markdown 持久层、LanceDB 索引与情节和原子事实产线。"
        "本库静态测量使用 SQLite、FTS5 和哈希向量，无模型作答、图数据库或网络服务。"
        " [AMB 流程](https://github.com/vectorize-io/agent-memory-benchmark) ·"
        " [EverOS 存储与索引](https://github.com/EverMind-AI/EverOS/blob/main/docs/how-memory-works.md)",
        "OmniMemEval 自己的复现实验在 LoCoMo 上报告平均作答阶段输入为 Hindsight 24,683、EverOS 8,559 Token，"
        "在 BEAM 100K 上分别为 24,085 和 7,393。外部测量含答题提示，且数据切分、官方裁判与本地估算器各异。"
        "跨系统可核对成本口径与结构，缺少同硬件、同检索集合的延迟和准确率对比。"
        " [外部口径](https://github.com/MemTensor/OmniMemEval/blob/main/docs/user_memory/results.md)"
        " · [LoCoMo 复现](https://github.com/MemTensor/OmniMemEval/blob/main/docs/user_memory/benchmark_results/locomo.md)"
        " · [BEAM 复现](https://github.com/MemTensor/OmniMemEval/blob/main/docs/user_memory/benchmark_results/beam.md)",
        "## 精简审计",
        "删除 migration 中 1 行未使用的 Token 估算导入。生产检索的标识符过滤与锚点打分分别改用与消融共享的函数，"
        "engine.py 对应两处逻辑从 14 行缩为 3 行，实际实现移至专用模块。保留公开 prompt alias 和受测试的兼容入口。"
        "L5 在冻结事实检索时提供侧路 Observation 及稳定前缀。"
        "预算更紧的调用可从默认关闭蒸馏开始。",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
