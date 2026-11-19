"""Run the four synthetic tests against four systems.

``--scale small`` uses about 10–20% of the spec sizes. Without an API key the
full system and variant C run on the deterministic mock; those scores are not
language-model quality numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.baselines import CrudMemory, NaiveRAG
from eval.generate import build_all
from eval.metrics import answer_correct, percentile, trap_hallucinated
from memengine.engine import CognitiveMemoryEngine
from memengine.llm import MockLLMClient, build_llm_from_env

ROOT = Path(__file__).resolve().parents[1]


def _systems(llm):
    def full():
        return CognitiveMemoryEngine(db_path=":memory:", llm=llm)

    def variant():
        return CognitiveMemoryEngine(
            db_path=":memory:",
            llm=llm,
            enable_timeline_buffer=False,
            enable_casual_bypass=False,
        )

    return [
        ("naive_rag", NaiveRAG, False),
        ("crud_baseline", CrudMemory, False),
        ("variant_c", variant, True),
        ("full_system", full, True),
    ]


def _eval_system(factory, datasets: dict) -> dict:
    per_test = {}
    casual_ms: list[float] = []
    task_ms: list[float] = []
    compressions: list[float] = []
    events = 0
    nodes = 0
    router_hit = 0
    router_total = 0
    hallucinations = 0
    trap_total = 0
    failures = []
    fallback = 0
    for name, dataset in datasets.items():
        system = factory()
        for turn in dataset["utterances"]:
            system.add(turn["text"], timestamp=turn["ts"])
        correct = 0
        scored = 0
        for question in dataset["questions"]:
            result = system.answer(question["query"], timestamp=question["ts"])
            route = result.get("route")
            if route == "CASUAL_2D_BYPASS":
                casual_ms.append(result["retrieval_ms"])
            elif route == "TASK_POISON_CIRCLE":
                task_ms.append(result["retrieval_ms"])
            else:
                task_ms.append(result["retrieval_ms"])
            if question.get("expected_route") and name == "test4":
                router_total += 1
                if route == question["expected_route"]:
                    router_hit += 1
            if question.get("trap"):
                trap_total += 1
                if trap_hallucinated(result.get("response") or "", question):
                    hallucinations += 1
                    failures.append(
                        {
                            "test": name,
                            "kind": "hallucination",
                            "query": question["query"],
                            "response": (result.get("response") or "")[:240],
                        }
                    )
            if question.get("keywords"):
                scored += 1
                ok = answer_correct(result.get("response") or "", question)
                correct += int(ok)
                if not ok:
                    failures.append(
                        {
                            "test": name,
                            "kind": "accuracy",
                            "query": question["query"],
                            "response": (result.get("response") or "")[:240],
                        }
                    )
            if question.get("measure_compression"):
                history = result.get("history_tokens") or 0
                if history > 0:
                    compressions.append(1 - (result.get("context_tokens") or 0) / history)
            fallback += int(result.get("llm_fallback") or 0)
        stats = system.timeline_stats()
        if stats:
            events += stats["events"]
            nodes += stats["nodes"]
        per_test[name] = {
            "accuracy": (correct / scored) if scored else None,
            "correct": correct,
            "n": scored,
            "utterances": len(dataset["utterances"]),
        }
    reduction = None if events == 0 else 1 - (nodes / events)
    return {
        "test1_accuracy": per_test["test1"]["accuracy"],
        "test1_correct": per_test["test1"]["correct"],
        "test1_n": per_test["test1"]["n"],
        "test2_accuracy": per_test["test2"]["accuracy"],
        "test2_correct": per_test["test2"]["correct"],
        "test2_n": per_test["test2"]["n"],
        "test3_accuracy": per_test["test3"]["accuracy"],
        "test3_correct": per_test["test3"]["correct"],
        "test3_n": per_test["test3"]["n"],
        "test4_router_accuracy": (router_hit / router_total) if router_total else None,
        "test4_router_correct": router_hit,
        "test4_router_n": router_total,
        "test4_hallucination_rate": (hallucinations / trap_total) if trap_total else None,
        "test4_hallucinations": hallucinations,
        "test4_trap_n": trap_total,
        "context_compression": (sum(compressions) / len(compressions)) if compressions else None,
        "timeline_node_reduction": reduction,
        "timeline_events": events,
        "timeline_nodes": nodes,
        "p95_retrieval_ms_casual": percentile(casual_ms, 0.95),
        "p95_retrieval_ms_task": percentile(task_ms, 0.95),
        "casual_n": len(casual_ms),
        "task_n": len(task_ms),
        "llm_fallback_count": fallback,
        "failures": failures[:30],
        "per_test_utterances": {key: value["utterances"] for key, value in per_test.items()},
    }


def _fmt_pct(value) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _fmt_ms(value) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _markdown(payload: dict) -> str:
    systems = payload["systems"]
    order = ["naive_rag", "crud_baseline", "variant_c", "full_system"]
    labels = {
        "naive_rag": "Baseline A 朴素 RAG",
        "crud_baseline": "Baseline B 自写 CRUD",
        "variant_c": "Variant C",
        "full_system": "Full System",
    }
    lines = [
        "# 小规模实测（mock 模式，不是质量分数）",
        "",
        payload["note"],
        "",
        f"- 规模：`{payload['scale']}`，种子 `{payload['seed']}`，模式 `{payload['mode']}`，模型 `{payload['model']}`",
        f"- 配置：`{json.dumps(payload['scale_config'], ensure_ascii=False)}`",
        "",
        "| 系统 | Test1 孤立事实 | Test2 动态更新 | Test3 多跳/时间 | Test4 路由准确率 | Test4 陷阱幻觉率 | 任务态上下文压缩 | 时间轴节点减少 | P95 闲聊 ms | P95 任务 ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in order:
        row = systems[key]
        router = "—" if not payload["system_has_router"][key] else _fmt_pct(row["test4_router_accuracy"])
        reduction = "—" if row["timeline_node_reduction"] is None else _fmt_pct(row["timeline_node_reduction"])
        lines.append(
            "| {name} | {t1} | {t2} | {t3} | {router} | {hall} | {comp} | {red} | {casual} | {task} |".format(
                name=labels[key],
                t1=_fmt_pct(row["test1_accuracy"]),
                t2=_fmt_pct(row["test2_accuracy"]),
                t3=_fmt_pct(row["test3_accuracy"]),
                router=router,
                hall=_fmt_pct(row["test4_hallucination_rate"]),
                comp=_fmt_pct(row["context_compression"]),
                red=reduction,
                casual=_fmt_ms(row["p95_retrieval_ms_casual"]),
                task=_fmt_ms(row["p95_retrieval_ms_task"]),
            )
        )
    lines.extend(
        [
            "",
            "口径：准确率是关键词全中且禁用词不出现；陷阱幻觉率是无否定地断言诱饵，或在零关联问题上吐出受保护的稀有词。",
            "朴素 RAG 的回复是 Top-5 块拼接。CRUD 是按实体键取最新记录并做一层共享键连接，原始窗口只计入上下文 token。",
            "时间轴节点减少 = 1 - 节点数/记录次数。缓冲规则按规格伪代码比较 `now - t_end`，会话持续有写入时会收成很少的节点。",
            "P95 只统计检索，不含慢系统生成。mock 延迟不能和规格里的毫秒目标比较。",
            "",
            "规格说明中的目标分数没有抄进本文件。",
        ]
    )
    return "\n".join(lines) + "\n"


def run(scale: str = "small", seed: int = 42, mode: str = "auto") -> dict:
    if mode == "mock":
        llm, resolved, model = MockLLMClient(), "mock", None
    elif mode == "live":
        llm, resolved, model = build_llm_from_env()
        if resolved != "live":
            raise SystemExit("没有可用的 LLM API key，无法按 live 模式运行。")
    else:
        llm, resolved, model = build_llm_from_env()
    bundle = build_all(scale, seed)
    payload = {
        "scale": scale,
        "seed": seed,
        "mode": resolved,
        "model": model,
        "meaningful_as_quality": resolved == "live",
        "note": (
            "本次为 mock 模式：未使用语言模型，数字来自确定性规则与检索基线，不能当作记忆质量分数。"
            if resolved == "mock"
            else f"本次调用了 OpenAI 兼容接口，模型 {model}。基线 A/B 仍然不调用模型。"
        ),
        "scale_config": bundle["config"],
        "system_has_router": {},
        "systems": {},
    }
    for name, factory, has_router in _systems(llm):
        payload["system_has_router"][name] = has_router
        payload["systems"][name] = _eval_system(factory, bundle["tests"])
        if not has_router:
            payload["systems"][name]["test4_router_accuracy"] = None
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cognitive-memory synthetic eval.")
    parser.add_argument("--scale", choices=["small", "full"], default="small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["auto", "mock", "live"], default="auto")
    parser.add_argument(
        "--out",
        default=str(ROOT / "eval" / "results" / "small_eval.json"),
        help="JSON output path",
    )
    args = parser.parse_args()
    payload = run(scale=args.scale, seed=args.seed, mode=args.mode)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = _markdown(payload)
    out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
