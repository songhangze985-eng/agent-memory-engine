"""Step D: RealMem-style multi-session project-oriented mixed-profile benchmark."""
from __future__ import annotations

import argparse
import json
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any

from agentmem_core import AgentEvent, UnifiedMemoryEngine, capture_anchor
from eval.qa.provenance import build_provenance, render_provenance_markdown


def run_realmem_hybrid_eval(*, seed: int = 42) -> dict[str, Any]:
    """Simulate a multi-session evolving 2-project lifecycle mounting hybrid conversational + coding memory."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        atlas_repo = root / "atlas-api"
        zephyr_repo = root / "zephyr-ui"
        (atlas_repo / "src").mkdir(parents=True)
        (zephyr_repo / "src").mkdir(parents=True)

        atlas_deps = atlas_repo / "pyproject.toml"
        atlas_deps.write_text('[project]\ndependencies = ["pydantic==1.10.15"]\n', encoding="utf-8")
        atlas_auth = atlas_repo / "src" / "auth.py"
        atlas_auth.write_text("def issue_jwt():\n    return 'hs256'\n", encoding="utf-8")
        (zephyr_repo / "src" / "router.ts").write_text("export const routes = [];\n", encoding="utf-8")

        engine = UnifiedMemoryEngine(root / "realmem.db")
        with closing(engine):
            # Single agent mounts conversational, coding(atlas), coding(zephyr), and hybrid(atlas) bindings
            conv = engine.bind_profile("conversational", user_id="maya", session_id="s1_chat")
            code_atlas = engine.bind_profile("coding", user_id="maya", repo_id="atlas-api", project_id="atlas", session_id="s3_atlas")
            code_zephyr = engine.bind_profile("coding", user_id="maya", repo_id="zephyr-ui", project_id="zephyr", session_id="s5_zephyr")
            hybrid_atlas = engine.bind_profile("hybrid", user_id="maya", repo_id="atlas-api", project_id="atlas", session_id="s6_hybrid")

            # Session 1 (Conversational): global user preferences
            conv.ingest_turn(
                "I prefer concise bullet-point explanations and dark mode dashboards.",
                scope_key="user:maya",
                timestamp=1_700_000_000.0,
            )
            engine.ingest_rule(
                "/prefs/test_cmd.md",
                title="Maya Default Test Command",
                trigger_condition="Running unit tests",
                action="Run plain pytest -q by default",
                verification_result="User global preference",
                user_id="maya",
                scope_key="user:maya",
                override_key="test_cmd",
                timestamp=1_700_000_010.0,
            )

            # Session 2 (Conversational -> Project Atlas): initial project constraints
            hybrid_atlas.ingest_turn(
                "Project Atlas launch budget is 15000 USD and target launch city is Tokyo.",
                scope_key="project:atlas",
                timestamp=1_700_010_000.0,
            )

            # Session 3 (Coding -> Repo Atlas): repo override + anchored rule + fail->pass capture
            code_atlas.ingest_rule(
                "/prefs/test_cmd.md",
                title="Atlas CI Test Command",
                trigger_condition="Running unit tests in atlas-api",
                action="Run uv run pytest -m integration --tb=short in atlas-api",
                verification_result="Required by Atlas CI",
                override_key="test_cmd",
                timestamp=1_700_020_000.0,
            )
            pkg_anchor = capture_anchor(atlas_repo, file_path="pyproject.toml", package_name="pydantic")
            code_atlas.ingest_rule(
                "/rules/atlas_pydantic.md",
                title="Atlas Pydantic v1 serialization",
                trigger_condition="Serializing models in atlas-api",
                action="Call legacy_v1_dict() on Pydantic 1.10 models",
                verification_result="Works on pydantic==1.10.15",
                anchors=[pkg_anchor],
                repo_root=atlas_repo,
                timestamp=1_700_020_100.0,
            )
            code_atlas.ingest_event(
                AgentEvent(
                    event_type="test_result",
                    tool_name="bash",
                    command="pytest tests/test_auth.py",
                    exit_code=1,
                    text="KeyError: JWT_SIGNING_SECRET missing in atlas-api auth",
                    file_paths=("src/auth.py",),
                    timestamp=1_700_020_200.0,
                ),
                repo_root=atlas_repo,
            )
            code_atlas.ingest_event(
                AgentEvent(
                    event_type="file_edit",
                    text="Load JWT_SIGNING_SECRET from .env.atlas in src/auth.py",
                    file_paths=("src/auth.py",),
                    timestamp=1_700_020_210.0,
                ),
                repo_root=atlas_repo,
            )
            code_atlas.ingest_event(
                AgentEvent(
                    event_type="test_result",
                    tool_name="bash",
                    command="uv run pytest tests/test_auth.py",
                    exit_code=0,
                    text="passed",
                    file_paths=("src/auth.py",),
                    timestamp=1_700_020_220.0,
                ),
                repo_root=atlas_repo,
            )

            # Session 4 (Mid-project evolution): budget updated + pydantic upgraded -> stale anchor!
            hybrid_atlas.ingest_turn(
                "Update: Project Atlas launch budget is now 28000 USD after Q2 approval.",
                scope_key="project:atlas",
                timestamp=1_700_050_000.0,
            )
            atlas_deps.write_text('[project]\ndependencies = ["pydantic==2.8.0"]\n', encoding="utf-8")
            drift = code_atlas.verify_anchors(repo_root=atlas_repo, timestamp=1_700_050_100.0)

            # Session 5 (Project Zephyr): isolated repo conventions
            code_zephyr.ingest_rule(
                "/rules/zephyr_router.md",
                title="Zephyr Router Guard",
                trigger_condition="Editing src/router.ts in zephyr-ui",
                action="Wrap lazy routes in ZephyrSuspenseBoundary",
                verification_result="Zero hydration error",
                anchors=["src/router.ts"],
                repo_root=zephyr_repo,
                timestamp=1_700_060_000.0,
            )

            # ---- Evaluation Assertions (8 RealMem-style multi-dimensional checks) ----
            checks: list[dict[str, Any]] = []

            # 1. Shared user preference visible from coding profile
            c1 = code_atlas.recall("What explanation format and dashboard mode do I prefer?", reference_time=1_700_070_000.0)
            ok1 = any("bullet-point" in f["plain"].lower() for f in c1["ranked"])
            checks.append({"id": "shared_user_pref_in_coding", "passed": ok1, "detail": "user:maya pref inherited in coding(atlas)"})

            # 2. Project/repo rule overrides user global preference inside atlas-api
            note_atlas = code_atlas.read_note("/prefs/test_cmd.md")
            rec_atlas_cmd = code_atlas.recall("Running unit tests", reference_time=1_700_070_000.0)
            ok2 = (
                note_atlas is not None
                and note_atlas["scope_key"] == "repo:atlas-api"
                and "uv run pytest -m integration" in note_atlas["content"]
                and all(f["scope_key"] != "user:maya" for f in rec_atlas_cmd["ranked"] if f.get("override_key") == "test_cmd")
            )
            checks.append({"id": "repo_overrides_user_pref", "passed": ok2, "detail": "repo:atlas-api test_cmd overrides user:maya"})

            # 3. User global preference remains intact in zephyr-ui (where no repo override exists)
            note_zephyr = code_zephyr.read_note("/prefs/test_cmd.md")
            ok3 = note_zephyr is not None and note_zephyr["scope_key"] == "user:maya" and "plain pytest -q" in note_zephyr["content"]
            checks.append({"id": "user_pref_preserved_in_other_repo", "passed": ok3, "detail": "user:maya test_cmd intact in zephyr-ui"})

            # 4. Zero cross-project contamination: zephyr rule NOT visible in atlas; atlas rule NOT visible in zephyr
            amb_atlas_z = code_atlas.ambient_context("ZephyrSuspenseBoundary", active_files=["src/router.ts"])
            amb_zephyr_a = code_zephyr.ambient_context("JWT_SIGNING_SECRET", active_files=["src/auth.py"])
            ok4 = ("ZephyrSuspenseBoundary" not in amb_atlas_z["dynamic_memory"]) and ("JWT_SIGNING_SECRET" not in amb_zephyr_a["dynamic_memory"])
            checks.append({"id": "zero_cross_repo_contamination", "passed": ok4, "detail": "atlas-api and zephyr-ui strictly isolated"})

            # 5. Learned coding pitfall retrieved in zero-LLM ambient path inside atlas-api
            amb_atlas_auth = code_atlas.ambient_context("JWT_SIGNING_SECRET missing", active_files=["src/auth.py"])
            ok5 = amb_atlas_auth["llm_calls"] == 0 and ".env.atlas" in amb_atlas_auth["dynamic_memory"]
            checks.append({"id": "coding_signal_ambient_hit", "passed": ok5, "detail": "fail_to_pass rule recalled with 0 LLM calls"})

            # 6. Stale rule excluded after mid-project dependency upgrade (pydantic 1.10 -> 2.8)
            amb_stale = code_atlas.ambient_context("Serializing models in atlas-api", active_files=["pyproject.toml"])
            ok6 = drift["stale_count"] >= 1 and "legacy_v1_dict()" not in amb_stale["dynamic_memory"]
            checks.append({"id": "mid_project_stale_excluded", "passed": ok6, "detail": "pydantic upgrade flagged & excluded stale rule"})

            # 7. Hybrid conversational knowledge-update: latest Atlas budget (28000) beats older (15000)
            ans_budget = hybrid_atlas.answer("What is the current Project Atlas launch budget?", reference_time=1_700_070_000.0)
            ok7 = (not ans_budget["abstained"]) and ("28000" in ans_budget["response"])
            checks.append({"id": "hybrid_knowledge_update", "passed": ok7, "detail": f"response={ans_budget['response'][:80]}"})

            # 8. Conversational profile does NOT leak coding directory outline or repo-only rules
            ctx_conv = conv.build_context("What dashboard mode do I prefer?", reference_time=1_700_070_000.0)
            ok8 = (
                ctx_conv["profile"] == "conversational"
                and "Directory Outline" not in ctx_conv["dynamic_memory"]
                and "ZephyrSuspenseBoundary" not in ctx_conv["dynamic_memory"]
                and "dark mode" in ctx_conv["dynamic_memory"].lower()
            )
            checks.append({"id": "dual_profile_non_interference", "passed": ok8, "detail": "conversational routing unaffected by coding rules"})

            passed_n = sum(1 for c in checks if c["passed"])
            prov = build_provenance(
                engine="UnifiedMemoryEngine (conversational + coding + hybrid bindings)",
                embedder="HashedTfidfEmbedder",
                reranker="CalibratedLocalReranker",
                llm="MockLLMClient/DeterministicReader",
                extractor_model="DeterministicRuleExtractor + C2 SignalExtractor",
                reader_model="MockLLMClient/DeterministicReader",
                judge="deterministic-assertion",
                protocol="Step D RealMem-Style Mixed-Profile Project Benchmark",
                granularity="fact + rule + code-anchor + scope-chain",
                seed=seed,
                split="full",
                dataset=f"realmem_hybrid_project_suite(checks={len(checks)})",
            )
            return {
                "provenance": prov,
                "gate_pass": passed_n == len(checks),
                "passed": passed_n,
                "total": len(checks),
                "pass_rate": round(passed_n / len(checks), 4),
                "checks": checks,
            }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Step D Mixed-Profile & Scope-Chain Verification Report (RealMem-Style)",
        "",
        render_provenance_markdown(report["provenance"]),
        "",
        f"- **Gate Status**: `{'PASS' if report['gate_pass'] else 'FAIL'}` (`{report['passed']}/{report['total']}` checks passed)",
        "",
        "| Check ID | Status | Detail |",
        "| :--- | :---: | :--- |",
    ]
    for c in report["checks"]:
        lines.append(f"| `{c['id']}` | **{'PASS' if c['passed'] else 'FAIL'}** | {c['detail']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("eval/results/step_d_hybrid_realmem_report.json"))
    parser.add_argument("--md-out", type=Path, default=Path("eval/results/step_d_hybrid_realmem_report.md"))
    args = parser.parse_args()

    rep = run_realmem_hybrid_eval(seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = render_markdown(rep)
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(md, encoding="utf-8")
    print(md)
    return 0 if rep["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
