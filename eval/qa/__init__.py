"""End-to-end QA evaluation suite with pluggable LLM/judge and 6-bucket attribution."""
from eval.qa.harness import QAItem, evaluate_group_qa, render_qa_markdown, summarize_qa_rows
from eval.qa.judge import DeterministicQAJudge, JudgeVerdict, OpenAICompatibleJudge
from eval.qa.provenance import build_provenance, render_provenance_markdown
from eval.qa.reader import DeterministicEngineReader, OpenAICompatibleReader, ReaderOutput

__all__ = [
    "DeterministicEngineReader",
    "DeterministicQAJudge",
    "JudgeVerdict",
    "OpenAICompatibleJudge",
    "OpenAICompatibleReader",
    "QAItem",
    "ReaderOutput",
    "build_provenance",
    "evaluate_group_qa",
    "render_provenance_markdown",
    "render_qa_markdown",
    "summarize_qa_rows",
]
