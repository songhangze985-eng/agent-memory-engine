"""Load the three spec prompts and fill their template slots."""

from __future__ import annotations

import json
from pathlib import Path

_FILES = {
    "origin": "origin_boundary.txt",
    "reason": "system2_reasoner.txt",
    "layout": "notebook_layout.txt",
    "query": "query_parser.txt",
}


def prompt_dir() -> Path:
    """Resolve ``prompts/`` for editable installs and packaged wheels."""
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "prompts",  # repo / editable layout
        here / "prompts",  # packaged next to the module
    ]
    for path in candidates:
        if path.is_dir() and (path / _FILES["origin"]).is_file():
            return path
    raise FileNotFoundError(
        "prompts/ not found beside the package; expected origin_boundary.txt"
    )


def load_prompt(name: str) -> str:
    return (prompt_dir() / _FILES[name]).read_text(encoding="utf-8")


def render_prompt(name: str, **kwargs) -> str:
    text = load_prompt(name)
    for key, value in kwargs.items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, indent=2)
        token = "{" + key + "}"
        if token not in text:
            raise KeyError(token)
        text = text.replace(token, value)
    return text
