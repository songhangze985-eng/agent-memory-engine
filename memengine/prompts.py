"""Load the three spec prompts and fill their template slots."""

from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path

_FILES = {
    "origin": "origin_boundary.txt",
    "reason": "system2_reasoner.txt",
    "layout": "notebook_layout.txt",
    "query": "query_parser.txt",
    "controller": "query_origin_controller.txt",
}


def prompt_dir() -> Path:
    """Resolve prompt templates packaged inside memengine."""
    root = files("memengine") / "prompt_templates"
    if (root / _FILES["origin"]).is_file():
        return root
    raise FileNotFoundError(
        "prompt_templates/ not found in the memengine package; expected origin_boundary.txt"
    )


def load_prompt(name: str) -> str:
    return (prompt_dir() / _FILES[name]).read_text(encoding="utf-8")


def render_prompt(name: str, **kwargs) -> str:
    text = load_prompt(name)
    if name == "layout":
        kwargs = {"role": "user", "categories": [], "timestamp": 0, "current_date_iso": "1970-01-01",
                  "recent_turns": [], **kwargs}
    if name == "reason":
        kwargs = {"matched_entities": [], **kwargs}
    values = {}
    for key, value in kwargs.items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, indent=2)
        token = "{" + key + "}"
        if token not in text:
            raise KeyError(token)
        values[token] = value
    return re.sub(r"\{[a-z_0-9]+\}", lambda match: values.get(match[0], match[0]), text)
