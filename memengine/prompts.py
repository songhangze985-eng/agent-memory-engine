"""Load the three spec prompts and fill their template slots."""

from __future__ import annotations

import json
from pathlib import Path

_DIR = Path(__file__).resolve().parents[1] / "prompts"

_FILES = {
    "origin": "origin_boundary.txt",
    "reason": "system2_reasoner.txt",
    "layout": "notebook_layout.txt",
}


def load_prompt(name: str) -> str:
    return (_DIR / _FILES[name]).read_text(encoding="utf-8")


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
