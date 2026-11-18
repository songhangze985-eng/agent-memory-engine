"""One pluggable LLM client: OpenAI-compatible HTTP, or a deterministic mock."""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.request

from memengine.policy import decide_layout, decide_origin, plan_answer


def loads_json_object(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM JSON must be an object")
    return data


def _labeled_json(prompt: str, label: str):
    index = prompt.find(label)
    if index < 0:
        raise KeyError(label)
    rest = prompt[index + len(label) :].lstrip()
    value, _ = json.JSONDecoder().raw_decode(rest)
    return value


def _labeled_line(prompt: str, label: str) -> str:
    index = prompt.find(label)
    if index < 0:
        raise KeyError(label)
    return prompt[index + len(label) :].splitlines()[0].strip()


class OpenAICompatibleClient:
    """POST ``{base_url}/chat/completions``.

    Environment: ``LLM_BASE_URL`` (default ``https://api.openai.com/v1``),
    ``LLM_API_KEY`` or ``OPENAI_API_KEY``, ``LLM_MODEL``
    (default ``Qwen2.5-7B-Instruct``).
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None, timeout: float = 60):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.model = model or os.environ.get("LLM_MODEL") or "Qwen2.5-7B-Instruct"
        self.timeout = timeout
        self.usage_totals = {"prompt_tokens": 0, "completion_tokens": 0}
        self._usage_lock = threading.Lock()

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("missing LLM API key")
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        usage = payload.get("usage") or {}
        with self._usage_lock:
            self.usage_totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.usage_totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        return payload["choices"][0]["message"]["content"]


class MockLLMClient:
    """Offline stand-in. Reads the filled prompt and runs the deterministic policy."""

    def complete(self, prompt: str) -> str:
        if "Origin & Boundary Controller" in prompt:
            data = decide_origin(
                _labeled_json(prompt, "当前历史原点："),
                _labeled_json(prompt, "2D 笔记本主题索引目录："),
                _labeled_json(prompt, "Z 轴命中实体列表："),
                _labeled_line(prompt, "用户当前输入："),
            )
            return json.dumps(data, ensure_ascii=False)
        if "System 2 Anti-Intuitive Reasoner" in prompt:
            data = plan_answer(
                _labeled_json(prompt, "当前决策原点："),
                _labeled_json(prompt, "毒圈外·高权反直觉事实（例外/冲突/单次约束，优先审查）："),
                _labeled_json(prompt, "毒圈内·高压缩直觉事实（渐近常规背景，仅供参考）："),
                _labeled_line(prompt, "用户当前问题："),
            )
            return json.dumps(data, ensure_ascii=False)
        if "3D Notebook Layout Router" in prompt:
            data = decide_layout(
                _labeled_json(prompt, "局部相似 2D 段落块（Top-3）："),
                _labeled_json(prompt, "本轮对话抽取的客观事实列表："),
                _labeled_line(prompt, "当前时间戳："),
            )
            return json.dumps(data, ensure_ascii=False)
        raise ValueError("unrecognized prompt")


def build_llm_from_env():
    """Return ``(client, mode, model_name)``. Mode is ``live`` only when a key exists."""
    key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return MockLLMClient(), "mock", None
    client = OpenAICompatibleClient()
    return client, "live", client.model
