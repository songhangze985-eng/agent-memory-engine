"""One pluggable LLM client: OpenAI-compatible HTTP, or a deterministic mock."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

from memengine.extract import extract_facts
from memengine.policy import decide_layout, decide_origin, plan_answer
from memengine.queryparse import parse_query_rules


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



def _section_after(prompt: str, label: str) -> str:
    index = prompt.find(label)
    if index < 0:
        raise KeyError(label)
    lines = prompt[index + len(label) :].splitlines()
    body: list[str] = []
    started = False
    for line in lines:
        if not started and not line.strip():
            continue
        if line.startswith("# "):
            break
        started = True
        body.append(line)
    return "\n".join(body).strip()


RETRY_STATUSES = {429, 500, 502, 503, 504}
FATAL_STATUSES = {401, 402}


class FatalLLMError(RuntimeError):
    """Auth or payment error: stop the run, do not retry."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace").strip().replace("\n", " ")[:200]
    except Exception:
        return ""
    finally:
        try:
            exc.close()
        except Exception:
            pass


class OpenAICompatibleClient:
    """POST ``{base_url}/chat/completions``.

    Environment: ``LLM_BASE_URL`` (default ``https://api.openai.com/v1``),
    ``LLM_API_KEY`` or ``OPENAI_API_KEY``, ``LLM_MODEL``
    (default ``Qwen2.5-7B-Instruct``).
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None, timeout: float = 60, max_retries: int = 4, retry_backoff: float = 2.0):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.model = model or os.environ.get("LLM_MODEL") or "Qwen2.5-7B-Instruct"
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.usage_totals = {"prompt_tokens": 0, "completion_tokens": 0}
        self._usage_lock = threading.Lock()
        self._local = threading.local()

    def _add_usage(self, usage: dict) -> None:
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        with self._usage_lock:
            self.usage_totals["prompt_tokens"] += prompt
            self.usage_totals["completion_tokens"] += completion
        local = getattr(self._local, "usage", None)
        if local is not None:
            local["prompt_tokens"] += prompt
            local["completion_tokens"] += completion

    @contextmanager
    def track_usage(self):
        previous = getattr(self._local, "usage", None)
        current = {"prompt_tokens": 0, "completion_tokens": 0}
        self._local.usage = current
        try:
            yield current
        finally:
            self._local.usage = previous

    def _retry_delay(self, attempt: int, exc: urllib.error.HTTPError) -> float:
        delay = self.retry_backoff * (2 ** attempt)
        header = exc.headers.get("Retry-After") if exc.headers else None
        if header:
            try:
                delay = max(delay, float(header))
            except ValueError:
                pass
        return min(delay, 60.0)

    def _complete_once(self, prompt: str) -> str:
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
        self._add_usage(payload.get("usage") or {})
        return payload["choices"][0]["message"]["content"]

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("missing LLM API key")
        attempts = self.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._complete_once(prompt)
            except FatalLLMError:
                raise
            except urllib.error.HTTPError as exc:
                status = exc.code
                body = _http_error_body(exc)
                if status in FATAL_STATUSES:
                    detail = f": {body}" if body else ""
                    raise FatalLLMError(status, f"LLM HTTP {status}{detail}") from None
                last_error = RuntimeError(f"LLM HTTP {status}" + (f": {body}" if body else ""))
                if status not in RETRY_STATUSES or attempt >= attempts - 1:
                    raise last_error from None
                delay = self._retry_delay(attempt, exc)
                print(f"LLM HTTP {status}, retry in {delay:.1f}s ({attempt + 1}/{attempts})", flush=True)
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt >= attempts - 1:
                    raise
                delay = min(self.retry_backoff * (2 ** attempt), 60.0)
                print(f"LLM network error, retry in {delay:.1f}s ({attempt + 1}/{attempts})", flush=True)
                time.sleep(delay)
        raise last_error if last_error else RuntimeError("LLM request failed")


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
        if "查询路由与上位类目解析器" in prompt:
            stem = _section_after(prompt, "用户问题题干：")
            return json.dumps(parse_query_rules(stem), ensure_ascii=False)
        if "慢系统综合推理与作答器" in prompt or "System 2 Anti-Intuitive Reasoner" in prompt:
            if "慢系统综合推理与作答器" in prompt:
                memories = _labeled_json(prompt, "检出的历史记忆片段（按相关度与双轨权重排序，含 step_id 时序）：")
                question = _section_after(prompt, "用户问题（及选项，如有）：")
                if not isinstance(memories, list):
                    memories = []
                outer = [item for item in memories if item.get("zone") == "OUTER_ANTI_INTUITIVE"]
                inner = [item for item in memories if item.get("zone") != "OUTER_ANTI_INTUITIVE"]
                if not outer and not inner:
                    inner = memories
                data = plan_answer(None, outer, inner, question)
                return json.dumps(
                    {
                        "evidence_synthesis": data.get("final_response") or "",
                        "anti_intuitive_override": bool((data.get("reasoning_trace") or {}).get("anti_intuitive_overrides")),
                        "final_answer": data.get("final_response") or "",
                        "reasoning_trace": data.get("reasoning_trace") or {},
                    },
                    ensure_ascii=False,
                )
            data = plan_answer(
                _labeled_json(prompt, "当前决策原点："),
                _labeled_json(prompt, "毒圈外·高权反直觉事实（例外/冲突/单次约束，优先审查）："),
                _labeled_json(prompt, "毒圈内·高压缩直觉事实（渐近常规背景，仅供参考）："),
                _labeled_line(prompt, "用户当前问题："),
            )
            return json.dumps(data, ensure_ascii=False)
        if "无损事实抽取与分类写入器" in prompt or "3D Notebook Layout Router" in prompt:
            if "无损事实抽取与分类写入器" in prompt:
                try:
                    candidates = _labeled_json(prompt, "局部相似已有 2D 段落（Top-3）：")
                except Exception:
                    candidates = []
                raw = _section_after(prompt, "当前轮次原文：")
                facts = extract_facts(raw)
                data = decide_layout(candidates if isinstance(candidates, list) else [], facts, 0)
                for op, fact in zip(data.get("write_operations") or [], facts):
                    zkeys = list(fact.get("zkeys") or op.get("z_axis_keys") or [])
                    op["z_axis_keys"] = zkeys
                    op["entity_z_keys"] = [key for key in zkeys if not str(key).startswith("类目:")]
                    op["category_z_keys"] = [key for key in zkeys if str(key).startswith("类目:")]
                return json.dumps(data, ensure_ascii=False)
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
