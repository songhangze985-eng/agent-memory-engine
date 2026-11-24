"""Core data models, salience write gate, structured rule schema, and path invariants for VFS v2."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

LAYERS = frozenset({"L1", "L2", "L3"})
VERIFICATION_STATUSES = frozenset({"unverified", "verified"})
SALIENCE_TAGS = frozenset({"normal", "preference", "taboo", "core_rule", "warning"})
MAX_INDEX_SUMMARY_CHARS = 50

SALIENCE_WRITE_GUIDE = (
    "写入记忆前请先自问三个问题（不要机械限定为“只记失败”）：\n"
    "1. 这个信息在当前界面/上下文里能直接看到吗？（能直接看到就不写）\n"
    "2. 这是一次性状态还是可复用的规律？\n"
    "3. 未来新会话没有它会不会踩坑？\n"
    "记录规则或假设时请使用结构化三段式：[触发条件] + [动作] + [验证结果]；"
    "无验证结果时自动标记为 unverified（未验证假设），切勿把猜测伪装为已验证规则。"
)

_INVALID_SEGMENT = re.compile(r"[\x00-\x1f\\]")
_MD_STRIP = re.compile(r"^[#>\-\*\d.\s`]+")
_TRIGGER_LINE = re.compile(r"^(?:[-*]\s*)?\[(?:触发条件|Trigger)\]\s*(.+)$", re.I | re.M)
_ACTION_LINE = re.compile(r"^(?:[-*]\s*)?\[(?:动作|Action)\]\s*(.+)$", re.I | re.M)
_VERIFY_LINE = re.compile(r"^(?:[-*]\s*)?\[(?:验证结果|Verification)\]\s*(.+)$", re.I | re.M)


class OptimisticLockError(RuntimeError):
    """Raised when a concurrent writer mutated a memory node before this write committed."""

    def __init__(self, path: str, expected_version: int, actual_version: int):
        self.path = path
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"optimistic lock conflict on {path}: expected version {expected_version}, found {actual_version}"
        )


def normalize_dir_path(raw: str | None) -> str:
    """Normalize a folder path to canonical form like '/' or '/strategies/database/'."""
    text = (raw or "/").strip()
    if not text:
        return "/"
    if len(text) > 512 or _INVALID_SEGMENT.search(text):
        raise ValueError(f"invalid VFS directory path: {raw!r}")
    parts = [segment for segment in text.split("/") if segment]
    if any(segment in {".", ".."} for segment in parts):
        raise ValueError(f"path traversal is not allowed in VFS path: {raw!r}")
    return "/" if not parts else "/" + "/".join(parts) + "/"


def normalize_node_path(raw: str) -> tuple[str, str]:
    """Normalize a file node path like '/strategies/database/pg.md' -> (parent_path, full_path)."""
    text = (raw or "").strip()
    if not text or len(text) > 512 or _INVALID_SEGMENT.search(text):
        raise ValueError(f"invalid VFS node path: {raw!r}")
    parts = [segment for segment in text.split("/") if segment]
    if not parts or any(segment in {".", ".."} for segment in parts):
        raise ValueError(f"invalid VFS node path: {raw!r}")
    filename = parts[-1]
    if not filename.endswith(".md"):
        filename = f"{filename}.md"
    parent_parts = parts[:-1]
    parent_path = "/" if not parent_parts else "/" + "/".join(parent_parts) + "/"
    return parent_path, parent_path + filename


def validate_node_fields(
    *,
    layer: str,
    confidence_score: float,
    verification_status: str,
    salience_tag: str,
) -> None:
    if layer not in LAYERS:
        raise ValueError(f"layer must be one of {sorted(LAYERS)}, got {layer!r}")
    if not (0.0 <= float(confidence_score) <= 1.0):
        raise ValueError(f"confidence_score must be in [0, 1], got {confidence_score!r}")
    if verification_status not in VERIFICATION_STATUSES:
        raise ValueError(
            f"verification_status must be one of {sorted(VERIFICATION_STATUSES)}, got {verification_status!r}"
        )
    if salience_tag not in SALIENCE_TAGS:
        raise ValueError(f"salience_tag must be one of {sorted(SALIENCE_TAGS)}, got {salience_tag!r}")


def evaluate_salience_gate(
    *,
    visible_in_context: bool = False,
    reusable_pattern: bool = True,
    future_pitfall_risk: bool = False,
) -> tuple[bool, str]:
    """Salience write gate based on the 3 self-check questions (no rigid 'failures-only' cap)."""
    if visible_in_context:
        return False, "visible_in_current_context"
    if not reusable_pattern and not future_pitfall_risk:
        return False, "ephemeral_low_salience"
    return True, "salient"


def format_structured_rule(
    *,
    title: str,
    trigger_condition: str,
    action: str,
    verification_result: str = "",
    notes: str = "",
) -> str:
    """Format a high-density structured Markdown rule: [触发条件] + [动作] + [验证结果]."""
    trig = trigger_condition.strip()
    act = action.strip()
    if not trig or not act:
        raise ValueError("structured rule requires nonempty trigger_condition and action")
    ver = verification_result.strip() or "待验证（临时假设）"
    lines = [
        f"# {title.strip() or act[:32]}",
        "",
        f"- [触发条件] {trig}",
        f"- [动作] {act}",
        f"- [验证结果] {ver}",
    ]
    if notes and notes.strip():
        lines.extend(["", notes.strip()])
    return "\n".join(lines)


def parse_structured_rule(content: str) -> dict[str, str]:
    """Extract [触发条件], [动作], [验证结果] fields from Markdown content when present."""
    trig = _TRIGGER_LINE.search(content or "")
    act = _ACTION_LINE.search(content or "")
    ver = _VERIFY_LINE.search(content or "")
    return {
        "trigger_condition": trig.group(1).strip() if trig else "",
        "action": act.group(1).strip() if act else "",
        "verification_result": ver.group(1).strip() if ver else "",
    }


def enforce_rule_state_machine(
    *,
    verification_status: str,
    confidence_score: float,
    verification_result: str = "",
) -> tuple[str, float]:
    """Enforce hypothesis vs verified-rule boundary: cannot mark verified without verification outcome."""
    outcome = (verification_result or "").strip()
    is_placeholder = not outcome or outcome in {"待验证", "待验证（临时假设）", "unverified", "pending", "TBD"}
    if is_placeholder:
        return "unverified", min(float(confidence_score), 0.65)
    return verification_status, float(confidence_score)


def make_index_summary(
    title: str,
    content: str,
    explicit_summary: str = "",
    *,
    max_chars: int = MAX_INDEX_SUMMARY_CHARS,
) -> str:
    """Produce a concise <=50-char one-sentence core takeaway for progressive disclosure."""
    candidate = " ".join((explicit_summary or "").split()).strip()
    if not candidate:
        parsed = parse_structured_rule(content)
        if parsed["action"]:
            candidate = parsed["action"]
        elif title and title.strip():
            candidate = " ".join(title.split()).strip()
        else:
            for raw in (content or "").splitlines():
                cleaned = _MD_STRIP.sub("", raw).strip()
                if cleaned:
                    candidate = " ".join(cleaned.split())
                    break
    candidate = _MD_STRIP.sub("", candidate).strip()
    if len(candidate) <= max_chars:
        return candidate
    # Prefer ending cleanly at a sentence/clause boundary before falling back to an ellipsis.
    slice_text = candidate[: max_chars - 1]
    for sep in ("。", "；", ".", ";", "，", ",", " "):
        idx = slice_text.rfind(sep)
        if idx >= max_chars // 2:
            return slice_text[:idx].rstrip(" ,，;；") + "…"
    return slice_text.rstrip() + "…"


@dataclass(frozen=True)
class TranscriptRecord:
    transcript_id: str
    user_id: str
    agent_id: str
    session_id: str
    role: str
    text: str
    timestamp: float
    speaker: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryNode:
    node_id: str
    user_id: str
    agent_id: str
    parent_path: str
    path: str
    title: str
    summary: str
    content: str
    layer: str
    confidence_score: float
    verification_status: str
    salience_tag: str
    source_transcript_ids: tuple[str, ...]
    version: int
    is_deleted: bool
    created_at: float
    updated_at: float
    always_active: bool = False
    occurrence_start: float | None = None
    occurrence_end: float | None = None
    attribute_slot: str = ""

    @property
    def index_summary(self) -> str:
        return self.summary

    @property
    def detail_body(self) -> str:
        return self.content

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "parent_path": self.parent_path,
            "path": self.path,
            "title": self.title,
            "summary": self.summary,
            "index_summary": self.summary,
            "content": self.content,
            "detail_body": self.content,
            "layer": self.layer,
            "confidence_score": self.confidence_score,
            "verification_status": self.verification_status,
            "salience_tag": self.salience_tag,
            "always_active": self.always_active,
            "occurrence_start": self.occurrence_start,
            "occurrence_end": self.occurrence_end,
            "attribute_slot": self.attribute_slot,
            "source_transcript_ids": list(self.source_transcript_ids),
            "version": self.version,
            "is_deleted": self.is_deleted,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
