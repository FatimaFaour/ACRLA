"""Shared JSON parsing + compact serialization helpers.

Used by planner.py, evidence_validator.py, and response_generator.py so each
component does not need its own copy of "strip markdown fences, find the
first {...} block, json.loads it" boilerplate.
"""

from __future__ import annotations

import json
import re
from typing import Any


class AgentJSONError(ValueError):
    """Raised when an LLM response cannot be parsed even after one repair."""


def extract_message_text(response: Any) -> str:
    """Extract the plain text of a chat-model response, regardless of
    whether the provider represents `.content` as a plain string or a list
    of text parts.

    `langchain_google_genai.ChatGoogleGenerativeAI` builds `AIMessage.content`
    as a `list[str]` whenever a response candidate has more than one text
    part (confirmed by reading `_parse_response_candidate` in
    `langchain_google_genai/chat_models.py`: `content = [content, text]` the
    moment a second text part appears) -- observed in practice with
    "thinking"-capable Gemini models, where an internal reasoning fragment
    can arrive as its own part ahead of the actual answer. Every call site in
    this codebase used to do `str(getattr(response, "content", response))`,
    which on a list produces Python's list repr (e.g.
    `"['...reasoning...', '{\\"goal\\": ...}']"`) instead of the underlying
    text -- exactly the "malformed/list-like fragments" JSON-parse failures
    reported against the Gemini provider. Joining the real text parts with
    newlines instead keeps any JSON object embedded in the text syntactically
    intact, so the existing balanced-brace extraction in `parse_json_object`
    below still finds it even with a leading reasoning fragment -- no new
    parsing/extraction logic needed, no assumption about which part is which.

    A plain string, or any other shape (a provider that hands back the
    response object itself when it has no `.content`), passes through via
    the existing `str(...)` fallback unchanged -- this is purely additive
    for the list case.
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse one JSON object from an LLM response with one safe repair attempt."""
    text = str(raw or "").strip()
    if not text:
        raise AgentJSONError("empty_model_output")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    candidate = _extract_balanced_json_object(text) or text
    try:
        data = json.loads(candidate)
    except Exception as first_error:
        try:
            data = _repair_and_parse_json(candidate)
        except Exception as repair_error:
            raise AgentJSONError(f"json_parse_failed:{first_error}; repair_failed:{repair_error}") from repair_error
    if not isinstance(data, dict):
        raise AgentJSONError(f"json_root_not_object:{type(data).__name__}")
    return data


def _extract_balanced_json_object(text: str) -> str | None:
    """Return the first balanced {...} object, respecting quoted strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return text[start:]


def _repair_and_parse_json(candidate: str) -> dict[str, Any]:
    """Use LangChain's partial JSON parser once when ordinary json.loads fails."""
    try:
        from langchain_core.utils.json import parse_partial_json
    except Exception as exc:
        raise AgentJSONError(f"repair_parser_unavailable:{exc}") from exc
    repaired = parse_partial_json(candidate)
    if not isinstance(repaired, dict):
        raise AgentJSONError(f"repair_root_not_object:{type(repaired).__name__}")
    return repaired


def model_validate(model_cls, data: dict[str, Any]):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def model_dump(model) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def compact_json(value: Any, limit: int = 2200) -> str:
    """Serialize valid compact JSON without cutting through the JSON syntax."""
    compacted = _compact_value(value)
    try:
        text = json.dumps(compacted, ensure_ascii=True, default=str, separators=(",", ":"))
    except TypeError:
        text = json.dumps(str(value), ensure_ascii=True)
    if len(text) <= limit:
        return text
    fallback = {
        "_truncated": True,
        "original_type": type(value).__name__,
        "preview": str(value)[: max(80, limit - 120)],
    }
    return json.dumps(fallback, ensure_ascii=True, separators=(",", ":"))


def compact_observations(observations: list[dict[str, Any]], limit: int = 2600) -> str:
    """Serialize tool observations as small valid JSON summaries for planner prompts."""
    summaries = []
    for observation in observations or []:
        result = observation.get("result") or {}
        summaries.append({
            "tool": observation.get("tool"),
            "rejected": bool(observation.get("rejected")),
            "arguments": _compact_value(observation.get("arguments") or {}, depth=0, max_string=120),
            "result_summary": summarize_tool_result(result),
        })
    return compact_json(summaries, limit=limit)


def summarize_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep only decision-relevant result facts; omit full RAG chunk text."""
    if not isinstance(result, dict):
        return {"value": _compact_value(result)}
    summary: dict[str, Any] = {"keys": sorted(result.keys())}
    # Truthy check: some tools always include an "error" key set to None on
    # success, and "error" in result would misreport that as a real error.
    if result.get("error"):
        summary["error"] = str(result.get("error"))[:240]
    for key in [
        "success", "query", "requested_concepts", "concepts_found", "sources",
        "selected_concept", "recommended_concept", "recommendation_reason",
        "mastery", "items", "evidence", "selected_pipeline", "reply",
    ]:
        if key in result:
            summary[key] = _compact_value(result.get(key), max_list=6, max_string=260)
    if "chunks" in result:
        chunks = result.get("chunks") or []
        summary["chunk_count"] = len(chunks) if isinstance(chunks, list) else 0
        summary["chunks"] = [
            {
                "concept": chunk.get("concept"),
                "sources": chunk.get("sources") or [],
                "text_preview": str(chunk.get("text") or "")[:260],
            }
            for chunk in chunks[:3]
            if isinstance(chunk, dict)
        ]
    return summary


def _compact_value(value: Any, depth: int = 0, max_list: int = 8, max_string: int = 500) -> Any:
    # Leaf scalars (str/int/float/bool/None) are handled BEFORE the depth
    # check, never after -- a chunk's retrieved text, a concept name, or a
    # source filename is exactly this shape, and none of them grow deeper
    # by existing at depth 5 instead of depth 3 (only nesting a container
    # one level deeper does that). The depth cutoff below still applies to
    # every actual container type (dict/list/tuple/set), which is what
    # this function is protecting against runaway/excessively deep nested
    # structures -- it was previously checked first and applied to EVERY
    # value regardless of type, which silently replaced a plain string
    # sitting 5 levels deep (e.g. tool_results -> observation -> result ->
    # chunks -> chunk -> "text") with a "<str:truncated-depth>" placeholder,
    # even though that string itself was never large or deeply nested --
    # only its position in the surrounding structure was.
    if isinstance(value, str):
        return value if len(value) <= max_string else value[:max_string] + "...(truncated)"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if depth > 4:
        return f"<{type(value).__name__}:truncated-depth>"
    if isinstance(value, dict):
        items = list(value.items())
        compacted = {str(key): _compact_value(item, depth + 1, max_list=max_list, max_string=max_string) for key, item in items[:20]}
        if len(items) > 20:
            compacted["_truncated_keys"] = len(items) - 20
        return compacted
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        compacted = [_compact_value(item, depth + 1, max_list=max_list, max_string=max_string) for item in values[:max_list]]
        if len(values) > max_list:
            compacted.append({"_truncated_items": len(values) - max_list})
        return compacted
    return str(value)[:max_string]
