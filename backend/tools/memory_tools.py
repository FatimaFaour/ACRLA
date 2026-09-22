"""Recent-conversation, last-reference, and structured-turn lookup tools.

Structured turns (see `agents.agent_models.ConversationTurn`) are the backing
store behind "which one should I start with?", "why these?", and "what source
did you use?" style follow-ups: instead of re-deriving that context from raw
text, the agent reads the last turn(s) that actually recorded a recommendation,
resolved entities, or response metadata.
"""

from __future__ import annotations

from typing import Any

from services.memory_manager import get_buffer


MAX_STORED_TURNS = 20

_structured_turns: dict[str, list[dict[str, Any]]] = {}


def save_structured_turn(session_id: str, turn: dict[str, Any]) -> None:
    """Append one structured turn for a session, capping history length."""
    if not session_id:
        return
    turns = _structured_turns.setdefault(session_id, [])
    turns.append(turn)
    if len(turns) > MAX_STORED_TURNS:
        del turns[: len(turns) - MAX_STORED_TURNS]


def get_recent_structured_turns(session_id: str, limit: int = 6) -> list[dict[str, Any]]:
    return list(_structured_turns.get(session_id, []))[-limit:]


def get_last_structured_turn(session_id: str) -> dict[str, Any] | None:
    turns = _structured_turns.get(session_id) or []
    return turns[-1] if turns else None


def _turns_for_context(context: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    """Prefer turns already loaded onto the context packet for this turn."""
    turns = context.get("recent_structured_turns")
    if turns is not None:
        return list(turns)[-limit:]
    return get_recent_structured_turns(context.get("session_id", ""), limit)


def get_recent_conversation_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    limit = int(arguments.get("limit") or 8)
    messages = context.get("recent_messages") or get_buffer(context.get("session_id", "")).last_n(limit)
    return {"messages": messages[-limit:]}


def get_last_reference_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    return {"last_reference": context.get("last_reference") or {}}


def get_recent_structured_turns_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    limit = int(arguments.get("limit") or 6)
    return {"turns": _turns_for_context(context, limit)}


def get_last_recommendation_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Return the most recent turn that recorded a recommendation + reason."""
    for turn in reversed(_turns_for_context(context, 10)):
        if turn.get("recommendation"):
            return {
                "recommendation": turn["recommendation"],
                "recommendation_reason": turn.get("recommendation_reason"),
                "source_turn_goal": turn.get("goal"),
            }
    return {"recommendation": None, "recommendation_reason": None}


def get_last_response_metadata_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Return pipeline/sources/evidence metadata for the previous turn only."""
    turns = _turns_for_context(context, 1)
    if not turns:
        return {"selected_pipeline": None, "sources": [], "evidence": {}, "goal": None}
    last = turns[-1]
    return {
        "selected_pipeline": last.get("selected_pipeline"),
        "sources": last.get("sources") or [],
        "evidence": last.get("evidence") or {},
        "goal": last.get("goal"),
    }
