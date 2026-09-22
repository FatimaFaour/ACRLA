"""Dialogue tools: agent-callable wrappers around deterministic policies.

These give the planner tool-shaped access (chosen by meaning, not by adding
another exact-phrase regex) to behavior that used to live only as
phrase-matched handlers inside chat_orchestrator.py: source provenance,
methodology explanations, recommendation explanations, mastery-guard wording,
course/chapter scope lookup, preference lookup, and focus switching.

Like `tools.analytics_tools.run_analytics_query_tool`, most of these return a
pre-formatted `"reply"` string alongside structured fields -- the underlying
policy in `agents.policies` is itself the complete, deterministic, safe
answer (no LLM involved), so there is nothing left for a response generator to
add. The one exception below is `switch_focus_tool`, which has a real (scope-
checked, non-mastery) side effect: it moves the session's active topic.
"""

from __future__ import annotations

from typing import Any

from agents import policies
from tools.memory_tools import _turns_for_context


def get_source_provenance_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Answer "was that from the PDF / what source did you use?" from stored metadata."""
    turns = _turns_for_context(context, 1)
    metadata = turns[-1] if turns else {}
    reply = policies.source_provenance_response(metadata)
    return {
        "tool": "get_source_provenance",
        "success": True,
        "selected_pipeline": metadata.get("selected_pipeline"),
        "sources": metadata.get("sources") or [],
        "reply": reply,
    }


_METHODOLOGY_RESPONSES = {
    "scoring": policies.scoring_methodology_response,
    "tutoring": policies.tutoring_methodology_response,
    "policy": policies.mastery_policy_response,
    "meta": policies.methodology_meta_response,
}


def explain_methodology_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Explain mastery scoring, tutoring methodology, mastery policy bands, or the
    difference between them (`arguments["topic"]` in scoring|tutoring|policy|meta).
    Ambiguous/missing topic asks a clarifying question instead of guessing.
    """
    topic = str(arguments.get("topic") or "").strip().lower()
    responder = _METHODOLOGY_RESPONSES.get(topic)
    if not responder:
        return {
            "tool": "explain_methodology",
            "success": True,
            "topic": topic or None,
            "reply": policies.methodology_clarification_response(),
        }
    return {
        "tool": "explain_methodology",
        "success": True,
        "topic": topic,
        "reply": responder(),
    }


def explain_recommendation_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Answer "why these?" / "why that one?" from the last stored recommendation."""
    for turn in reversed(_turns_for_context(context, 10)):
        if turn.get("recommendation"):
            reply = policies.recommendation_explanation_response(
                turn.get("recommendation"), turn.get("recommendation_reason"),
            )
            return {
                "tool": "explain_recommendation",
                "success": True,
                "recommendation": turn.get("recommendation"),
                "recommendation_reason": turn.get("recommendation_reason"),
                "reply": reply,
            }
    return {
        "tool": "explain_recommendation",
        "success": True,
        "recommendation": None,
        "recommendation_reason": None,
        "reply": policies.recommendation_explanation_response(None, None),
    }


def get_mastery_guard_response_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Return the canned refusal for a mastery-modification request.

    Detection of whether a message IS a mastery-modification request stays a
    deterministic regex check in chat_orchestrator.py (a hard safety rule);
    this tool only supplies the reusable response text.
    """
    return {
        "tool": "get_mastery_guard_response",
        "success": True,
        "reply": policies.mastery_modification_guard_response(),
    }


def get_course_structure_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """List the tracked concepts for the active chapter/course/overall scope."""
    concepts = sorted(context.get("available_concepts") or [])
    return {
        "tool": "get_course_structure",
        "success": True,
        "remediation_level": context.get("remediation_level"),
        "available_concepts": concepts,
        "reply": policies.course_structure_response(concepts),
    }


def get_preferences_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Report the student's saved preferences (difficulty, name, routing mode). Read-only."""
    memory = context.get("memory")
    student_id = context.get("student_id")
    if not memory or not student_id:
        return {"tool": "get_preferences", "success": False, "preferences": {}, "reply": "I do not have your preferences available right now."}
    prefs = memory.get_profile_preferences(student_id)
    difficulty = str(prefs.get("difficulty") or context.get("difficulty") or "medium").replace("_", " ").strip().capitalize()
    name = prefs.get("name")
    parts = []
    if name:
        parts.append(f"name = {name}")
    parts.append(f"difficulty = {difficulty}")
    parts.append("response routing = Automatic")
    return {
        "tool": "get_preferences",
        "success": True,
        "preferences": {"name": name, "difficulty": difficulty},
        "reply": "Your saved preferences are: " + ", ".join(parts) + ".",
    }


def switch_focus_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Switch the active tutoring focus to a concept within the current scope.

    Only ever moves within `context["available_concepts"]` (the scope
    chat_orchestrator already resolved for this turn), so it cannot bypass
    chapter/course/overall scope, and it never touches mastery. It does not
    reproduce the legacy chapter-lock redirect (that depends on Moodle-manifest
    resolution that lives in chat_orchestrator.py) -- a session locked to one
    chapter should keep using the legacy NAVIGATION path for that extra check.
    """
    from services.course_concepts import canonicalize_concept

    requested = arguments.get("concept") or context.get("message") or ""
    available = list(context.get("available_concepts") or [])
    concept = canonicalize_concept(requested)
    if concept not in available:
        concept = next((c for c in available if c.lower() == str(requested).strip().lower()), None)
    if not concept:
        return {
            "tool": "switch_focus",
            "success": False,
            "switched_to": None,
            "available_concepts": available,
            "reply": f"Which course topic should we switch to? Choose one of: {', '.join(available)}.",
        }

    memory = context.get("memory")
    session_id = context.get("session_id")
    if memory and session_id:
        memory.set_current_topic(session_id, concept)
    return {
        "tool": "switch_focus",
        "success": True,
        "switched_to": concept,
        "reply": f"Focus switched to {concept}.",
    }
