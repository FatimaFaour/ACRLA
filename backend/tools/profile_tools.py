"""Student profile and study-recommendation tools.

Recommendations must be driven by remediation scope, mastery, profile/
preferences, and conversation context -- never by "this concept happens to
have a PDF." Neither tool here touches RAG/course-material retrieval at all;
`run_study_recommendation_tool` only ever ranks concepts that are already in
the active scope by their stored mastery.
"""

from __future__ import annotations

from typing import Any

from services.memory_manager import MemoryManager
from tools.mastery_tools import get_mastery_for_concepts_tool


def get_student_learning_profile_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Read-only snapshot of who the student is and what scope applies.

    Recommendation/tutoring reasoning should start here (and/or
    `run_study_recommendation`), not with course-material search.
    """
    memory: MemoryManager | None = context.get("memory")
    student_id = context.get("student_id")
    name = None
    preferred_difficulty = context.get("difficulty")
    if memory and student_id:
        prefs = memory.get_profile_preferences(student_id)
        name = prefs.get("name")
        preferred_difficulty = prefs.get("difficulty") or preferred_difficulty
    return {
        "tool": "get_student_learning_profile",
        "success": True,
        "name": name,
        "preferred_difficulty": preferred_difficulty,
        "remediation_level": context.get("remediation_level"),
        "current_course": context.get("current_course"),
        "available_concepts": context.get("available_concepts") or [],
        "weak_concepts": context.get("weak_concepts") or [],
        "tutoring_strategy": context.get("tutoring_strategy") or {},
    }


def run_study_recommendation_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Recommend a concept to study next, ranked by stored mastery only.

    Candidate concepts come from (in priority order): an explicit
    `arguments["candidate_concepts"]` list, the most recent structured turn
    that named two or more concepts (e.g. a just-discussed comparison -- this
    is what backs "which one should I start with?"/"explain the weaker one"),
    or otherwise every concept in the current remediation scope. Whichever
    candidate has the lowest current mastery is recommended -- pedagogical
    dependency data is deliberately not fabricated since ACRLA does not store
    real concept-dependency data.
    """
    candidates = arguments.get("candidate_concepts")
    basis = "lowest_mastery_among_candidates"
    if not candidates:
        for turn in reversed(context.get("recent_structured_turns") or []):
            turn_concepts = (turn.get("resolved_entities") or {}).get("concepts") or []
            if len(turn_concepts) >= 2:
                candidates = turn_concepts
                break
    if not candidates:
        candidates = context.get("available_concepts") or []
        basis = "lowest_mastery_in_scope"

    if not candidates:
        return {
            "tool": "run_study_recommendation",
            "success": False,
            "recommended_concept": None,
            "current_mastery": None,
            "basis": "no_candidates_in_scope",
            "candidates": [],
        }

    mastery_result = get_mastery_for_concepts_tool(context, {"concepts": candidates})
    rows = mastery_result.get("mastery") or []
    selected = min(rows, key=lambda item: item.get("current_mastery", 0.0), default=None)
    return {
        "tool": "run_study_recommendation",
        "success": bool(selected),
        "recommended_concept": (selected or {}).get("concept"),
        "current_mastery": (selected or {}).get("current_mastery"),
        "basis": basis,
        "candidates": rows,
    }
