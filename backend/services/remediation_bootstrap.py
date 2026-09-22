"""Deterministic proactive remediation bootstrap.

When a student opens ACRLA through a Moodle remediation launch (chapter,
course, or overall) and has not yet typed anything, the tutor should start
teaching immediately instead of waiting for a first question ("explain
recursion", "help me learn", "where should I start?"). This module decides,
from structured launch/session state -- remediation_level, tutor state, a
one-shot session-scoped marker -- whether THIS turn should bootstrap a fresh
tutoring session, and if so which concept (and, for course/overall
launches, which course) to start with.

Bootstrap only ever fires when this turn's message is empty/whitespace-only
(see `get_bootstrap_target`'s first content check) -- a genuine, purposeful
message of any kind must always reach the real planner's classification
first, never be silently overridden by a hardcoded EXPLAIN plan. This is
deliberately content-agnostic: no keyword list decides whether a message
"counts" as purposeful, only whether one was typed at all.

Concept selection is fully deterministic, no LLM call:
- chapter launch: the concept is already known from the launch context
  (`context["current_concept"]`) -- no weakest-concept search.
- course launch: the lowest-mastery concept within the current course,
  via `tools.analytics_tools.execute_analytics_query` (the same canonical,
  manifest-backed, stale-duplicate-safe course data `run_analytics_query`
  already trusts) scoped to `current_course`.
- overall launch: the lowest-mastery concept across every canonical synced
  course the student is authorized to see, scope `all_courses` -- the
  returned row carries its own real `course_db_id`/`course_id`/`course_name`
  from the canonical course list, so the concept/course mapping is never
  invented.

This module never writes tutor state itself. It only decides whether/what to
bootstrap and marks the session so it never re-fires; the actual EXPLAIN
turn is produced by handing a concept-scoped `SemanticPlan` to the EXISTING
tutor-state-machine fresh-start compilation
(`agents.plan_compiler._compile_tutor_state`) exactly the way an ordinary
"explain <concept>" turn already works -- see
`agents.simple_agent._try_remediation_bootstrap_fast_path`, which is the one
and only caller. No new database column: the one-shot marker lives in the
same `StudentLongTermMemory.learning_state` JSON as tutor state and Quick
Progress Check state.
"""

from __future__ import annotations

from typing import Any

from tools.analytics_tools import execute_analytics_query


def get_bootstrap_target(context: dict[str, Any]) -> dict[str, Any] | None:
    """Return the deterministic bootstrap decision for this turn, or None if
    remediation should NOT be auto-started right now: already bootstrapped
    this session, a tutoring/practice/assessment flow is already active, or
    there is no usable launch context/concept to teach."""
    memory = context.get("memory")
    student_id = context.get("student_id")
    course_id = context.get("course_db_id")
    session_id = context.get("session_id")
    if not memory or not student_id or not course_id or not session_id:
        return None

    # A genuine, purposeful message this turn must ALWAYS reach normal
    # intent/goal classification first -- this module's whole job is to
    # teach BEFORE the student has to type anything (see the module
    # docstring), never to override or silently discard what they actually
    # typed once they have. Deliberately content-agnostic: no keyword list.
    # ANY non-empty message -- "explain X", "give me an example", "check my
    # progress", "recommend a video", even just "hi" -- must be classified
    # by the real planner, the only place equipped to understand what was
    # actually asked; only a truly empty/whitespace-only message (a
    # system-triggered call with no student input yet, e.g. an auto-launch
    # immediately after a fresh Moodle session start, via `ChatRequest.message
    # = ""` -- which the API layer already accepts, `models/schemas.py:
    # ChatRequest.message: str` has no `min_length`) means there is nothing
    # to classify, so it is safe to proactively teach. This turn is not
    # marked bootstrapped in that case (see `mark_bootstrapped`), so a later
    # turn can still bootstrap if it also arrives with no message.
    if str(context.get("message") or "").strip():
        return None

    # Never hijack a turn where the tutor loop or Quick Progress Check is
    # already mid-flow -- bootstrap is only for a genuinely fresh session.
    # Belt-and-suspenders alongside the session-scoped marker below (the two
    # normally become true on the exact same turn, but checking both keeps
    # this safe even if only one write of that turn ever landed).
    if context.get("tutor_state") or context.get("quick_progress_check"):
        return None

    # Only the session's genuinely FIRST turn -- an empty structured-turn
    # history is what "before tutoring begins" actually means. Without this,
    # bootstrap would also hijack a student's very first message even when
    # it is a specific, purposeful request for something else (e.g. "check
    # my progress" as literally the first thing typed) -- that must still
    # reach its own real classification instead of being silently overridden
    # by a proactive EXPLAIN turn. Once any turn has completed,
    # recent_structured_turns is never empty again for this session.
    if context.get("recent_structured_turns"):
        return None

    if _already_bootstrapped(memory, student_id, course_id, session_id):
        return None

    level = str(context.get("remediation_level") or "chapter").strip().lower()
    if level == "chapter":
        return _chapter_target(context, memory, student_id, course_id)
    if level in ("course", "overall"):
        return _weakest_concept_target(context, memory, student_id, course_id, level)
    return None


def mark_bootstrapped(context: dict[str, Any]) -> None:
    """Persist the one-shot, session-scoped marker so this session's
    remaining turns never re-trigger the bootstrap -- a NEW Moodle launch
    always creates a new session_id (see services.chat_orchestrator /
    routers.api's session_start), so a genuinely new remediation session
    still bootstraps normally."""
    memory = context.get("memory")
    student_id = context.get("student_id")
    course_id = context.get("course_db_id")
    session_id = context.get("session_id")
    if memory and student_id and course_id and session_id:
        memory.update_course_memory(student_id, course_id, {
            "remediation_bootstrap": {"session_id": session_id, "bootstrapped": True},
        })


def _already_bootstrapped(memory, student_id: str, course_id: str, session_id: str) -> bool:
    state = memory.get_course_memory(student_id, course_id).get("remediation_bootstrap") or {}
    return state.get("session_id") == session_id and bool(state.get("bootstrapped"))


def _chapter_target(context: dict[str, Any], memory, student_id: str, course_id: str) -> dict[str, Any] | None:
    concept = context.get("current_concept")
    if not concept:
        return None
    current_course = context.get("current_course") or {}
    mastery = round(float(memory.get_mastery(student_id, course_id, concept) or 0.0) * 100, 2)
    return {
        "started": True,
        "level": "chapter",
        "course_id": current_course.get("moodle_course_id"),
        "course_db_id": course_id,
        "course_name": current_course.get("name"),
        "course_mastery": _course_average_mastery(context, memory, student_id, course_id),
        "concept": concept,
        "mastery": mastery,
        "initial_state": "EXPLAIN",
        "selection_reason": "chapter_launch_selected_concept",
    }


def _weakest_concept_target(
    context: dict[str, Any], memory, student_id: str, course_id: str, level: str,
) -> dict[str, Any] | None:
    courses = context.get("canonical_courses") or []
    if not courses:
        return None
    scope = "current_course" if level == "course" else "all_courses"
    plan = {
        "operation": "rank", "scope": scope, "entity": "concept",
        "metrics": ["current_mastery"], "filters": {}, "group_by": [],
        "sort_by": "current_mastery", "sort_order": "asc", "limit": 1,
        "include_course_average": False, "include_overall_average": False,
        "include_weakest": False, "include_strongest": False, "confidence": 1.0,
    }
    result = execute_analytics_query(plan, student_id, memory, courses, current_course_id=course_id)
    items = result.get("items") or []
    if result.get("error") or not items:
        return None
    row = items[0]
    concept = row.get("concept")
    if not concept:
        return None
    target_course_db_id = row.get("course_db_id") or course_id
    reason = "lowest_mastery_in_current_course" if level == "course" else "lowest_mastery_across_all_courses"
    return {
        "started": True,
        "level": level,
        "course_id": row.get("course_id"),
        "course_db_id": target_course_db_id,
        "course_name": row.get("course_name"),
        "course_mastery": _course_average_mastery(context, memory, student_id, target_course_db_id),
        "concept": concept,
        "mastery": round(float(row.get("current_mastery") or 0.0) * 100, 2),
        "initial_state": "EXPLAIN",
        "selection_reason": reason,
    }


def _course_average_mastery(context: dict[str, Any], memory, student_id: str, target_course_db_id: str) -> float | None:
    """Course-level average mastery for the target concept's OWNING course --
    presentation-only (distinguishing "course mastery" from "topic mastery" in
    the UI, never changes how either is calculated). Reuses
    `execute_analytics_query`'s existing `entity="course"` aggregation, the
    same one `run_analytics_query`'s own course-average answers already use."""
    courses = context.get("canonical_courses") or []
    if not courses:
        return None
    plan = {
        "operation": "list", "scope": "current_course", "entity": "course",
        "metrics": ["course_average"], "filters": {}, "group_by": [],
        "sort_by": None, "sort_order": None, "limit": None,
        "include_course_average": True, "include_overall_average": False,
        "include_weakest": False, "include_strongest": False, "confidence": 1.0,
    }
    result = execute_analytics_query(plan, student_id, memory, courses, current_course_id=target_course_db_id)
    items = result.get("items") or []
    if result.get("error") or not items:
        return None
    return round(float(items[0].get("course_average") or 0.0) * 100, 2)
