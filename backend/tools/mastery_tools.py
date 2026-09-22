"""Mastery lookup tools.

Read-only: these tools report mastery, they never write it. Mastery is
assessment-gated elsewhere (Quick Progress Check only); nothing here may
change a stored mastery value.
"""

from __future__ import annotations

from typing import Any

from services.memory_manager import MemoryManager
from tools.course_tools import _concepts_from_arguments
from tools.text_utils import normalize_key


def get_mastery_for_concepts_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    memory: MemoryManager = context["memory"]
    concepts = _concepts_from_arguments(arguments, context)
    # course_id always comes from context (the already-resolved active course),
    # never from tool arguments -- otherwise a hallucinated/injected argument
    # could redirect this read to an out-of-scope course.
    course_id = context.get("course_db_id")
    reference_items = (context.get("last_reference") or {}).get("items") or []
    reference_by_concept = {
        normalize_key(item.get("concept")): item
        for item in reference_items
        if item.get("concept")
    }
    # For overall scope (multiple canonical courses), a concept requested by
    # name (not already in last_reference) belongs to whichever canonical,
    # deduplicated synced course actually teaches it -- not to the single
    # currently active course. Without this, an overall-scope recommendation
    # would look every concept's mastery up against the wrong course and get
    # back 0.0 for anything outside the currently clicked course.
    concept_to_canonical_course: dict[str, dict] = {}
    for course in context.get("canonical_courses") or []:
        for concept in course.get("concepts") or []:
            concept_to_canonical_course.setdefault(normalize_key(concept), course)

    rows = []
    for concept in concepts:
        reference_item = reference_by_concept.get(normalize_key(concept), {})
        canonical_course = concept_to_canonical_course.get(normalize_key(concept))
        item_course_id = (
            reference_item.get("course_db_id")
            or reference_item.get("course_id")
            or (canonical_course.get("db_course_id") if canonical_course else None)
            or course_id
        )
        current_mastery = reference_item.get("current_mastery")
        if current_mastery is None:
            current_mastery = memory.get_mastery(context["student_id"], item_course_id, concept) if item_course_id else 0.0
        rows.append({
            "concept": concept,
            "course_id": item_course_id,
            "course_name": (
                reference_item.get("course_name")
                or (canonical_course.get("name") if canonical_course else None)
                or (context.get("current_course") or {}).get("name")
            ),
            "current_mastery": current_mastery,
        })
    return {"mastery": rows}


def get_all_mastery_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """List mastery across the courses the active remediation scope allows.

    `course_ids` is never taken from `arguments` -- only from context, so a
    hallucinated/injected tool-call argument can never widen this beyond the
    student's own courses (and, for chapter/course scope, beyond the single
    active course).
    """
    memory: MemoryManager = context["memory"]
    remediation_level = str(context.get("remediation_level") or "chapter").lower()
    canonical_by_db_id = {
        course.get("db_course_id"): course
        for course in (context.get("canonical_courses") or [])
        if course.get("db_course_id")
    }
    if remediation_level == "overall":
        scoped_courses = list(canonical_by_db_id.values())
    else:
        current_course = canonical_by_db_id.get(context.get("course_db_id"))
        if not current_course and context.get("course_db_id"):
            current_course = {
                "db_course_id": context["course_db_id"],
                "moodle_course_id": (context.get("current_course") or {}).get("moodle_course_id"),
                "name": (context.get("current_course") or {}).get("name"),
                "concepts": context.get("available_concepts") or [],
                "merged_db_course_ids": [context["course_db_id"]],
            }
        scoped_courses = [current_course] if current_course else []

    rows = []
    for course in scoped_courses:
        if not course:
            continue
        allowed_concepts = list(dict.fromkeys(course.get("concepts") or []))
        for concept in allowed_concepts:
            rows.append({
                "concept": concept,
                "course_id": course.get("db_course_id"),
                "moodle_course_id": course.get("moodle_course_id"),
                "course_name": course.get("name"),
                "current_mastery": _merged_course_mastery_for_tool(memory, context["student_id"], course, concept),
            })
    return {"mastery": rows}


def _merged_course_mastery_for_tool(memory: MemoryManager, student_id: str, course: dict[str, Any], concept: str) -> float:
    values = []
    for db_course_id in course.get("merged_db_course_ids") or [course.get("db_course_id")]:
        if not db_course_id:
            continue
        value = memory.get_mastery(student_id, db_course_id, concept)
        if value is not None:
            values.append(value)
    return max(values) if values else 0.0


def get_mastery_policy_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "bands": [
            {"label": "Weak", "range": "0-49%"},
            {"label": "Moderate", "range": "50-79%"},
            {"label": "Strong", "range": "80-100%"},
        ]
    }


def get_mastery_scoring_methodology_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "rules": [
            "Initial Moodle mastery is the baseline.",
            "Current ACRLA mastery starts from that baseline.",
            "Chat explanations and ordinary practice do not update mastery.",
            "Only Quick Progress Check updates mastery.",
            "calculated_mastery = 0.7 * current_mastery + 0.3 * assessment_score",
            "updated_mastery = max(current_mastery, calculated_mastery, initial_moodle_mastery)",
            "Mastery never decreases in the MVP.",
        ]
    }


def select_lowest_mastery_concept_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Pick the lowest-mastery concept among available/requested concepts.

    RQ1 edge-case fix: when nothing supplies a candidate list (no explicit
    `concepts` argument, no `current_concept`, no `last_reference` item --
    the "help me study"/nothing-named case), `_concepts_from_arguments`
    alone returns `[]`. Falls back here to the current AUTHORIZED scope's
    own `context["available_concepts"]` (already resolved elsewhere for
    this exact remediation level -- chapter/course/overall -- never
    re-derived or widened here) so this tool still finds a real weakest
    concept instead of silently returning none. `_concepts_from_arguments`
    itself is unchanged -- every other caller keeps its original behavior.
    """
    if not _concepts_from_arguments(arguments, context):
        arguments = {**arguments, "concepts": list(context.get("available_concepts") or [])}
    mastery_result = get_mastery_for_concepts_tool(context, arguments)
    rows = mastery_result.get("mastery") or []
    selected = min(rows, key=lambda item: item.get("current_mastery", 0.0), default=None)
    return {"selected_concept": selected}


def select_lowest_mastery_among_previous_turn_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Pick the lower-mastery concept among a previously discussed set.

    Backs follow-ups like "which one should I start with?" or "explain the
    weaker one" after a comparison: the concepts come from the last structured
    conversation turn's `resolved_entities`, not from the current message
    (which may not name any concept at all).
    """
    concepts = arguments.get("concepts")
    if not concepts:
        recent_turns = context.get("recent_structured_turns") or []
        for turn in reversed(recent_turns):
            candidate = (turn.get("resolved_entities") or {}).get("concepts") or []
            if len(candidate) >= 2:
                concepts = candidate
                break
    if not concepts:
        return {"selected_concept": None, "candidates": []}

    mastery_result = get_mastery_for_concepts_tool(context, {**arguments, "concepts": concepts})
    rows = mastery_result.get("mastery") or []
    selected = min(rows, key=lambda item: item.get("current_mastery", 0.0), default=None)
    return {"selected_concept": selected, "candidates": rows}
