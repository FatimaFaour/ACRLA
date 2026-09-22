"""Course/concept/scope resolution tools.

These tools never change which chapter/course/overall scope is active; they
only read the scope and available-concept set that chat_orchestrator already
computed for the turn.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session as DBSession

from models.db_models import Course
from services.course_concepts import canonicalize_concept
from tools.text_utils import normalize_key


def _course_for_db_id(db: DBSession, course_id: str | None) -> Course | None:
    if not course_id:
        return None
    return db.query(Course).filter_by(id=course_id).first()


def _concepts_from_arguments(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    """Resolve a tool call's requested concepts against the active scope.

    Falls back to the last structured reference, then the current concept,
    but always canonicalizes against `available_concepts` so a stale or
    out-of-scope name can never leak into a tool result.
    """
    concepts = arguments.get("concepts")
    if isinstance(concepts, str):
        concepts = [concepts]
    if not concepts:
        reference = context.get("last_reference") or {}
        concepts = [item.get("concept") for item in reference.get("items", []) if item.get("concept")]
    if not concepts and context.get("current_concept"):
        concepts = [context["current_concept"]]

    available = context.get("available_concepts") or []
    resolved: list[str] = []
    for raw in concepts or []:
        canonical = canonicalize_concept(raw)
        if canonical and canonical in available and canonical not in resolved:
            resolved.append(canonical)
            continue
        raw_key = normalize_key(raw)
        for candidate in available:
            candidate_key = normalize_key(candidate)
            if candidate_key == raw_key or raw_key in candidate_key or candidate_key in raw_key:
                if candidate not in resolved:
                    resolved.append(candidate)
                break
    return resolved


def get_current_scope_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "remediation_level": context.get("remediation_level"),
        "current_course": context.get("current_course"),
        "current_concept": context.get("current_concept"),
        "available_concepts": context.get("available_concepts") or [],
    }


def get_current_course_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    return {"course": context.get("current_course") or {}}


def get_available_concepts_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    return {"concepts": context.get("available_concepts") or []}


def get_course_for_concepts_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    db: DBSession = context["db"]
    concepts = _concepts_from_arguments(arguments, context)
    rows = []
    for concept in concepts:
        matches = []
        concept_key = normalize_key(concept)
        for course in db.query(Course).all():
            course_context = context.get("course_context_by_db_id", {}).get(course.id)
            course_concepts = course_context.get("concepts") if course_context else []
            if any(normalize_key(item) == concept_key for item in course_concepts):
                matches.append({
                    "course_id": course.id,
                    "moodle_course_id": course.moodle_course_id,
                    "course_name": course.name,
                })
        rows.append({"concept": concept, "courses": matches})
    return {"concept_courses": rows}
