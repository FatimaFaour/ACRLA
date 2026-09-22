"""
ACRLA FastAPI route layer.

Purpose:
    Exposes HTTP endpoints used by the standalone frontend and Moodle plugin.

Role in ACRLA:
    This file translates Moodle launches, material syncs, chat messages,
    analytics requests, and Quick Progress Check submissions into backend
    service calls. It is also where chapter/course/overall remediation scope is
    initialized before the chat orchestrator takes over.

Main responsibilities:
    - Moodle profile/mastery/material sync
    - Moodle launch redirect and session setup
    - chat endpoint response serialization
    - RAG collection diagnostics and rebuild helpers
    - assessment generation and mastery update persistence
"""

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session as DBSession
from pathlib import Path
import tempfile, os, shutil, re, json
import random
from urllib.parse import urlencode
from datetime import datetime
from uuid import uuid4

from models.db_models import (
    get_db,
    Course,
    Student,
    Session as SessionModel,
    AssessmentRecord,
    AssessmentQuestionVariant,
)
from models.schemas import (
    SessionStartRequest, SessionStartResponse,
    ChatRequest, ChatResponse,
    IngestResponse, StudentAnalytics,
    MoodleSyncRequest, MoodleSyncResponse,
    AssessmentStartRequest, AssessmentStartResponse,
    AssessmentSubmitRequest, AssessmentSubmitResponse,
)
from services.memory_manager import MemoryManager
from services.adaptive_tutor import generate_greeting
from services.chat_orchestrator import handle_message
from services.course_concepts import ALLOWED_CONCEPTS, canonicalize_concept, concepts_for_course, sub_concepts_for
from services.llm_factory import get_llm
from pipelines.rag_pipeline import (
    get_vectorstore,
    ingest_documents,
    rebuild_course_collection,
    reset_course_collection,
    retrieve_context,
)

router = APIRouter()
_active_assessments: dict[str, dict] = {}


# ==========================================================
# Shared Normalization and Mastery Helpers
# ==========================================================


def _clean_concept_name(raw: str) -> str:
    concept = canonicalize_concept(raw)
    if concept:
        return concept
    text = str(raw).replace("_", " ").strip()
    return " ".join(part.capitalize() for part in text.split())


def _score_to_mastery(score) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    return value / 100 if value > 1 else value


def _mastery_to_percent(value: float) -> float:
    percent = round(max(0.0, min(1.0, float(value or 0.0))) * 100, 2)
    return int(percent) if percent.is_integer() else percent


def _normalize_level_type(value: str | None) -> str:
    level = str(value or "chapter").strip().lower()
    return level if level in {"overall", "course", "chapter"} else "chapter"


def _norm_label(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _demo_initial_mastery_for_concept(course: Course, concept: str) -> float | None:
    """Fixed thesis-demo Moodle mastery baselines.

    These are initial Moodle values only. Current ACRLA mastery may increase
    after assessments, but must never fall below these values.
    """
    course_name = _norm_label(course.name)
    concept_name = _assessment_concept(concept) or str(concept or "").strip()
    concept_norm = _norm_label(concept_name)

    if "data science" in course_name:
        return 55.0

    if "computer science" in course_name:
        cs_values = {
            "recursion": 40.0,
            "sorting algorithms": 50.0,
            "pointers and memory management": 55.0,
            "binary trees and bsts": 60.0,
        }
        return cs_values.get(concept_norm)

    if "mathematics" in course_name or course_name == "math":
        math_values = {
            "logic": 76.0,
            "sets": 88.0,
            "graphs": 90.0,
            "relations functions": 33.0,
        }
        return math_values.get(concept_norm)

    return None


def _demo_initial_mastery_for_scope(
    memory: MemoryManager,
    student_id: str,
    course: Course,
    level_type: str,
    concepts: list[str],
) -> float | None:
    level_type = _normalize_level_type(level_type)
    if level_type == "chapter" and concepts:
        return _demo_initial_mastery_for_concept(course, concepts[0])

    course_concepts = _assessment_concepts_for_course(memory, student_id, course)
    values = [
        value for value in (
            _demo_initial_mastery_for_concept(course, concept)
            for concept in (course_concepts or concepts)
        )
        if value is not None
    ]
    if level_type == "course" and values:
        return round(sum(values) / len(values), 2)
    if values and not concepts:
        return round(sum(values) / len(values), 2)
    if values and level_type == "chapter":
        return round(sum(values) / len(values), 2)
    return None


def _demo_baseline_map_for_course(memory: MemoryManager, student_id: str, course: Course) -> dict[str, float]:
    course_name = _norm_label(course.name)

    if "computer science" in course_name:
        preferred = [
            "Recursion",
            "Sorting Algorithms",
            "Pointers and Memory Management",
            "Binary Trees and BSTs",
        ]
        return {
            concept: _demo_initial_mastery_for_concept(course, concept)
            for concept in preferred
            if _demo_initial_mastery_for_concept(course, concept) is not None
        }
    elif "mathematics" in course_name or course_name == "math":
        preferred = ["Logic", "Sets", "Graphs", "Relations Functions"]
        return {
            concept: _demo_initial_mastery_for_concept(course, concept)
            for concept in preferred
            if _demo_initial_mastery_for_concept(course, concept) is not None
        }
    elif "data science" in course_name:
        concepts = _material_concepts_for_course(course) or [
            record.concept for record in memory.get_all_mastery(student_id, course.id)
        ]
        baseline = {}
        for concept in concepts:
            canonical = _assessment_concept(concept)
            if canonical:
                baseline[canonical] = 55.0
        return baseline

    return {}


def _weakest_concept_for_course(
    memory: MemoryManager,
    student_id: str,
    course_id: str,
    moodle_course_id: int | None = None,
    course_name: str | None = None,
) -> str:
    allowed = set()
    records = memory.get_all_mastery(student_id, course_id)
    dynamic_records = [
        record for record in records
        if _assessment_concept(record.concept)
        and (not allowed or _assessment_concept(record.concept) in allowed)
    ]
    records = dynamic_records
    if records:
        weakest = min(records, key=lambda record: record.mastery_level or 0.0)
        concept = _assessment_concept(weakest.concept)
        if concept:
            return concept
    return "General Course Material"


def _weakest_course_for_student(db: DBSession, memory: MemoryManager, student_id: str) -> Course | None:
    courses = _available_courses_for_student(db, memory, student_id)
    courses_with_scores = []
    for course in courses:
        records = memory.get_all_mastery(student_id, course.id)
        if not records:
            continue
        avg = sum(record.mastery_level or 0.0 for record in records) / len(records)
        courses_with_scores.append((avg, course))
    if courses_with_scores:
        return min(courses_with_scores, key=lambda item: item[0])[1]
    return courses[0] if courses else None


def _percent_from_mastery(value: float) -> float:
    return round(max(0.0, min(1.0, float(value or 0.0))) * 100, 2)


def _avg_mastery_percent(memory: MemoryManager, student_id: str, course_id: str, concepts: list[str]) -> float:
    canonical = [concept for concept in (_assessment_concept(c) for c in concepts) if concept]
    if not canonical:
        return 0.0
    scores = [memory.get_mastery(student_id, course_id, concept) for concept in canonical]
    return _percent_from_mastery(sum(scores) / len(scores))


def _avg_mapped_mastery_percent(
    memory: MemoryManager,
    student_id: str,
    concepts: list[str],
    concept_internal_course_ids: dict[str, str],
    fallback_course_id: str,
) -> float:
    canonical = [concept for concept in (_assessment_concept(c) for c in concepts) if concept]
    if not canonical:
        return 0.0
    scores = [
        memory.get_mastery(
            student_id,
            concept_internal_course_ids.get(concept) or fallback_course_id,
            concept,
        )
        for concept in canonical
    ]
    return _percent_from_mastery(sum(scores) / len(scores))


def _mastery_scope_key(level_type: str, concepts: list[str]) -> str:
    canonical = sorted(concept for concept in (_assessment_concept(c) for c in concepts) if concept)
    return f"{_normalize_level_type(level_type)}:{'|'.join(canonical)}"


def _chapter_mastery_key(course: Course | None, concept: str) -> str:
    course_id = course.moodle_course_id if course and course.moodle_course_id is not None else "unknown"
    canonical = _assessment_concept(concept) or str(concept or "").strip()
    return f"chapter:{course_id}:{canonical}"


def _course_mastery_key(course: Course | None) -> str:
    course_id = course.moodle_course_id if course and course.moodle_course_id is not None else "unknown"
    return f"course:{course_id}"


def _overall_mastery_key(student_id: str) -> str:
    return f"overall:{student_id}"


def _level_mastery_key(student_id: str, course: Course | None, level_type: str, concepts: list[str]) -> str:
    level_type = _normalize_level_type(level_type)
    if level_type == "overall":
        return _overall_mastery_key(student_id)
    if level_type == "course":
        return _course_mastery_key(course)
    concept = concepts[0] if concepts else ""
    return _chapter_mastery_key(course, concept)


def _memory_mastery_maps(course_memory: dict) -> tuple[dict, dict]:
    initial = course_memory.get("moodle_initial_mastery") or {}
    current = course_memory.get("current_acrla_mastery") or {}
    return dict(initial), dict(current)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mastery_floor(initial_map: dict, key: str) -> float:
    return _safe_float(initial_map.get(key), 0.0)


def _floor_current_mastery(initial_map: dict, current_value: float, key: str) -> float:
    return round(max(_safe_float(current_value), _mastery_floor(initial_map, key)), 2)


def _concept_current_mastery_percent(
    memory: MemoryManager,
    student_id: str,
    course: Course,
    concept: str,
) -> float:
    course_memory = memory.get_course_memory(student_id, course.id)
    initial_map, current_map = _memory_mastery_maps(course_memory)
    key = _chapter_mastery_key(course, concept)
    demo_initial = _demo_initial_mastery_for_concept(course, concept)
    if demo_initial is not None:
        initial_map[key] = demo_initial
    if key in current_map:
        current = _safe_float(current_map.get(key), 0.0)
    else:
        current = demo_initial if demo_initial is not None else _mastery_to_percent(memory.get_mastery(student_id, course.id, concept))
    current = round(max(current, _safe_float(initial_map.get(key), 0.0)), 2)
    current_map[key] = current
    memory.update_course_memory(student_id, course.id, {
        "moodle_initial_mastery": initial_map,
        "current_acrla_mastery": current_map,
    })
    return current


def _course_initial_mastery_percent(memory: MemoryManager, student_id: str, course: Course) -> float:
    course_memory = memory.get_course_memory(student_id, course.id)
    initial_map, _current_map = _memory_mastery_maps(course_memory)
    course_key = _course_mastery_key(course)
    demo_course_initial = _demo_initial_mastery_for_scope(memory, student_id, course, "course", [])
    if demo_course_initial is not None:
        initial_map[course_key] = demo_course_initial
        return round(demo_course_initial, 2)
    explicit = _safe_float(initial_map.get(course_key), 0.0)
    if explicit > 0:
        return round(explicit, 2)
    concepts = _assessment_concepts_for_course(memory, student_id, course)
    initial_scores = []
    for concept in concepts:
        chapter_key = _chapter_mastery_key(course, concept)
        if chapter_key in initial_map:
            initial_scores.append(_safe_float(initial_map.get(chapter_key), 0.0))
        else:
            initial_scores.append(_mastery_to_percent(memory.get_mastery(student_id, course.id, concept)))
    return round(sum(initial_scores) / len(initial_scores), 2) if initial_scores else 0.0


def _course_current_mastery_percent(memory: MemoryManager, student_id: str, course: Course) -> float:
    course_memory = memory.get_course_memory(student_id, course.id)
    initial_map, current_map = _memory_mastery_maps(course_memory)
    key = _course_mastery_key(course)
    initial = _course_initial_mastery_percent(memory, student_id, course)
    initial_map[key] = initial
    if key in current_map:
        current = round(max(_safe_float(current_map.get(key), 0.0), initial), 2)
        current_map[key] = current
        memory.update_course_memory(student_id, course.id, {
            "moodle_initial_mastery": initial_map,
            "current_acrla_mastery": current_map,
        })
        return current
    current_map[key] = initial
    memory.update_course_memory(student_id, course.id, {
        "moodle_initial_mastery": initial_map,
        "current_acrla_mastery": current_map,
    })
    return round(initial, 2)


def _overall_initial_mastery_percent(db: DBSession, memory: MemoryManager, student_id: str) -> float:
    courses = [
        course for course in _available_courses_for_student(db, memory, student_id)
        if _assessment_concepts_for_course(memory, student_id, course)
    ]
    scores = [_course_initial_mastery_percent(memory, student_id, course) for course in courses]
    return round(sum(scores) / len(scores), 2) if scores else 0.0


def _overall_current_mastery_percent(db: DBSession, memory: MemoryManager, student_id: str) -> float:
    courses = _available_courses_for_student(db, memory, student_id)
    key = _overall_mastery_key(student_id)
    initial = _overall_initial_mastery_percent(db, memory, student_id)
    for course in courses:
        course_memory = memory.get_course_memory(student_id, course.id)
        initial_map, current_map = _memory_mastery_maps(course_memory)
        initial_map[key] = initial
        if key in current_map:
            current = round(max(_safe_float(current_map.get(key), 0.0), initial), 2)
            current_map[key] = current
            memory.update_course_memory(student_id, course.id, {
                "moodle_initial_mastery": initial_map,
                "current_acrla_mastery": current_map,
            })
            return current
    if courses:
        course_memory = memory.get_course_memory(student_id, courses[0].id)
        initial_map, current_map = _memory_mastery_maps(course_memory)
        initial_map[key] = initial
        current_map[key] = initial
        memory.update_course_memory(student_id, courses[0].id, {
            "moodle_initial_mastery": initial_map,
            "current_acrla_mastery": current_map,
        })
    return round(initial, 2)


def _canonical_current_mastery_percent(
    db: DBSession,
    memory: MemoryManager,
    student_id: str,
    course: Course,
    level_type: str,
    concepts: list[str],
    concept_internal_course_ids: dict[str, str],
) -> float:
    level_type = _normalize_level_type(level_type)
    if level_type == "overall":
        return _overall_current_mastery_percent(db, memory, student_id)
    if level_type == "course":
        return _course_current_mastery_percent(memory, student_id, course)
    scores = []
    for concept in concepts:
        target_course_id = concept_internal_course_ids.get(concept) or course.id
        target_course = memory.db.query(Course).filter_by(id=target_course_id).first() or course
        scores.append(_concept_current_mastery_percent(memory, student_id, target_course, concept))
    return round(sum(scores) / len(scores), 2) if scores else 0.0


def _ensure_mastery_baseline(
    db: DBSession,
    memory: MemoryManager,
    student_id: str,
    course_id: str,
    course: Course,
    level_type: str,
    concepts: list[str],
    concept_internal_course_ids: dict[str, str],
    clicked_score: float | None = None,
) -> tuple[float, float, str]:
    course_memory = memory.get_course_memory(student_id, course_id)
    initial_map, current_map = _memory_mastery_maps(course_memory)
    level_type = _normalize_level_type(level_type)
    key = _level_mastery_key(student_id, course, level_type, concepts)
    computed_mastery = _canonical_current_mastery_percent(
        db,
        memory,
        student_id,
        course,
        level_type,
        concepts,
        concept_internal_course_ids,
    )
    source_of_truth = "current_acrla_mastery"
    demo_initial = _demo_initial_mastery_for_scope(memory, student_id, course, level_type, concepts)
    initial_value = demo_initial if demo_initial is not None else (clicked_score if clicked_score is not None else computed_mastery)
    if key not in initial_map or demo_initial is not None:
        initial_map[key] = initial_value
        source_of_truth = "clicked_moodle_mastery" if clicked_score is not None else "fallback_moodle_mastery"
    if key in current_map:
        try:
            current_mastery = round(float(current_map[key]), 2)
        except (TypeError, ValueError):
            current_mastery = computed_mastery
    else:
        current_mastery = computed_mastery
    current_mastery = _floor_current_mastery(initial_map, current_mastery, key)
    current_map[key] = current_mastery
    memory.update_course_memory(student_id, course_id, {
        "moodle_initial_mastery": initial_map,
        "current_acrla_mastery": current_map,
    })
    print(
        "[ACRLA] mastery_baseline "
        f"student_id={student_id} "
        f"level_type={level_type} "
        f"course_id={course.moodle_course_id} "
        f"concept={', '.join(concepts)} "
        f"mastery_key={key} "
        f"clicked_score={clicked_score if clicked_score is not None else 'none'} "
        f"demo_initial={demo_initial if demo_initial is not None else 'none'} "
        f"computed_mastery={computed_mastery} "
        f"current_mastery={current_mastery} "
        f"source_of_truth={source_of_truth}"
    )
    return round(float(initial_map[key]), 2), round(float(current_mastery), 2), source_of_truth


def _set_current_acrla_mastery(
    memory: MemoryManager,
    student_id: str,
    course_id: str,
    level_type: str,
    concepts: list[str],
    updated_mastery: float,
    concept_internal_course_ids: dict[str, str] | None = None,
) -> None:
    concept_internal_course_ids = concept_internal_course_ids or {}
    course_memory = memory.get_course_memory(student_id, course_id)
    initial_map, current_map = _memory_mastery_maps(course_memory)
    course = memory.db.query(Course).filter_by(id=course_id).first()
    level_type = _normalize_level_type(level_type)
    key = _level_mastery_key(student_id, course, level_type, concepts)
    demo_initial = _demo_initial_mastery_for_scope(memory, student_id, course, level_type, concepts) if course else None
    if demo_initial is not None:
        initial_map[key] = demo_initial
    updated_mastery = _floor_current_mastery(initial_map, updated_mastery, key)
    current_map[key] = updated_mastery
    memory.update_course_memory(student_id, course_id, {
        "moodle_initial_mastery": initial_map,
        "current_acrla_mastery": current_map,
    })
    if level_type != "chapter":
        print(
            "[ACRLA] set_current_acrla_mastery "
            f"student_id={student_id} "
            f"level_type={level_type} "
            f"course_id={course.moodle_course_id if course else course_id} "
            f"mastery_key={key} "
            f"updated_mastery={updated_mastery} "
            "chapter_records_updated=False"
        )
        return
    for concept in concepts:
        canonical = _assessment_concept(concept)
        if not canonical:
            continue
        concept_course_id = concept_internal_course_ids.get(canonical) or course_id
        concept_memory = memory.get_course_memory(student_id, concept_course_id)
        concept_initial, concept_current = _memory_mastery_maps(concept_memory)
        concept_course = memory.db.query(Course).filter_by(id=concept_course_id).first()
        concept_demo_initial = _demo_initial_mastery_for_concept(concept_course, canonical) if concept_course else None
        concept_key = _chapter_mastery_key(concept_course, canonical)
        if concept_demo_initial is not None:
            concept_initial[concept_key] = concept_demo_initial
        concept_updated = _floor_current_mastery(concept_initial, updated_mastery, concept_key)
        concept_current[concept_key] = concept_updated
        memory.update_course_memory(student_id, concept_course_id, {
            "moodle_initial_mastery": concept_initial,
            "current_acrla_mastery": concept_current,
        })


def _weakest_concepts_for_course(
    memory: MemoryManager,
    student_id: str,
    course: Course,
    limit: int = 3,
) -> list[str]:
    allowed = _assessment_concepts_for_course(memory, student_id, course)
    records = {
        _assessment_concept(record.concept): record.mastery_level or 0.0
        for record in memory.get_all_mastery(student_id, course.id)
        if _assessment_concept(record.concept) in set(allowed)
    }
    ordered = sorted(allowed, key=lambda concept: records.get(concept, 0.0))
    return ordered[:limit]


def _course_scope_concepts(
    memory: MemoryManager,
    student_id: str,
    course: Course,
) -> list[str]:
    """Course concepts from the authoritative source (synced material), sorted weakest-first.

    Uses _material_concepts_for_course — immune to cross-course mastery record
    contamination. A previous overall session may have written CS/Math mastery records
    under a Data Science course_id; reading from mastery records would surface those
    alien concepts. The manifest/ChromaDB only contains what was actually synced for
    this specific course.

    If no material exists, return an empty scope instead of falling back to
    mastery records. Course-level remediation must be anchored to the clicked
    course material; stale records can contain concepts from prior overall runs.
    """
    material = _material_concepts_for_course(course)
    if not material:
        return []
    material_set = set(material)
    mastery_map = {
        _assessment_concept(r.concept): r.mastery_level or 0.0
        for r in memory.get_all_mastery(student_id, course.id)
        if _assessment_concept(r.concept) in material_set
    }
    return sorted(material, key=lambda c: mastery_map.get(c, 0.0))


def _available_courses_for_student(db: DBSession, memory: MemoryManager, student_id: str) -> list[Course]:
    courses = db.query(Course).all()
    real_courses = []
    for course in courses:
        course_name = _norm_label(course.name)
        if course.moodle_course_id == 1:
            continue
        if course.moodle_course_id and course.moodle_course_id >= 70000:
            continue
        if "mastery test" in course_name or course_name == "intro cs":
            continue
        real_courses.append(course)
    return real_courses if real_courses else courses


def _material_concepts_for_course(course: Course) -> list[str]:
    concepts: list[str] = []

    # Primary source: materials manifest written atomically during sync.
    # Reliable even when ChromaDB is temporarily unavailable or the vectorstore
    # cache on this worker hasn't yet seen the newly ingested documents.
    project_root = Path(__file__).resolve().parents[2]
    docs_dir = project_root / "course_docs" / f"moodle_course_{course.moodle_course_id}"
    manifest_path = docs_dir / "materials_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest_data = json.load(mf)
            files = manifest_data.get("files", manifest_data) if isinstance(manifest_data, dict) else {}
            for entry in (files or {}).values():
                if not isinstance(entry, dict):
                    continue
                # concept: already processed at sync time — use directly.
                raw_concept = entry.get("concept")
                if raw_concept:
                    c = _assessment_concept(str(raw_concept))
                    if c and c not in concepts and c.lower() != "moodle":
                        concepts.append(c)
                    # Explicit Moodle concept metadata wins. Avoid inferring
                    # from names like chapter1_logic, where "chapter1" can
                    # match legacy Computer Science chapter aliases.
                    continue
                # display_title: may contain colons — avoid Path.stem, use directly.
                raw_title = entry.get("display_title")
                if raw_title:
                    c = _assessment_concept(str(raw_title))
                    if c and c not in concepts and c.lower() != "moodle":
                        concepts.append(c)
                # original_file_name: strip extension before processing.
                raw_fn = entry.get("original_file_name")
                if raw_fn:
                    c = _assessment_concept(Path(str(raw_fn)).stem)
                    if c and c not in concepts and c.lower() != "moodle":
                        concepts.append(c)
            if concepts:
                print(
                    "[ACRLA] material_concepts_manifest "
                    f"course_id={course.moodle_course_id} "
                    f"course_name={course.name!r} "
                    f"concepts={concepts}"
                )
                return concepts
        except Exception as exc:
            print(f"[ACRLA] manifest_concepts_lookup_failed course_id={course.moodle_course_id}: {exc}")

    # Secondary source: live ChromaDB collection.
    try:
        vectorstore = get_vectorstore(course.moodle_course_id)
        raw = vectorstore._collection.get(limit=1000, include=["metadatas"])
        for meta in raw.get("metadatas") or []:
            for value in (
                meta.get("concept"),
                meta.get("chapter"),
                meta.get("display_title"),
                meta.get("source_file"),
                meta.get("document_origin"),
            ):
                if not value:
                    continue
                concept = _assessment_concept(Path(str(value)).stem)
                if concept and concept not in concepts and concept.lower() != "moodle":
                    concepts.append(concept)
    except Exception as exc:
        print(f"[ACRLA] material_concepts_lookup_failed course_id={course.moodle_course_id}: {exc}")

    print(
        "[ACRLA] material_concepts_chroma "
        f"course_id={course.moodle_course_id} "
        f"course_name={course.name!r} "
        f"concepts={concepts}"
    )
    return concepts


def _assessment_concepts_for_course(
    memory: MemoryManager,
    student_id: str,
    course: Course,
) -> list[str]:
    records = [
        concept for concept in (_assessment_concept(record.concept) for record in memory.get_all_mastery(student_id, course.id))
        if concept
    ]
    material = _material_concepts_for_course(course)
    # Union both sources so newly synced material is accepted even before any mastery record exists for it.
    return list(dict.fromkeys(records + material))


def _overall_remediation_concepts_for_student(
    db: DBSession,
    memory: MemoryManager,
    student_id: str,
) -> tuple[list[str], dict[str, int], dict[str, str], Course | None]:
    courses = _available_courses_for_student(db, memory, student_id)
    scored_courses = []
    for course in courses:
        concepts = _assessment_concepts_for_course(memory, student_id, course)
        if not concepts:
            continue
        avg = _avg_mastery_percent(memory, student_id, course.id, list(concepts))
        scored_courses.append((avg, course))
    scored_courses.sort(key=lambda pair: pair[0])

    concepts: list[str] = []
    concept_course_ids: dict[str, int] = {}
    concept_internal_course_ids: dict[str, str] = {}
    for _avg, course in scored_courses:
        weakest = _weakest_concept_for_course(
            memory,
            student_id,
            course.id,
            course.moodle_course_id,
            course.name,
        )
        if weakest and weakest not in concepts:
            concepts.append(weakest)
            concept_course_ids[weakest] = course.moodle_course_id
            concept_internal_course_ids[weakest] = course.id

    if len(concepts) < 2 and scored_courses:
        for _avg, course in scored_courses:
            for concept in _weakest_concepts_for_course(memory, student_id, course, limit=3):
                if concept not in concepts:
                    concepts.append(concept)
                    concept_course_ids[concept] = course.moodle_course_id
                    concept_internal_course_ids[concept] = course.id
                if len(concepts) >= 3:
                    break
            if len(concepts) >= 3:
                break

    primary_course = scored_courses[0][1] if scored_courses else None
    return concepts[:3], concept_course_ids, concept_internal_course_ids, primary_course


def _all_enrolled_scope_concepts(
    db: DBSession,
    memory: MemoryManager,
    student_id: str,
) -> tuple[list[str], dict[str, int], dict[str, str], Course | None]:
    """Return ALL concepts from ALL enrolled courses for broad scope.

    Courses sorted by avg mastery (weakest first). Concepts within each course
    sorted by mastery (weakest first). Result is interleaved round-robin across
    courses so canonical[:2] in the assessment generator naturally spans different
    courses — required for the cross-course integrated question check.
    """
    courses = _available_courses_for_student(db, memory, student_id)
    scored_courses: list[tuple[float, Course, list[str]]] = []
    for course in courses:
        concepts = _assessment_concepts_for_course(memory, student_id, course)
        if not concepts:
            continue
        avg = _avg_mastery_percent(memory, student_id, course.id, list(concepts))
        mastery_map = {
            _assessment_concept(r.concept): r.mastery_level or 0.0
            for r in memory.get_all_mastery(student_id, course.id)
            if _assessment_concept(r.concept)
        }
        sorted_concepts = sorted(concepts, key=lambda c: mastery_map.get(c, 0.0))
        scored_courses.append((avg, course, sorted_concepts))
    scored_courses.sort(key=lambda t: t[0])

    all_concepts: list[str] = []
    concept_course_ids: dict[str, int] = {}
    concept_internal_course_ids: dict[str, str] = {}
    max_per_course = max((len(t[2]) for t in scored_courses), default=0)
    for i in range(max_per_course):
        for _avg, course, course_concepts in scored_courses:
            if i < len(course_concepts):
                concept = course_concepts[i]
                if concept not in all_concepts:
                    all_concepts.append(concept)
                    concept_course_ids[concept] = course.moodle_course_id
                    concept_internal_course_ids[concept] = course.id

    primary_course = scored_courses[0][1] if scored_courses else None
    return all_concepts, concept_course_ids, concept_internal_course_ids, primary_course


def _course_maps_for_concepts(
    db: DBSession,
    concepts: list[str],
    default_course: Course,
) -> tuple[dict[str, int], dict[str, str]]:
    concept_course_ids: dict[str, int] = {}
    concept_internal_course_ids: dict[str, str] = {}
    courses = db.query(Course).all()
    for raw in concepts:
        concept = _assessment_concept(raw)
        if not concept:
            continue
        matched_course = None
        default_concepts = set(_assessment_concepts_for_course(MemoryManager(db), "", default_course))
        if concept in default_concepts:
            matched_course = default_course
        for course in courses:
            if matched_course:
                break
            if course.id == default_course.id:
                continue
            course_concepts = set(_assessment_concepts_for_course(MemoryManager(db), "", course))
            # Unknown/default courses expose every concept; do not let them steal
            # chapter mastery from the actual clicked Moodle course.
            if course_concepts == set(ALLOWED_CONCEPTS):
                continue
            if concept in course_concepts:
                matched_course = course
                break
        matched_course = matched_course or default_course
        concept_course_ids[concept] = matched_course.moodle_course_id
        concept_internal_course_ids[concept] = matched_course.id
    return concept_course_ids, concept_internal_course_ids


def _maps_for_single_course(concepts: list[str], course: Course) -> tuple[dict[str, int], dict[str, str]]:
    clean = [concept for concept in (_assessment_concept(c) for c in concepts) if concept]
    return (
        {concept: course.moodle_course_id for concept in clean},
        {concept: course.id for concept in clean},
    )


def _stored_remediation_concepts(course_memory: dict) -> list[str]:
    raw_concepts = course_memory.get("remediation_concepts") or []
    concepts = []
    for raw in raw_concepts:
        concept = _assessment_concept(raw)
        if concept and concept not in concepts:
            concepts.append(concept)
    return concepts


def _assessment_scope(
    db: DBSession,
    memory: MemoryManager,
    session: SessionModel,
) -> tuple[str, Course, list[str], dict[str, int], dict[str, str]]:
    course = session.course
    course_memory = memory.get_course_memory(session.student_id, session.course_id)
    launch_context = course_memory.get("launch_context") or {}
    level_type = _normalize_level_type(
        launch_context.get("level_type")
        or course_memory.get("level_type")
        or "chapter"
    )
    stored_concepts = _stored_remediation_concepts(course_memory)
    print(
        "[ACRLA] assessment_scope_entry "
        f"level_type={level_type} "
        f"course_id={course.moodle_course_id} "
        f"course_name={course.name!r} "
        f"concept={launch_context.get('concept') or course_memory.get('selected_concept') or 'none'} "
        f"stored_concepts={stored_concepts} "
        f"stored_scope_course_ids={course_memory.get('remediation_course_ids') or {}} "
        f"launch_context={launch_context}"
    )
    if level_type == "course":
        # Rebuild course-level scope from the clicked course every time. Stored
        # memory can contain stale overall concepts, but the course material
        # manifest is the authoritative boundary for a course remediation launch.
        concepts = _course_scope_concepts(memory, session.student_id, course)
        concept_course_ids, concept_internal_course_ids = _maps_for_single_course(concepts, course)
        print(
            "[ACRLA] assessment_scope_course_strict "
            f"level_type=course "
            f"course_id={course.moodle_course_id} "
            f"course_name={course.name!r} "
            f"stored_concepts={stored_concepts} "
            f"scope_concepts={concepts} "
            f"scope_course_ids={list(set(concept_course_ids.values()))}"
        )
        return "course", course, concepts, concept_course_ids, concept_internal_course_ids
    if stored_concepts:
        # For course level: filter stored_concepts through the course's synced material.
        # A prior overall session may have written cross-course concepts into this
        # course's memory; the material manifest is the authoritative scope boundary.
        if level_type == "course":
            stored_concepts = _course_scope_concepts(memory, session.student_id, course)
            concept_course_ids, concept_internal_course_ids = _maps_for_single_course(stored_concepts, course)
            print(
                "[ACRLA] assessment_scope "
                f"level_type=course "
                f"course_id={course.moodle_course_id} "
                f"scope_concepts={stored_concepts} "
                f"scope_course_ids={list(set(concept_course_ids.values()))}"
            )
            return level_type, course, stored_concepts, concept_course_ids, concept_internal_course_ids
            material = _material_concepts_for_course(course)
            if material:
                material_set = set(material)
                filtered = [c for c in stored_concepts if c in material_set]
                if filtered:
                    stored_concepts = filtered
                else:
                    # All stored concepts are alien — rebuild from material.
                    stored_concepts = _course_scope_concepts(memory, session.student_id, course)
        concept_course_ids = {
            concept: int(course_memory.get("remediation_course_ids", {}).get(concept))
            for concept in stored_concepts
            if str(course_memory.get("remediation_course_ids", {}).get(concept, "")).isdigit()
        }
        concept_internal_course_ids = {
            concept: str(course_memory.get("remediation_internal_course_ids", {}).get(concept))
            for concept in stored_concepts
            if course_memory.get("remediation_internal_course_ids", {}).get(concept)
        }
        if len(concept_course_ids) != len(stored_concepts) or len(concept_internal_course_ids) != len(stored_concepts):
            concept_course_ids, concept_internal_course_ids = _course_maps_for_concepts(db, stored_concepts, course)
        return level_type, course, stored_concepts, concept_course_ids, concept_internal_course_ids

    if level_type == "overall":
        concepts, concept_course_ids, concept_internal_course_ids, primary_course = (
            _all_enrolled_scope_concepts(db, memory, session.student_id)
        )
        if primary_course:
            course = primary_course
        if not concepts:
            concepts = _weakest_concepts_for_course(memory, session.student_id, course, limit=100)
            concept_course_ids = {concept: course.moodle_course_id for concept in concepts}
            concept_internal_course_ids = {concept: course.id for concept in concepts}
        print(
            "[ACRLA] assessment_scope "
            f"level_type=overall "
            f"scope_concepts={concepts} "
            f"scope_course_ids={list(concept_course_ids.values())} "
            f"retrieval_course_ids={list(set(concept_course_ids.values()))}"
        )
        return "overall", course, concepts, concept_course_ids, concept_internal_course_ids

    if level_type == "course":
        concepts = _course_scope_concepts(memory, session.student_id, course)
        print(
            "[ACRLA] assessment_scope "
            f"level_type=course "
            f"course_id={course.moodle_course_id} "
            f"scope_concepts={concepts}"
        )
        return (
            "course",
            course,
            concepts,
            {concept: course.moodle_course_id for concept in concepts},
            {concept: course.id for concept in concepts},
        )

    raw_concept = (
        course_memory.get("locked_concept")
        or launch_context.get("locked_concept")
        or launch_context.get("concept")
        or course_memory.get("launch_concept")
        or course_memory.get("selected_concept")
        or course_memory.get("last_concept")
    )
    concept = _assessment_concept(raw_concept)
    allowed = set(_assessment_concepts_for_course(memory, session.student_id, course))
    if not concept or concept not in allowed:
        concept = _weakest_concept_for_course(
            memory,
            session.student_id,
            course.id,
            course.moodle_course_id,
            course.name,
        )
    return (
        "chapter",
        course,
        [concept],
        {concept: course.moodle_course_id},
        {concept: course.id},
    )


def _assessment_question_bank(concept: str) -> list[dict]:
    bank = {
        "Recursion": [
            {
                "prompt": "A recursive function processes a list by handling the first item and then calling itself on the rest. What prevents it from running forever?",
                "options": ["A. A loop counter", "B. A base case", "C. A pointer address", "D. A sorting rule"],
                "correct": "B",
            },
            {
                "prompt": "In recursive factorial, what should factorial(0) return?",
                "options": ["A. 0", "B. 1", "C. n - 1", "D. It should call itself again"],
                "correct": "B",
            },
            {
                "prompt": "A DFS function calls itself on each unvisited neighbor. What is the recursive step?",
                "options": ["A. Stop immediately", "B. Call DFS on an unvisited neighbor", "C. Sort all vertices alphabetically", "D. Allocate heap memory only"],
                "correct": "B",
            },
        ],
        "Sorting Algorithms": [
            {
                "prompt": "You have exam scores [72, 91, 65] and want them from lowest to highest. What is the main goal of sorting here?",
                "options": ["A. Put values into a chosen order", "B. Store addresses", "C. Create tree nodes", "D. Prove a logical statement"],
                "correct": "A",
            },
            {
                "prompt": "A list is split around a pivot, then each side is sorted recursively. Which sorting algorithm is this describing?",
                "options": ["A. Bubble sort", "B. Quick sort", "C. Linear search", "D. Tree insertion"],
                "correct": "B",
            },
            {
                "prompt": "When comparing two numbers during sorting, what does the comparison decide?",
                "options": ["A. Which item should come before the other", "B. Whether a pointer is null", "C. Which graph edge exists", "D. Whether a set is empty"],
                "correct": "A",
            },
        ],
        "Pointers and Memory Management": [
            {
                "prompt": "In C-style code, int *p = &x; what does p store?",
                "options": ["A. The value of x copied twice", "B. The memory address of x", "C. The name of x", "D. A sorted version of x"],
                "correct": "B",
            },
            {
                "prompt": "If p points to x, what does *p let the program access?",
                "options": ["A. The value stored at the address in p", "B. The file name", "C. The next recursive call", "D. The graph edge count"],
                "correct": "A",
            },
            {
                "prompt": "A program calls malloc to reserve memory but never calls free. What problem can this cause?",
                "options": ["A. Memory leak", "B. Automatic sorting", "C. Faster recursion", "D. A true logic statement"],
                "correct": "A",
            },
        ],
        "Binary Trees and BSTs": [
            {
                "prompt": "A node in a binary tree can have at most how many children?",
                "options": ["A. One", "B. Two", "C. Four", "D. Unlimited"],
                "correct": "B",
            },
            {
                "prompt": "In an inorder traversal of a binary search tree, which part is visited between the left and right subtrees?",
                "options": ["A. The current node/root", "B. A null pointer", "C. A pivot array", "D. The course grade"],
                "correct": "A",
            },
            {
                "prompt": "In a Binary Search Tree, where should a value smaller than the current node usually go?",
                "options": ["A. Left subtree", "B. Right subtree", "C. Heap memory", "D. The base case"],
                "correct": "A",
            },
        ],
        "Logic": [
            {
                "prompt": "If P means 'x is even' and Q means 'x is divisible by 2', which statement says P guarantees Q?",
                "options": ["A. P implies Q", "B. P and not Q", "C. Q is false", "D. P is a set"],
                "correct": "A",
            },
            {
                "prompt": "If a statement is true only when both P and Q are true, which logical operator is being used?",
                "options": ["A. AND", "B. OR", "C. NOT", "D. UNION"],
                "correct": "A",
            },
        ],
        "Sets": [
            {
                "prompt": "Let A = {1, 2} and B = {2, 3}. What is A union B?",
                "options": ["A. {2}", "B. {1, 2, 3}", "C. {1, 3}", "D. {}"],
                "correct": "B",
            },
            {
                "prompt": "Let S = {A, C}. Which statement is true about A?",
                "options": ["A. A is an element of S", "B. A is the empty set", "C. A is a pointer", "D. A is a sorting pivot"],
                "correct": "A",
            },
        ],
        "Graphs": [
            {
                "prompt": "A graph has vertices A and B with an edge between them. What does the edge represent?",
                "options": ["A. A relationship or connection", "B. A recursive base case", "C. A memory address", "D. A sorted order"],
                "correct": "A",
            },
            {
                "prompt": "In a graph, what are vertices usually used to represent?",
                "options": ["A. Objects or points", "B. Only memory addresses", "C. Only sorted numbers", "D. Only false statements"],
                "correct": "A",
            },
        ],
    }
    bank.update({
        "Pointers and Memory Management": [
            {
                "question_id": "pointers_basics",
                "variant_id": "pointers_basics_v1",
                "sub_concept": "pointer basics",
                "prompt": "In C-style code, int *p = &x; what does p store?",
                "options": ["A. The value of x copied twice", "B. The memory address of x", "C. The name of x", "D. A sorted version of x"],
                "correct": "B",
            },
            {
                "question_id": "pointers_basics",
                "variant_id": "pointers_basics_v2",
                "sub_concept": "pointer basics",
                "prompt": "In C-style code, what does the expression &x return?",
                "options": ["A. The address of x", "B. A sorted copy of x", "C. The next tree node", "D. The size of the course"],
                "correct": "A",
            },
            {
                "question_id": "pointers_basics",
                "variant_id": "pointers_basics_v3",
                "sub_concept": "pointer basics",
                "prompt": "Which statement correctly describes a pointer?",
                "options": ["A. It stores a memory address", "B. It automatically sorts arrays", "C. It proves logical conditions", "D. It is always a graph edge"],
                "correct": "A",
            },
            {
                "question_id": "pointers_deref",
                "variant_id": "pointers_deref_v1",
                "sub_concept": "dereferencing",
                "prompt": "If p points to x, what does *p let the program access?",
                "options": ["A. The value stored at the address in p", "B. The file name", "C. The next recursive call", "D. The graph edge count"],
                "correct": "A",
            },
            {
                "question_id": "pointers_deref",
                "variant_id": "pointers_deref_v2",
                "sub_concept": "dereferencing",
                "prompt": "If p stores the address of x, what happens when the program executes *p = 5?",
                "options": ["A. x becomes 5", "B. p becomes null", "C. the list is sorted", "D. a new graph vertex is created"],
                "correct": "A",
            },
            {
                "question_id": "pointers_memory",
                "variant_id": "pointers_memory_v1",
                "sub_concept": "memory allocation",
                "prompt": "A program calls malloc to reserve memory but never calls free. What problem can this cause?",
                "options": ["A. Memory leak", "B. Automatic sorting", "C. Faster recursion", "D. A true logic statement"],
                "correct": "A",
            },
            {
                "question_id": "pointers_memory",
                "variant_id": "pointers_memory_v2",
                "sub_concept": "memory leaks",
                "prompt": "What is a memory leak?",
                "options": ["A. Allocated memory that is no longer reachable or freed", "B. A pointer that sorts values", "C. A binary tree traversal", "D. A set union operation"],
                "correct": "A",
            },
            {
                "question_id": "pointers_null",
                "variant_id": "pointers_null_v1",
                "sub_concept": "null pointers",
                "prompt": "Why is dereferencing a null pointer dangerous?",
                "options": ["A. It accesses no valid object and can crash the program", "B. It always creates a sorted array", "C. It proves P implies Q", "D. It inserts a BST node"],
                "correct": "A",
            },
        ],
        "Binary Trees and BSTs": [
            {
                "question_id": "trees_children",
                "variant_id": "trees_children_v1",
                "sub_concept": "root and leaf nodes",
                "prompt": "A node in a binary tree can have at most how many children?",
                "options": ["A. One", "B. Two", "C. Four", "D. Unlimited"],
                "correct": "B",
            },
            {
                "question_id": "trees_traversal",
                "variant_id": "trees_traversal_v1",
                "sub_concept": "tree traversal",
                "prompt": "In an inorder traversal of a binary search tree, which part is visited between the left and right subtrees?",
                "options": ["A. The current node/root", "B. A null pointer", "C. A pivot array", "D. The course grade"],
                "correct": "A",
            },
            {
                "question_id": "trees_traversal",
                "variant_id": "trees_traversal_v2",
                "sub_concept": "tree traversal",
                "prompt": "Which traversal visits the root before its left and right subtrees?",
                "options": ["A. Preorder", "B. Inorder", "C. Postorder", "D. Pointer arithmetic"],
                "correct": "A",
            },
            {
                "question_id": "trees_bst_property",
                "variant_id": "trees_bst_property_v1",
                "sub_concept": "BST property",
                "prompt": "In a Binary Search Tree, where should a value smaller than the current node usually go?",
                "options": ["A. Left subtree", "B. Right subtree", "C. Heap memory", "D. The base case"],
                "correct": "A",
            },
            {
                "question_id": "trees_bst_property",
                "variant_id": "trees_bst_property_v2",
                "sub_concept": "BST property",
                "prompt": "Why can a BST search discard one subtree after each comparison?",
                "options": ["A. The BST ordering rule tells which side could contain the value", "B. Pointers automatically delete nodes", "C. Sorting is disabled", "D. Every set is empty"],
                "correct": "A",
            },
        ],
        "Sorting Algorithms": [
            {
                "question_id": "sorting_goal",
                "variant_id": "sorting_goal_v1",
                "sub_concept": "comparison",
                "prompt": "You have exam scores [72, 91, 65] and want them from lowest to highest. What is the main goal of sorting here?",
                "options": ["A. Put values into a chosen order", "B. Store addresses", "C. Create tree nodes", "D. Prove a logical statement"],
                "correct": "A",
            },
            {
                "question_id": "sorting_pivot",
                "variant_id": "sorting_pivot_v1",
                "sub_concept": "pivot selection",
                "prompt": "A list is split around a pivot, then each side is sorted recursively. Which sorting algorithm is this describing?",
                "options": ["A. Bubble sort", "B. Quick sort", "C. Linear search", "D. Tree insertion"],
                "correct": "B",
            },
            {
                "question_id": "sorting_merge",
                "variant_id": "sorting_merge_v1",
                "sub_concept": "partitioning",
                "prompt": "Merge sort repeatedly splits a list, sorts the parts, and then does what?",
                "options": ["A. Merges the sorted parts back together", "B. Frees every pointer", "C. Removes all vertices", "D. Turns comparisons into sets"],
                "correct": "A",
            },
            {
                "question_id": "sorting_complexity",
                "variant_id": "sorting_complexity_v1",
                "sub_concept": "time complexity",
                "prompt": "When comparing two numbers during sorting, what does the comparison decide?",
                "options": ["A. Which item should come before the other", "B. Whether a pointer is null", "C. Which graph edge exists", "D. Whether a set is empty"],
                "correct": "A",
            },
        ],
    })
    return bank.get(concept, [
        {
            "question_id": f"{_concept_slug(concept)}_fallback_core",
            "variant_id": f"{_concept_slug(concept)}_fallback_core_v1",
            "sub_concept": "core idea",
            "prompt": f"Which option best matches the key idea of {concept}?",
            "options": ["A. A course concept to reason about", "B. A browser setting", "C. A food recipe", "D. A Moodle password"],
            "correct": "A",
        },
        {
            "question_id": f"{_concept_slug(concept)}_fallback_application",
            "variant_id": f"{_concept_slug(concept)}_fallback_application_v1",
            "sub_concept": "application",
            "prompt": f"In a course example about {concept}, what should you do first?",
            "options": ["A. Identify the relevant rule or definition", "B. Switch to an unrelated course", "C. Ignore the question context", "D. Guess without reading"],
            "correct": "A",
        },
        {
            "question_id": f"{_concept_slug(concept)}_fallback_check",
            "variant_id": f"{_concept_slug(concept)}_fallback_check_v1",
            "sub_concept": "understanding check",
            "prompt": f"What is a good way to prove you understand {concept}?",
            "options": ["A. Apply it to a specific example", "B. Repeat an unrelated phrase", "C. Avoid examples", "D. Use another student's password"],
            "correct": "A",
        },
    ])


def _normalize_question_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _assessment_concept(raw: str | None) -> str | None:
    concept = canonicalize_concept(raw)
    if concept:
        return concept
    text = str(raw or "").replace("_", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    return " ".join(part[:1].upper() + part[1:] for part in text.split())


def _concept_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "concept"


def _dynamic_material_context(course_id: int, concept: str) -> tuple[str, str | None, str | None, list[dict]]:
    """Returns (full_local_context, source_file, display_title, chunks).

    `full_local_context` is the complete, unfiltered joined text -- kept
    exactly as before for LOCAL-only uses (the RQ3 deterministic validity
    gate, the deterministic fallback template's anchor snippet). Neither
    of those sends this text externally, so neither needs it filtered.

    `chunks` is the same underlying material as a list of {"text",
    "sensitivity", "source"} dicts (built from the same already-fetched
    metadata, no extra query) -- RQ2 institutional-privacy gateway input.
    The caller (`_stored_or_generated_variants`) runs this through the
    existing `services.privacy_context.filter_course_chunks_for_external`
    before anything reaches Gemini. This function itself makes no privacy
    decision and sends nothing anywhere -- it only preserves the metadata
    that already existed but was previously discarded.
    """
    try:
        vectorstore = get_vectorstore(course_id)
        raw = vectorstore._collection.get(
            where={"concept": concept},
            limit=4,
            include=["documents", "metadatas"],
        )
        docs = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        if not docs:
            found = vectorstore.similarity_search(concept, k=4)
            docs = [doc.page_content for doc in found]
            metadatas = [doc.metadata or {} for doc in found]
        chunks = [
            {
                "text": str(doc)[:700],
                "sensitivity": (metadatas[index] or {}).get("sensitivity") if index < len(metadatas) else None,
                "source": (metadatas[index] or {}).get("source_file") if index < len(metadatas) else None,
            }
            for index, doc in enumerate(docs) if doc
        ]
        context = "\n\n".join(chunk["text"] for chunk in chunks)
        first_meta = metadatas[0] if metadatas else {}
        return (
            context,
            first_meta.get("source_file") if first_meta else None,
            first_meta.get("display_title") if first_meta else None,
            chunks,
        )
    except Exception as exc:
        print(f"[ACRLA] assessment_variant_context_failed course_id={course_id} concept={concept}: {exc}")
        return "", None, None, []


def _fallback_dynamic_variants(
    course_id: int,
    concept: str,
    context: str,
    source_file: str | None,
    display_title: str | None,
) -> list[dict]:
    snippets = [line.strip() for line in re.split(r"[\n.]+", context) if len(line.strip()) > 35]
    anchor = snippets[0][:140] if snippets else f"{concept} in the synced course material"
    slug = _concept_slug(concept)
    return [
        {
            "question_id": f"{slug}_core",
            "variant_id": f"dyn_{course_id}_{slug}_core_{uuid4().hex[:8]}",
            "sub_concept": "core idea",
            "prompt": f"According to the synced material, which statement best matches this idea: {anchor}?",
            "options": [
                f"A. It describes an important idea in {concept}",
                "B. It is unrelated to the course material",
                "C. It only describes a Moodle setting",
                "D. It is a password rule",
            ],
            "correct": "A",
            "source_file": source_file,
            "display_title": display_title,
        },
        {
            "question_id": f"{slug}_application",
            "variant_id": f"dyn_{course_id}_{slug}_application_{uuid4().hex[:8]}",
            "sub_concept": "application",
            "prompt": f"A student is working with {concept}. What should they do first when applying the course material?",
            "options": [
                "A. Identify the relevant definition or rule from the material",
                "B. Ignore the course context",
                "C. Switch to an unrelated chapter",
                "D. Answer without checking conditions",
            ],
            "correct": "A",
            "source_file": source_file,
            "display_title": display_title,
        },
        {
            "question_id": f"{slug}_misconception",
            "variant_id": f"dyn_{course_id}_{slug}_misconception_{uuid4().hex[:8]}",
            "sub_concept": "misconception check",
            "prompt": f"Which answer is the safest way to check understanding of {concept}?",
            "options": [
                "A. Explain the concept using a specific example from the course",
                "B. Memorize a random unrelated sentence",
                "C. Use another course's topic instead",
                "D. Skip the source material",
            ],
            "correct": "A",
            "source_file": source_file,
            "display_title": display_title,
        },
    ]


def _extract_llm_text(response) -> str:
    """`response.content` from `get_llm().invoke(...)` is usually a plain
    string, but was found (RQ3 QPC-dynamic diagnostic, 2026-09-10) to come
    back as a list of text chunks for at least one Gemini model/response
    shape (e.g. `['```json\\n', '[...']`) -- `re.search`/`json.loads`
    require a string, so a bare `getattr(response, "content", ...)` would
    raise a TypeError on that shape (caught by the same broad
    `except Exception` in `_llm_dynamic_variants` either way, but as a
    confusing, undiagnosed error rather than a clean parse attempt).
    Joins any non-string chunks into one string; leaves an already-string
    `content` untouched."""
    content = getattr(response, "content", None)
    if content is None:
        return str(response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else str(part.get("text", "") if isinstance(part, dict) else part)
            for part in content
        )
    return str(content)


def _llm_dynamic_variants(course_id: int, concept: str, context: str, source_file: str | None, display_title: str | None) -> list[dict]:
    if not context.strip():
        return []
    prompt = (
        "Generate 5 varied multiple-choice assessment questions from this course material. "
        "Return only JSON array. Each item must have: sub_concept, prompt, options, correct. "
        "Options must be an array of exactly 4 strings beginning with A., B., C., D.; correct must be A/B/C/D. "
        f"Concept: {concept}\nCourse material:\n{context[:2500]}"
    )
    try:
        # max_tokens raised from the original 900: a live diagnostic (RQ3
        # Step 4, 2026-09-10) confirmed `finish_reason: MAX_TOKENS` --
        # Gemini was being cut off before completing even the first of 5
        # requested MCQ objects, every time, causing every dynamic QPC
        # generation attempt to fail parsing and silently fall back to the
        # deterministic template. 900 was simply too small a budget for 5
        # full MCQ objects (sub_concept + prompt + 4 options + correct)
        # plus markdown-fence wrapping, on a model that also spends some of
        # its token budget on internal reasoning before emitting visible
        # text. No other part of this call (prompt, temperature, model,
        # validation) changed.
        response = get_llm(temperature=0.4, max_tokens=3500).invoke(prompt)
        text = _extract_llm_text(response)
        match = re.search(r"\[[\s\S]*\]", text)
        data = json.loads(match.group(0) if match else text)
    except Exception as exc:
        print(f"[ACRLA] assessment_variant_llm_failed course_id={course_id} concept={concept}: {exc}")
        return []
    variants = []
    slug = _concept_slug(concept)
    for index, item in enumerate(data if isinstance(data, list) else []):
        options = item.get("options") if isinstance(item, dict) else None
        correct = str(item.get("correct", "A")).strip().upper()[:1] if isinstance(item, dict) else "A"
        prompt_text = str(item.get("prompt", "")).strip() if isinstance(item, dict) else ""
        if not prompt_text or not isinstance(options, list) or len(options) != 4 or correct not in {"A", "B", "C", "D"}:
            continue
        variants.append({
            "question_id": f"{slug}_generated_{index + 1}",
            "variant_id": f"dyn_{course_id}_{slug}_{uuid4().hex[:10]}",
            "sub_concept": str(item.get("sub_concept") or f"{concept} concept").strip(),
            "prompt": prompt_text,
            "options": [str(option) for option in options],
            "correct": correct,
            "source_file": source_file,
            "display_title": display_title,
        })
    return variants


def _stored_or_generated_variants(
    db: DBSession,
    course_internal_id: str,
    moodle_course_id: int,
    concept: str,
) -> list[dict]:
    concept = _assessment_concept(concept) or str(concept)
    records = db.query(AssessmentQuestionVariant).filter_by(
        course_id=course_internal_id,
        concept=concept,
    ).all()
    if records:
        return [_variant_record_to_item(record) for record in records]

    context, source_file, display_title, chunks = _dynamic_material_context(moodle_course_id, concept)

    # RQ2 institutional-privacy gateway: the SAME deterministic function
    # pipelines.rag_pipeline.retrieve_context already applies for every
    # other external-LLM-bound retrieval (`for_external=True`) -- reused
    # here, not reimplemented, so this path stops bypassing it. RESTRICTED
    # chunks are dropped entirely; PUBLIC/INTERNAL chunks are kept up to
    # the same MAX_EXTERNAL_RAG_CHARS budget. `context` (the full, LOCAL,
    # unfiltered text) is passed to `validate_generated_questions` below
    # completely unchanged -- that call never leaves this process, so it
    # correctly keeps seeing everything retrieved, not just what's safe to
    # send externally. Only `external_context` (the gateway's output) goes
    # to `_llm_dynamic_variants`, i.e. to Gemini.
    from services.privacy_context import filter_course_chunks_for_external, log_external_content_decision
    allowed_chunks, gateway_audit = filter_course_chunks_for_external(chunks)
    external_context = "\n\n---\n\n".join(chunk["text"] for chunk in allowed_chunks)
    if gateway_audit:
        log_external_content_decision(course_id=course_internal_id, audit=gateway_audit)
    raw_variants = _llm_dynamic_variants(moodle_course_id, concept, external_context, source_file, display_title)

    # RQ3 question-validity gate: a question that merely PARSED into valid
    # JSON (_llm_dynamic_variants' own check) is not yet trusted -- verify
    # it is actually about this concept, grounded in the retrieved course
    # material, unambiguous, and that the indicated correct answer is at
    # least as well-supported by that material as every distractor. A
    # candidate that fails is dropped here -- never shown to a student,
    # never cached, never able to influence mastery -- and generation falls
    # through to the SAME deterministic fallback template this code path
    # already used whenever the LLM call returned nothing at all, rather
    # than a new/separate failure path.
    from services.question_validity import validate_generated_questions
    variants, rejected = validate_generated_questions(raw_variants, concept=concept, course_context=context)
    if rejected:
        print(
            "[ACRLA] assessment_variant_validation_rejected "
            f"course_id={moodle_course_id} concept={concept} "
            f"rejected_count={len(rejected)} accepted_count={len(variants)} "
            f"reasons={[r['reason'] for r in rejected]}"
        )

    used_fallback_template = not variants  # true only if nothing survived validation (or nothing was generated at all)
    variants = variants or _fallback_dynamic_variants(moodle_course_id, concept, context, source_file, display_title)
    origin = "fallback_template" if used_fallback_template else "llm_generated_validated"
    for item in variants:
        db.add(AssessmentQuestionVariant(
            course_id=course_internal_id,
            moodle_course_id=moodle_course_id,
            concept=concept,
            sub_concept=item.get("sub_concept"),
            question_id=item.get("question_id") or _normalize_question_text(item.get("prompt"))[:80],
            variant_id=item.get("variant_id") or f"dyn_{moodle_course_id}_{_concept_slug(concept)}_{uuid4().hex[:10]}",
            difficulty="easy",
            prompt=item["prompt"],
            options=item["options"],
            correct=item.get("correct", "A"),
            source_file=item.get("source_file"),
            display_title=item.get("display_title"),
        ))
    db.commit()
    # RQ3 research observability (Question Validity Rate): concept,
    # difficulty, generation origin, and validation outcome -- never raw
    # question text, student data, or secrets (see
    # evaluation/rq3_accuracy/qpc_question_validity.md).
    print(
        "[ACRLA] assessment_variants_generated "
        f"course_id={moodle_course_id} concept={concept} difficulty=easy "
        f"origin={origin} variants_accepted={len(variants)} "
        f"variants_rejected={len(rejected)} source={display_title or source_file or 'none'}"
    )
    return variants


def _variant_record_to_item(record: AssessmentQuestionVariant) -> dict:
    return {
        "question_id": record.question_id,
        "variant_id": record.variant_id,
        "sub_concept": record.sub_concept,
        "prompt": record.prompt,
        "options": record.options or [],
        "correct": record.correct or "A",
        "source_file": record.source_file,
        "display_title": record.display_title,
    }


def _question_reuse_keys(item: dict) -> set[str]:
    keys = set()
    for key in ("variant_id",):
        value = item.get(key)
        if value:
            keys.add(str(value))
    normalized = _normalize_question_text(item.get("prompt"))
    if normalized:
        keys.add(normalized)
    return keys


def _recent_assessment_question_keys(
    db: DBSession,
    student_id: str,
    level_type: str,
    course_id: str,
    concepts: list[str],
    limit: int = 8,
) -> set[str]:
    concept_set = {concept for concept in (_assessment_concept(c) for c in concepts) if concept}
    records = (
        db.query(AssessmentRecord)
        .filter_by(student_id=student_id, level_type=level_type)
        .order_by(AssessmentRecord.timestamp.desc())
        .limit(limit)
        .all()
    )
    keys: set[str] = set()
    for record in records:
        if level_type != "overall" and record.course_id != course_id:
            continue
        record_concepts = {
            concept for concept in (_assessment_concept(c) for c in str(record.concept or "").split(","))
            if concept
        }
        if concept_set and record_concepts and not concept_set.intersection(record_concepts):
            continue
        details = record.details or {}
        for item in details.get("questions") or []:
            keys.update(_question_reuse_keys(item))
    return keys


def _integrated_assessment_question(concepts: list[str]) -> dict:
    canonical = [concept for concept in (_assessment_concept(c) for c in concepts) if concept]
    pair = canonical[:2]
    concept_key = " + ".join(pair) if pair else "Course concepts"

    if "Pointers and Memory Management" in pair and "Binary Trees and BSTs" in pair:
        return {
            "prompt": "In a pointer-based binary tree, why are pointers needed?",
            "options": ["A. To connect nodes to left and right children", "B. To sort values automatically", "C. To remove all recursion", "D. To ignore memory"],
            "correct": "A",
            "concept": concept_key,
        }
    if "Logic" in canonical and "Sets" in canonical and "Graphs" in canonical:
        return {
            "prompt": "A graph has vertices V = {A, B, C}. Let S = {A, C}. Which statement says every vertex in S has at least one edge?",
            "options": ["A. For every v in S, v is incident to some edge", "B. S is empty", "C. No vertex belongs to V", "D. Every edge is a pointer"],
            "correct": "A",
            "concept": "Logic + Sets + Graphs",
        }
    if "Sorting Algorithms" in pair and "Pointers and Memory Management" in pair:
        return {
            "prompt": "When sorting a linked list, why are pointers useful?",
            "options": ["A. They let nodes be rearranged without copying all data", "B. They remove the need for comparisons", "C. They automatically choose the smallest value", "D. They prevent all memory allocation"],
            "correct": "A",
            "concept": concept_key,
        }
    if "Graphs" in pair and "Recursion" in pair:
        return {
            "prompt": "In depth-first search on a graph, what role does recursion play?",
            "options": ["A. It visits neighboring vertices until a base case is reached", "B. It sorts vertices alphabetically", "C. It removes every edge", "D. It replaces the graph with a set"],
            "correct": "A",
            "concept": concept_key,
        }
    if "Pointers and Memory Management" in pair and "Graphs" in pair:
        return {
            "prompt": "A graph is stored as linked adjacency lists. Why might pointers be useful in this representation?",
            "options": ["A. They connect each vertex to dynamically allocated neighbor nodes", "B. They prove every logical statement true", "C. They sort all vertices automatically", "D. They remove the need for edges"],
            "correct": "A",
            "concept": concept_key,
        }
    if "Recursion" in pair and "Logic" in pair:
        return {
            "prompt": "A recursive proof checks a base case, then proves the next case follows. Which logical idea matches the recursive step?",
            "options": ["A. If the smaller case is true, then the next case follows", "B. The set must be empty", "C. A pointer stores the proof", "D. A graph edge sorts values"],
            "correct": "A",
            "concept": concept_key,
        }
    if "Binary Trees and BSTs" in pair and "Sets" in pair:
        return {
            "prompt": "A binary search tree stores values from the set S = {4, 2, 7}. What must be true after inserting these values?",
            "options": ["A. Each stored node value belongs to S and smaller values go left", "B. The set becomes empty", "C. Every node must have three children", "D. Logic rules remove the root"],
            "correct": "A",
            "concept": concept_key,
        }
    if "Sorting Algorithms" in pair and "Logic" in pair:
        return {
            "prompt": "A sorting algorithm swaps two items only if condition P is true: left value > right value. What does P control?",
            "options": ["A. Whether the algorithm should reorder that pair", "B. Whether memory is freed", "C. Whether a graph has vertices", "D. Whether a set contains A"],
            "correct": "A",
            "concept": concept_key,
        }
    return {
        "prompt": f"A task uses both {pair[0]} and {pair[1] if len(pair) > 1 else 'another weak concept'}. What is the best first step?",
        "options": ["A. Identify how the concepts interact in the scenario", "B. Ignore one concept completely", "C. Change the course topic", "D. Avoid checking the conditions"],
        "correct": "A",
        "concept": concept_key,
    }


def _question_metadata(
    concepts_used: list[str],
    concept_course_ids: dict[str, int],
    question_type: str,
) -> dict:
    concepts = [concept for concept in (_assessment_concept(c) for c in concepts_used) if concept]
    course_ids = []
    for concept in concepts:
        course_id = concept_course_ids.get(concept)
        if course_id is not None and course_id not in course_ids:
            course_ids.append(course_id)
    return {
        "concepts_used": concepts,
        "course_ids_used": course_ids,
        "question_type": question_type,
    }


def _add_unique_question(
    questions: list[dict],
    item: dict,
    concept: str | None = None,
    concept_course_ids: dict[str, int] | None = None,
    question_type: str = "single_concept",
) -> bool:
    normalized = _normalize_question_text(item.get("prompt"))
    if not normalized or any(_normalize_question_text(q.get("prompt")) == normalized for q in questions):
        return False
    clone = item.copy()
    concepts_used = [concept] if concept else [c.strip() for c in str(clone.get("concept", "")).split("+")]
    if concept:
        clone["concept"] = concept
    clone.update(_question_metadata(concepts_used, concept_course_ids or {}, question_type))
    clone["question_id"] = clone.get("question_id") or _normalize_question_text(clone.get("prompt"))[:80]
    clone["variant_id"] = clone.get("variant_id") or clone["question_id"]
    clone["sub_concept"] = clone.get("sub_concept") or (concepts_used[0] if concepts_used else None)
    clone["id"] = f"q{len(questions) + 1}"
    questions.append(clone)
    return True


def _candidate_questions(
    concept: str,
    recent_keys: set[str],
    db: DBSession | None = None,
    course_internal_id: str | None = None,
    moodle_course_id: int | None = None,
) -> list[dict]:
    concept = _assessment_concept(concept) or str(concept)
    candidates: list[dict] = []
    if db and course_internal_id and moodle_course_id is not None:
        candidates.extend(_stored_or_generated_variants(db, course_internal_id, moodle_course_id, concept))
    if not candidates or concept in set(ALLOWED_CONCEPTS):
        candidates.extend(item.copy() for item in _assessment_question_bank(concept))
    deduped: list[dict] = []
    seen = set()
    for item in candidates:
        key = item.get("variant_id") or _normalize_question_text(item.get("prompt"))
        if key and key not in seen:
            seen.add(key)
            deduped.append(item.copy())
    candidates = deduped
    random.shuffle(candidates)
    fresh = [item for item in candidates if not (_question_reuse_keys(item) & recent_keys)]
    return fresh or candidates


def _build_assessment_questions(
    concepts: list[str],
    level_type: str = "chapter",
    concept_course_ids: dict[str, int] | None = None,
    concept_internal_course_ids: dict[str, str] | None = None,
    recent_question_keys: set[str] | None = None,
    db: DBSession | None = None,
) -> list[dict]:
    canonical = [concept for concept in (_assessment_concept(c) for c in concepts) if concept]
    concept_course_ids = concept_course_ids or {}
    concept_internal_course_ids = concept_internal_course_ids or {}
    recent_question_keys = recent_question_keys or set()
    questions: list[dict] = []

    if not canonical:
        return questions

    if level_type == "chapter":
        concept = canonical[0]
        used_sub_concepts = set()
        for item in _candidate_questions(
            concept,
            recent_question_keys,
            db=db,
            course_internal_id=concept_internal_course_ids.get(concept),
            moodle_course_id=concept_course_ids.get(concept),
        ):
            sub_concept = item.get("sub_concept")
            if sub_concept and sub_concept in used_sub_concepts and len(questions) < 3:
                continue
            _add_unique_question(questions, item, concept, concept_course_ids, "single_concept")
            if sub_concept:
                used_sub_concepts.add(sub_concept)
            if len(questions) == 3:
                return questions
        for item in _candidate_questions(
            concept,
            set(),
            db=db,
            course_internal_id=concept_internal_course_ids.get(concept),
            moodle_course_id=concept_course_ids.get(concept),
        ):
            _add_unique_question(questions, item, concept, concept_course_ids, "single_concept")
            if len(questions) == 3:
                return questions
        return questions

    for concept in canonical[:2]:
        for item in _candidate_questions(
            concept,
            recent_question_keys,
            db=db,
            course_internal_id=concept_internal_course_ids.get(concept),
            moodle_course_id=concept_course_ids.get(concept),
        ):
            if _add_unique_question(questions, item, concept, concept_course_ids, "single_concept"):
                break

    if len(canonical) >= 2:
        integrated_concepts = canonical
        if len(canonical) > 2:
            integrated_concepts = [canonical[-1], canonical[0]]
        integrated_item = _integrated_assessment_question(integrated_concepts)
        if _question_reuse_keys(integrated_item) & recent_question_keys:
            integrated_item = _integrated_assessment_question(list(reversed(integrated_concepts)))
        if _question_reuse_keys(integrated_item) & recent_question_keys:
            integrated_item = {
                "question_id": "integrated_scope",
                "variant_id": f"integrated_scope_{random.randint(1000, 9999)}",
                "sub_concept": "integrated reasoning",
                "prompt": (
                    f"A remediation task combines {integrated_concepts[0]} with "
                    f"{integrated_concepts[1]}. Which answer best uses both in the same scenario?"
                ),
                "options": ["A. Apply both concepts to explain the scenario", "B. Ignore one concept", "C. Switch to an unrelated topic", "D. Treat the answer as a password"],
                "correct": "A",
                "concept": " + ".join(integrated_concepts[:2]),
            }
        _add_unique_question(
            questions,
            integrated_item,
            concept_course_ids=concept_course_ids,
            question_type="integrated",
        )

    concept_index = 0
    while len(questions) < 3:
        concept = canonical[concept_index % len(canonical)]
        added = False
        for item in _candidate_questions(
            concept,
            recent_question_keys,
            db=db,
            course_internal_id=concept_internal_course_ids.get(concept),
            moodle_course_id=concept_course_ids.get(concept),
        ):
            if _add_unique_question(questions, item, concept, concept_course_ids, "single_concept"):
                added = True
                break
        if not added:
            fallback = {
                "prompt": f"In a concrete remediation scenario, what is one correct use of {concept}?",
                "options": ["A. Apply the concept to solve the stated problem", "B. Ignore the scenario", "C. Use an unrelated topic", "D. Skip all reasoning"],
                "correct": "A",
            }
            _add_unique_question(questions, fallback, concept, concept_course_ids, "single_concept")
        concept_index += 1
        if concept_index > 12:
            break
    return questions[:3]


def _public_assessment_questions(questions: list[dict]) -> list[dict]:
    return [
        {
            "id": item["id"],
            "question_id": item.get("question_id"),
            "variant_id": item.get("variant_id"),
            "sub_concept": item.get("sub_concept"),
            "prompt": item["prompt"],
            "options": item["options"],
            "concepts_used": item.get("concepts_used", []),
            "course_ids_used": item.get("course_ids_used", []),
            "question_type": item.get("question_type", "single_concept"),
        }
        for item in questions
    ]


def _validate_assessment_scope(
    level_type: str,
    questions: list[dict],
    selected_concepts: list[str],
    concept_course_ids: dict[str, int],
) -> None:
    canonical_selected = [concept for concept in (_assessment_concept(c) for c in selected_concepts) if concept]
    selected_set = set(canonical_selected)
    selected_course_ids = {concept_course_ids.get(concept) for concept in canonical_selected}
    selected_course_ids.discard(None)

    prompts = [_normalize_question_text(item.get("prompt")) for item in questions]
    if len(prompts) != len(set(prompts)):
        raise HTTPException(status_code=500, detail="Assessment generator produced duplicate questions.")

    for item in questions:
        concepts_used = set(item.get("concepts_used") or [])
        course_ids_used = set(item.get("course_ids_used") or [])
        if not concepts_used:
            raise HTTPException(status_code=500, detail="Assessment question is missing concept metadata.")
        if not course_ids_used:
            raise HTTPException(status_code=500, detail="Assessment question is missing course metadata.")
        if not concepts_used.issubset(selected_set):
            raise HTTPException(status_code=500, detail="Assessment question escaped the selected concept scope.")

    assessment_concepts = set().union(*(set(item.get("concepts_used") or []) for item in questions))
    # Assessment is allowed to cover a SUBSET of the scope (weak focus within broad scope).
    # Reject only if it escapes the scope entirely or leaves concepts from the scope uncovered
    # in a way that violates level-specific rules below.
    if not assessment_concepts.issubset(selected_set):
        print(
            "[ACRLA] assessment_scope_violation "
            f"scope={sorted(selected_set)} "
            f"assessment_concepts={sorted(assessment_concepts)} "
            f"escaped={sorted(assessment_concepts - selected_set)}"
        )
        raise HTTPException(status_code=500, detail="Assessment question escaped the selected concept scope.")
    if assessment_concepts != selected_set:
        print(
            "[ACRLA] assessment_scope_partial_coverage "
            f"scope={sorted(selected_set)} "
            f"assessment_concepts={sorted(assessment_concepts)}"
        )

    if level_type == "chapter":
        locked = canonical_selected[0] if canonical_selected else None
        for item in questions:
            if set(item.get("concepts_used") or []) != {locked}:
                raise HTTPException(status_code=500, detail="Chapter assessment included another chapter concept.")
        return

    if level_type == "course":
        if len(selected_set) >= 2:
            covered = set().union(*(set(item.get("concepts_used") or []) for item in questions))
            if len(covered) < 2:
                raise HTTPException(status_code=500, detail="Course assessment did not cover multiple course concepts.")
        for item in questions:
            if not set(item.get("course_ids_used") or []).issubset(selected_course_ids):
                raise HTTPException(status_code=500, detail="Course assessment included another course.")
        if len(selected_set) >= 2 and not any(item.get("question_type") == "integrated" for item in questions):
            raise HTTPException(status_code=500, detail="Course assessment is missing an integrated question.")
        return

    if level_type == "overall":
        all_course_ids = set().union(*(set(item.get("course_ids_used") or []) for item in questions))
        if len(selected_course_ids) >= 2 and len(all_course_ids) < 2:
            raise HTTPException(status_code=500, detail="Overall assessment did not cover multiple courses.")
        if len(selected_set) >= 2 and not any(item.get("question_type") == "integrated" for item in questions):
            raise HTTPException(status_code=500, detail="Overall assessment is missing an integrated question.")
        if len(selected_course_ids) >= 2 and not any(
            item.get("question_type") == "integrated"
            and len(set(item.get("course_ids_used") or [])) >= 2
            for item in questions
        ):
            raise HTTPException(status_code=500, detail="Overall integrated question did not span multiple courses.")


@router.post("/moodle/sync", response_model=MoodleSyncResponse)
def sync_moodle(payload: MoodleSyncRequest, db: DBSession = Depends(get_db)):
    memory = MemoryManager(db)

    student = memory.get_or_create_student(
        moodle_user_id=payload.student_id,
        username=payload.student_name,
    )
    memory.set_profile_name(student.id, payload.student_name)

    course = db.query(Course).filter_by(moodle_course_id=payload.course_id).first()
    if not course:
        course = Course(moodle_course_id=payload.course_id, name=payload.course_name)
        db.add(course)
        db.commit()
        db.refresh(course)
    elif course.name != payload.course_name:
        course.name = payload.course_name
        db.commit()
        db.refresh(course)

    if payload.learning_mode or payload.difficulty:
        memory.save_preferences(
            student_id=student.id,
            mode=payload.learning_mode,
            difficulty=payload.difficulty,
        )
        memory.set_profile_preferences(
            student.id,
            difficulty=payload.difficulty,
            learning_mode=payload.learning_mode,
        )

    updated_concepts = []
    payload_concepts = [
        concept for concept in (_assessment_concept(raw) for raw in payload.mastery.keys())
        if concept
    ]
    course_concepts = payload_concepts or _assessment_concepts_for_course(memory, student.id, course)
    allowed_concepts = set(course_concepts)
    existing_records = {
        _assessment_concept(record.concept): record
        for record in memory.get_all_mastery(student.id, course.id)
        if _assessment_concept(record.concept)
    }

    # Load the canonical ACRLA mastery map — the single source of truth kept
    # current by _ensure_mastery_baseline and _set_current_acrla_mastery.
    course_memory_data = memory.get_course_memory(student.id, course.id)
    initial_acrla_map, current_acrla_map = _memory_mastery_maps(course_memory_data)

    for raw_concept, raw_score in payload.mastery.items():
        concept = _assessment_concept(raw_concept)
        if not concept or concept not in allowed_concepts:
            continue
        # Guard: never write a Moodle grade when ACRLA already has mastery for
        # this concept — current_acrla_map is checked first because MasteryRecord
        # mastery_level can be 0.5 even after real assessments (cross-course or
        # timing scenarios), while current_acrla_mastery always reflects the
        # last real assessment outcome.
        scope_key = _chapter_mastery_key(course, concept)
        demo_pct = _demo_initial_mastery_for_concept(course, concept)
        moodle_pct = demo_pct if demo_pct is not None else _mastery_to_percent(_score_to_mastery(raw_score))
        initial_acrla_map[scope_key] = moodle_pct if demo_pct is not None else max(_safe_float(initial_acrla_map.get(scope_key)), moodle_pct)
        if scope_key in current_acrla_map:
            current_acrla_map[scope_key] = max(_safe_float(current_acrla_map.get(scope_key)), initial_acrla_map[scope_key])
        has_acrla_mastery = float(current_acrla_map.get(scope_key) or 0) > 0
        existing = existing_records.get(concept)
        if not existing and not has_acrla_mastery:
            memory.set_mastery(
                student_id=student.id,
                course_id=course.id,
                concept=concept,
                mastery_level=_score_to_mastery(raw_score),
            )
        elif (
            existing
            and not has_acrla_mastery
            and float(existing.mastery_level or 0.0) == 0.5
            and int(existing.attempts or 0) == 0
            and int(existing.correct or 0) == 0
        ):
            memory.set_mastery(
                student_id=student.id,
                course_id=course.id,
                concept=concept,
                mastery_level=_score_to_mastery(raw_score),
            )
        updated_concepts.append(concept)

    memory.update_course_memory(student.id, course.id, {
        "moodle_initial_mastery": initial_acrla_map,
        "current_acrla_mastery": current_acrla_map,
    })

    # Resolve per-concept mastery using current_acrla_mastery as primary source
    # so that weak/strongest sorting matches what the modal and analytics show.
    mastery_by_concept: dict[str, float] = {}
    for concept in course_concepts:
        scope_key = _chapter_mastery_key(course, concept)
        acrla_pct = current_acrla_map.get(scope_key)
        initial_pct = _safe_float(initial_acrla_map.get(scope_key), 0.0)
        mastery_by_concept[concept] = max(float(acrla_pct) / 100 if acrla_pct is not None else memory.get_mastery(student.id, course.id, concept), initial_pct / 100)

    ordered_by_mastery = sorted(mastery_by_concept.items(), key=lambda item: item[1])
    weak_concepts = [concept for concept, score in ordered_by_mastery if score < 0.6]
    strongest_concepts = [
        concept
        for concept, _score in sorted(mastery_by_concept.items(), key=lambda item: item[1], reverse=True)
    ][:2]

    if updated_concepts:
        weakest = weak_concepts[0] if weak_concepts else updated_concepts[-1]
        memory.update_course_memory(student.id, course.id, {
            "last_concept": weakest,
            "last_activity": "synced mastery data from Moodle",
            "weak_concepts": weak_concepts,
            "strongest_concepts": strongest_concepts,
            "next_recommended_action": f"continue with {weakest}",
        })

    # Build response mastery keyed by original payload keys with current_acrla_mastery
    # as the canonical source. Fallback chain: ACRLA → MasteryRecord → Moodle grade → 50%.
    response_mastery: dict[str, float] = {}
    for raw_key in payload.mastery:
        canonical = _assessment_concept(raw_key)
        if not canonical:
            response_mastery[raw_key] = 50.0
            continue
        scope_key = _chapter_mastery_key(course, canonical)
        acrla_pct = current_acrla_map.get(scope_key)
        record_mastery_01 = mastery_by_concept.get(canonical, 0.0)
        if acrla_pct is not None:
            button_mastery = round(float(acrla_pct), 2)
        elif record_mastery_01 > 0.0:
            button_mastery = _mastery_to_percent(record_mastery_01)
        else:
            moodle_01 = _score_to_mastery(payload.mastery.get(raw_key))
            button_mastery = _mastery_to_percent(moodle_01) if moodle_01 > 0 else 50.0
        response_mastery[raw_key] = button_mastery
        rec = existing_records.get(canonical)
        print(
            f"[ACRLA] button_mastery "
            f"concept={canonical} "
            f"course_id={course.id} "
            f"button_mastery={button_mastery} "
            f"resolver_mastery={acrla_pct} "
            f"mastery_record_id={rec.id if rec else 'none'}"
        )

    payload_canonical_set = {_assessment_concept(k) for k in payload.mastery if _assessment_concept(k)}
    for concept, record_mastery_01 in mastery_by_concept.items():
        if concept not in payload_canonical_set and concept not in response_mastery:
            scope_key = _chapter_mastery_key(course, concept)
            acrla_pct = current_acrla_map.get(scope_key)
            if acrla_pct is not None:
                response_mastery[concept] = round(float(acrla_pct), 2)
            elif record_mastery_01 > 0.0:
                response_mastery[concept] = _mastery_to_percent(record_mastery_01)
            else:
                response_mastery[concept] = 50.0

    return MoodleSyncResponse(
        status="ok",
        student_id=payload.student_id,
        course_id=payload.course_id,
        student_name=payload.student_name,
        weak_concepts=weak_concepts,
        strongest_concepts=strongest_concepts,
        mastery=response_mastery,
    )


@router.get("/mastery/student/{student_id}")
def get_student_mastery(student_id: int, db: DBSession = Depends(get_db)):
    memory = MemoryManager(db)
    student = db.query(Student).filter_by(moodle_user_id=student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    courses = _available_courses_for_student(db, memory, student.id)
    course_rows = []
    all_concepts = []

    for course in courses:
        demo_baseline = _demo_baseline_map_for_course(memory, student.id, course)
        course_concepts = list(demo_baseline.keys()) or _assessment_concepts_for_course(memory, student.id, course)
        if not course_concepts:
            continue

        course_mem = memory.get_course_memory(student.id, course.id)
        concept_initial_map, concept_acrla_map = _memory_mastery_maps(course_mem)

        concept_rows = {}
        for concept in course_concepts:
            scope_key = _chapter_mastery_key(course, concept)
            initial_pct = _safe_float(concept_initial_map.get(scope_key), 0.0)
            acrla_pct = concept_acrla_map.get(scope_key)
            current = _concept_current_mastery_percent(memory, student.id, course, concept)
            concept_rows[concept] = {
                "current_acrla_mastery": current,
                "fallback_moodle_mastery": initial_pct or current,
                "source_of_truth": "current_acrla_mastery" if acrla_pct is not None else "mastery_records_or_initial",
            }
            all_concepts.append({
                "course_id": course.moodle_course_id,
                "course_name": course.name,
                "concept": concept,
                "current_acrla_mastery": current,
            })

        course_current = _course_current_mastery_percent(memory, student.id, course)
        course_rows.append({
            "course_id": course.moodle_course_id,
            "course_name": course.name,
            "course_current_acrla_mastery": course_current,
            "concept_current_acrla_mastery": concept_rows,
        })

    overall = _overall_current_mastery_percent(db, memory, student.id)
    print(
        "[ACRLA] mastery_endpoint "
        f"student_id={student_id} "
        f"overall_current_acrla_mastery={overall} "
        f"courses={[(row['course_id'], row['course_current_acrla_mastery']) for row in course_rows]}"
    )

    return {
        "status": "ok",
        "student_id": student_id,
        "overall_current_acrla_mastery": overall,
        "courses": course_rows,
        "concepts": all_concepts,
    }


# ── POST /session/start ───────────────────────────────────────────────────────

@router.post("/moodle/materials/sync")
async def sync_moodle_materials(
    course_id: int = Form(...),
    course_name: str = Form(...),
    file: UploadFile = File(...),
    student_id: int | None = Form(None),
    concept: str | None = Form(None),
    original_file_name: str | None = Form(None),
    moodle_resource_title: str | None = Form(None),
    sensitivity: str | None = Form(None),
    db: DBSession = Depends(get_db),
):
    """Receive a Moodle course PDF and ingest it into the course RAG collection.

    `sensitivity` (RQ2 institutional-privacy step, optional): the
    institution's PUBLIC/INTERNAL/RESTRICTED classification for this
    specific document -- this is the request-time equivalent of hand-
    editing a course's materials_manifest.json "sensitivity" key (see
    pipelines.rag_pipeline.rebuild_course_collection); both write/read the
    SAME manifest file, so a document classified here stays correctly
    classified across a later manifest-driven rebuild too. Omitted/blank
    normalizes to INTERNAL (services.privacy_context.normalize_sensitivity),
    never PUBLIC.
    """
    filename = Path(original_file_name or file.filename or "moodle_material.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    selected_concept = _assessment_concept(concept) if concept else None

    course = db.query(Course).filter_by(moodle_course_id=course_id).first()
    if not course:
        course = Course(moodle_course_id=course_id, name=course_name)
        db.add(course)
        db.commit()
        db.refresh(course)
    elif course.name != course_name:
        course.name = course_name
        db.commit()
        db.refresh(course)

    safe_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", filename).strip(" .") or "moodle_material.pdf"
    project_root = Path(__file__).resolve().parents[2]
    dest_dir = project_root / "course_docs" / f"moodle_course_{course_id}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / safe_name
    display_title = (moodle_resource_title or "").strip() or filename

    try:
        with open(dest_path, "wb") as output:
            shutil.copyfileobj(file.file, output)
        _update_material_manifest(
            dest_dir,
            safe_name,
            original_file_name=filename,
            display_title=display_title,
            concept=selected_concept,
            sensitivity=sensitivity,
        )
        result = ingest_documents(
            course_id,
            [str(dest_path)],
            concept=selected_concept,
            display_title=display_title,
            original_file_name=filename,
            sensitivity=sensitivity,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to ingest Moodle material: {exc}") from exc

    # Invalidate the per-worker vectorstore cache so that subsequent concept-
    # availability checks always open a fresh ChromaDB connection and see the
    # newly ingested chunks, even across multiple uvicorn worker processes.
    get_vectorstore.cache_clear()

    if student_id is not None and selected_concept:
        memory = MemoryManager(db)
        student = memory.get_or_create_student(
            moodle_user_id=student_id,
            username=f"Moodle Student {student_id}",
        )
        if memory.get_mastery(student.id, course.id, selected_concept) == 0.0:
            memory.set_mastery(
                student_id=student.id,
                course_id=course.id,
                concept=selected_concept,
                mastery_level=0.5,
            )
        memory.update_course_memory(student.id, course.id, {
            "last_concept": selected_concept,
            "selected_concept": selected_concept,
            "last_activity": "synced Moodle course material",
        })

    print(
        "[ACRLA] moodle_material_sync "
        f"student_id={student_id or 'unknown'} "
        f"course_id={course_id} "
        f"concept={selected_concept or 'none'} "
        f"file={safe_name} "
        f"display_title={display_title} "
        f"chunks={result.get('chunks_created', 0)}"
    )

    return {
        "status": "ok",
        "course_id": course_id,
        "files_ingested": result.get("files_processed", []),
        "original_file_name": filename,
        "display_title": display_title,
        "chunks_created": result.get("chunks_created", 0),
    }


def _update_material_manifest(
    dest_dir: Path,
    saved_file_name: str,
    original_file_name: str,
    display_title: str,
    concept: str | None,
    sensitivity: str | None = None,
) -> None:
    manifest_path = dest_dir / "materials_manifest.json"
    manifest = {"files": {}}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                loaded = json.load(manifest_file)
            if isinstance(loaded, dict):
                manifest = loaded
                manifest.setdefault("files", {})
        except Exception as exc:
            print(f"[ACRLA] Failed to read material manifest {manifest_path}: {exc}")
    manifest["files"][saved_file_name] = {
        "original_file_name": original_file_name,
        "display_title": display_title,
        "concept": concept,
        # RQ2 institutional-privacy step: persisted so a later
        # rebuild_course_collection (manifest-driven re-ingest) keeps this
        # document's classification instead of silently reverting to the
        # INTERNAL default. Stored as given (None if not supplied) --
        # normalize_sensitivity is applied wherever this value is READ
        # (ingest_documents), not here, so the manifest file itself stays a
        # faithful, human-editable record of exactly what was set.
        "sensitivity": sensitivity,
    }
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)


# ==========================================================
# Moodle Launch and Session Setup
# ==========================================================

@router.get("/moodle/launch")
def launch_from_moodle(
    request: Request,
    student_id: int,
    course_id: int | None = None,
    student_name: str | None = None,
    course_name: str | None = None,
    concept: str | None = None,
    score: float | None = None,
    course_score: float | None = None,
    overall_score: float | None = None,
    level_type: str = "chapter",
    source: str = "moodle",
    embedded: str | None = None,
    hide_header: str | None = None,
    status: str | None = None,
    db: DBSession = Depends(get_db),
):
    """Prepare an ACRLA remediation launch from Moodle.

    Moodle may launch ACRLA from an overall, course, or chapter grade. Every
    launch resets the persisted remediation scope for the clicked context so
    concepts from a previous Moodle course cannot leak into the new session.

    This is defense-critical behavior: opening Data Science after Computer
    Science must rebuild scope from Data Science material only, not from any
    CS concepts left in long-term memory or chat state.
    """
    memory = MemoryManager(db)
    level_type = _normalize_level_type(level_type)
    launch_score = score if score is not None else (course_score if course_score is not None else overall_score)
    print(
        "[ACRLA] moodle_launch_payload "
        f"level_type={level_type} "
        f"student_id={student_id} "
        f"course_id={course_id if course_id is not None else 'none'} "
        f"course_name={course_name!r} "
        f"concept={concept or 'none'} "
        f"score={score if score is not None else 'none'} "
        f"course_score={course_score if course_score is not None else 'none'} "
        f"overall_score={overall_score if overall_score is not None else 'none'}"
    )

    student = db.query(Student).filter_by(moodle_user_id=student_id).first()
    if not student:
        student = memory.get_or_create_student(
            moodle_user_id=student_id,
            username=student_name or f"Moodle Student {student_id}",
        )
    elif student_name and student.username != student_name:
        student.username = student_name
        db.commit()
        db.refresh(student)
    if student_name:
        memory.set_profile_name(student.id, student_name)

    course = None
    if level_type == "overall":
        course = _weakest_course_for_student(db, memory, student.id)  # noqa: F841 (course resolved below)
    elif course_id is not None:
        course = db.query(Course).filter_by(moodle_course_id=course_id).first()
    if not course:
        if course_id is None and not course:
            raise HTTPException(status_code=404, detail="No Moodle course context is available for this student.")
    if not course:
        course = Course(moodle_course_id=course_id, name=course_name or f"Moodle Course {course_id}")
        db.add(course)
        db.commit()
        db.refresh(course)
    elif course_name and course.name != course_name:
        course.name = course_name
        db.commit()
        db.refresh(course)

    requested_concept = _assessment_concept(concept)
    allowed_for_course = set(_assessment_concepts_for_course(memory, student.id, course))
    if level_type == "chapter" and concept and not requested_concept:
        raise HTTPException(status_code=400, detail="Unknown ACRLA course concept.")
    if level_type == "chapter" and requested_concept and allowed_for_course and requested_concept not in allowed_for_course:
        raise HTTPException(status_code=400, detail="Concept is not available in this Moodle course.")
    if level_type == "chapter":
        selected_concept = requested_concept
    elif level_type == "overall":
        remediation_concepts, remediation_course_ids, remediation_internal_course_ids, primary_course = (
            _all_enrolled_scope_concepts(db, memory, student.id)
        )
        if primary_course:
            course = primary_course
        selected_concept = remediation_concepts[0] if remediation_concepts else None
        print(
            "[ACRLA] launch_scope "
            f"level_type=overall "
            f"selected_concept={selected_concept} "
            f"scope_concepts={remediation_concepts} "
            f"scope_course_ids={list(set(remediation_course_ids.values()))} "
            f"remediation_concepts={remediation_concepts}"
        )
    else:
        # Resolve concepts strictly from this course's synced material only.
        # Do NOT fall back to mastery records — they may include concepts from
        # other courses that were written to this course's records by cross-course
        # sessions, leading to stale remediation context (e.g. Pointers showing
        # up in Data Science because a prior CS session contaminated its records).
        # In plain terms, the clicked Moodle course material is the source of
        # truth for course-level remediation scope.
        prev_course_mem = memory.get_course_memory(student.id, course.id)
        course_material_concepts = _material_concepts_for_course(course)
        print(
            "[ACRLA] course_launch_context "
            f"clicked_course_id={course.moodle_course_id} "
            f"clicked_course_name={course.name!r} "
            f"resolved_course_material_concepts={course_material_concepts} "
            f"prev_remediation_concepts={prev_course_mem.get('remediation_concepts')} "
            f"prev_last_concept={prev_course_mem.get('last_concept')!r}"
        )
        if course_material_concepts:
            course_mastery_map = {
                _assessment_concept(r.concept): r.mastery_level or 0.0
                for r in memory.get_all_mastery(student.id, course.id)
                if _assessment_concept(r.concept) in set(course_material_concepts)
            }
            ordered_material = sorted(
                course_material_concepts,
                key=lambda c: course_mastery_map.get(c, 0.0),
            )
            selected_concept = ordered_material[0]
        else:
            ordered_material = []
            selected_concept = None

    if (
        selected_concept
        and level_type == "chapter"
        and launch_score is not None
        and not memory._mastery_records_for_concept(student.id, course.id, selected_concept)
    ):
        mastery = _score_to_mastery(launch_score)
        memory.set_mastery(
            student_id=student.id,
            course_id=course.id,
            concept=selected_concept,
            mastery_level=mastery,
        )

    if selected_concept:
        selected_mastery = memory.get_mastery(student.id, course.id, selected_concept)
        if level_type == "chapter" and launch_score is not None:
            selected_mastery = _score_to_mastery(launch_score)
        if level_type == "overall":
            remediation_concepts, remediation_course_ids, remediation_internal_course_ids, primary_course = (
                _all_enrolled_scope_concepts(db, memory, student.id)
            )
            if primary_course:
                course = primary_course
        elif level_type == "course":
            # Scope = ALL material concepts for the course (sorted weakest first).
            # The assessment generator will choose which weak ones to quiz on.
            remediation_concepts = ordered_material
            remediation_course_ids, remediation_internal_course_ids = _maps_for_single_course(remediation_concepts, course)
            print(
                "[ACRLA] launch_scope "
                f"level_type=course "
                f"clicked_course_id={course.moodle_course_id} "
                f"selected_concept={selected_concept} "
                f"scope_concepts={remediation_concepts} "
                f"scope_course_ids={list(set(remediation_course_ids.values()))} "
                f"remediation_concepts={remediation_concepts}"
            )
        else:
            remediation_concepts = [selected_concept]
            remediation_course_ids, remediation_internal_course_ids = _maps_for_single_course(remediation_concepts, course)
        clicked_percent = _mastery_to_percent(_score_to_mastery(launch_score)) if launch_score is not None else None
        resolved_current = _canonical_current_mastery_percent(
            db,
            memory,
            student.id,
            course,
            level_type,
            remediation_concepts,
            remediation_internal_course_ids,
        )
        resolved_initial = clicked_percent if clicked_percent is not None else resolved_current
        source_used = "current_acrla_mastery" if resolved_current else "fallback_moodle_mastery"
        normalized_concept = _assessment_concept(selected_concept)
        mastery_record_key = (
            f"{remediation_internal_course_ids.get(normalized_concept) or course.id}:"
            f"{normalized_concept}"
        ) if normalized_concept else "none"
        print(
            "[ACRLA] launch_mastery_resolution "
            f"level_type={level_type} "
            f"student_id={student.moodle_user_id} "
            f"course_id={course.moodle_course_id} "
            f"clicked_concept={concept or 'none'} "
            f"selected_concept={selected_concept or 'none'} "
            f"launch_concept={selected_concept or 'none'} "
            f"normalized_concept={normalized_concept or 'none'} "
            f"mastery_record_key_used={mastery_record_key} "
            f"concept={', '.join(remediation_concepts)} "
            f"clicked_score={clicked_percent if clicked_percent is not None else 'none'} "
            f"button_displayed_mastery={clicked_percent if clicked_percent is not None else 'none'} "
            f"resolved_initial_moodle_mastery={resolved_initial} "
            f"resolved_current_acrla_mastery={resolved_current} "
            f"source_of_truth_used={source_used}"
        )
        # Moodle launches can arrive after a different course was used in the
        # same browser session. Overwrite all scope keys here because
        # update_course_memory merges dictionaries by design.
        memory.update_course_memory(student.id, course.id, {
            "remediation_level": level_type,
            "remediation_concepts": remediation_concepts,
            "scope_concepts": remediation_concepts,
            "remediation_course_ids": remediation_course_ids,
            "remediation_internal_course_ids": remediation_internal_course_ids,
            "level_type": level_type,
            "student_id": student.moodle_user_id,
            "course_id": course.moodle_course_id,
            "course_name": course.name,
            "last_concept": selected_concept,
            "launch_concept": selected_concept,
            "selected_concept": selected_concept,
            "active_tutoring_concept": selected_concept,
            "previous_concepts": [],
            "cached_retrieval_scope": None,
            "locked_concept": selected_concept if level_type == "chapter" else None,
            "locked_level_type": level_type if level_type == "chapter" else None,
            "sub_concepts": sub_concepts_for(selected_concept),
            "last_activity": f"clicked Moodle {level_type} remediation link",
            "launch_context": {
                "level_type": level_type,
                "student_id": student.moodle_user_id,
                "course_id": course.moodle_course_id,
                "course_name": course.name,
                "concept": selected_concept,
                "remediation_concepts": remediation_concepts,
                "remediation_course_ids": remediation_course_ids,
                "score": _mastery_to_percent(selected_mastery),
                "clicked_score": _mastery_to_percent(_score_to_mastery(launch_score)) if launch_score is not None else None,
                "locked_concept": selected_concept if level_type == "chapter" else None,
                "sub_concepts": sub_concepts_for(selected_concept),
            },
            "next_recommended_action": f"start remediation for {selected_concept}",
        })
        print(
            "[ACRLA] moodle_launch "
            f"clicked_course_id={course.moodle_course_id} "
            f"scope_course_ids={list(set(remediation_course_ids.values()))} "
            f"scope_concepts={remediation_concepts} "
            f"launch_concept={selected_concept} "
            f"selected_concept={selected_concept} "
            f"active_tutoring_concept={selected_concept}"
        )

    # For course-level launches with no synced material, explicitly wipe stale
    # context so the chatbot doesn't inherit a previous session's concepts.
    # This also prevents the "Pointers in Data Science" class of errors where a
    # concept from a prior course bleeds into a new course's remediation context.
    if level_type == "course" and not selected_concept:
        memory.update_course_memory(student.id, course.id, {
            "remediation_level": "course",
            "level_type": "course",
            "remediation_concepts": [],
            "scope_concepts": [],
            "remediation_course_ids": {},
            "remediation_internal_course_ids": {},
            "course_id": course.moodle_course_id,
            "course_name": course.name,
            "last_concept": None,
            "launch_concept": None,
            "selected_concept": None,
            "active_tutoring_concept": None,
            "previous_concepts": [],
            "cached_retrieval_scope": None,
            "locked_concept": None,
            "locked_level_type": None,
            "launch_context": {
                "level_type": "course",
                "student_id": student.moodle_user_id,
                "course_id": course.moodle_course_id,
                "course_name": course.name,
                "concept": None,
                "remediation_concepts": [],
                "remediation_course_ids": {},
                "locked_concept": None,
            },
            "last_activity": "clicked course remediation link — no synced material found",
        })

    prefs = memory.get_profile_preferences(student.id)
    mode = prefs.get("learning_mode") or "internal"
    difficulty = prefs.get("difficulty") or "easy"
    active_session = db.query(SessionModel).filter_by(
        student_id=student.id,
        course_id=course.id,
        is_active=True,
    ).first()
    if not active_session:
        active_session = memory.start_session(
            student_id=student.id,
            course_id=course.id,
            mode=mode,
            difficulty=difficulty,
            learning_goal=f"Moodle {level_type} remediation" if selected_concept else "Moodle launch",
        )
    memory.set_current_topic(active_session.id, selected_concept)

    params = {
        "student_id": student_id,
        "course_id": course.moodle_course_id,
        "student_name": student_name or student.username,
        "course_name": course.name,
        "source": source,
        "launch_session_id": active_session.id,
    }
    if selected_concept and level_type != "overall":
        params["concept"] = selected_concept
        params["score"] = _mastery_to_percent(memory.get_mastery(student.id, course.id, selected_concept))
    if launch_score is not None:
        params["clicked_score"] = _mastery_to_percent(_score_to_mastery(launch_score))
    params["level_type"] = level_type
    if embedded is not None:
        params["embedded"] = embedded
    if hide_header is not None:
        params["hide_header"] = hide_header
    if status is not None:
        params["status"] = status
    if level_type == "course" and not selected_concept:
        params["message"] = (
            f"No synced course material found for {course.name} yet. "
            "Please sync Moodle materials first."
        )

    frontend_url = f"{str(request.base_url).rstrip('/')}/?{urlencode(params)}"
    print(
        "[ACRLA] moodle_launch_redirect "
        f"level_type={level_type} "
        f"course_id={course.moodle_course_id} "
        f"course_name={course.name!r} "
        f"concept={selected_concept or 'none'} "
        f"url={frontend_url}"
    )
    return RedirectResponse(frontend_url)


@router.post("/session/start", response_model=SessionStartResponse)
def start_session(request: SessionStartRequest, db: DBSession = Depends(get_db)):
    """Start a chatbot session using Moodle profile and launch context.

    The endpoint seeds student/course state, loads persisted mastery and
    preferences, computes the active remediation scope, and returns the initial
    greeting plus available concepts. It does not perform tutoring itself.

    Session start repeats Moodle launch scope resolution because the frontend
    may start directly from iframe query parameters. The backend therefore
    never relies on stale browser/UI state to decide the active course scope.
    """
    payload = request.moodle_payload
    memory = MemoryManager(db)
    launch_preferences = payload.learning_preferences or {}
    launch_concept = _assessment_concept(
        launch_preferences.get("launch_concept")
        or launch_preferences.get("selected_concept")
    )
    launch_level_type = _normalize_level_type(launch_preferences.get("level_type"))
    print(
        "[ACRLA] session_start_payload "
        f"launch_level_type={launch_level_type} "
        f"payload_student_id={payload.student_id} "
        f"payload_course_id={payload.course_id} "
        f"payload_course_name={payload.course_name!r} "
        f"launch_concept={launch_concept or 'none'} "
        f"payload_scores={payload.scores} "
        f"payload_weak_concepts={payload.weak_concepts}"
    )

    # Upsert student
    student = memory.get_or_create_student(
        moodle_user_id=payload.student_id,
        username=payload.username,
        email=payload.email,
    )

    # Upsert course
    course = db.query(Course).filter_by(moodle_course_id=payload.course_id).first()
    if not course:
        course = Course(moodle_course_id=payload.course_id, name=payload.course_name)
        db.add(course)
        db.commit()
        db.refresh(course)
    elif course.name != payload.course_name:
        course.name = payload.course_name
        db.commit()
        db.refresh(course)

    # Load preferences
    prefs = memory.get_preferences(student.id)
    profile_prefs = memory.get_profile_preferences(student.id)
    mode = "automatic"
    difficulty = (
        profile_prefs.get("difficulty")
        or request.difficulty
        or (prefs.preferred_difficulty if prefs else "medium")
    )

    # Seed/update mastery profile from Moodle payload scores.
    accepted_scores = []
    payload_concepts = [
        concept for concept in (_assessment_concept(raw) for raw in payload.scores.keys())
        if concept
    ]
    if launch_level_type == "course":
        course_concepts = _course_scope_concepts(memory, student.id, course) or payload_concepts
    else:
        course_concepts = payload_concepts or _assessment_concepts_for_course(memory, student.id, course)
    allowed_concepts = set(course_concepts)
    if launch_concept and launch_concept not in allowed_concepts:
        if launch_level_type == "course":
            # A course launch must be rebuilt from the clicked course material.
            # Keeping a stale launch_concept would reintroduce previous-course
            # topics into the new course scope.
            print(
                "[ACRLA] session_start_drop_stale_launch_concept "
                f"clicked_course_id={payload.course_id} "
                f"course_name={payload.course_name!r} "
                f"stale_launch_concept={launch_concept} "
                f"scope_concepts={course_concepts}"
            )
            launch_concept = None
        else:
            course_concepts = list(dict.fromkeys(course_concepts + [launch_concept]))
            allowed_concepts.add(launch_concept)
    course_memory_data = memory.get_course_memory(student.id, course.id)
    initial_acrla_map, current_acrla_map = _memory_mastery_maps(course_memory_data)
    existing_concepts = {record.concept for record in memory.get_all_mastery(student.id, course.id)}
    for raw_concept, score in payload.scores.items():
        concept = _assessment_concept(raw_concept)
        if not concept or concept not in allowed_concepts:
            continue
        scope_key = _chapter_mastery_key(course, concept)
        demo_pct = _demo_initial_mastery_for_concept(course, concept)
        moodle_pct = demo_pct if demo_pct is not None else _mastery_to_percent(_score_to_mastery(score))
        initial_acrla_map[scope_key] = moodle_pct if demo_pct is not None else max(_safe_float(initial_acrla_map.get(scope_key)), moodle_pct)
        if scope_key in current_acrla_map:
            current_acrla_map[scope_key] = max(_safe_float(current_acrla_map.get(scope_key)), initial_acrla_map[scope_key])
        acrla_pct = current_acrla_map.get(scope_key)
        if acrla_pct is not None:
            mastery = float(acrla_pct) / 100
            accepted_scores.append(mastery)
            print(
                "[ACRLA] session_start_mastery_load "
                f"student_id={payload.student_id} "
                f"course_id={payload.course_id} "
                f"concept={concept} "
                f"mastery_record_key={course.id}:{concept} "
                f"source=current_acrla_mastery "
                f"loaded_mastery={round(float(acrla_pct), 2)} "
                f"moodle_payload_score={score}"
            )
            continue
        existing = next((record for record in memory.get_all_mastery(student.id, course.id) if _assessment_concept(record.concept) == concept), None)
        if concept in existing_concepts and not (
            existing
            and float(existing.mastery_level or 0.0) == 0.5
            and int(existing.attempts or 0) == 0
            and int(existing.correct or 0) == 0
        ):
            mastery = memory.get_mastery(student.id, course.id, concept)
            accepted_scores.append(mastery)
            print(
                "[ACRLA] session_start_mastery_load "
                f"student_id={payload.student_id} "
                f"course_id={payload.course_id} "
                f"concept={concept} "
                f"mastery_record_key={course.id}:{concept} "
                f"source=mastery_records "
                f"loaded_mastery={_mastery_to_percent(mastery)} "
                f"moodle_payload_score={score}"
            )
            continue
        mastery = _score_to_mastery(score)
        accepted_scores.append(mastery)
        memory.set_mastery(
            student_id=student.id,
            course_id=course.id,
            concept=concept,
            mastery_level=mastery,
        )
        print(
            "[ACRLA] session_start_mastery_load "
            f"student_id={payload.student_id} "
            f"course_id={payload.course_id} "
            f"concept={concept} "
            f"mastery_record_key={course.id}:{concept} "
            f"source=moodle_payload_fallback "
            f"loaded_mastery={_mastery_to_percent(mastery)} "
            f"moodle_payload_score={score}"
        )
        existing_concepts.add(concept)

    memory.update_course_memory(student.id, course.id, {
        "moodle_initial_mastery": initial_acrla_map,
        "current_acrla_mastery": current_acrla_map,
    })

    for concept in payload.weak_concepts:
        clean = _assessment_concept(concept)
        if not clean or clean not in allowed_concepts:
            continue
        if clean not in existing_concepts:
            memory.set_mastery(student.id, course.id, clean, 0.25)
            existing_concepts.add(clean)

    # Merge weak concepts from the persisted mastery profile, ordered by lowest mastery.
    mastery_records = sorted(memory.get_all_mastery(student.id, course.id), key=lambda r: r.mastery_level)
    weak_concepts = [
        r.concept for r in mastery_records
        if _assessment_concept(r.concept) in allowed_concepts and r.mastery_level < 0.6
    ]

    # Start session
    chapter_score = min(accepted_scores, default=0.0)
    suggested_difficulty = _suggest_difficulty(weak_concepts, chapter_score)
    if request.difficulty is None and not profile_prefs.get("difficulty"):
        difficulty = suggested_difficulty

    session = memory.start_session(
        student_id=student.id,
        course_id=course.id,
        mode=mode,
        difficulty=difficulty,
        learning_goal=request.learning_goal,
    )
    if launch_concept:
        if launch_level_type == "overall":
            remediation_concepts, remediation_course_ids, remediation_internal_course_ids, _primary_course = (
                _all_enrolled_scope_concepts(db, memory, student.id)
            )
            remediation_concepts = remediation_concepts or course_concepts or [launch_concept]
            if not remediation_course_ids:
                remediation_course_ids, remediation_internal_course_ids = _course_maps_for_concepts(
                    db, remediation_concepts, course,
                )
        elif launch_level_type == "chapter":
            remediation_concepts = [launch_concept]
            remediation_course_ids, remediation_internal_course_ids = _maps_for_single_course(remediation_concepts, course)
        else:
            # Course level: material-authoritative scope, sorted weakest-first.
            # Do NOT use mastery records to derive the concept list — prior overall
            # sessions may have contaminated course.id with alien concepts.
            remediation_concepts = _course_scope_concepts(memory, student.id, course)
            remediation_course_ids, remediation_internal_course_ids = _maps_for_single_course(remediation_concepts, course)
        print(
            "[ACRLA] session_start_scope "
            f"level_type={launch_level_type} "
            f"student_id={payload.student_id} "
            f"clicked_course_id={payload.course_id} "
            f"selected_concept={launch_concept} "
            f"scope_concepts={remediation_concepts} "
            f"scope_course_ids={list(set(remediation_course_ids.values()))} "
            f"remediation_concepts={remediation_concepts} "
            f"retrieval_course_ids={list(set(remediation_internal_course_ids.values()))} "
            f"active_tutoring_concept={launch_concept}"
        )
        memory.set_current_topic(session.id, launch_concept)
        memory.update_course_memory(student.id, course.id, {
            "remediation_level": launch_level_type,
            "remediation_concepts": remediation_concepts,
            "scope_concepts": remediation_concepts,
            "remediation_course_ids": remediation_course_ids,
            "remediation_internal_course_ids": remediation_internal_course_ids,
            "level_type": launch_level_type,
            "last_concept": launch_concept,
            "launch_concept": launch_concept,
            "selected_concept": launch_concept,
            "active_tutoring_concept": launch_concept,
            "previous_concepts": [],
            "cached_retrieval_scope": None,
            "locked_concept": launch_concept if launch_level_type == "chapter" else None,
            "locked_level_type": launch_level_type if launch_level_type == "chapter" else None,
            "sub_concepts": sub_concepts_for(launch_concept),
            "last_activity": "started Moodle remediation launch",
            "launch_context": {
                "level_type": launch_level_type,
                "student_id": payload.student_id,
                "course_id": payload.course_id,
                "course_name": payload.course_name,
                "concept": launch_concept,
                "remediation_concepts": remediation_concepts,
                "remediation_course_ids": remediation_course_ids,
                "score": launch_preferences.get("launch_score"),
                "locked_concept": launch_concept if launch_level_type == "chapter" else None,
                "sub_concepts": sub_concepts_for(launch_concept),
            },
            "next_recommended_action": f"continue with {launch_concept}",
        })
    else:
        default_concept = weak_concepts[0] if weak_concepts else (course_concepts[0] if course_concepts else "Course Material")
        if launch_level_type == "overall":
            remediation_concepts, remediation_course_ids, remediation_internal_course_ids, _primary_course = (
                _all_enrolled_scope_concepts(db, memory, student.id)
            )
            remediation_concepts = remediation_concepts or course_concepts or [default_concept]
            if not remediation_course_ids:
                remediation_course_ids, remediation_internal_course_ids = _course_maps_for_concepts(
                    db, remediation_concepts, course,
                )
        elif launch_level_type == "chapter":
            remediation_concepts = [default_concept]
            remediation_course_ids, remediation_internal_course_ids = _maps_for_single_course(remediation_concepts, course)
        else:
            # Course level: material-authoritative scope, sorted weakest-first.
            remediation_concepts = _course_scope_concepts(memory, student.id, course)
            remediation_course_ids, remediation_internal_course_ids = _maps_for_single_course(remediation_concepts, course)
            if remediation_concepts:
                default_concept = remediation_concepts[0]
        print(
            "[ACRLA] session_start_scope "
            f"level_type={launch_level_type} "
            f"student_id={payload.student_id} "
            f"clicked_course_id={payload.course_id} "
            f"selected_concept={default_concept} "
            f"scope_concepts={remediation_concepts} "
            f"scope_course_ids={list(set(remediation_course_ids.values()))} "
            f"remediation_concepts={remediation_concepts} "
            f"retrieval_course_ids={list(set(remediation_internal_course_ids.values()))} "
            f"active_tutoring_concept={default_concept}"
        )
        memory.set_current_topic(session.id, default_concept)
        memory.update_course_memory(student.id, course.id, {
            "remediation_level": launch_level_type,
            "remediation_concepts": remediation_concepts,
            "scope_concepts": remediation_concepts,
            "remediation_course_ids": remediation_course_ids,
            "remediation_internal_course_ids": remediation_internal_course_ids,
            "level_type": launch_level_type,
            "last_concept": default_concept,
            "launch_concept": None,
            "selected_concept": default_concept,
            "active_tutoring_concept": default_concept,
            "previous_concepts": [],
            "cached_retrieval_scope": None,
            "locked_concept": None,
            "locked_level_type": None,
            "sub_concepts": sub_concepts_for(default_concept),
            "last_activity": "started Moodle course session",
            "launch_context": {
                "level_type": launch_level_type,
                "student_id": payload.student_id,
                "course_id": payload.course_id,
                "course_name": payload.course_name,
                "concept": default_concept,
                "remediation_concepts": remediation_concepts,
                "remediation_course_ids": remediation_course_ids,
                "locked_concept": None,
                "sub_concepts": sub_concepts_for(default_concept),
            },
            "next_recommended_action": f"continue with {default_concept}",
        })
    print(
        "[ACRLA] session_start_preferences "
        f"profile_difficulty={profile_prefs.get('difficulty') or 'none'} "
        f"session_difficulty={session.difficulty} "
        f"welcome_difficulty={difficulty} "
        f"profile_mode={profile_prefs.get('learning_mode') or 'none'} "
        f"session_mode={session.mode}"
    )

    # Count previous sessions
    from models.db_models import Session as SessionModel
    session_count = db.query(SessionModel).filter_by(student_id=student.id).count()
    persistent_name = memory.get_profile_name(student.id) or student.username
    course_memory = memory.get_course_memory(student.id, course.id)
    last_concept = canonicalize_concept(course_memory.get("last_concept"))
    if last_concept not in allowed_concepts:
        last_concept = None

    if session_count > 1 and last_concept:
        last_mastery = memory.get_mastery(student.id, course.id, last_concept)
        greeting = (
            f"Welcome back, {persistent_name}. Last time, you were practicing "
            f"{last_concept}. Your mastery is now {last_mastery:.0%}. "
            f"Let's continue with a new question on {last_concept}."
        )
    else:
        # Generate personalized greeting
        greeting = generate_greeting(
            username=persistent_name,
            chapter_name=payload.chapter_name or payload.course_name,
            score=chapter_score,
            weak_concepts=weak_concepts,
            session_count=session_count,
        )

    if launch_concept:
        launch_sub_concepts = sub_concepts_for(launch_concept)
        if launch_sub_concepts:
            greeting += (
                f"\n\nWithin {launch_concept}, we'll check smaller skills like: "
                f"{', '.join(launch_sub_concepts[:5])}."
            )

    # Generate opening question suggestion
    if launch_concept or weak_concepts:
        concept = launch_concept or weak_concepts[0]
        greeting += f"\n\n💡 Suggested start: Try asking me *'Explain {concept} to me'* or say *'Test me on {concept}'*"

    greeting += f"\n\nYour saved preferences are: {_label(difficulty)} difficulty. Response routing is automatic."

    # Save greeting to buffer
    memory.save_message(session.id, "assistant", greeting, intent="greeting")

    return SessionStartResponse(
        session_id=session.id,
        greeting=greeting,
        weak_concepts=weak_concepts,
        available_concepts=list(course_concepts),
        suggested_difficulty=suggested_difficulty,
        difficulty=session.difficulty,
        mode="automatic",
        learning_mode="automatic",
    )


def _label(value: str) -> str:
    if not value:
        return ""
    return str(value).replace("_", " ").strip().capitalize()


def _suggest_difficulty(weak_concepts: list, score: float) -> str:
    if score < 0.4 or len(weak_concepts) > 3:
        return "easy"
    if score > 0.75 and len(weak_concepts) == 0:
        return "hard"
    return "medium"


# ── POST /chat ────────────────────────────────────────────────────────────────

# ==========================================================
# Chat API Response Construction
# ==========================================================

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: DBSession = Depends(get_db)):
    try:
        result = handle_message(
            session_id=request.session_id,
            student_moodle_id=request.student_id,
            message=request.message,
            db=db,
        )
        return ChatResponse(**result)
    except Exception as exc:
        print(f"[CHAT] Failed to handle message: {exc}")
        return ChatResponse(
            reply=(
                "I could not generate a response just now. Please check the backend "
                "LLM settings and try again."
            ),
            intent="error",
            strategy="none",
            sources=[],
        )


# ── POST /session/end ─────────────────────────────────────────────────────────

# ==========================================================
# Assessment / Progress Check
# ==========================================================

@router.post("/assessment/start", response_model=AssessmentStartResponse)
def start_assessment(request: AssessmentStartRequest, db: DBSession = Depends(get_db)):
    """Create a scoped Quick Progress Check assessment.

    Assessment scope is taken from the active session/remediation launch:
    chapter stays inside one concept, course stays inside one Moodle course,
    and overall may span all enrolled courses. Starting an assessment does not
    update mastery.
    """
    student = db.query(Student).filter_by(moodle_user_id=request.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    session = db.query(SessionModel).filter_by(
        id=request.session_id,
        student_id=student.id,
        is_active=True,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Active session not found")

    memory = MemoryManager(db)
    level_type, course, concepts, concept_course_ids, concept_internal_course_ids = _assessment_scope(db, memory, session)
    concepts = [concept for concept in (_assessment_concept(c) for c in concepts) if concept]
    if not concepts:
        raise HTTPException(status_code=400, detail="No remediation concept is available for assessment.")
    print(
        "[ACRLA] assessment_start_scope "
        f"level_type={level_type} "
        f"course_id={course.moodle_course_id} "
        f"course_name={course.name!r} "
        f"concept={', '.join(concepts)} "
        f"scope_course_ids={list(set(concept_course_ids.values()))} "
        f"scope_concepts={concepts}"
    )

    recent_question_keys = _recent_assessment_question_keys(
        db,
        student.id,
        level_type,
        course.id,
        concepts,
    )
    questions = _build_assessment_questions(
        concepts,
        level_type=level_type,
        concept_course_ids=concept_course_ids,
        concept_internal_course_ids=concept_internal_course_ids,
        recent_question_keys=recent_question_keys,
        db=db,
    )
    assessment_concepts = sorted({
        concept
        for item in questions
        for concept in (item.get("concepts_used") or [])
    })
    selected_remediation_concepts = sorted(concepts)
    print(
        "[ACRLA] assessment_generation "
        f"selected_remediation_concepts={selected_remediation_concepts} "
        f"assessment_concepts={assessment_concepts}"
    )
    if assessment_concepts != selected_remediation_concepts:
        print(
            "[ACRLA] assessment_scope_warning "
            f"selected_remediation_concepts={selected_remediation_concepts} "
            f"assessment_concepts={assessment_concepts}"
        )
    _validate_assessment_scope(level_type, questions, concepts, concept_course_ids)
    course_memory = memory.get_course_memory(student.id, course.id)
    launch_context = course_memory.get("launch_context") or {}
    clicked_score = launch_context.get("clicked_score")
    try:
        clicked_score = float(clicked_score) if clicked_score is not None else None
    except (TypeError, ValueError):
        clicked_score = None
    moodle_initial_mastery, current_acrla_mastery, source_of_truth = _ensure_mastery_baseline(
        db,
        memory,
        student.id,
        course.id,
        course,
        level_type,
        concepts,
        concept_internal_course_ids,
        clicked_score=clicked_score,
    )
    launch_concept = _assessment_concept(launch_context.get("concept"))
    selected_concept = _assessment_concept(course_memory.get("selected_concept"))
    normalized_concept = concepts[0] if level_type == "chapter" and concepts else _assessment_concept(
        selected_concept or launch_concept
    )
    mastery_record_key = (
        f"{concept_internal_course_ids.get(normalized_concept) or course.id}:"
        f"{normalized_concept}"
    ) if normalized_concept else "none"
    print(
        "[ACRLA] assessment_start_mastery "
        f"level_type={level_type} "
        f"student_id={student.moodle_user_id} "
        f"course_id={course.moodle_course_id} "
        f"clicked_concept={launch_context.get('concept') or 'none'} "
        f"selected_concept={selected_concept or 'none'} "
        f"launch_concept={launch_concept or 'none'} "
        f"normalized_concept={normalized_concept or 'none'} "
        f"mastery_record_key_used={mastery_record_key} "
        f"concept={', '.join(concepts)} "
        f"clicked_score={clicked_score if clicked_score is not None else 'none'} "
        f"button_displayed_mastery={clicked_score if clicked_score is not None else 'none'} "
        f"resolved_initial_moodle_mastery={moodle_initial_mastery} "
        f"resolved_current_acrla_mastery={current_acrla_mastery} "
        f"assessment_modal_current_mastery={current_acrla_mastery} "
        f"source_of_truth_used={source_of_truth}"
    )
    assessment_id = str(uuid4())
    _active_assessments[assessment_id] = {
        "student_id": student.id,
        "moodle_student_id": request.student_id,
        "session_id": session.id,
        "course_id": course.id,
        "moodle_course_id": course.moodle_course_id,
        "level_type": level_type,
        "concepts": concepts,
        "concept_course_ids": concept_course_ids,
        "concept_internal_course_ids": concept_internal_course_ids,
        "moodle_initial_mastery": moodle_initial_mastery,
        "current_acrla_mastery": current_acrla_mastery,
        "questions": questions,
        "created_at": datetime.utcnow().isoformat(),
    }

    return AssessmentStartResponse(
        assessment_id=assessment_id,
        level_type=level_type,
        course_id=course.moodle_course_id,
        concept=", ".join(concepts),
        moodle_initial_mastery=moodle_initial_mastery,
        current_acrla_mastery=current_acrla_mastery,
        questions=_public_assessment_questions(questions),
    )


@router.post("/assessment/submit", response_model=AssessmentSubmitResponse)
def submit_assessment(request: AssessmentSubmitRequest, db: DBSession = Depends(get_db)):
    """Score a Quick Progress Check and persist current ACRLA mastery.

    This is the only MVP flow that changes mastery. The update uses current
    ACRLA mastery as the baseline and never lowers the value below previous
    ACRLA mastery or the initial Moodle mastery.
    """
    assessment = _active_assessments.get(request.assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found or expired")

    student = db.query(Student).filter_by(moodle_user_id=request.student_id).first()
    if not student or student.id != assessment["student_id"]:
        raise HTTPException(status_code=403, detail="Assessment does not belong to this student")

    if request.session_id != assessment["session_id"]:
        raise HTTPException(status_code=403, detail="Assessment does not belong to this session")

    memory = MemoryManager(db)
    questions = assessment["questions"]
    total = len(questions)
    correct = 0
    details = []
    for item in questions:
        submitted = str(request.answers.get(item["id"], "")).strip().upper()[:1]
        expected = str(item["correct"]).strip().upper()
        is_correct = submitted == expected
        if is_correct:
            correct += 1
        details.append({
            "id": item["id"],
            "question_id": item.get("question_id"),
            "variant_id": item.get("variant_id"),
            "sub_concept": item.get("sub_concept"),
            "concept": item.get("concept"),
            "concepts_used": item.get("concepts_used", []),
            "course_ids_used": item.get("course_ids_used", []),
            "question_type": item.get("question_type", "single_concept"),
            "submitted": submitted,
            "correct_answer": expected,
            "correct": is_correct,
        })

    assessment_score = round((correct / total) * 100, 2) if total else 0.0
    concepts = [
        concept for concept in (_assessment_concept(c) for c in assessment.get("concepts", []))
        if concept
    ]
    concept_internal_course_ids = assessment.get("concept_internal_course_ids") or {}
    moodle_initial_mastery = round(float(assessment.get("moodle_initial_mastery", 0.0)), 2)
    assessment_course = db.query(Course).filter_by(id=assessment["course_id"]).first()
    previous_mastery = round(float(
        assessment.get("current_acrla_mastery")
        if assessment.get("current_acrla_mastery") is not None
        else _canonical_current_mastery_percent(
            db,
            memory,
            student.id,
            assessment_course,
            assessment["level_type"],
            concepts,
            concept_internal_course_ids,
        )
    ), 2)
    # Moodle grade is the floor, current ACRLA mastery is the baseline, and the
    # assessment score is new evidence. The MVP intentionally never lowers
    # mastery because progress checks are framed as remediation feedback rather
    # than punitive grading.
    previous_mastery = max(previous_mastery, moodle_initial_mastery)
    calculated_mastery = round((0.7 * previous_mastery) + (0.3 * assessment_score), 2)
    updated_mastery = max(previous_mastery, calculated_mastery, moodle_initial_mastery)
    mastery_delta = round(updated_mastery - previous_mastery, 2)

    persisted_rows = []
    if assessment["level_type"] == "chapter":
        for concept in concepts:
            target_course_id = concept_internal_course_ids.get(concept) or assessment["course_id"]
            record = memory.set_mastery(
                student_id=student.id,
                course_id=target_course_id,
                concept=concept,
                mastery_level=updated_mastery / 100,
            )
            persisted_value = memory.get_mastery(student.id, target_course_id, concept)
            persisted_rows.append({
                "concept": concept,
                "mastery_record_key": f"{target_course_id}:{concept}",
                "mastery_record_id": record.id if record else "none",
                "persisted_mastery": _mastery_to_percent(persisted_value),
            })
            print(
                "[ACRLA] assessment_submit_persist_row "
                f"previous_mastery={previous_mastery} "
                f"updated_mastery={updated_mastery} "
                f"level_type={assessment['level_type']} "
                f"course_id={assessment['moodle_course_id']} "
                f"concept={concept} "
                f"mastery_record_key={target_course_id}:{concept} "
                f"database_row_updated={record.id if record else 'none'} "
                f"persisted_value={_mastery_to_percent(persisted_value)}"
            )
    else:
        print(
            "[ACRLA] assessment_submit_level_isolated "
            f"previous_mastery={previous_mastery} "
            f"updated_mastery={updated_mastery} "
            f"level_type={assessment['level_type']} "
            f"course_id={assessment['moodle_course_id']} "
            f"concept={', '.join(concepts)} "
            "chapter_mastery_records_updated=False"
        )
    _set_current_acrla_mastery(
        memory,
        student.id,
        assessment["course_id"],
        assessment["level_type"],
        concepts,
        updated_mastery,
        concept_internal_course_ids,
    )

    record = AssessmentRecord(
        student_id=student.id,
        course_id=assessment["course_id"],
        session_id=assessment["session_id"],
        concept=", ".join(concepts),
        level_type=assessment["level_type"],
        previous_mastery=previous_mastery,
        assessment_score=assessment_score,
        updated_mastery=updated_mastery,
        details={
            "questions": details,
            "moodle_course_id": assessment["moodle_course_id"],
            "concept_course_ids": assessment.get("concept_course_ids", {}),
            "concept_internal_course_ids": concept_internal_course_ids,
            "moodle_initial_mastery": moodle_initial_mastery,
            "current_acrla_mastery": updated_mastery,
            "calculated_mastery": calculated_mastery,
            "mastery_delta": mastery_delta,
            "non_decreasing_mvp": True,
            "persisted_rows": persisted_rows,
        },
    )
    db.add(record)
    db.commit()

    memory.update_course_memory(student.id, assessment["course_id"], {
        "last_assessment": {
            "previous_mastery": previous_mastery,
            "moodle_initial_mastery": moodle_initial_mastery,
            "current_acrla_mastery": updated_mastery,
            "assessment_score": assessment_score,
            "calculated_mastery": calculated_mastery,
            "updated_mastery": updated_mastery,
            "mastery_delta": mastery_delta,
            "level_type": assessment["level_type"],
            "course_id": assessment["moodle_course_id"],
            "concept": ", ".join(concepts),
            "timestamp": datetime.utcnow().isoformat(),
        },
        "last_concept": concepts[0] if concepts else None,
        "selected_concept": concepts[0] if concepts else None,
        "last_activity": "completed quick progress assessment",
    })
    resolver_value = _canonical_current_mastery_percent(
        db,
        memory,
        student.id,
        assessment_course,
        assessment["level_type"],
        concepts,
        concept_internal_course_ids,
    )
    print(
        "[ACRLA] assessment_submit_resolver "
        f"previous_mastery={previous_mastery} "
        f"updated_mastery={updated_mastery} "
        f"level_type={assessment['level_type']} "
        f"course_id={assessment['moodle_course_id']} "
        f"concept={', '.join(concepts)} "
        f"resolver_current_acrla_mastery={resolver_value} "
        f"persisted_rows={persisted_rows}"
    )
    _active_assessments.pop(request.assessment_id, None)

    return AssessmentSubmitResponse(
        status="ok",
        moodle_initial_mastery=moodle_initial_mastery,
        current_acrla_mastery=updated_mastery,
        previous_mastery=previous_mastery,
        assessment_score=assessment_score,
        calculated_mastery=calculated_mastery,
        updated_mastery=updated_mastery,
        mastery_delta=mastery_delta,
        level_type=assessment["level_type"],
        course_id=assessment["moodle_course_id"],
        concept=", ".join(concepts),
        correct=correct,
        total=total,
    )


@router.post("/demo/reset-mastery/{student_id}")
def reset_demo_mastery(student_id: int = 2, db: DBSession = Depends(get_db)):
    """Reset thesis-demo mastery state for one Moodle student.

    This intentionally resets only the demo baseline state used by the Moodle
    buttons and assessment modal. It does not delete students, sessions,
    messages, PDFs, or Chroma collections.
    """
    memory = MemoryManager(db)
    student = db.query(Student).filter_by(moodle_user_id=student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    reset_courses = []
    course_scores = []
    all_concepts = []
    all_courses = db.query(Course).all()
    courses = _available_courses_for_student(db, memory, student.id)
    overall_key = _overall_mastery_key(student.id)

    # First clear stale explicit overall values everywhere for this student, so
    # /mastery/student recomputes overall from the reset course values.
    for course in all_courses:
        course_memory = memory.get_course_memory(student.id, course.id)
        initial_map, current_map = _memory_mastery_maps(course_memory)
        initial_map = {k: v for k, v in initial_map.items() if not str(k).startswith("overall:")}
        current_map = {k: v for k, v in current_map.items() if not str(k).startswith("overall:")}
        memory.update_course_memory(student.id, course.id, {
            "moodle_initial_mastery": initial_map,
            "current_acrla_mastery": current_map,
        })

    for course in courses:
        baseline = _demo_baseline_map_for_course(memory, student.id, course)
        if not baseline:
            continue

        initial_map = {}
        current_map = {}
        for concept, value in baseline.items():
            chapter_key = _chapter_mastery_key(course, concept)
            initial_map[chapter_key] = value
            current_map[chapter_key] = value

            record = memory.set_mastery(
                student_id=student.id,
                course_id=course.id,
                concept=concept,
                mastery_level=value / 100,
            )
            if record:
                record.attempts = 0
                record.correct = 0
                record.last_updated = datetime.utcnow()
            all_concepts.append({
                "course_id": course.moodle_course_id,
                "course_name": course.name,
                "concept": concept,
                "current_acrla_mastery": value,
            })

        db.commit()

        concepts = list(baseline.keys())
        course_score = round(sum(baseline.values()) / len(baseline), 2)
        course_key = _course_mastery_key(course)
        initial_map[course_key] = course_score
        current_map[course_key] = course_score
        ordered = sorted(baseline.items(), key=lambda item: item[1])
        weak_concepts = [concept for concept, value in ordered if value < 60]
        strongest_concepts = [concept for concept, _value in sorted(baseline.items(), key=lambda item: item[1], reverse=True)[:2]]

        memory.update_course_memory(student.id, course.id, {
            "moodle_initial_mastery": initial_map,
            "current_acrla_mastery": current_map,
            "weak_concepts": weak_concepts,
            "strongest_concepts": strongest_concepts,
            "last_concept": weak_concepts[0] if weak_concepts else concepts[0],
            "selected_concept": weak_concepts[0] if weak_concepts else concepts[0],
            "last_activity": "demo mastery reset to fixed Moodle baselines",
            "next_recommended_action": f"continue with {weak_concepts[0] if weak_concepts else concepts[0]}",
        })

        reset_courses.append({
            "course_id": course.moodle_course_id,
            "course_name": course.name,
            "course_current_acrla_mastery": course_score,
            "concept_current_acrla_mastery": {
                concept: {
                    "current_acrla_mastery": value,
                    "fallback_moodle_mastery": value,
                    "source_of_truth": "demo_reset",
                }
                for concept, value in baseline.items()
            },
        })
        course_scores.append(course_score)

    overall = round(sum(course_scores) / len(course_scores), 2) if course_scores else 0.0
    if courses:
        first_course = courses[0]
        course_memory = memory.get_course_memory(student.id, first_course.id)
        initial_map, current_map = _memory_mastery_maps(course_memory)
        initial_map[overall_key] = overall
        current_map[overall_key] = overall
        memory.update_course_memory(student.id, first_course.id, {
            "moodle_initial_mastery": initial_map,
            "current_acrla_mastery": current_map,
        })
    print(
        "[ACRLA] demo_reset_mastery "
        f"student_id={student_id} "
        f"overall_current_acrla_mastery={overall} "
        f"courses={[(row['course_id'], row['course_current_acrla_mastery']) for row in reset_courses]}"
    )

    return {
        "status": "ok",
        "student_id": student_id,
        "overall_current_acrla_mastery": overall,
        "courses": reset_courses,
        "concepts": all_concepts,
    }


@router.post("/session/end")
def end_session(session_id: str, db: DBSession = Depends(get_db)):
    memory = MemoryManager(db)
    memory.end_session(session_id)
    return {"status": "ended", "session_id": session_id}


# ── POST /ingest ──────────────────────────────────────────────────────────────

@router.post("/ingest/{course_id}", response_model=IngestResponse)
async def ingest(
    course_id: int,
    files: list[UploadFile] = File(...),
    db: DBSession = Depends(get_db),
):
    """Upload course documents (PDF/TXT) to ingest into ChromaDB."""
    tmp_dir = tempfile.mkdtemp()
    file_paths = []

    try:
        for upload in files:
            dest = os.path.join(tmp_dir, upload.filename)
            with open(dest, "wb") as f:
                shutil.copyfileobj(upload.file, f)
            file_paths.append(dest)

        result = ingest_documents(course_id, file_paths)
        return IngestResponse(**result)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── GET /analytics/{student_id} ───────────────────────────────────────────────

@router.get("/rag/debug/{course_id}")
def debug_rag_retrieval(
    course_id: int,
    query: str,
    concept: str | None = None,
):
    """Show retrieval diagnostics without calling the LLM."""
    selected_concept = canonicalize_concept(concept) if concept else canonicalize_concept(query)
    context, sources = retrieve_context(course_id, query, selected_concept=selected_concept)
    return {
        "course_id": course_id,
        "query": query,
        "concept_detected": selected_concept,
        "sources": sources,
        "retrieved_chunk_count": context.count("\n\n---\n\n") + 1 if context else 0,
        "context_preview": context[:1200],
    }


@router.get("/rag/collection/{course_id}")
def inspect_rag_collection(course_id: int, limit: int = 100):
    """Inspect source metadata currently stored in a course Chroma collection."""
    vectorstore = get_vectorstore(course_id)
    try:
        count = vectorstore._collection.count()
        raw = vectorstore._collection.get(limit=max(1, min(limit, 500)), include=["metadatas"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to inspect Chroma collection: {exc}") from exc

    files: dict[str, int] = {}
    concepts: dict[str, int] = {}
    document_origins: dict[str, int] = {}
    display_titles: dict[str, int] = {}
    source_labels: dict[str, int] = {}
    for metadata in raw.get("metadatas") or []:
        if not metadata:
            continue
        source = metadata.get("source_file") or metadata.get("source") or "unknown"
        files[source] = files.get(source, 0) + 1
        concept = metadata.get("concept")
        if concept:
            concepts[concept] = concepts.get(concept, 0) + 1
        origin = metadata.get("document_origin") or "unknown"
        document_origins[origin] = document_origins.get(origin, 0) + 1
        display_title = metadata.get("display_title") or "unknown"
        display_titles[display_title] = display_titles.get(display_title, 0) + 1
        origin_for_label = metadata.get("document_origin")
        if origin_for_label and str(origin_for_label).strip().lower() in {"moodle", "bundled", "unknown"}:
            origin_for_label = ""
        label = metadata.get("display_title") or origin_for_label or source
        source_labels[label] = source_labels.get(label, 0) + 1

    return {
        "course_id": course_id,
        "collection": f"course_{course_id}",
        "count": count,
        "sampled": len(raw.get("metadatas") or []),
        "files": files,
        "concepts": concepts,
        "document_origins": document_origins,
        "display_titles": display_titles,
        "source_labels": source_labels,
    }


@router.post("/rag/reset/{course_id}")
def reset_rag(course_id: int):
    """Delete a course Chroma collection."""
    return {"status": "ok", **reset_course_collection(course_id)}


@router.post("/rag/rebuild/{course_id}")
def rebuild_rag(course_id: int, document_dir: str | None = Form(None)):
    """Rebuild a course Chroma collection from current PDFs/TXT files only."""
    result = rebuild_course_collection(course_id, document_dir=document_dir)
    return {"status": "ok", **result}


@router.get("/analytics/{student_id}", response_model=StudentAnalytics)
def get_analytics(student_id: int, course_id: int, db: DBSession = Depends(get_db)):
    student = db.query(Student).filter_by(moodle_user_id=student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    from models.db_models import Course as CourseModel, Session as SessionModel
    course = db.query(CourseModel).filter_by(moodle_course_id=course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    memory = MemoryManager(db)
    all_mastery = [
        record for record in memory.get_all_mastery(student.id, course.id)
        if canonicalize_concept(record.concept)
    ]
    weak = sorted(
        [record for record in all_mastery if record.mastery_level < 0.6],
        key=lambda r: r.mastery_level,
    )
    session_count = db.query(SessionModel).filter_by(student_id=student.id).count()

    from models.db_models import SessionAnalytics, ConversationMessage
    total_messages = db.query(ConversationMessage).join(SessionModel).filter(
        SessionModel.student_id == student.id
    ).count()

    return StudentAnalytics(
        student_id=student_id,
        weak_concepts=[{"concept": r.concept, "mastery": r.mastery_level} for r in weak],
        mastery_by_concept={r.concept: r.mastery_level for r in all_mastery},
        session_count=session_count,
        total_messages=total_messages,
    )
