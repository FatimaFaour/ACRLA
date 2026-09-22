"""
Chat orchestration: routes every message through intent → strategy → pipeline.
This is the central nervous system of ACRLA.
"""

import re
import json
import time
from pathlib import Path
from typing import Any

# ==========================================================
# File Purpose
# ==========================================================
# Coordinates one student chat turn from intent detection to final API reply.
# This service is the backend traffic controller for adaptive tutoring: it
# enforces Moodle launch scope, chooses internal RAG vs external fallback,
# applies mastery-based tutoring strategy, and keeps analytics/preference
# responses separate from document-grounded tutoring.

from langchain.prompts import ChatPromptTemplate
from sqlalchemy.orm import Session as DBSession

from services.intent_classifier import classify_intent, get_strategy_instruction, select_strategy, Intent, Strategy
from services.memory_manager import MemoryManager, get_buffer
from services.strategy_selector import TutoringStrategy, select_tutoring_strategy
from services.llm_factory import get_llm, get_json_llm
from services.course_concepts import (
    ALLOWED_CONCEPTS,
    canonicalize_concept,
    concepts_in_text,
    concepts_for_course,
    display_concept_name,
    sub_concepts_for,
)
from pipelines.rag_pipeline import (
    generate_rag_response,
    retrieve_context_for_scope,
    retrieve_scored_context_for_scope,
    get_vectorstore,
)
from pipelines.hybrid_pipeline import generate_hybrid_response
from models.db_models import Session as SessionModel
from agents.conversation_agent import run_conversation_agent
from agents.simple_agent import run_simple_conversation_agent
from agents.agent_json import compact_json as agent_compact_json, extract_message_text, parse_json_object as agent_parse_json_object
from agents.llm_errors import FINAL_ANSWER_MAX_TOKENS, invoke_with_json_mode_retry, log_llm_provider_error
from agents.debug_log import vprint
from agents import policies
from tools.analytics_tools import (
    plan_analytics_query,
    execute_analytics_query,
    format_analytics_result,
    _normalize_analytics_plan,
)
from tools.memory_tools import get_recent_structured_turns, save_structured_turn
from agents.agent_models import ConversationTurn
from agents.agent_json import model_dump


# ==========================================================
# Constants and Difficulty Rules
# ==========================================================

DIFFICULTY_FORMATS = {
    "easy": "simple multiple-choice question with exactly four options",
    "medium": "open-ended explanation question",
    "hard": "code-writing or algorithm-design question",
}


# ==========================================================
# Scope and Concept Resolution
# ==========================================================

def _normalize_difficulty(difficulty: str | None) -> str:
    """Normalize UI/backend difficulty labels to the three supported levels.

    Moodle and the frontend sometimes use the student-facing word "moderate".
    The backend stores the same level as "medium" so prompts, deterministic
    practice templates, and API responses all agree.
    """
    value = str(difficulty or "medium").strip().lower()
    if value in {"moderate", "mod"}:
        return "medium"
    if value not in DIFFICULTY_FORMATS:
        return "medium"
    return value


def _course_local_concept(raw: str | None) -> str | None:
    """Convert Moodle/resource labels into a stable course-local concept name.

    Built-in demo concepts are canonicalized by `course_concepts.py`. Newly
    synced Moodle courses may contain arbitrary resource titles, so this helper
    also preserves unknown concepts in title case instead of discarding them.
    """
    concept = canonicalize_concept(raw)
    if concept:
        return concept
    text = str(raw or "").replace("_", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    return " ".join(part[:1].upper() + part[1:] for part in text.split())


def _material_concepts_for_course(course) -> list[str]:
    """Load concepts discovered from synced Moodle material for one course.

    The manifest is checked before Chroma because it is written during Moodle
    material sync and is the cleanest record of which resources belong to the
    clicked course. Chroma metadata is only a fallback for older ingestions.
    """
    concepts: list[str] = []
    project_root = Path(__file__).resolve().parents[2]
    manifest_path = project_root / "course_docs" / f"moodle_course_{course.moodle_course_id}" / "materials_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest_data = json.load(manifest_file)
            files = manifest_data.get("files", manifest_data) if isinstance(manifest_data, dict) else {}
            for entry in (files or {}).values():
                if not isinstance(entry, dict):
                    continue
                for raw in (
                    entry.get("concept"),
                    entry.get("display_title"),
                    Path(str(entry.get("original_file_name"))).stem if entry.get("original_file_name") else None,
                ):
                    concept = _course_local_concept(raw)
                    if concept and concept.lower() != "moodle" and concept not in concepts:
                        concepts.append(concept)
                        break
            if concepts:
                vprint(
                    "[ACRLA] chat_material_concepts_manifest "
                    f"course_id={getattr(course, 'moodle_course_id', 'unknown')} "
                    f"course_name={getattr(course, 'name', '')!r} "
                    f"concepts={concepts}"
                )
                return concepts
        except Exception as exc:
            print(f"[ACRLA] chat_manifest_concepts_lookup_failed course_id={getattr(course, 'moodle_course_id', 'unknown')}: {exc}")

    try:
        vectorstore = get_vectorstore(course.moodle_course_id)
        raw = vectorstore._collection.get(limit=1000, include=["metadatas"])
        for metadata in raw.get("metadatas") or []:
            for value in (
                metadata.get("concept"),
                metadata.get("display_title"),
                metadata.get("source_file"),
                metadata.get("document_origin"),
            ):
                if not value:
                    continue
                concept = _course_local_concept(Path(str(value)).stem)
                if concept and concept.lower() != "moodle" and concept not in concepts:
                    concepts.append(concept)
    except Exception as exc:
        print(f"[ACRLA] chat_dynamic_concepts_lookup_failed course_id={getattr(course, 'moodle_course_id', 'unknown')}: {exc}")
    return concepts


def _dynamic_concepts_for_course(
    memory: MemoryManager,
    student_id: str,
    course,
) -> list[str]:
    """Return the course-local concepts that chat is allowed to use.

    The material manifest is authoritative when it exists. This prevents stale
    mastery records from a previous Moodle course from becoming available
    concepts after the student switches courses.
    """
    material = _material_concepts_for_course(course)
    if material:
        material_set = set(material)
        records = [
            concept
            for concept in (_course_local_concept(record.concept) for record in memory.get_all_mastery(student_id, course.id))
            if concept and concept in material_set
        ]
        ordered = list(dict.fromkeys(records + material))
        vprint(
            "[ACRLA] chat_course_scope_concepts "
            f"course_id={getattr(course, 'moodle_course_id', 'unknown')} "
            f"course_name={getattr(course, 'name', '')!r} "
            f"scope_concepts={ordered}"
        )
        return ordered

    records = [
        concept for concept in (_course_local_concept(record.concept) for record in memory.get_all_mastery(student_id, course.id))
        if concept
    ]
    if records:
        vprint(
            "[ACRLA] chat_course_scope_records_fallback "
            f"course_id={getattr(course, 'moodle_course_id', 'unknown')} "
            f"course_name={getattr(course, 'name', '')!r} "
            f"scope_concepts={list(dict.fromkeys(records))}"
        )
        return list(dict.fromkeys(records))
    return []


def _concepts_in_message_from_available(message: str, available_concepts) -> list[str]:
    """Find explicitly named concepts, but only inside the active scope.

    This prevents "test me on pointers" from being accepted while the active
    course is Mathematics, even if Pointers exists in a previous course.
    """
    msg = re.sub(r"\s+", " ", str(message or "").lower())
    matches = []
    for concept in available_concepts or []:
        normalized = re.sub(r"\s+", " ", str(concept).lower()).strip()
        if normalized and re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", msg):
            matches.append(concept)
    return matches


def _difficulty_instruction_block(difficulty: str | None) -> str:
    selected = _normalize_difficulty(difficulty)
    label = "Moderate" if selected == "medium" else selected.capitalize()
    return (
        "BACKEND-CONTROLLED QUESTION DIFFICULTY:\n"
        f"Use the student's selected difficulty: {label}.\n"
        "Do not downgrade the question to Easy because mastery is low.\n"
        "Do not upgrade the question because mastery is high.\n"
        f"Required question format: {DIFFICULTY_FORMATS[selected]}."
    )


_practice_state: dict[str, dict] = {}


def _get_practice_state(session_id: str) -> dict:
    return _practice_state.setdefault(
        session_id,
        {
            "topic": None,
            "counts": {},
            "asked": {},
            "streak": {},
            "awaiting_answer": False,
            "current_question": None,
            "session_memory": {},
        },
    )


def _get_session_memory(session_id: str) -> dict:
    return _get_practice_state(session_id).setdefault("session_memory", {})


def _set_active_question(session_id: str, question: str) -> None:
    state = _get_practice_state(session_id)
    state["awaiting_answer"] = True
    state["current_question"] = question


def _clear_question_state(session_id: str) -> None:
    state = _get_practice_state(session_id)
    state["awaiting_answer"] = False
    state["current_question"] = None


def _awaiting_answer(session_id: str) -> bool:
    state = _practice_state.get(session_id, {})
    return bool(state.get("awaiting_answer") and state.get("current_question"))


def _normalize_learning_mode(session: SessionModel) -> str:
    raw_mode = getattr(session, "learning_mode", None) or getattr(session, "mode", None) or "internal"
    mode = str(raw_mode).strip().lower()
    if mode not in {"internal", "external"}:
        print(
            f"[ACRLA] invalid learning mode for session={getattr(session, 'id', 'unknown')}: "
            f"{raw_mode!r}; defaulting to internal"
        )
        return "internal"
    return mode


# ==========================================================
# Automatic Response Routing
# ==========================================================

SEMANTIC_INTENTS = {
    "greeting",
    "analytics",
    "mastery_policy",
    "mastery_modification_request",
    "course_structure",
    "reference_followup",
    "navigation",
    "preference",
    "practice_request",
    "course_tutoring",
    "general_external",
    "unclear",
}

SEMANTIC_INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You classify one ACRLA student message. Return strict JSON only.
Do not answer the student. Do not update mastery. Do not include markdown.

Allowed intent values:
greeting, analytics, mastery_policy, mastery_modification_request, course_structure,
reference_followup, navigation, preference, practice_request, course_tutoring,
general_external, unclear.

Use these meanings:
- analytics: asks for current mastery/progress/performance/weakest/strongest data.
- mastery_policy: asks what mastery levels mean, such as what counts as weak/low/strong.
- mastery_modification_request: asks to update, improve, bump, mark complete, or change mastery/score.
- course_structure: asks which chapters/concepts/topics exist in the current scope.
- reference_followup: asks about a previous list/reference using wording like those, they, them, their, which one, from what courses, scores, explain them.
- course_tutoring: asks to learn/explain a course concept.
- general_external: asks a factual/casual question unrelated to the course.

Return exactly this JSON shape. Use secondary_intents for compound requests:
{{
  "primary_intent": "one allowed intent",
  "secondary_intents": ["zero or more allowed intents"],
  "is_followup": true/false,
  "refers_to_last_reference": true/false,
  "followup_target": "concept_courses, concept_scores, scoring_methodology, concept_explanations, recommendation, or null",
  "needs_student_data": true/false,
  "needs_course_rag": true/false,
  "requested_concept": string or null,
  "confidence": number
}}"""),
    ("human", """SESSION CONTEXT:
Remediation level: {remediation_level}
Current course: {course_name}
Current topic: {current_topic}
Available concepts: {available_concepts}
Weak concepts: {weak_concepts}
Last structured reference: {last_reference}
Dialogue state: {dialogue_state}
Previous assistant reply summary: {previous_assistant_reply}

CURRENT USER MESSAGE:
{message}"""),
])

def _log_pipeline_selection(
    session_id: str,
    active_mode: str,
    selected_pipeline: str,
    selected_concept: str | None,
    strategy: str,
) -> None:
    """Log the selected response pipeline for a chat turn."""
    print(
        "[ACRLA] "
        f"session_id={session_id} "
        f"active_learning_mode={active_mode} "
        f"selected_pipeline={selected_pipeline} "
        f"selected_concept={selected_concept or 'none'} "
        f"strategy={strategy}"
    )


def _select_strategy_for_turn(
    memory: MemoryManager,
    student_id: str,
    course_id: str,
    selected_concept: str | None,
    fallback_mastery: float,
    difficulty: str,
) -> TutoringStrategy:
    mastery_level = fallback_mastery
    if selected_concept:
        mastery_level = memory.get_mastery(student_id, course_id, selected_concept)
    return select_tutoring_strategy(
        mastery_score=mastery_level,
        difficulty=difficulty,
        concept=selected_concept,
    )


def analyze_user_turn(
    message: str,
    session_context: dict,
    last_reference: dict | None,
) -> dict:
    """Use the LLM as a classifier-only semantic understanding layer.

    The result is allowed to choose an existing deterministic handler, but it
    never generates the final student answer and never mutates mastery. If the
    response is malformed or low confidence, handle_message falls back to the
    older rule-based classifier.
    """
    llm = get_json_llm(temperature=0, max_tokens=300)
    prompt_values = {
        "remediation_level": session_context.get("remediation_level", "chapter"),
        "course_name": session_context.get("course_name", "current Moodle course"),
        "current_topic": session_context.get("current_topic") or "not set",
        "available_concepts": _compact_json(session_context.get("available_concepts", [])),
        "weak_concepts": _compact_json(session_context.get("weak_concepts", [])),
        "last_reference": _compact_json(last_reference or {}),
        "dialogue_state": _compact_json(session_context.get("dialogue_state", {})),
        "previous_assistant_reply": session_context.get("previous_assistant_reply", ""),
        "message": message,
    }
    try:
        response, _json_mode_fallback_used = invoke_with_json_mode_retry(
            SEMANTIC_INTENT_PROMPT, prompt_values, llm=llm, temperature=0, max_tokens=300,
            stage="semantic_intent_analysis",
        )
    except Exception as exc:
        # The call to the LLM provider itself never completed here (or the
        # one non-JSON-mode retry for a json-mode error also failed) -- kept
        # separate from the JSON-parse try/except below so a provider outage
        # is never misreported as "the model returned bad JSON".
        log_llm_provider_error(
            stage="semantic_intent_analysis", exc=exc, prompt_values=prompt_values,
            json_mode_enabled=True, llm=llm, response_length=0,
            final_fallback_reason="semantic_intent_provider_error",
        )
        return _empty_semantic_intent()

    raw_output = extract_message_text(response)
    try:
        return _parse_semantic_intent_json(raw_output)
    except Exception as exc:
        print(f"[ACRLA] semantic_intent_failed error={exc}")
        return _empty_semantic_intent()


def _parse_semantic_intent_json(raw_response) -> dict:
    """Parse and validate the strict JSON returned by the semantic classifier."""
    raw = str(raw_response or "").strip()
    try:
        data = agent_parse_json_object(raw)
    except Exception as exc:
        print(f"[ACRLA] semantic_intent_parse_failed error={exc} raw_output_len={len(raw)} raw_output_preview={raw[:240]!r}")
        raise
    intent = str(data.get("primary_intent") or data.get("intent") or "unclear").strip().lower()
    if intent not in SEMANTIC_INTENTS:
        intent = "unclear"
    secondary_intents = []
    for item in data.get("secondary_intents") or []:
        secondary = str(item or "").strip().lower()
        if secondary in SEMANTIC_INTENTS and secondary not in secondary_intents:
            secondary_intents.append(secondary)
    confidence = data.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "intent": intent,
        "primary_intent": intent,
        "secondary_intents": secondary_intents,
        "is_followup": bool(data.get("is_followup")),
        "refers_to_last_reference": bool(data.get("refers_to_last_reference")),
        "followup_target": data.get("followup_target") if data.get("followup_target") else None,
        "needs_student_data": bool(data.get("needs_student_data")),
        "needs_course_rag": bool(data.get("needs_course_rag")),
        "requested_concept": data.get("requested_concept") if data.get("requested_concept") else None,
        "confidence": confidence,
    }


def _empty_semantic_intent() -> dict:
    return {
        "intent": "unclear",
        "primary_intent": "unclear",
        "secondary_intents": [],
        "is_followup": False,
        "refers_to_last_reference": False,
        "followup_target": None,
        "needs_student_data": False,
        "needs_course_rag": False,
        "requested_concept": None,
        "confidence": 0.0,
    }


def _compact_json(value) -> str:
    """Serialize classifier context compactly so the prompt stays small."""
    return agent_compact_json(value, limit=1800)


def _previous_assistant_reply_summary(session_id: str) -> str:
    """Return a short previous assistant message for semantic follow-up detection."""
    for msg in reversed(get_buffer(session_id).messages):
        if msg.get("role") == "assistant":
            content = re.sub(r"\s+", " ", str(msg.get("content") or "")).strip()
            return content[:500]
    return ""


def _explicit_analytics_entities(message: str, available_courses: list[dict]) -> dict:
    """Extract explicit current-message concepts/courses so stale references lose."""
    msg = str(message or "")
    concepts = _resolve_explicit_concepts_from_message(msg, available_courses)
    courses = []
    for course in available_courses or []:
        course_name = course.get("course_name") or course.get("name") or ""
        if course_name and _text_mentions_label(msg, course_name):
            courses.append(course)
    concepts = list(dict.fromkeys(concepts))
    operation = _explicit_analytics_operation_in_message(msg)
    filters = _explicit_analytics_filters_in_message(msg)
    return {
        "detected": bool(concepts or courses or filters),
        "concepts": concepts,
        "courses": courses,
        "operation": operation,
        "filters": filters,
    }


def _resolve_explicit_concepts_from_message(message: str, available_courses: list[dict]) -> list[str]:
    """Resolve concept aliases in the current message against available concepts."""
    all_concepts = []
    for course in available_courses or []:
        for concept in course.get("concepts") or []:
            if concept not in all_concepts:
                all_concepts.append(concept)

    resolved = []
    for concept in concepts_in_text(message, all_concepts):
        if concept in all_concepts and concept not in resolved:
            resolved.append(concept)

    for concept in all_concepts:
        if concept in resolved:
            continue
        if _text_mentions_label(message, concept):
            resolved.append(concept)
    vprint(
        "[ACRLA] explicit_concept_resolution "
        f"message={message!r} "
        f"available_concepts={all_concepts} "
        f"explicit_concepts={resolved}"
    )
    return resolved


def _text_mentions_label(text: str, label: str) -> bool:
    text_key = _concept_key(text)
    label_key = _concept_key(label)
    if not label_key:
        return False
    if label_key in text_key:
        return True
    # Also allow concise resource labels like "sorting" for "Sorting Algorithms".
    label_tokens = set(label_key.split())
    text_tokens = set(text_key.split())
    text_token_roots = {_singular_token(token) for token in text_tokens}
    generic_tokens = {"algorithms", "management", "functions", "chapter", "chapters"}
    meaningful = {token for token in label_tokens if len(token) > 3 and token not in generic_tokens}
    meaningful_roots = {_singular_token(token) for token in meaningful}
    return bool(meaningful and (meaningful & text_tokens or meaningful_roots & text_token_roots))


def _singular_token(token: str) -> str:
    token = str(token or "").strip().lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _explicit_analytics_operation_in_message(message: str) -> str | None:
    msg = str(message or "").lower()
    if re.search(r"\bcompare|versus| vs \b|difference\b", msg):
        return "compare"
    if re.search(r"\b(lowest|weakest|highest|strongest|top|bottom)\b", msg):
        return "rank"
    if re.search(r"\b(list|show|display|give me)\b", msg):
        return "list"
    return None


def _explicit_analytics_filters_in_message(message: str) -> dict:
    msg = str(message or "").lower()
    filters = {}
    below = re.search(r"\bbelow\s+(\d+(?:\.\d+)?)\s*%?", msg)
    if below:
        filters["current_mastery_lt"] = float(below.group(1)) / 100
    return filters


def _apply_explicit_entities_to_plan(plan: dict, explicit_entities: dict) -> dict:
    """Let current-message entities override last_reference/planner drift."""
    if not explicit_entities.get("detected"):
        return plan
    plan = dict(plan or {})
    filters = dict(plan.get("filters") or {})
    if explicit_entities.get("concepts"):
        filters["concepts"] = explicit_entities["concepts"]
        plan["entity"] = "concept"
        plan["scope"] = "all_courses"
    if explicit_entities.get("filters"):
        filters.update(explicit_entities["filters"])
    if explicit_entities.get("operation"):
        plan["operation"] = explicit_entities["operation"]
    if plan.get("operation") == "compare":
        plan["scope"] = "all_courses"
        plan["entity"] = "concept"
        plan["metrics"] = ["current_mastery"]
    plan["filters"] = filters
    plan["confidence"] = max(float(plan.get("confidence") or 0.0), 0.9)
    return _normalize_analytics_plan(plan)


def _analytics_available_course_context(student_id: str, memory: MemoryManager, db: DBSession) -> list[dict]:
    return [
        {
            "course_id": course["moodle_course_id"],
            "course_name": course["name"],
            "concepts": course["concepts"],
        }
        for course in _canonical_synced_courses(student_id, memory, db)
    ]


def _canonical_synced_courses(student_id: str, memory: MemoryManager, db: DBSession) -> list[dict]:
    """Return deduplicated synced Moodle courses with valid manifest concepts."""
    from models.db_models import Course

    discovered_rows = []
    grouped_rows: dict[str, list[dict]] = {}
    excluded = []
    for course in db.query(Course).all():
        manifest_moodle_id = _manifest_moodle_id_for_course(course)
        concepts = _material_concepts_for_course(course)
        concept_signature = _concept_set_signature(concepts)
        canonical_key = f"name:{_concept_key(course.name)}|concepts:{concept_signature}"
        row = {
            "db_course_id": course.id,
            "moodle_course_id": course.moodle_course_id,
            "manifest_moodle_id": manifest_moodle_id,
            "course_name": course.name,
            "concepts": concepts,
            "concept_signature": concept_signature,
            "canonical_course_key": canonical_key,
            "has_manifest": manifest_moodle_id is not None,
            "created_at": getattr(course, "created_at", None),
        }
        discovered_rows.append(row)
        moodle_id = manifest_moodle_id or course.moodle_course_id
        if moodle_id is None:
            excluded.append((course.name, "missing_moodle_course_id"))
            continue
        if not concepts:
            excluded.append((course.name, moodle_id))
            continue
        grouped_rows.setdefault(canonical_key, []).append(row)

    canonical = []
    merged = []
    for canonical_key, rows in grouped_rows.items():
        selected = _select_canonical_course_row(rows)
        duplicates = [row for row in rows if row["db_course_id"] != selected["db_course_id"]]
        merged.extend((row["course_name"], row["moodle_course_id"], row["db_course_id"]) for row in duplicates)
        canonical.append({
            "db_course_id": selected["db_course_id"],
            "moodle_course_id": selected["manifest_moodle_id"] or selected["moodle_course_id"],
            "name": selected["course_name"],
            "concepts": selected["concepts"],
            "merged_db_course_ids": [row["db_course_id"] for row in rows],
            "canonical_course_key": canonical_key,
        })
    canonical = sorted(canonical, key=lambda item: item["moodle_course_id"] or 0)
    _assert_no_duplicate_canonical_courses(canonical)
    vprint(
        "[ACRLA] analytics_courses "
        f"discovered_course_rows={discovered_rows} "
        f"canonical_course_key={[course['canonical_course_key'] for course in canonical]} "
        f"selected_canonical_course_row={[(course['name'], course['db_course_id']) for course in canonical]} "
        f"canonical_course_ids={[course['moodle_course_id'] for course in canonical]} "
        f"merged_duplicate_rows={merged} "
        f"excluded_stale_rows={excluded}"
    )
    return canonical


def _manifest_moodle_id_for_course(course) -> int | None:
    project_root = Path(__file__).resolve().parents[2]
    docs_dir = project_root / "course_docs" / f"moodle_course_{course.moodle_course_id}"
    manifest_path = docs_dir / "materials_manifest.json"
    if manifest_path.exists():
        match = re.search(r"moodle_course_(\d+)$", docs_dir.name)
        if match:
            return int(match.group(1))
    return None


def _concept_set_signature(concepts: list[str]) -> str:
    return ",".join(sorted(_concept_key(concept) for concept in concepts or []))


def _select_canonical_course_row(rows: list[dict]) -> dict:
    """Prefer manifest-backed synced rows over stale legacy duplicates."""
    return sorted(
        rows,
        key=lambda row: (
            0 if row.get("has_manifest") else 1,
            0 if row.get("manifest_moodle_id") == row.get("moodle_course_id") else 1,
            -_timestamp_for_sort(row.get("created_at")),
            str(row.get("db_course_id") or ""),
        ),
    )[0]


def _timestamp_for_sort(value) -> float:
    try:
        return float(value.timestamp())
    except Exception:
        return 0.0


def _assert_no_duplicate_canonical_courses(canonical: list[dict]) -> None:
    seen = {}
    duplicates = []
    for course in canonical:
        key = (_concept_key(course.get("name")), _concept_set_signature(course.get("concepts") or []))
        if key in seen:
            duplicates.append((seen[key], course))
        else:
            seen[key] = course
    if duplicates:
        print(f"[ACRLA] analytics_canonical_duplicate_error duplicates={duplicates}")
        raise ValueError("Duplicate canonical courses with identical name and concept set remain after deduplication.")


def _recent_messages_for_agent(session_id: str, limit: int = 10) -> list[dict]:
    """Return recent chat turns in the compact shape expected by the agent."""
    return [
        {"role": item.get("role"), "content": item.get("content")}
        for item in get_buffer(session_id).last_n(limit)
        if item.get("role") and item.get("content")
    ]


def _agent_course_context_by_db_id(student_id: str, memory: MemoryManager, db: DBSession) -> dict[str, dict]:
    """Expose canonical course/concept context to read-only agent tools."""
    return {
        course["db_course_id"]: course
        for course in _canonical_synced_courses(student_id, memory, db)
    }


def _proactive_bootstrap_payload(tutor_bootstrap: dict | None) -> dict | None:
    """Map services.remediation_bootstrap's internal decision dict to the
    external chat-response shape the frontend's one-time "focus card" reads
    -- pure relabeling/renaming for the API boundary, never touches
    selection logic (services.remediation_bootstrap computes every value
    here; this only picks which keys are public and what they're called).
    None whenever no bootstrap happened this turn, which is every turn
    except the one that started it."""
    if not tutor_bootstrap:
        return None
    return {
        "active": True,
        "level": tutor_bootstrap.get("level"),
        "concept": display_concept_name(tutor_bootstrap.get("concept")),
        "concept_mastery": tutor_bootstrap.get("mastery"),
        "course_name": tutor_bootstrap.get("course_name"),
        "course_mastery": tutor_bootstrap.get("course_mastery"),
        "selection_reason": tutor_bootstrap.get("selection_reason"),
        "tutor_state": tutor_bootstrap.get("initial_state"),
    }


def _session_scoped_quick_progress_check(memory: MemoryManager, student_id: str, course_id: str, session_id: str) -> dict:
    """Read the active Quick Progress Check state (tools.assessment_tools),
    treating a different/stale session's leftover state as absent -- same
    session-scoping precedent as services.tutor_state_machine.load_tutor_state."""
    state = memory.get_quick_progress_check(student_id, course_id)
    if state.get("session_id") != session_id:
        return {}
    return state


def _build_agent_context(
    *,
    session_id: str,
    message: str,
    session: SessionModel,
    student,
    memory: MemoryManager,
    db: DBSession,
    remediation_level: str,
    course_moodle_id: int,
    available_concepts: set[str],
    weak_concepts: list[str],
    current_topic: str | None,
    last_reference: dict | None,
    dialogue_state: dict,
    strategy,
) -> dict:
    """Build the read-only context packet for the LLM-orchestrated agent."""
    from services.tutor_state_machine import load_tutor_state

    current_course = {
        "db_course_id": session.course_id,
        "moodle_course_id": course_moodle_id,
        "name": getattr(getattr(session, "course", None), "name", "") or "current Moodle course",
    }
    retrieval_course_ids = _retrieval_course_ids_for_level(remediation_level, course_moodle_id, db)
    return {
        "session_id": session_id,
        "message": message,
        "student_id": student.id,
        "student_name": student.username,
        "course_db_id": session.course_id,
        "current_course": current_course,
        "current_concept": current_topic,
        "tutor_state": load_tutor_state(memory, student.id, session.course_id, session_id),
        "quick_progress_check": _session_scoped_quick_progress_check(memory, student.id, session.course_id, session_id),
        "remediation_level": remediation_level,
        "available_concepts": sorted(available_concepts),
        "weak_concepts": weak_concepts,
        "last_reference": last_reference or {},
        "last_answer_type": dialogue_state.get("last_answer_type"),
        "last_discussed_metric": dialogue_state.get("last_discussed_metric"),
        "difficulty": session.difficulty,
        "tutoring_strategy": {
            "name": getattr(strategy, "name", str(strategy)),
            "reason": getattr(strategy, "reason", ""),
            # RQ1 fix: `strategy` here is a services.intent_classifier.Strategy
            # enum member (from select_strategy, the mastery-aware classifier
            # this active path actually uses) -- it has no .prompt_block
            # attribute (that belongs to the unrelated, legacy-path-only
            # services.strategy_selector.TutoringStrategy dataclass), so the
            # previous getattr(..., "") silently dropped the mastery-band
            # teaching instructions before they ever reached the prompt.
            # get_strategy_instruction is the existing, correct accessor for
            # this exact enum (services/intent_classifier.py) -- no new
            # strategy system, no duplication of strategy_selector.py.
            "instructions": get_strategy_instruction(strategy),
        },
        "recent_messages": _recent_messages_for_agent(session_id),
        "recent_structured_turns": get_recent_structured_turns(session_id, limit=6),
        "retrieval_course_ids": retrieval_course_ids,
        "scope_rules": _remediation_scope_instruction(remediation_level, available_concepts),
        "course_context_by_db_id": _agent_course_context_by_db_id(student.id, memory, db),
        "canonical_courses": _canonical_synced_courses(student.id, memory, db),
        "memory": memory,
        "db": db,
    }


_agent_usage_metrics: dict[str, int] = {"agent_used": 0, "legacy_fallback": 0, "blocked": 0}


def _record_agent_usage_metric(outcome: str) -> None:
    """In-memory counter for how often the new agent actually handles a turn
    vs. falls back to legacy routing (or never runs at all because a safety
    guard blocked it). No external metrics backend is wired up in this repo,
    so this is logged every turn -- `legacy_fallback_rate` can be aggregated
    straight from logs. Resets on process restart; that is acceptable for a
    "how often is the fallback used" signal during migration.
    """
    _agent_usage_metrics[outcome] = _agent_usage_metrics.get(outcome, 0) + 1
    total = sum(_agent_usage_metrics.values())
    non_agent = _agent_usage_metrics.get("legacy_fallback", 0) + _agent_usage_metrics.get("blocked", 0)
    print(
        "[ACRLA] agent_usage_metrics "
        f"outcome={outcome} "
        f"agent_used_count={_agent_usage_metrics.get('agent_used', 0)} "
        f"legacy_fallback_count={_agent_usage_metrics.get('legacy_fallback', 0)} "
        f"blocked_count={_agent_usage_metrics.get('blocked', 0)} "
        f"total_turns={total} "
        f"legacy_fallback_rate={(non_agent / total) if total else 0.0:.2f}"
    )


def _try_conversation_agent(context: dict) -> dict:
    """Run the new agent layer -- the primary conversational path.

    Always returns a dict with a `success` key so the caller can branch with
    `agent_result.get("success")` (per the handle_message/legacy_handle_message
    migration flow) without needing a separate None check. Pre-agent blocking
    (safety guards, name capture/recall) happens in the caller before this is
    ever invoked -- this function itself never decides to skip the agent.

    `ACRLA_AGENT_MODE` picks which agent architecture actually runs this
    turn: "simple" (default -- one planner call, execute everything the plan
    needs, one final-answer call, see agents.simple_agent) or "iterative"
    (the original agent_brain.decide() -> tool -> observe loop, kept as a
    fallback behind this flag until the simple path is fully validated).
    Both return the same `AgentResult` contract, so nothing downstream of
    this function needs to know or care which one ran.
    """
    from config import get_settings

    agent_mode = str(get_settings().acrla_agent_mode or "simple").strip().lower()
    turn_started = time.monotonic()
    if agent_mode == "iterative":
        result = run_conversation_agent(context)
    else:
        result = run_simple_conversation_agent(context)
    turn_latency_ms = (time.monotonic() - turn_started) * 1000
    print(
        "[ACRLA] turn_usage "
        f"agent_mode={result.agent_mode} "
        f"goal={result.goal} "
        f"planner_calls={result.planner_call_count} "
        f"response_calls={result.response_call_count} "
        f"total_llm_calls={result.total_llm_call_count} "
        f"planner_prompt_tokens={result.planner_prompt_tokens} "
        f"planner_completion_tokens={result.planner_completion_tokens} "
        f"response_prompt_tokens={result.response_prompt_tokens} "
        f"response_completion_tokens={result.response_completion_tokens} "
        f"total_tokens={result.total_tokens} "
        f"tools={result.tools_executed} "
        f"optional_replan_used={result.optional_replan_used} "
        f"selected_pipeline={result.selected_pipeline} "
        f"latency_ms={turn_latency_ms:.1f}"
    )
    print(
        "[ACRLA] conversation_agent_result "
        f"agent_mode={result.agent_mode} "
        f"agent_goal={result.goal} "
        f"agent_success={result.agent_used} "
        f"agent_confidence={result.confidence:.2f} "
        f"tool_requested={result.tools_requested} "
        f"evidence_reliable={result.evidence_reliable} "
        f"evidence_coverage={result.evidence_coverage} "
        f"answer_basis={result.reasoning_basis} "
        f"final_selected_pipeline={result.selected_pipeline} "
        f"legacy_fallback_reason={result.fallback_reason or 'none'} "
        f"final_selected_handler={'conversation_agent' if result.agent_used else 'legacy_orchestrator'} "
        f"planner_call_count={result.planner_call_count} "
        f"response_call_count={result.response_call_count} "
        f"total_llm_call_count={result.total_llm_call_count} "
        f"total_tokens={result.total_tokens} "
        f"optional_replan_used={result.optional_replan_used}"
    )
    for step in result.steps:
        print(
            "[ACRLA] agent_step "
            f"agent_iteration={step.get('step')} "
            f"phase={step.get('phase')} "
            f"agent_decision_type={step.get('decision_type', step.get('planner_decision', 'n/a'))} "
            f"tool_requested={step.get('tool_requested', 'n/a')} "
            f"tool_success={step.get('tool_success', 'n/a')} "
            f"tool_result_summary={step.get('tool_result_summary', 'n/a')} "
            f"agent_goal={step.get('agent_goal', step.get('final_goal', 'n/a'))} "
            f"selected_pipeline={step.get('selected_pipeline', 'n/a')}"
        )
    _record_agent_usage_metric("agent_used" if result.agent_used else "legacy_fallback")
    payload = result.dict() if hasattr(result, "dict") else result.model_dump()
    payload["success"] = result.agent_used
    return payload


def _is_source_provenance_question(message: str) -> bool:
    """Detect questions about whether the previous answer used PDFs/RAG/fallback."""
    msg = re.sub(r"\s+", " ", str(message or "").lower()).strip(" ?.!") 
    asks_source = bool(re.search(r"\b(source|sources|pdf|material|course material|came from|come from|using|used|grounded|internal|external|rag|fallback|knowledge)\b", msg))
    refers_answer = bool(re.search(r"\b(this|that|it|answer|response|explanation|you)\b", msg))
    return asks_source and refers_answer


def _store_last_response_metadata(
    session_id: str,
    *,
    selected_pipeline: str | None,
    sources: list[str],
    agent_goal: str | None,
    concepts: list[str],
    resolved_concepts: list[str] | None = None,
    evidence_reliable: bool | None = None,
    evidence_reason: str | None = None,
) -> None:
    metadata = {
        "selected_pipeline": selected_pipeline or "deterministic",
        "sources": list(dict.fromkeys(sources or [])),
        "agent_goal": agent_goal,
        "concepts": list(dict.fromkeys(concepts or [])),
        "resolved_concepts": list(dict.fromkeys(resolved_concepts or [])),
        "evidence_reliable": evidence_reliable,
        "evidence_reason": evidence_reason,
    }
    _get_session_memory(session_id)["last_response_metadata"] = metadata
    print(f"[ACRLA] last_response_metadata={metadata}")


def _get_last_response_metadata(session_id: str) -> dict:
    metadata = _get_session_memory(session_id).get("last_response_metadata") if session_id else None
    return metadata if isinstance(metadata, dict) else {}


def _handle_source_provenance_question(session_id: str) -> str:
    # Wording now lives in agents.policies (also reachable as the
    # get_source_provenance agent tool) so both paths stay identical.
    return policies.source_provenance_response(_get_last_response_metadata(session_id))


def _metadata_concepts_for_turn(
    message: str,
    available_concepts: set[str],
    agent_result: dict | None,
) -> list[str]:
    if agent_result and agent_result.get("concepts"):
        return list(agent_result.get("concepts") or [])
    concepts = []
    for concept in concepts_in_text(message, available_concepts):
        if concept not in concepts:
            concepts.append(concept)
    return concepts


def _save_structured_turn_for_response(
    session_id: str,
    *,
    message: str,
    reply: str,
    selected_pipeline: str | None,
    sources: list[str],
    last_reference: dict | None,
    available_concepts: set[str],
    agent_result: dict | None,
) -> None:
    """Persist one ConversationTurn so later follow-ups can read structured state.

    This backs "which one should I start with?", "why these?", and "what
    source did you use?" style follow-ups (see tools/memory_tools.py) without
    needing to re-derive that context from raw text each time.
    """
    resolved_concepts = _metadata_concepts_for_turn(message, available_concepts, agent_result)
    observations: list[dict[str, Any]] = []
    if agent_result:
        for step in agent_result.get("steps") or []:
            if step.get("phase") == "tool_execution":
                observations.append({
                    "tool": step.get("tool_requested"),
                    "result_summary": step.get("tool_result_summary") or {},
                    "success": step.get("tool_success"),
                })
    goal = (agent_result or {}).get("goal") or "unclear"
    comparison = (
        {"concepts": resolved_concepts, "basis": "concept_comparison"}
        if goal == "concept_comparison" and len(resolved_concepts) >= 2
        else None
    )
    turn = ConversationTurn(
        user_message=message,
        assistant_reply=reply,
        goal=goal,
        resolved_entities={"concepts": resolved_concepts, "courses": [], "metrics": []},
        references=last_reference or {},
        tools_used=(agent_result or {}).get("tools_executed") or [],
        observations=observations,
        answer_basis=(agent_result or {}).get("reasoning_basis"),
        selected_pipeline=selected_pipeline,
        sources=list(dict.fromkeys(sources or [])),
        evidence={
            "reliable": (agent_result or {}).get("evidence_reliable"),
            "coverage": (agent_result or {}).get("evidence_coverage"),
            "reason": (agent_result or {}).get("evidence_reason"),
        },
        recommendation=(agent_result or {}).get("recommendation"),
        recommendation_reason=(agent_result or {}).get("recommendation_reason"),
        comparison=comparison,
        analytics_request=(agent_result or {}).get("analytics_request"),
    )
    save_structured_turn(session_id, model_dump(turn))


# ==========================================================
# One Chat Turn: End-to-End Flow
# ==========================================================
# handle_message() (below) is the thin primary-path entry point: deterministic
# safety/scope checks, then the iterative conversation agent
# (agents.conversation_agent.run_conversation_agent). Only when the agent does
# not produce a usable result does it call legacy_handle_message() -- the
# original rule-based pipeline, preserved unchanged as the fallback:
#
# 1. Load the active DB session and student/course identity.
# 2. Resolve the Moodle launch level (chapter, course, overall).
# 3. Build `available_concepts` from that launch scope so previous courses
#    cannot leak into the current remediation session.
# 4. Classify the user message as analytics, preference, navigation, greeting,
#    engagement, or tutoring.
# 5. Handle analytics/preferences/navigation deterministically when possible.
#    These paths bypass RAG and do not display PDF sources.
# 6. For tutoring messages, resolve the current/launch/explicit concept.
# 7. Probe Chroma with the raw current message to decide whether internal RAG is
#    reliable. The contextual query is used later for generation only.
# 8. If reliable material exists, call the internal RAG pipeline and show
#    sources. Otherwise, call the fallback pipeline and clear sources.
# 9. Save messages for continuity, but force mastery_update=None. Chatting and
#    practice questions prepare the student; Quick Progress Check is the only
#    MVP component that changes mastery.

# agent_result["fallback_reason"] values that mean an LLM call itself never
# completed due to a provider-level condition -- never a low-confidence/
# evidence-gap outcome the legacy orchestrator could meaningfully redo.
# "decision_provider_error": the semantic planner call itself failed (see
# agents.simple_agent's early return on a simple_planner.plan() exception).
# "final_answer_provider_error": the plan succeeded but the final-answer
# call then failed with a provider-level error and no deterministic
# grounded fallback was available (see agents.simple_agent._finalize_answer/
# _finalize_clarification and agents.response_generator's
# provider_error_category signal). Either way, legacy_handle_message would
# very likely hit the SAME failing provider again in its own semantic-
# classification stage -- see the handling below.
_PROVIDER_UNAVAILABLE_FALLBACK_REASONS = {"decision_provider_error", "final_answer_provider_error"}


def handle_message(
    session_id: str,
    student_moodle_id: int,
    message: str,
    db: DBSession,
) -> dict:
    """
    Primary conversational path for one student turn.

        handle_message
        -> load session/scope context
        -> deterministic safety guards
        -> build agent context
        -> run the iterative conversation agent (agent_brain -> tool ->
           observation -> agent_brain -> ... -> answer)
        -> if the agent produced a usable result, return it
        -> otherwise, legacy_handle_message() (rule-based fallback)

    Only deterministic safety/integrity checks and the session/scope/profile
    data every path (agent or legacy) needs are resolved here -- NOT
    conversational intent classification. classify_intent, analyze_user_turn,
    and the large rule-based routing chain live only in legacy_handle_message
    and run only when the agent could not handle the turn.
    """
    memory = MemoryManager(db)

    session: SessionModel = memory.get_active_session(session_id)
    if not session:
        return {
            "reply": "Your session has expired. Please start a new session from Moodle.",
            "intent": "error",
            "strategy": "none",
        }

    student = session.student
    course_id_str = session.course_id
    db.refresh(session)

    # ── Deterministic safety guards (must run before the agent) ─────────────
    early_guard_reply = _run_safety_guards(message)
    if early_guard_reply is not None:
        memory.save_message(session_id, "user", message, intent="mastery_modification_request")
        memory.save_message(session_id, "assistant", early_guard_reply, intent="mastery_modification_request", strategy="mastery_update_guard")
        _record_agent_usage_metric("blocked")
        _log_primary_path_result(
            primary_path="blocked", agent_attempted=False, agent_success=False,
            agent_failure_reason=None, legacy_fallback_used=False, legacy_fallback_reason=None,
            pre_agent_safety_guard="mastery_modification_request", agent_latency_ms=0.0, legacy_latency_ms=0.0,
        )
        return {
            "reply": early_guard_reply,
            "intent": "mastery_modification_request",
            "strategy": "mastery_update_guard",
            "mastery_update": None,
            "difficulty": session.difficulty,
            "mode": "automatic",
            "learning_mode": "automatic",
            # RQ1.B root-cause fix (see rq1b_root_cause_analysis.md, Root
            # cause 3): this pre-agent guard's outcome is an intentional,
            # deterministic reply -- the same conceptual outcome
            # agents.plan_compiler's own mastery_modification_request ->
            # get_mastery_guard_response mapping produces when it is the one
            # that ends up handling this goal -- so it is tagged with the
            # same "deterministic_reply" pipeline value that path already
            # uses, instead of None, which was indistinguishable from an
            # actual execution failure to any consumer reading this field.
            # No safety/routing behavior changes; this is observability only.
            "selected_pipeline": "deterministic_reply",
            "source_mode": None,
            "sources": [],
        }

    # A name-capture/recall turn writes/reads session+profile state that has
    # no corresponding agent tool (the agent's tool registry is read-only --
    # see agents/agent_tools.py). Keeping this deterministic and pre-agent
    # avoids the agent silently "answering" a name update without ever
    # persisting it -- the same reasoning as the mastery guard above, just
    # for a profile write instead of a mastery write.
    name_update = _extract_session_name_update(message)
    if name_update:
        _get_session_memory(session_id)["name"] = name_update
        memory.set_profile_name(student.id, name_update)
        reply = f"Nice to meet you, {name_update}. I'll remember your name during this session."
        memory.save_message(session_id, "user", message, intent="preference")
        memory.save_message(session_id, "assistant", reply, intent="preference", strategy="name_capture")
        _record_agent_usage_metric("blocked")
        _log_primary_path_result(
            primary_path="blocked", agent_attempted=False, agent_success=False,
            agent_failure_reason=None, legacy_fallback_used=False, legacy_fallback_reason=None,
            pre_agent_safety_guard="name_capture", agent_latency_ms=0.0, legacy_latency_ms=0.0,
        )
        return {
            "reply": reply,
            "intent": "preference",
            "strategy": "name_capture",
            "mastery_update": None,
            "difficulty": session.difficulty,
            "mode": "automatic",
            "learning_mode": "automatic",
            "selected_pipeline": None,
            "source_mode": None,
            "sources": [],
        }

    if _is_name_recall_question(message):
        remembered_name = _get_session_memory(session_id).get("name") or memory.get_profile_name(student.id)
        reply = f"Your name is {remembered_name}." if remembered_name else "I don't know your name yet in this session."
        memory.save_message(session_id, "user", message, intent="preference")
        memory.save_message(session_id, "assistant", reply, intent="preference", strategy="name_recall")
        _record_agent_usage_metric("blocked")
        _log_primary_path_result(
            primary_path="blocked", agent_attempted=False, agent_success=False,
            agent_failure_reason=None, legacy_fallback_used=False, legacy_fallback_reason=None,
            pre_agent_safety_guard="name_recall", agent_latency_ms=0.0, legacy_latency_ms=0.0,
        )
        return {
            "reply": reply,
            "intent": "preference",
            "strategy": "name_recall",
            "mastery_update": None,
            "difficulty": session.difficulty,
            "mode": "automatic",
            "learning_mode": "automatic",
            "selected_pipeline": None,
            "source_mode": None,
            "sources": [],
        }

    # ── Required scope/profile context (deterministic; not conversational
    #    routing) -- the Moodle launch-scope guardrail every path, agent or
    #    legacy, must respect ───────────────────────────────────────────────
    course_moodle_id = _get_moodle_course_id(session, db)
    remediation_level = _launch_level_for_session(memory, student.id, course_id_str)
    available_concepts = set(_concepts_for_level(remediation_level, course_moodle_id, db, memory, student.id))
    avg_mastery = memory.get_average_mastery(student.id, course_id_str)
    weak_concepts = [
        concept for concept in memory.get_weak_concepts(student.id, course_id_str)
        if (_course_local_concept(concept) or concept) in available_concepts
    ]
    recent_errors = max(0, int((1 - avg_mastery) * 3))
    is_confused = any(w in message.lower() for w in ["confused", "don't understand", "lost", "stuck"])
    in_recovery = avg_mastery < 0.2 and recent_errors >= 2
    strategy = select_strategy(
        mastery_level=avg_mastery,
        recent_errors=recent_errors,
        is_confused=is_confused,
        is_exam_mode="exam" in (session.learning_goal or "").lower(),
        in_recovery=in_recovery,
    )
    current_topic = _current_topic_for_session(memory, student.id, course_id_str, session_id)
    last_reference = _get_last_reference(session_id)
    dialogue_state = _get_dialogue_state(session_id)

    # ── Primary path: the iterative conversation agent ───────────────────────
    agent_started = time.monotonic()
    agent_result = _try_conversation_agent(_build_agent_context(
        session_id=session_id,
        message=message,
        session=session,
        student=student,
        memory=memory,
        db=db,
        remediation_level=remediation_level,
        course_moodle_id=course_moodle_id,
        available_concepts=available_concepts,
        weak_concepts=weak_concepts,
        current_topic=current_topic,
        last_reference=last_reference,
        dialogue_state=dialogue_state,
        strategy=strategy,
    ))
    agent_latency_ms = (time.monotonic() - agent_started) * 1000

    if agent_result.get("success"):
        reply = agent_result.get("reply", "")
        sources = agent_result.get("sources", [])
        retrieved_sources = list(sources)
        selected_pipeline = agent_result.get("selected_pipeline")
        response_strategy = "conversation_agent"
        final_selected_intent = "conversation_agent"
        intent_label = agent_result.get("goal") or "conversation_agent"

        if selected_pipeline == "internal_rag":
            sources = sources or retrieved_sources
            retrieved_sources = sources

        if agent_result.get("analytics_items"):
            # goal/analytics_request/resolved_concepts are stored alongside the
            # rows themselves so a LATER turn's reuse decision
            # (_reference_reuse_decision) can check semantic compatibility
            # against what THIS turn was actually about, instead of only
            # having the raw rows to go on.
            _store_last_reference(session_id, {
                "type": "analytics_result",
                "operation": agent_result.get("analytics_operation") or agent_result.get("goal"),
                "items": agent_result.get("analytics_items") or [],
                "source": "conversation_agent",
                "goal": agent_result.get("goal"),
                "analytics_request": agent_result.get("analytics_request"),
                "resolved_concepts": agent_result.get("resolved_concepts") or [],
            })

        _store_last_response_metadata(
            session_id,
            selected_pipeline=selected_pipeline,
            sources=sources,
            agent_goal=agent_result.get("goal"),
            concepts=_metadata_concepts_for_turn(message, available_concepts, agent_result),
            resolved_concepts=agent_result.get("resolved_concepts"),
            evidence_reliable=agent_result.get("evidence_reliable"),
            evidence_reason=agent_result.get("evidence_reason"),
        )
        _save_structured_turn_for_response(
            session_id=session_id,
            message=message,
            reply=reply,
            selected_pipeline=selected_pipeline,
            sources=sources,
            last_reference=last_reference,
            available_concepts=available_concepts,
            agent_result=agent_result,
        )
        _store_dialogue_state(session_id, {
            "last_user_intent": final_selected_intent,
            "last_assistant_intent": response_strategy,
            "last_answer_type": _answer_type_for_response_strategy(response_strategy, _get_dialogue_state(session_id)),
            "last_discussed_metric": None,
        })
        memory.save_message(session_id, "user", message, intent=intent_label)
        memory.save_message(session_id, "assistant", reply, intent=intent_label, strategy=response_strategy)

        _log_primary_path_result(
            primary_path="agent", agent_attempted=True, agent_success=True,
            agent_failure_reason=None, legacy_fallback_used=False, legacy_fallback_reason=None,
            pre_agent_safety_guard=None, agent_latency_ms=agent_latency_ms, legacy_latency_ms=0.0,
        )

        return {
            "reply": reply,
            "intent": intent_label,
            "strategy": response_strategy,
            "mastery_update": None,
            "difficulty": session.difficulty,
            "mode": "automatic",
            "learning_mode": "automatic",
            "selected_pipeline": selected_pipeline,
            "source_mode": selected_pipeline,
            "retrieved_sources": retrieved_sources if selected_pipeline == "internal_rag" else [],
            "sources": sources,
            "agent_used": True,
            "agent_goal": agent_result.get("goal"),
            "agent_tools_used": agent_result.get("tools_executed", []),
            "agent_knowledge_strategy": agent_result.get("knowledge_strategy"),
            "agent_resolved_concepts": agent_result.get("resolved_concepts", []),
            "evidence_reliable": agent_result.get("evidence_reliable"),
            "evidence_reason": agent_result.get("evidence_reason"),
            "recommendation": agent_result.get("recommendation"),
            "recommendation_reason": agent_result.get("recommendation_reason"),
            "agent_fallback_reason": agent_result.get("fallback_reason"),
            "tutor_state": agent_result.get("tutor_state"),
            "proactive_bootstrap": _proactive_bootstrap_payload(agent_result.get("tutor_bootstrap")),
        }

    # ── Legacy Conversation Fallback ──────────────────────────────────────────
    # Only reached when the agent could not produce a usable result this turn
    # (low confidence, step limit without sufficient evidence, tool failure,
    # or empty final answer -- see run_conversation_agent). This is the only
    # place classify_intent / analyze_user_turn / the rule-based routing
    # chain still run.
    agent_failure_reason = agent_result.get("fallback_reason") or "unknown"
    current_goal = agent_result.get("goal")
    current_analytics_request = agent_result.get("analytics_request")
    current_resolved_concepts = agent_result.get("resolved_concepts") or []
    reuse_allowed, reuse_reason = _reference_reuse_decision(
        current_goal=current_goal, current_analytics_request=current_analytics_request,
        current_resolved_concepts=current_resolved_concepts, last_reference=last_reference,
        failure_reason=agent_failure_reason,
    )
    print(
        "[ACRLA] reference_reuse_check "
        f"current_goal={current_goal} previous_goal={(last_reference or {}).get('goal')} "
        f"reference_reuse_allowed={reuse_allowed} reference_reuse_reason={reuse_reason} "
        f"current_entities={current_resolved_concepts} reused_entities={(last_reference or {}).get('resolved_concepts')}"
    )
    deterministic_reference_reply = _answer_from_last_analytics_reference(last_reference) if reuse_allowed else ""
    if deterministic_reference_reply:
        memory.save_message(session_id, "user", message, intent="analytics_query")
        memory.save_message(session_id, "assistant", deterministic_reference_reply, intent="analytics_query", strategy="analytics_reference_fallback")
        _log_primary_path_result(
            primary_path="agent_reference_fallback", agent_attempted=True, agent_success=False,
            agent_failure_reason=agent_failure_reason, legacy_fallback_used=False, legacy_fallback_reason=None,
            pre_agent_safety_guard=None, agent_latency_ms=agent_latency_ms, legacy_latency_ms=0.0,
        )
        return {
            "reply": deterministic_reference_reply,
            "intent": "analytics_query",
            "strategy": "analytics_reference_fallback",
            "mastery_update": None,
            "difficulty": session.difficulty,
            "mode": "automatic",
            "learning_mode": "automatic",
            "selected_pipeline": "agent_tools",
            "source_mode": "agent_tools",
            "retrieved_sources": [],
            "sources": [],
            "agent_used": False,
            "agent_goal": agent_result.get("goal"),
            "agent_tools_used": agent_result.get("tools_executed", []),
            "agent_knowledge_strategy": "analytics",
            "agent_resolved_concepts": agent_result.get("resolved_concepts", []),
            "evidence_reliable": None,
            "evidence_reason": "reused_last_authoritative_analytics_reference",
            "agent_fallback_reason": agent_failure_reason,
        }

    if agent_failure_reason in _PROVIDER_UNAVAILABLE_FALLBACK_REASONS:
        # A genuine provider-level failure this turn -- either the semantic
        # planner call itself never completed ("decision_provider_error"),
        # or it succeeded but the final-answer call then failed with a
        # provider-level error and no deterministic grounded fallback was
        # available ("final_answer_provider_error"; see
        # agents.simple_agent._finalize_answer/_finalize_clarification and
        # agents.response_generator's provider_error_category signal -- when
        # a grounded fallback WAS possible, e.g. internal_rag with reliable
        # retrieved chunks, the agent already returned a normal successful
        # reply and never reaches this branch at all). Either way, falling
        # through to legacy_handle_message would very likely hit the SAME
        # failing provider again in its own semantic-classification stages
        # (classify_intent/analyze_user_turn), and reuse was just ruled out
        # above as unsafe (unrelated goal/entities, or nothing to reuse at
        # all). Say so plainly instead of guessing from unrelated stale
        # state or attempting another LLM-dependent path.
        provider_unavailable_reply = (
            "I'm temporarily unable to reach the AI service right now. Please try again in a moment."
        )
        memory.save_message(session_id, "user", message, intent="provider_unavailable")
        memory.save_message(session_id, "assistant", provider_unavailable_reply, intent="provider_unavailable", strategy="provider_unavailable")
        _log_primary_path_result(
            primary_path="provider_unavailable", agent_attempted=True, agent_success=False,
            agent_failure_reason=agent_failure_reason, legacy_fallback_used=False, legacy_fallback_reason=None,
            pre_agent_safety_guard=None, agent_latency_ms=agent_latency_ms, legacy_latency_ms=0.0,
        )
        return {
            "reply": provider_unavailable_reply,
            "intent": "provider_unavailable",
            "strategy": "provider_unavailable",
            "mastery_update": None,
            "difficulty": session.difficulty,
            "mode": "automatic",
            "learning_mode": "automatic",
            "selected_pipeline": None,
            "source_mode": None,
            "retrieved_sources": [],
            "sources": [],
            "agent_used": False,
            "agent_goal": agent_result.get("goal"),
            "agent_tools_used": agent_result.get("tools_executed", []),
            "agent_knowledge_strategy": None,
            "agent_resolved_concepts": agent_result.get("resolved_concepts", []),
            "evidence_reliable": None,
            "evidence_reason": "provider_unavailable",
            "agent_fallback_reason": agent_failure_reason,
        }

    legacy_started = time.monotonic()
    legacy_result = legacy_handle_message(session_id, student_moodle_id, message, db)
    legacy_latency_ms = (time.monotonic() - legacy_started) * 1000
    _log_primary_path_result(
        primary_path="legacy", agent_attempted=True, agent_success=False,
        agent_failure_reason=agent_failure_reason, legacy_fallback_used=True, legacy_fallback_reason=agent_failure_reason,
        pre_agent_safety_guard=None, agent_latency_ms=agent_latency_ms, legacy_latency_ms=legacy_latency_ms,
    )
    return legacy_result


def _log_primary_path_result(
    *,
    primary_path: str,
    agent_attempted: bool,
    agent_success: bool,
    agent_failure_reason: str | None,
    legacy_fallback_used: bool,
    legacy_fallback_reason: str | None,
    pre_agent_safety_guard: str | None,
    agent_latency_ms: float,
    legacy_latency_ms: float,
) -> None:
    print(
        "[ACRLA] primary_path_result "
        f"primary_path={primary_path} "
        f"agent_attempted={agent_attempted} "
        f"agent_success={agent_success} "
        f"agent_failure_reason={agent_failure_reason or 'none'} "
        f"legacy_fallback_used={legacy_fallback_used} "
        f"legacy_fallback_reason={legacy_fallback_reason or 'none'} "
        f"pre_agent_safety_guard={pre_agent_safety_guard or 'none'} "
        f"agent_latency_ms={agent_latency_ms:.1f} "
        f"legacy_latency_ms={legacy_latency_ms:.1f}"
    )


def legacy_handle_message(
    session_id: str,
    student_moodle_id: int,
    message: str,
    db: DBSession,
) -> dict:
    """
    Legacy Conversation Fallback: the original rule-based routing pipeline.

    Only called by handle_message() when the iterative conversation agent
    (the primary path) did not produce a usable result this turn -- low
    confidence, step limit reached without sufficient evidence, tool
    failure, or empty final answer. Reloads session/scope state independently
    so it is self-contained and byte-for-byte identical to how this pipeline
    behaved before the agent became the primary path; it does not invoke the
    agent itself (that already happened once, in handle_message()).

    The pipeline loads session state, classifies the intent, resolves the
    active Moodle remediation scope, chooses the tutoring strategy, and then
    routes to analytics, preference handling, internal RAG, or fallback LLM.
    Chat/practice turns may update conversation memory, but they do not update
    mastery; mastery changes only through assessment submission.
    """
    memory = MemoryManager(db)

    # Load active session
    session: SessionModel = memory.get_active_session(session_id)
    if not session:
        return {
            "reply": "Your session has expired. Please start a new session from Moodle.",
            "intent": "error",
            "strategy": "none",
        }

    student = session.student
    course_id_str = session.course_id
    db.refresh(session)

    # ── Hard safety guard, before intent classification / the agent ────────
    # Runs before anything else in the pipeline, including the conversation
    # agent. See _run_safety_guards for why this does not replace (only
    # front-runs) the semantic-intent-aware guard further below.
    early_guard_reply = _run_safety_guards(message)
    if early_guard_reply is not None:
        memory.save_message(session_id, "user", message, intent="mastery_modification_request")
        memory.save_message(session_id, "assistant", early_guard_reply, intent="mastery_modification_request", strategy="mastery_update_guard")
        return {
            "reply": early_guard_reply,
            "intent": "mastery_modification_request",
            "strategy": "mastery_update_guard",
            "mastery_update": None,
            "difficulty": session.difficulty,
            "mode": "automatic",
            "learning_mode": "automatic",
            # RQ1.B root-cause fix, Root cause 3 -- see the identical
            # early-return in handle_message above for the rationale. This
            # copy inside legacy_handle_message is normally unreachable for
            # this exact message shape (handle_message's own earlier guard
            # already returns first), kept consistent for defense-in-depth.
            "selected_pipeline": "deterministic_reply",
            "source_mode": None,
            "sources": [],
        }

    active_mode = "automatic"
    selected_pipeline: str | None = None
    course_moodle_id = _get_moodle_course_id(session, db)
    remediation_level = _launch_level_for_session(memory, student.id, course_id_str)
    # The available concept set is the guardrail for the whole turn. It is
    # derived from the Moodle launch level, not from free-form chat history, so
    # opening Data Science after Computer Science cannot inherit CS topics.
    available_concepts = set(_concepts_for_level(remediation_level, course_moodle_id, db, memory, student.id))

    # Compute current mastery state
    avg_mastery = memory.get_average_mastery(student.id, course_id_str)
    # Weak concepts are filtered to the active scope. Persisted mastery may
    # contain old records, but the tutor should only talk about concepts that
    # are valid for the currently opened Moodle course/remediation level.
    weak_concepts = [
        concept for concept in memory.get_weak_concepts(student.id, course_id_str)
        if (_course_local_concept(concept) or concept) in available_concepts
    ]

    # Recent performance (last 3 answer evals — simplified: use mastery delta proxy)
    recent_errors = max(0, int((1 - avg_mastery) * 3))
    is_confused = any(w in message.lower() for w in ["confused", "don't understand", "lost", "stuck"])
    in_recovery = avg_mastery < 0.2 and recent_errors >= 2

    # Classify intent
    intent = classify_intent(message)
    rule_intent = intent.value
    current_topic_for_semantics = _current_topic_for_session(memory, student.id, course_id_str, session_id)
    last_reference = _get_last_reference(session_id)
    dialogue_state = _get_dialogue_state(session_id)
    available_course_context = _analytics_available_course_context(student.id, memory, db)
    explicit_entities = _explicit_analytics_entities(message, available_course_context)
    explicit_entities_detected = bool(explicit_entities.get("detected"))
    explicit_concepts = explicit_entities.get("concepts") or []
    semantic_result = analyze_user_turn(
        message=message,
        session_context={
            "remediation_level": remediation_level,
            "course_name": getattr(getattr(session, "course", None), "name", "") or "current Moodle course",
            "current_topic": current_topic_for_semantics,
            "available_concepts": sorted(available_concepts),
            "weak_concepts": weak_concepts,
            "dialogue_state": dialogue_state,
            "previous_assistant_reply": _previous_assistant_reply_summary(session_id),
        },
        last_reference=last_reference,
    )
    semantic_intent = semantic_result.get("intent", "unclear")
    semantic_intents = {semantic_intent, *(semantic_result.get("secondary_intents") or [])}
    semantic_confidence = float(semantic_result.get("confidence") or 0.0)
    use_semantic_intent = semantic_confidence >= 0.75
    if use_semantic_intent:
        if semantic_intent == "greeting":
            intent = Intent.GREETING
        elif semantic_intent == "navigation":
            intent = Intent.NAVIGATION
        elif semantic_intent == "preference":
            intent = Intent.PREFERENCE
        else:
            intent = Intent.TUTORING

    # Select strategy
    strategy = select_strategy(
        mastery_level=avg_mastery,
        recent_errors=recent_errors,
        is_confused=is_confused,
        is_exam_mode="exam" in (session.learning_goal or "").lower(),
        in_recovery=in_recovery,
    )

    # ── Route by intent ──────────────────────────────────────────────────────

    reply = ""
    sources = []
    retrieved_sources = []
    mastery_update = None
    explicit_correction = _detect_explicit_correction(message)
    explicit_correction_detected = bool(explicit_correction.get("detected"))
    corrected_target = explicit_correction.get("target")
    methodology_meta_request = _is_methodology_meta_question(message)
    mastery_modification_request = (
        "mastery_modification_request" in semantic_intents
        if use_semantic_intent
        else _is_mastery_modification_request(message)
    )
    mastery_policy_request = (
        not explicit_correction_detected
        and not methodology_meta_request
        and (bool(use_semantic_intent and "mastery_policy" in semantic_intents) or _is_mastery_policy_question(message))
    )
    scoring_methodology_request = (
        corrected_target == "scoring_methodology"
        or _is_scoring_methodology_question(message)
        or (
            not explicit_correction_detected
            and not methodology_meta_request
            and _is_scoring_methodology_followup(message, semantic_result, dialogue_state)
        )
    )
    tutoring_methodology_request = corrected_target == "tutoring_methodology"
    chapter_correction_request = corrected_target == "chapter"
    course_correction_request = corrected_target == "course"
    unknown_correction_request = explicit_correction_detected and not corrected_target
    methodology_clarification_request = (
        not explicit_correction_detected
        and not methodology_meta_request
        and _is_ambiguous_methodology_followup(message, dialogue_state)
        and not scoring_methodology_request
    )
    reference_followup_request = (
        not explicit_correction_detected
        and not methodology_meta_request
        and not explicit_entities_detected
        and (bool(use_semantic_intent and "reference_followup" in semantic_intents) or _is_reference_followup(message))
    )
    source_provenance_request = _is_source_provenance_question(message)
    course_structure_request = bool(use_semantic_intent and "course_structure" in semantic_intents) or _is_course_structure_question(message)
    practice_request = bool(use_semantic_intent and "practice_request" in semantic_intents) or _is_practice_request(message)
    analytics_only = (
        ("analytics" in semantic_intents if use_semantic_intent else _is_analytics_only_question(message, intent))
        and not mastery_modification_request
        and not mastery_policy_request
        and not scoring_methodology_request
        and not tutoring_methodology_request
        and not chapter_correction_request
        and not course_correction_request
        and not unknown_correction_request
        and not methodology_clarification_request
        and not methodology_meta_request
        and not source_provenance_request
    )
    analytics_plan = None
    analytics_plan_confidence = 0.0
    analytics_planned_request = False
    analytics_clarification_request = False
    # Explicit concept/course extraction is an analytics-plan *modifier*, not
    # a router. A concept mention such as "recursion" may belong to tutoring,
    # comparison, or analytics; only the semantic analytics decision (or the
    # pre-existing rule fallback when semantic planning is unavailable) is
    # allowed to create an analytics plan.
    explicit_analytics_entity_request = explicit_entities_detected and analytics_only
    if analytics_only:
        analytics_plan = plan_analytics_query(
            message=message,
            remediation_level=remediation_level,
            current_course=getattr(getattr(session, "course", None), "name", "") or "current Moodle course",
            current_concept=current_topic_for_semantics,
            available_courses=available_course_context,
            last_reference=None if explicit_entities_detected else last_reference,
        )
        analytics_plan = _apply_explicit_entities_to_plan(analytics_plan, explicit_entities)
        analytics_plan_confidence = float(analytics_plan.get("confidence") or 0.0)
        analytics_planned_request = analytics_plan_confidence >= 0.75
        analytics_clarification_request = not analytics_planned_request
    final_selected_intent = semantic_intent if use_semantic_intent else rule_intent
    previous_followup_target = dialogue_state.get("followup_target") or semantic_result.get("followup_target")
    final_followup_target = corrected_target or semantic_result.get("followup_target") or dialogue_state.get("followup_target")
    routing_precedence_reason = "semantic_or_rules"
    if explicit_correction_detected:
        final_selected_intent = "explicit_user_correction"
        routing_precedence_reason = "explicit_correction_overrides_dialogue_state"
    elif methodology_meta_request:
        final_selected_intent = "methodology_meta"
        routing_precedence_reason = "explicit_current_message_intent"
    elif mastery_modification_request:
        final_selected_intent = "mastery_modification_request"
    elif mastery_policy_request:
        final_selected_intent = "mastery_policy"
    elif scoring_methodology_request:
        final_selected_intent = "scoring_methodology"
    elif tutoring_methodology_request:
        final_selected_intent = "tutoring_methodology"
    elif chapter_correction_request:
        final_selected_intent = "chapter_correction"
    elif course_correction_request:
        final_selected_intent = "course_correction"
    elif unknown_correction_request:
        final_selected_intent = "correction_clarification"
    elif methodology_clarification_request:
        final_selected_intent = "methodology_clarification"
    elif analytics_planned_request:
        final_selected_intent = "analytics_query_plan"
    elif analytics_clarification_request:
        final_selected_intent = "analytics_clarification"
    elif reference_followup_request:
        final_selected_intent = "reference_followup"
    elif course_structure_request:
        final_selected_intent = "course_structure"
    print(
        "[ACRLA] semantic_turn "
        f"primary_intent={semantic_intent} "
        f"semantic_intent={semantic_intent} "
        f"semantic_confidence={semantic_confidence:.2f} "
        f"secondary_intents={semantic_result.get('secondary_intents', [])} "
        f"is_followup={semantic_result.get('is_followup')} "
        f"followup_target={semantic_result.get('followup_target')} "
        f"explicit_entities_detected={explicit_entities_detected} "
        f"explicit_concepts={explicit_concepts} "
        f"explicit_analytics_entity_request={explicit_analytics_entity_request} "
        f"reference_reuse_allowed={not explicit_entities_detected} "
        f"source_provenance_request={source_provenance_request} "
        f"explicit_correction_detected={explicit_correction_detected} "
        f"corrected_target={corrected_target or 'none'} "
        f"previous_followup_target={previous_followup_target or 'none'} "
        f"final_followup_target={final_followup_target or 'none'} "
        f"routing_precedence_reason={routing_precedence_reason} "
        f"analytics_plan_confidence={analytics_plan_confidence:.2f} "
        f"fallback_rule_intent={rule_intent} "
        f"final_selected_intent={final_selected_intent} "
        f"last_answer_type={dialogue_state.get('last_answer_type', 'none')} "
        f"last_reference_type={(last_reference or {}).get('type', 'none')} "
        f"last_reference_items={[(item.get('concept'), item.get('course_name')) for item in (last_reference or {}).get('items', [])[:5]]}"
    )
    response_strategy = strategy.value
    if _awaiting_answer(session_id) and _message_is_new_intent(message, intent):
        _clear_question_state(session_id)

    # The iterative conversation agent already ran once, as the primary path,
    # in handle_message() before this legacy fallback was ever reached -- it
    # is deliberately NOT invoked again here (that would be a second,
    # redundant LLM agent loop for the same turn). agent_result/agent_taken
    # stay fixed so every branch below behaves exactly as this pipeline did
    # before the agent existed.
    name_update = _extract_session_name_update(message)
    agent_result = None
    agent_taken = False
    if name_update:
        _get_session_memory(session_id)["name"] = name_update
        memory.set_profile_name(student.id, name_update)
        reply = f"Nice to meet you, {name_update}. I'll remember your name during this session."
        sources = []

    elif _is_name_recall_question(message):
        remembered_name = _get_session_memory(session_id).get("name") or memory.get_profile_name(student.id)
        if remembered_name:
            _get_session_memory(session_id)["name"] = remembered_name
            reply = f"Your name is {remembered_name}."
        else:
            reply = "I don't know your name yet in this session."
        sources = []

    elif source_provenance_request:
        reply = _handle_source_provenance_question(session_id)
        sources = []
        response_strategy = "source_provenance"
        final_selected_intent = "source_provenance"

    elif _is_preference_question(message):
        reply = _handle_preference_question(student.id, session, memory, message)
        sources = []

    elif explicit_correction_detected and tutoring_methodology_request:
        reply = _handle_tutoring_methodology_question()
        sources = []
        response_strategy = "tutoring_methodology"
        _store_dialogue_state(session_id, {
            "last_answer_type": "tutoring_methodology",
            "last_discussed_metric": "__CLEAR__",
            "followup_target": "tutoring_methodology",
        })

    elif explicit_correction_detected and scoring_methodology_request:
        reply = _handle_scoring_methodology_question(dialogue_state)
        sources = []
        response_strategy = "scoring_methodology"
        _store_dialogue_state(session_id, {
            "last_answer_type": "scoring_methodology",
            "last_discussed_metric": "mastery",
            "followup_target": "scoring_methodology",
        })

    elif explicit_correction_detected and chapter_correction_request:
        reply = _handle_vague_concept_followup(
            message="what is this chapter",
            remediation_level="chapter",
            available_concepts=available_concepts,
            weak_concepts=weak_concepts,
            memory=memory,
            student_id=student.id,
            course_id=course_id_str,
            session_id=session_id,
            db=db,
        )
        sources = []
        response_strategy = "chapter_correction"
        _store_dialogue_state(session_id, {"followup_target": "chapter"})

    elif explicit_correction_detected and course_correction_request:
        lines = "\n".join(
            f"{index}. {concept}"
            for index, concept in enumerate(sorted(available_concepts), start=1)
        )
        reply = f"The current course concepts are:\n\n{lines}"
        sources = []
        response_strategy = "course_correction"
        _store_dialogue_state(session_id, {"followup_target": "course"})

    elif explicit_correction_detected and unknown_correction_request:
        reply = "What should I correct that to?"
        sources = []
        response_strategy = "correction_clarification"

    elif mastery_modification_request:
        # Detection above is a deterministic regex check (a hard safety rule);
        # only the response wording is shared with agents.policies /
        # the get_mastery_guard_response agent tool.
        reply = policies.mastery_modification_guard_response()
        sources = []
        response_strategy = "mastery_update_guard"

    # NOTE: no `elif agent_result:` branch here -- agent_result is always
    # None in this legacy pipeline (see the comment above name_update). The
    # agent's own answer is returned directly by handle_message() before
    # this function is ever called.

    # ==========================================================
    # Legacy Conversation Fallback
    # Remove gradually after agent validation
    #
    # Everything below only runs when the conversation agent did not
    # produce a usable result (agent_blocked, low confidence, parse
    # failure, or an unsafe/incomplete answer -- see run_conversation_agent
    # and _try_conversation_agent). None of these branches were removed in
    # this pass; they are marked so each can be retired individually once
    # the agent path is proven equivalent for it (see the validation
    # scenarios in the migration report).
    # ==========================================================
    elif methodology_meta_request:
        reply = _handle_methodology_meta_question()
        sources = []
        response_strategy = "methodology_meta"

    elif tutoring_methodology_request:
        reply = _handle_tutoring_methodology_question()
        sources = []
        response_strategy = "tutoring_methodology"
        _store_dialogue_state(session_id, {
            "last_answer_type": "tutoring_methodology",
            "last_discussed_metric": "__CLEAR__",
            "followup_target": "tutoring_methodology",
        })

    elif chapter_correction_request:
        reply = _handle_vague_concept_followup(
            message="what is this chapter",
            remediation_level="chapter",
            available_concepts=available_concepts,
            weak_concepts=weak_concepts,
            memory=memory,
            student_id=student.id,
            course_id=course_id_str,
            session_id=session_id,
            db=db,
        )
        sources = []
        response_strategy = "chapter_correction"
        _store_dialogue_state(session_id, {"followup_target": "chapter"})

    elif course_correction_request:
        lines = "\n".join(
            f"{index}. {concept}"
            for index, concept in enumerate(sorted(available_concepts), start=1)
        )
        reply = f"The current course concepts are:\n\n{lines}"
        sources = []
        response_strategy = "course_correction"
        _store_dialogue_state(session_id, {"followup_target": "course"})

    elif unknown_correction_request:
        reply = "What should I correct that to?"
        sources = []
        response_strategy = "correction_clarification"

    elif mastery_policy_request:
        reply = _handle_mastery_policy_question()
        sources = []
        response_strategy = "mastery_policy"

    elif scoring_methodology_request:
        reply = _handle_scoring_methodology_question(dialogue_state)
        sources = []
        response_strategy = "scoring_methodology"
        _store_dialogue_state(session_id, {
            "last_answer_type": "scoring_methodology",
            "last_discussed_metric": "mastery",
            "followup_target": "scoring_methodology",
        })

    elif methodology_clarification_request:
        reply = policies.methodology_clarification_response()
        sources = []
        response_strategy = "methodology_clarification"

    elif reference_followup_request:
        reference_reply = _handle_reference_followup(
            message=message,
            semantic_result=semantic_result,
            session_id=session_id,
            memory=memory,
            student_id=student.id,
            course_id=course_id_str,
            remediation_level=remediation_level,
            available_concepts=available_concepts,
            weak_concepts=weak_concepts,
            db=db,
        )
        if reference_reply:
            reply = reference_reply
            sources = []
            response_strategy = "structured_reference_followup"

    elif analytics_planned_request:
        analytics_result = execute_analytics_query(
            plan=analytics_plan,
            student_id=student.id,
            memory=memory,
            courses=_canonical_synced_courses(student.id, memory, db),
            current_course_id=course_id_str,
        )
        reply = format_analytics_result(analytics_plan, analytics_result)
        sources = []
        response_strategy = "analytics_query_plan"
        _store_last_reference(session_id, {
            "type": "analytics_result",
            "operation": analytics_plan.get("operation"),
            "items": analytics_result.get("items", []),
            "source": "analytics_query_plan",
        })
        print(
            "[ACRLA] analytics_result "
            f"analytics_result_count={len(analytics_result.get('items', []))} "
            f"final_selected_handler={response_strategy}"
        )

    elif analytics_clarification_request:
        reply = "Which analytics view do you want: chapter scores, course averages, weakest topics, strongest topics, or one specific concept?"
        sources = []
        response_strategy = "analytics_clarification"

    elif _is_vague_concept_followup(message):
        reply = _handle_vague_concept_followup(
            message=message,
            remediation_level=remediation_level,
            available_concepts=available_concepts,
            weak_concepts=weak_concepts,
            memory=memory,
            student_id=student.id,
            course_id=course_id_str,
            session_id=session_id,
            db=db,
        )
        sources = []
        response_strategy = "remediation_context_followup"

    elif analytics_only:
        reply = _handle_analytics(student.id, course_id_str, memory, message, available_concepts, db)
        sources = []
        _maybe_store_analytics_reference(
            session_id=session_id,
            message=message,
            memory=memory,
            student_id=student.id,
            course_id=course_id_str,
            available_concepts=available_concepts,
            db=db,
        )

    elif course_structure_request:
        # Course structure is metadata from the active scope, not a content
        # question. Answer it deterministically (shared with the
        # get_course_structure agent tool) so the LLM cannot invent chapters
        # or cite unrelated PDFs.
        reply = policies.course_structure_response(sorted(available_concepts))
        sources = []
        _store_last_reference(session_id, {
            "type": "concept_list",
            "items": _concept_reference_items_for_course(
                memory=memory,
                student_id=student.id,
                course_id=course_id_str,
                concepts=sorted(available_concepts),
                db=db,
            ),
            "source": "course_structure",
        })

    elif intent == Intent.NAVIGATION:
        message_concepts = _concepts_in_message_from_available(message, available_concepts)
        selected_concept = message_concepts[0] if message_concepts else _resolve_selected_concept(message, weak_concepts, session_id)
        if selected_concept not in available_concepts:
            selected_concept = None
        if selected_concept:
            locked_concept = _locked_chapter_concept_for_session(memory, student.id, course_id_str)
            if _course_local_concept(locked_concept) not in available_concepts:
                locked_concept = None
            if locked_concept and selected_concept != locked_concept:
                reply = _locked_chapter_redirect(locked_concept)
                sources = []
            else:
                _get_practice_state(session_id)["topic"] = selected_concept
                memory.set_current_topic(session_id, selected_concept)
                reply = f"Focus switched to {selected_concept}."
                sources = []
        else:
            allowed = ", ".join(available_concepts)
            reply = f"Which course topic should we switch to? Choose one of: {allowed}."

    elif intent == Intent.PREFERENCE:
        reply, session = _handle_preference(message, session, memory, student.id, db, session_id)
        active_mode = "automatic"

    elif intent == Intent.ENGAGEMENT:
        hint = "Tell me the exact word, line, or idea that feels confusing, and I'll explain it with a simple example first."
        reply = (
            f"No worries — let's slow down. {hint}\n\n"
            "Would you like me to try explaining it a different way?"
        )

    elif intent == Intent.GREETING:
        reply = f"Hi {student.username}! Ready to continue? Ask me anything about your course."

    else:
        # Default: tutoring — run the appropriate pipeline
        # Build here so session.mode/difficulty reflect any preference update from this turn
        current_topic = _current_topic_for_session(memory, student.id, course_id_str, session_id)
        if _course_local_concept(current_topic) not in available_concepts:
            current_topic = None
        launch_concept = _launch_concept_for_session(memory, student.id, course_id_str, session_id)
        if _course_local_concept(launch_concept) not in available_concepts:
            launch_concept = None
        retrieval_course_ids = _retrieval_course_ids_for_level(remediation_level, course_moodle_id, db)
        requested_concepts = list(dict.fromkeys(
            list(concepts_in_text(message, available_concepts))
            + _concepts_in_message_from_available(message, available_concepts)
        ))
        locked_concept = _locked_chapter_concept_for_session(memory, student.id, course_id_str)
        if _course_local_concept(locked_concept) not in available_concepts:
            locked_concept = None
        if locked_concept:
            requested_concepts = [locked_concept]
        raw_explicit_concept = requested_concepts[0] if requested_concepts else canonicalize_concept(message)
        if raw_explicit_concept and raw_explicit_concept not in available_concepts:
            reply = _course_scope_redirect(current_topic, available_concepts)
            sources = []
            response_strategy = "scope_redirect"
            memory.save_message(session_id, "user", message, intent=intent.value)
            memory.save_message(session_id, "assistant", reply, intent=intent.value, strategy=response_strategy)
            return {
                "reply": reply,
                "intent": intent.value,
                "strategy": response_strategy,
                "mastery_update": None,
                "difficulty": session.difficulty,
                "mode": active_mode,
                "learning_mode": active_mode,
                "selected_pipeline": selected_pipeline,
                "source_mode": selected_pipeline,
                "sources": sources,
            }
        explicit_concept = raw_explicit_concept
        if explicit_concept not in available_concepts:
            explicit_concept = None
        if not explicit_concept and requested_concepts:
            explicit_concept = requested_concepts[0]
        unsupported_explicit_topic = _extract_unsupported_explicit_topic(message)
        if locked_concept and explicit_concept and explicit_concept != locked_concept:
            reply = _locked_chapter_redirect(locked_concept)
            sources = []
            response_strategy = "chapter_scope_lock"
            memory.save_message(session_id, "user", message, intent=intent.value)
            memory.save_message(session_id, "assistant", reply, intent=intent.value, strategy=response_strategy)
            return {
                "reply": reply,
                "intent": intent.value,
                "strategy": response_strategy,
                "mastery_update": None,
                "difficulty": session.difficulty,
                "mode": active_mode,
                "learning_mode": active_mode,
                "selected_pipeline": selected_pipeline,
                "source_mode": selected_pipeline,
                "sources": sources,
            }
        selected_concept = explicit_concept
        if unsupported_explicit_topic:
            selected_concept = None
            requested_concepts = []
            current_topic = None
        elif locked_concept:
            selected_concept = locked_concept
        if not unsupported_explicit_topic and not selected_concept and launch_concept and _should_stay_on_launch_concept(message):
            selected_concept = launch_concept
        if not unsupported_explicit_topic and not selected_concept:
            selected_concept = _resolve_selected_concept(message, weak_concepts, session_id)
        if selected_concept not in available_concepts:
            selected_concept = None
        if selected_concept:
            memory.set_current_topic(session_id, selected_concept)
            _get_practice_state(session_id)["topic"] = selected_concept
            if explicit_concept and explicit_concept != launch_concept:
                memory.update_course_memory(student.id, course_id_str, {
                    "launch_concept": explicit_concept,
                    "selected_concept": explicit_concept,
                    "last_concept": explicit_concept,
                    "last_activity": f"switched tutoring focus to {explicit_concept}",
                    "next_recommended_action": f"continue with {explicit_concept}",
                })
            current_topic = selected_concept
        elif current_topic and _is_vague_followup(message):
            selected_concept = current_topic
            _get_practice_state(session_id)["topic"] = current_topic

        strategy = _select_strategy_for_turn(
            memory=memory,
            student_id=student.id,
            course_id=course_id_str,
            selected_concept=selected_concept,
            fallback_mastery=avg_mastery,
            difficulty=session.difficulty,
        )
        response_strategy = strategy.name
        retrieval_query = _contextual_retrieval_query(message, " ".join(requested_concepts) or current_topic or selected_concept)
        # Routing must use the raw message. If we used the contextual query here,
        # a vague follow-up or previous topic could make an unrelated question
        # pass the RAG gate and reuse stale course chunks.
        routing_probe_query = message.strip()
        routing_probe_results = retrieve_scored_context_for_scope(
            retrieval_course_ids,
            routing_probe_query,
            requested_concepts=requested_concepts,
        )
        internal_context_relevant = _is_reliable_internal_context(message, routing_probe_results, available_concepts)
        selected_pipeline = "internal_rag" if internal_context_relevant else "external_fallback"
        _log_pipeline_selection(session_id, active_mode, selected_pipeline, selected_concept, strategy.name)
        top_sources = [item.get("source") or item.get("source_file") for item in routing_probe_results[:3]]
        top_scores = [round(float(item.get("score", 0.0)), 4) for item in routing_probe_results[:3]]
        top_score_kinds = [item.get("score_kind", "distance_lower_is_better") for item in routing_probe_results[:3]]
        retrieved_sources = _dedupe_labels(top_sources) if selected_pipeline == "internal_rag" else []
        print(
            "[ACRLA] knowledge_route "
            f"session_id={session_id} "
            f"relevance_gate_result={internal_context_relevant} "
            f"selected_pipeline={selected_pipeline}"
        )
        vprint(
            "[ACRLA] knowledge_route_detail "
            f"session_id={session_id} "
            f"raw_user_message={message!r} "
            f"retrieval_probe_query={routing_probe_query!r} "
            f"contextual_generation_query={retrieval_query!r} "
            f"top_retrieved_sources={top_sources} "
            f"top_scores={top_scores} "
            f"top_score_kinds={top_score_kinds}"
        )
        print(
            "[ACRLA] tutoring_focus "
            f"launch_concept={launch_concept or 'none'} "
            f"selected_concept={selected_concept or 'none'} "
            f"active_tutoring_concept={(selected_concept or current_topic or 'none')} "
            f"explicit_concept={explicit_concept or 'none'}"
        )
        # RQ2 (legacy-fallback privacy fix): this rule-based fallback's
        # student_context feeds an external-LLM prompt (pipelines.rag_pipeline
        # / pipelines.hybrid_pipeline), so it must carry the same minimized
        # shape the primary agent path's services.privacy_context.
        # build_llm_safe_student_context enforces -- no real Moodle name, no
        # student mastery percentage. strategy.reason is
        # "mastery for <concept> is N%" (services.strategy_selector) and
        # strategy.prompt_block embeds it, so the strategy block is rebuilt
        # here from the pedagogical instructions only, dropping that line
        # (mirrors build_llm_safe_student_context, which passes the strategy
        # name + instructions but never its internal "reason").
        _legacy_strategy_block = (
            "BACKEND-CONTROLLED ADAPTIVE STRATEGY:\n"
            f"Strategy: {strategy.name}\n\n"
            "You must follow this strategy:\n"
            + "\n".join(f"- {item}" for item in strategy.instructions)
        )
        student_context = {
            "session_id": session_id,
            "course_name": getattr(getattr(session, "course", None), "name", "") or "current Moodle course",
            "weak_concepts": weak_concepts,
            "selected_concept": selected_concept,
            "sub_concepts": sub_concepts_for(selected_concept),
            "requested_concepts": requested_concepts,
            "available_concepts": _concepts_for_level(remediation_level, course_moodle_id, db, memory, student.id),
            "remediation_level": remediation_level,
            "remediation_scope": _remediation_scope_instruction(remediation_level, available_concepts),
            "retrieval_course_ids": retrieval_course_ids,
            "difficulty": _normalize_difficulty(session.difficulty),
            "question_format": DIFFICULTY_FORMATS[_normalize_difficulty(session.difficulty)],
            "difficulty_instructions": _difficulty_instruction_block(session.difficulty),
            "strategy": strategy.name,
            "strategy_reason": "",
            "strategy_instructions": _legacy_strategy_block,
            "session_goal": session.learning_goal or "general revision",
            "mode": "internal" if selected_pipeline == "internal_rag" else "external",
            "skip_internal_context": selected_pipeline != "internal_rag",
            "current_topic": current_topic or selected_concept or "",
            "retrieval_query": retrieval_query,
        }

        answer_update = None
        if _awaiting_answer(session_id) and _message_is_new_intent(message, intent):
            _clear_question_state(session_id)
        else:
            answer_update = _evaluate_practice_answer(session_id, message, memory, student.id, course_id_str, session, db)
        if answer_update:
            mastery_update = answer_update["mastery_update"]
            reply = answer_update["reply"]
        elif _last_easy_mc_question(session_id) and _is_invalid_mc_answer(message):
            reply = "Please answer with A, B, C, or D."
        elif _last_assistant_question(session_id) and _looks_like_clarification(message.lower()):
            meta = _practice_question_meta(_last_assistant_question(session_id))
            concept = meta["concept"] if meta else "that topic"
            reply = (
                f"I meant {concept}. Please answer the question using the requested format. "
                "For multiple choice, reply with A, B, C, or D."
            )
        elif practice_request:
            practice_difficulty = _normalize_difficulty(session.difficulty)
            if selected_concept:
                memory.set_current_topic(session_id, selected_concept)
            reply = _build_practice_question(
                message,
                practice_difficulty,
                weak_concepts,
                session_id,
                memory=memory,
                student_id=student.id,
                course_id=course_id_str,
                strategy_name=strategy.name,
                strategy_reason=strategy.reason,
                remediation_level=remediation_level,
                available_concepts=list(available_concepts),
                all_level_concepts=_concepts_for_level(remediation_level, course_moodle_id, db, memory, student.id),
                requested_concepts=requested_concepts,
            )
        else:
            # If the student replied with a bare MC letter, inject the previous
            # question so the LLM knows what is being answered.
            effective_message = message
            if _is_mc_letter_answer(message):
                mc_question = _last_mc_question_from_buffer(session_id)
                if mc_question:
                    effective_message = (
                        f"The question you asked was:\n{mc_question}\n\n"
                        f"My answer is: {message.strip().upper()}"
                    )

            if selected_pipeline == "internal_rag":
                reply, sources = generate_rag_response(course_moodle_id, effective_message, student_context)
                sources = sources or retrieved_sources
            else:
                reply, sources = generate_hybrid_response(course_moodle_id, effective_message, student_context)
                # Fallback answers are not grounded in retrieved PDFs for this
                # turn, so expose no source labels. This prevents stale source
                # chips from a previous internal-RAG answer appearing in the UI.
                sources = []
            reply = _sanitize_reply(reply)

        # Mastery is intentionally assessment-gated. Chat/practice feedback
        # can prepare the student, but it must not change mastery scores.
        mastery_update = None
    # ==========================================================
    # End Legacy Conversation Fallback
    # ==========================================================

    # ── Save to memory ───────────────────────────────────────────────────────
    if response_strategy != "conversation_agent" and (analytics_only or course_structure_request):
        sources = []
        retrieved_sources = []
    elif selected_pipeline == "internal_rag":
        sources = sources or retrieved_sources
        retrieved_sources = sources

    if response_strategy != "source_provenance":
        _store_last_response_metadata(
            session_id,
            selected_pipeline=selected_pipeline,
            sources=sources,
            agent_goal=(agent_result or {}).get("goal") if agent_taken else None,
            concepts=_metadata_concepts_for_turn(message, available_concepts, agent_result if agent_taken else None),
            resolved_concepts=(agent_result or {}).get("resolved_concepts") if agent_taken else None,
            evidence_reliable=(agent_result or {}).get("evidence_reliable") if agent_taken else None,
            evidence_reason=(agent_result or {}).get("evidence_reason") if agent_taken else None,
        )
        _save_structured_turn_for_response(
            session_id=session_id,
            message=message,
            reply=reply,
            selected_pipeline=selected_pipeline,
            sources=sources,
            last_reference=last_reference,
            available_concepts=available_concepts,
            agent_result=agent_result if agent_taken else None,
        )

    _store_dialogue_state(session_id, {
        "last_user_intent": final_selected_intent,
        "last_assistant_intent": response_strategy,
        "last_answer_type": _answer_type_for_response_strategy(response_strategy, _get_dialogue_state(session_id)),
        "last_discussed_metric": "mastery" if response_strategy in {"structured_reference_followup", "scoring_methodology", "mastery_policy"} else None,
    })
    print(
        "[ACRLA] dialogue_state "
        f"final_selected_handler={response_strategy} "
        f"last_answer_type={_get_dialogue_state(session_id).get('last_answer_type', 'none')} "
        f"last_discussed_metric={_get_dialogue_state(session_id).get('last_discussed_metric', 'none')}"
    )

    memory.save_message(session_id, "user", message, intent=intent.value)
    memory.save_message(session_id, "assistant", reply, intent=intent.value, strategy=response_strategy)

    return {
        "reply": reply,
        "intent": intent.value,
        "strategy": response_strategy,
        "mastery_update": mastery_update,
        "difficulty": session.difficulty,
        "mode": active_mode,
        "learning_mode": active_mode,
        "selected_pipeline": selected_pipeline,
        "source_mode": selected_pipeline,
        "retrieved_sources": retrieved_sources if selected_pipeline == "internal_rag" else [],
        "sources": sources,
        "agent_used": agent_taken,
        "agent_goal": (agent_result or {}).get("goal") if agent_taken else None,
        "agent_tools_used": (agent_result or {}).get("tools_executed", []) if agent_taken else [],
        "agent_knowledge_strategy": (agent_result or {}).get("knowledge_strategy") if agent_taken else None,
        "agent_resolved_concepts": (agent_result or {}).get("resolved_concepts", []) if agent_taken else [],
        "evidence_reliable": (agent_result or {}).get("evidence_reliable") if agent_taken else None,
        "evidence_reason": (agent_result or {}).get("evidence_reason") if agent_taken else None,
        "recommendation": (agent_result or {}).get("recommendation") if agent_taken else None,
        "recommendation_reason": (agent_result or {}).get("recommendation_reason") if agent_taken else None,
        "agent_fallback_reason": (agent_result or {}).get("fallback_reason") if agent_taken else None,
    }


def _is_analytics_only_question(message: str, intent: Intent) -> bool:
    msg = message.lower()
    if _is_mastery_modification_request(message):
        return False
    if _is_preference_question(message):
        return False
    if _is_practice_request(message):
        return False
    if _is_all_chapter_mastery_question(message):
        return True
    if intent == Intent.ANALYTICS:
        return True
    analytics_markers = [
        "weakest concept",
        "weakest concepts",
        "weakest chapter",
        "weakest chapters",
        "strongest concept",
        "strongest concepts",
        "strongest chapter",
        "strongest chapters",
        "mastery level",
        "mastery levels",
        "chapter scores",
        "progress in",
        "performance by",
        "scores for",
    ]
    return any(marker in msg for marker in analytics_markers)


def _is_mastery_modification_request(message: str) -> bool:
    """Detect requests that try to change mastery outside assessment flow.

    Mastery is assessment-gated in ACRLA. These phrases should not be treated
    as analytics questions because the student is asking the chatbot to change
    a score, not merely report it.
    """
    msg = re.sub(r"\s+", " ", str(message or "").strip().lower())
    if not msg:
        return False
    patterns = [
        r"\b(update|increase|change|improve|raise|set)\b\s+(?:my\s+)?(mastery|score|progress)\b",
        r"\b(update|increase|change|improve|raise|set)\b.*\b(mastery|score|progress)\b.*\b(now|please)?\b",
        r"\b(mark|set)\b.*\b(this|it|topic|chapter|course)?\b.*\b(completed|complete|done)\b",
        r"\b(improve|raise)\b\s+(?:my\s+)?score\b",
    ]
    return any(re.search(pattern, msg) for pattern in patterns)


def _run_safety_guards(message: str) -> str | None:
    """Deterministic hard-safety pre-check, run before intent classification,
    semantic classification, and the conversation agent -- not after them.

    Only returns non-None for the unambiguous case a fast regex can catch
    (the same regex used by the rest of the pipeline, kept deterministic per
    the "hard safety checks stay deterministic" rule). This is additive: the
    slower semantic-intent-aware `mastery_modification_request` check inside
    `legacy_handle_message` remains exactly as-is as a safety net for
    phrasings this fast check misses AND the agent also failed to route
    safely -- so nothing here can regress existing coverage, it only lets the
    obvious cases return sooner, before the agent (or anything else) has a
    chance to run at all.
    """
    if _is_mastery_modification_request(message):
        return policies.mastery_modification_guard_response()
    return None


def _handle_mastery_policy_question() -> str:
    """Explain ACRLA's mastery bands without changing any mastery value."""
    return policies.mastery_policy_response()


def _handle_scoring_methodology_question(dialogue_state: dict | None = None) -> str:
    """Explain how ACRLA mastery scores are calculated in the MVP."""
    return policies.scoring_methodology_response()


def _handle_tutoring_methodology_question() -> str:
    """Explain how ACRLA teaches and adapts, separate from score calculation."""
    return policies.tutoring_methodology_response()


def _handle_methodology_meta_question() -> str:
    """Explain the difference between scoring and tutoring methodology."""
    return policies.methodology_meta_response()


def _detect_explicit_correction(message: str) -> dict:
    """Detect correction turns and extract the corrected target.

    Correction turns outrank previous dialogue state because the student is
    explicitly repairing our interpretation of the prior answer.
    """
    msg = re.sub(r"\s+", " ", str(message or "").strip().lower().strip(" ?.!"))
    if not msg:
        return {"detected": False, "target": None}
    correction_intro = bool(re.search(
        r"^(no\b|no,|not that\b|actually\b|i mean\b|i meant\b|i was referring to\b|rather\b)",
        msg,
    ))
    correction_body = bool(re.search(
        r"\b(i mean|i meant|asking about|referring to|not .* i mean|rather)\b",
        msg,
    ))
    negated_target = bool(re.search(r"^not the .+ i mean\b", msg))
    if not (correction_intro or correction_body or negated_target):
        return {"detected": False, "target": None}
    return {
        "detected": True,
        "target": _corrected_target_from_text(msg),
    }


def _corrected_target_from_text(msg: str) -> str | None:
    """Map a correction phrase to an ACRLA routing target."""
    if re.search(r"\b(tutoring|teaching|teach|adapt|explain|pedagog|remediation)\b", msg) and re.search(r"\b(methodology|method|approach|process|work|works)\b", msg):
        return "tutoring_methodology"
    if re.search(r"\b(scoring|score|scores|mastery|calculated|calculation|formula|baseline)\b", msg) and re.search(r"\b(methodology|method|calculated|calculation|formula|score|scores|mastery)\b", msg):
        return "scoring_methodology"
    if re.search(r"\bchapter\b", msg):
        return "chapter"
    if re.search(r"\bcourse\b", msg):
        return "course"
    if re.search(r"\b(concept|topic)\b", msg):
        return "reference_followup"
    return None


def _get_dialogue_state(session_id: str) -> dict:
    state = _get_session_memory(session_id).get("dialogue_state") if session_id else None
    return dict(state) if isinstance(state, dict) else {}


def _store_dialogue_state(session_id: str, updates: dict) -> None:
    if not session_id:
        return
    state = _get_dialogue_state(session_id)
    for key, value in (updates or {}).items():
        if value == "__CLEAR__":
            state.pop(key, None)
        elif value is not None:
            state[key] = value
    _get_session_memory(session_id)["dialogue_state"] = state


def _is_scoring_methodology_followup(message: str, semantic_result: dict, dialogue_state: dict) -> bool:
    """Resolve elliptical methodology questions from recent score context."""
    msg = re.sub(r"\s+", " ", str(message or "").strip().lower().strip(" ?.!"))
    target = str((semantic_result or {}).get("followup_target") or "").lower()
    if target == "scoring_methodology":
        return True
    methodology_marker = bool(re.search(
        r"\b(methodology|calculated|calculation|based on what|why that score|scoring methodology|how calculated)\b",
        msg,
    ))
    correction_marker = bool(re.search(r"^(no|no,|not)\b.*\b(scoring|methodology|calculation)\b", msg))
    if not (methodology_marker or correction_marker):
        return False
    return dialogue_state.get("last_answer_type") in {"concept_scores", "scoring_methodology"} or dialogue_state.get("last_discussed_metric") == "mastery"


def _is_scoring_methodology_question(message: str) -> bool:
    """Detect explicit questions about how mastery/scores are calculated."""
    msg = re.sub(r"\s+", " ", str(message or "").strip().lower())
    has_score_subject = bool(re.search(r"\b(mastery|score|scores|grade|acrla mastery)\b", msg))
    asks_calculation = bool(re.search(r"\b(calculated|calculate|calculation|formula|methodology|method|based on|derived|computed)\b", msg))
    asks_personal_value = bool(re.search(r"\b(what is my|show my|how am i|performing|doing)\b", msg))
    return has_score_subject and asks_calculation and not asks_personal_value


def _is_ambiguous_methodology_followup(message: str, dialogue_state: dict) -> bool:
    msg = re.sub(r"\s+", " ", str(message or "").strip().lower().strip(" ?.!"))
    if not re.search(r"\b(methodology|calculated|calculation|based on what)\b", msg):
        return False
    return not dialogue_state.get("last_answer_type") and not dialogue_state.get("last_discussed_metric")


def _is_methodology_meta_question(message: str) -> bool:
    """Detect clarification/meta questions that mention both methodology types."""
    msg = re.sub(r"\s+", " ", str(message or "").strip().lower())
    return bool(
        re.search(r"\bdo you mean\b", msg)
        and re.search(r"\b(scoring|mastery|score)\b", msg)
        and re.search(r"\b(tutoring|teaching)\b", msg)
        and re.search(r"\b(methodology|method)\b", msg)
    )


def _is_mastery_policy_question(message: str) -> bool:
    """Detect questions about mastery band definitions, not student progress."""
    msg = re.sub(r"\s+", " ", str(message or "").strip().lower())
    policy_marker = bool(re.search(r"\b(weak|low|moderate|strong|excellent|mastery|score)\b", msg))
    asks_threshold = bool(re.search(r"\b(what point|when|at what|counts? as|considered|no longer|threshold|range)\b", msg))
    personal_progress = bool(re.search(r"\b(my|i am|am i|how am i|performing|doing|progress)\b", msg))
    return policy_marker and asks_threshold and not personal_progress


def _is_vague_concept_followup(message: str) -> bool:
    """Detect follow-ups that refer to ACRLA's remediation focus.

    These questions often appear immediately after an external fallback answer
    says "continue working on your weakest concepts." They should resolve to
    the student's ACRLA scope, not to the unrelated external topic that was
    just answered.
    """
    msg = re.sub(r"\s+", " ", str(message or "").strip().lower().strip(" ?.!"))
    exact = {
        "what are these concepts",
        "what concepts",
        "which concepts",
        "what are they",
        "explain them",
        "tell me about them",
        "what are those concepts",
        "which are these concepts",
        "which are those concepts",
        "what is this chapter",
        "what is the current chapter",
        "which chapter is this",
    }
    if msg in exact:
        return True
    patterns = [
        r"\bwhat\b.*\b(these|those|the)\b.*\bconcepts?\b",
        r"\bwhat\b.*\b(this|current|the)\b.*\bchapter\b",
        r"\bwhich\b.*\bconcepts?\b",
        r"\bwhich\b.*\bchapter\b",
        r"\b(explain|describe|tell me about)\b\s+(them|these|those)\b",
    ]
    return any(re.search(pattern, msg) for pattern in patterns)


def _store_last_reference(session_id: str, reference: dict) -> None:
    """Remember the last deterministic list so follow-ups can resolve pronouns."""
    if not session_id or not reference:
        return
    items = reference.get("items") or []
    if not items:
        return
    _get_session_memory(session_id)["last_reference"] = reference


def _get_last_reference(session_id: str) -> dict | None:
    """Return the structured reference created by the last contextual reply."""
    reference = _get_session_memory(session_id).get("last_reference") if session_id else None
    return reference if isinstance(reference, dict) and reference.get("items") else None


# Goals whose own meaning has nothing to do with reusing a stale analytics
# result -- structural (the fixed SemanticGoal enum), never a phrase/keyword
# match on the message. Anything not in this allowlist (casual conversation,
# external knowledge, tutoring/concept explanation or comparison, personal
# profile, navigation, etc.) must never be answered from a previous turn's
# cached mastery rows.
_ANALYTICS_REUSE_COMPATIBLE_GOALS = {"analytics_query", "reference_followup", "study_recommendation"}

# Only these failure reasons represent a recoverable situation (the provider
# call itself, or the final phrasing step, did not complete) where reusing an
# already-authoritative prior result is a reasonable stand-in -- anything
# else (an entity gap, a rejected low-confidence decision, an unrelated
# goal) should go to legacy_handle_message and answer the CURRENT message
# fresh instead.
_RECOVERABLE_FAILURE_MARKERS = ("provider", "parse", "empty_final_answer", "rate", "connection", "max_agent_steps")


def _reference_reuse_decision(
    *,
    current_goal: str | None,
    current_analytics_request: dict | None,
    current_resolved_concepts: list[str] | None,
    last_reference: dict | None,
    failure_reason: str | None,
) -> tuple[bool, str]:
    """Deterministic compatibility gate: may THIS turn reuse a previous
    turn's stored analytics reference?

    Every check here is structural -- fixed goal-enum membership, entity-set
    containment, analytics_request.entity equality -- never a keyword or
    phrase match against the raw message text. `current_goal`/
    `current_analytics_request`/`current_resolved_concepts` come from the
    CURRENT turn's own agent classification (whatever the planner determined
    before the turn ultimately failed), not re-derived from wording here.

    Reuse is refused (not just "not attempted") whenever:
    - there is no prior analytics reference to reuse at all;
    - the failure this turn is not one of the recoverable provider/evidence
      categories (an entity gap or rejected decision should get a fresh
      legacy answer, not stale data);
    - nothing was classified this turn at all (a genuine provider failure
      before any understanding happened) -- there is no current-turn signal
      to verify compatibility against, so guessing would be unsafe;
    - the current goal is not one where reusing mastery/analytics rows makes
      sense (casual conversation, external knowledge, tutoring explanation,
      etc.);
    - the current turn names explicit entities that are not a subset of the
      stale reference's own entities (a new, explicit request must not be
      answered from someone else's old data);
    - the requested analytics dimension (entity: concept vs course vs
      overall) changed, e.g. "and chapters?" after "for courses?".
    """
    if not last_reference or last_reference.get("type") != "analytics_result":
        return False, "no_prior_analytics_reference"
    items = last_reference.get("items")
    if not isinstance(items, list) or not items:
        return False, "no_prior_analytics_reference"
    failure_text = str(failure_reason or "").lower()
    if not any(marker in failure_text for marker in _RECOVERABLE_FAILURE_MARKERS):
        return False, "agent_failure_not_a_recoverable_provider_or_evidence_issue"
    if not current_goal or current_goal == "unclear":
        return False, "no_current_turn_classification_available"
    if current_goal not in _ANALYTICS_REUSE_COMPATIBLE_GOALS:
        return False, "goal_incompatible_with_analytics_reuse"
    stale_concepts = {c for c in (last_reference.get("resolved_concepts") or []) if c}
    new_concepts = {c for c in (current_resolved_concepts or []) if c}
    if new_concepts and not new_concepts.issubset(stale_concepts):
        return False, "explicit_new_entities_present"
    stale_entity = (last_reference.get("analytics_request") or {}).get("entity")
    current_entity = (current_analytics_request or {}).get("entity")
    if current_entity and stale_entity and current_entity != stale_entity:
        return False, "analytics_dimension_changed"
    return True, "compatible_followup"


def _answer_from_last_analytics_reference(last_reference: dict | None) -> str:
    """Render the previously stored analytics rows as a reply.

    Callers must run `_reference_reuse_decision` first -- this function only
    formats, it does not decide whether reuse is safe.
    """
    items = (last_reference or {}).get("items") or []
    if not items:
        return ""
    operation = (last_reference or {}).get("operation") or "list"
    if operation in {"rank", "recommend"}:
        title = "Here is the latest ranked mastery result I have:"
    elif operation == "compare":
        title = "Here is the latest mastery comparison I have:"
    else:
        title = "Here is the latest mastery data I have:"
    return _format_analytics_reference_items(title, items)


def _format_analytics_reference_items(title: str, items: list[dict]) -> str:
    lines = []
    for index, item in enumerate(items, start=1):
        concept = item.get("concept") or item.get("course_name") or item.get("name") or "Item"
        course = item.get("course_name")
        value = item.get("current_mastery")
        if value is None:
            value = item.get("course_average") if item.get("course_average") is not None else item.get("overall_average")
        course_part = f" - {course}" if course and course != concept else ""
        lines.append(f"{index}. {concept}{course_part} - {_format_reference_percent(value)}")
    return title + "\n" + "\n".join(lines)


def _format_reference_percent(value) -> str:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        return "not available"
    if numeric <= 1.0:
        numeric *= 100
    return f"{numeric:.1f}%"


def _is_reference_followup(message: str) -> bool:
    """Detect follow-ups about a previously mentioned list or focus item.

    This is intentionally intent-based rather than exact-phrase based: it looks
    for reference language plus a question about source courses, scores,
    ranking, explanation, or next-step recommendation.
    """
    msg = re.sub(r"\s+", " ", str(message or "").strip().lower().strip(" ?.!"))
    if not msg:
        return False
    has_reference_language = bool(re.search(r"\b(they|them|these|those|it|their|one)\b", msg))
    has_followup_intent = bool(re.search(
        r"\b(from|where|course|courses|subject|belongs?|belong|contains?|score|scores|scored|mastery|weakest|strongest|start|practice|first|explain|describe|about|same course)\b",
        msg,
    ))
    if has_reference_language and has_followup_intent:
        return True
    return bool(re.search(
        r"\b(from what courses|which courses|what course.*from|which subject|what subject|where do .* belong|how are .* scored|what are their scores|which one.*weakest|which one should i start|what should i practice first)\b",
        msg,
    ))


def _handle_reference_followup(
    message: str,
    semantic_result: dict | None,
    session_id: str,
    memory: MemoryManager,
    student_id: str,
    course_id: str,
    remediation_level: str,
    available_concepts: set[str],
    weak_concepts: list[str],
    db: DBSession,
) -> str:
    """Answer follow-ups by reading the structured dialogue reference.

    If the previous turn did not store a reference yet, build one from the
    active remediation scope. This keeps "from what courses" and "explain them"
    anchored to ACRLA context instead of the last unrelated fallback topic.
    """
    reference = _get_last_reference(session_id)
    if not reference:
        reference = _build_reference_for_current_scope(
            memory=memory,
            student_id=student_id,
            course_id=course_id,
            remediation_level=remediation_level,
            available_concepts=available_concepts,
            weak_concepts=weak_concepts,
            session_id=session_id,
            db=db,
        )
        _store_last_reference(session_id, reference)

    items = list(reference.get("items") or [])
    if not items:
        return "I do not have a recent ACRLA concept list to refer to yet."

    operations = _reference_followup_operations(message, semantic_result)
    sections = []
    if "same_course" in operations:
        sections.append(_format_reference_same_course(items))
    if "courses" in operations:
        sections.append(_format_reference_courses(items))
    if "scores" in operations:
        sections.append(_format_reference_scores(items))
    if "recommendation" in operations:
        sections.append(_format_reference_recommendation(items))
    if "explain" in operations:
        sections.append(_format_reference_explanations(items))
    if "methodology" in operations:
        sections.append(_handle_scoring_methodology_question({"last_answer_type": "concept_scores"}))
    if not sections:
        sections.append(_format_reference_items("Here is the current reference list:", items))

    answer_type = _answer_type_for_reference_operations(operations)
    _store_dialogue_state(session_id, {
        "last_answer_type": answer_type,
        "last_assistant_intent": "reference_followup",
        "last_discussed_metric": "mastery" if {"scores", "methodology"} & set(operations) else None,
        "last_discussed_concepts": [item.get("concept") for item in items if item.get("concept")],
    })
    return "\n\n".join(sections)


def _reference_followup_operations(message: str, semantic_result: dict | None = None) -> list[str]:
    """Return all deterministic operations requested by one follow-up message."""
    msg = re.sub(r"\s+", " ", str(message or "").strip().lower())
    operations = []
    target = str((semantic_result or {}).get("followup_target") or "").lower()
    target_map = {
        "concept_courses": "courses",
        "concept_scores": "scores",
        "scoring_methodology": "methodology",
        "concept_explanations": "explain",
        "recommendation": "recommendation",
    }
    if target in target_map:
        operations.append(target_map[target])
    if re.search(r"\bsame course\b", msg):
        operations.append("same_course")
    if re.search(r"\b(from|where|course|courses|subject|belongs?|belong|contains?)\b", msg):
        operations.append("courses")
    if re.search(r"\b(score|scores|scored|mastery)\b", msg):
        operations.append("scores")
    if re.search(r"\b(weakest|strongest|start|practice|first)\b", msg):
        operations.append("recommendation")
    if re.search(r"\b(explain|describe|about)\b", msg):
        operations.append("explain")
    if re.search(r"\b(methodology|calculated|calculation|based on what|why that score)\b", msg):
        operations.append("methodology")
    return list(dict.fromkeys(operations))


def _answer_type_for_reference_operations(operations: list[str]) -> str:
    if "scores" in operations:
        return "concept_scores"
    if "courses" in operations or "same_course" in operations:
        return "concept_courses"
    if "explain" in operations:
        return "concept_explanations"
    if "recommendation" in operations:
        return "concept_recommendation"
    if "methodology" in operations:
        return "scoring_methodology"
    return "reference_list"


def _answer_type_for_response_strategy(response_strategy: str, current_state: dict | None = None) -> str | None:
    """Map assistant handler names to dialogue-state answer types."""
    if response_strategy == "scoring_methodology":
        return "scoring_methodology"
    if response_strategy == "tutoring_methodology":
        return "tutoring_methodology"
    if response_strategy == "mastery_policy":
        return "mastery_policy"
    if response_strategy == "remediation_context_followup":
        return "concept_list"
    if response_strategy == "methodology_clarification":
        return "methodology_clarification"
    if response_strategy == "structured_reference_followup":
        return (current_state or {}).get("last_answer_type") or "reference_followup"
    return (current_state or {}).get("last_answer_type")


def _format_reference_courses(items: list[dict]) -> str:
    lines = []
    for index, item in enumerate(items, start=1):
        concept = item.get("concept") or "Concept"
        course_name = item.get("course_name") or "Unknown course"
        lines.append(f"{index}. {concept} - {course_name}")
    return "These focus concepts come from:\n" + "\n".join(lines)


def _format_reference_same_course(items: list[dict]) -> str:
    courses = [item.get("course_name") or "Unknown course" for item in items]
    unique_courses = list(dict.fromkeys(courses))
    if len(unique_courses) == 1:
        return f"Yes. They are all from {unique_courses[0]}."
    return "No. They come from multiple courses:\n" + "\n".join(
        f"- {course}" for course in unique_courses
    )


def _format_reference_scores(items: list[dict]) -> str:
    lines = []
    for index, item in enumerate(items, start=1):
        concept = item.get("concept") or "Concept"
        mastery = item.get("mastery", item.get("current_mastery"))
        score = f"{float(mastery):.0%}" if mastery is not None else "not tracked yet"
        lines.append(f"{index}. {concept}: {score}")
    return "Their current mastery scores are:\n" + "\n".join(lines)


def _format_reference_recommendation(items: list[dict]) -> str:
    scored = [item for item in items if item.get("mastery", item.get("current_mastery")) is not None]
    if not scored:
        first = items[0].get("concept") or "the first concept"
        return f"I would start with {first}, then work through the rest one by one."
    weakest = min(scored, key=lambda item: float(item.get("mastery", item.get("current_mastery")) or 0.0))
    concept = weakest.get("concept") or "the weakest concept"
    course_name = weakest.get("course_name")
    score = f"{float(weakest.get('mastery', weakest.get('current_mastery')) or 0.0):.0%}"
    course_note = f" from {course_name}" if course_name else ""
    return f"Start with {concept}{course_note}. It currently has the lowest mastery in this list ({score})."


def _format_reference_explanations(items: list[dict]) -> str:
    if len(items) > 5:
        names = ", ".join(str(item.get("concept") or "Concept") for item in items[:5])
        return f"There are several concepts here. We can take them one by one. The first few are: {names}."
    lines = []
    for index, item in enumerate(items, start=1):
        concept = item.get("concept") or "Concept"
        course_name = item.get("course_name")
        course_note = f" ({course_name})" if course_name else ""
        lines.append(f"{index}. {concept}{course_note}: {_short_concept_explanation(concept)}")
    return "Here is a short explanation of each:\n" + "\n".join(lines)


def _format_reference_items(intro: str, items: list[dict]) -> str:
    lines = []
    for index, item in enumerate(items, start=1):
        concept = item.get("concept") or "Concept"
        course_name = item.get("course_name")
        suffix = f" - {course_name}" if course_name else ""
        lines.append(f"{index}. {concept}{suffix}")
    return f"{intro}\n" + "\n".join(lines)


def _short_concept_explanation(concept: str) -> str:
    """Brief deterministic explanations for reference follow-ups."""
    topic_kind = _topic_kind(concept)
    if topic_kind == "recursion":
        return "solving a problem by calling the same process on smaller cases until a base case stops it."
    if topic_kind == "sorting":
        return "arranging data into a useful order, often by comparing and moving values."
    if topic_kind == "pointers":
        return "using memory addresses to access, update, or manage data safely."
    if topic_kind == "binary_trees":
        return "organizing data in nodes where each node has at most two children."
    normalized = str(concept or "").lower()
    if "relation" in normalized or "function" in normalized:
        return "describing how elements from one set are connected or mapped to elements of another set."
    if "logic" in normalized:
        return "reasoning with statements, truth values, and valid arguments."
    if "set" in normalized:
        return "working with collections of objects and operations such as union or intersection."
    if "graph" in normalized:
        return "modeling objects as vertices and their connections as edges."
    return "a course concept that ACRLA is tracking for remediation and practice."


def _build_reference_for_current_scope(
    memory: MemoryManager,
    student_id: str,
    course_id: str,
    remediation_level: str,
    available_concepts: set[str],
    weak_concepts: list[str],
    session_id: str,
    db: DBSession,
) -> dict:
    """Create a structured reference from the current remediation scope."""
    level = str(remediation_level or "chapter").strip().lower()
    if level == "overall":
        return {
            "type": "concept_list",
            "items": _overall_focus_items_for_student(memory, student_id, db),
            "source": "overall_focus_concepts",
        }
    if level == "course":
        concepts = _ordered_concepts(weak_concepts) or _ordered_concepts(available_concepts)
        return {
            "type": "concept_list",
            "items": _concept_reference_items_for_course(memory, student_id, course_id, concepts, db),
            "source": "course_focus_concepts",
        }

    chapter = (
        _locked_chapter_concept_for_session(memory, student_id, course_id)
        or _launch_concept_for_session(memory, student_id, course_id, session_id)
        or _current_topic_for_session(memory, student_id, course_id, session_id)
    )
    chapter = _course_local_concept(chapter) or chapter
    concepts = [chapter] if chapter else _ordered_concepts(available_concepts)[:1]
    return {
        "type": "concept_list",
        "items": _concept_reference_items_for_course(memory, student_id, course_id, concepts, db),
        "source": "chapter_focus_concept",
    }


def _overall_focus_items_for_student(
    memory: MemoryManager,
    student_id: str,
    db: DBSession,
    limit: int = 3,
) -> list[dict]:
    """Return weakest concepts across all courses with course metadata."""
    from models.db_models import Course

    rows = []
    for course in db.query(Course).all():
        for concept in _dynamic_concepts_for_course(memory, student_id, course):
            mastery = memory.get_mastery(student_id, course.id, concept)
            rows.append({
                "concept": concept,
                "course_name": course.name,
                "course_id": course.moodle_course_id,
                "mastery": mastery,
            })
    unique = []
    seen = set()
    for item in sorted(rows, key=lambda row: row["mastery"]):
        key = (item["concept"], item["course_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def _concept_reference_items_for_course(
    memory: MemoryManager,
    student_id: str,
    course_id: str,
    concepts,
    db: DBSession,
) -> list[dict]:
    """Build reference items for concepts known to belong to one course."""
    from models.db_models import Course

    course = db.query(Course).filter_by(id=course_id).first()
    course_name = course.name if course else "Current course"
    moodle_course_id = course.moodle_course_id if course else None
    items = []
    for concept in _ordered_concepts(concepts):
        items.append({
            "concept": concept,
            "course_name": course_name,
            "course_id": moodle_course_id,
            "mastery": memory.get_mastery(student_id, course_id, concept),
        })
    return items


def _maybe_store_analytics_reference(
    session_id: str,
    message: str,
    memory: MemoryManager,
    student_id: str,
    course_id: str,
    available_concepts: set[str],
    db: DBSession,
) -> None:
    """Store a reference when deterministic analytics returns concept lists."""
    if _is_global_overall_mastery_question(message):
        items = _overall_focus_items_for_student(memory, student_id, db)
    elif _is_all_chapter_mastery_question(message) or _is_weakest_concepts_question(message) or _is_strongest_concepts_question(message):
        items = _concept_reference_items_for_course(memory, student_id, course_id, sorted(available_concepts), db)
    else:
        return
    _store_last_reference(session_id, {
        "type": "concept_list",
        "items": items,
        "source": "analytics_concept_list",
    })


def _handle_vague_concept_followup(
    message: str,
    remediation_level: str,
    available_concepts: set[str],
    weak_concepts: list[str],
    memory: MemoryManager,
    student_id: str,
    course_id: str,
    session_id: str,
    db: DBSession,
) -> str:
    """Answer "what are these concepts?" from remediation scope only.

    This deterministic path deliberately bypasses RAG and fallback generation.
    The phrase "these concepts" refers to ACRLA's current remediation target,
    especially after external fallback redirects the student back to course
    work. It must not inherit the external topic, such as Japan or quantum
    computing.
    """
    level = str(remediation_level or "chapter").strip().lower()

    if level == "overall":
        items = _overall_focus_items_for_student(memory, student_id, db)
        if not items:
            items = _concept_reference_items_by_real_course(
                memory=memory,
                student_id=student_id,
                concepts=_ordered_concepts(weak_concepts) or _ordered_concepts(available_concepts),
                db=db,
                fallback_course_id=course_id,
            )
        _store_last_reference(session_id, {
            "type": "concept_list",
            "items": items,
            "source": "overall_focus_concepts",
        })
        return _format_focus_concepts_reply(
            "These are your current focus concepts across your courses:",
            [item["concept"] for item in items],
        )

    if level == "course":
        concepts = _ordered_concepts(weak_concepts) or _ordered_concepts(available_concepts)
        _store_last_reference(session_id, {
            "type": "concept_list",
            "items": _concept_reference_items_for_course(memory, student_id, course_id, concepts, db),
            "source": "course_focus_concepts",
        })
        return _format_focus_concepts_reply(
            "These are the current focus concepts for this course:",
            concepts,
        )

    chapter = (
        _locked_chapter_concept_for_session(memory, student_id, course_id)
        or _launch_concept_for_session(memory, student_id, course_id, session_id)
        or _current_topic_for_session(memory, student_id, course_id, session_id)
    )
    chapter = _course_local_concept(chapter) or chapter
    if not chapter:
        concepts = _ordered_concepts(available_concepts)
        chapter = concepts[0] if concepts else "the clicked chapter"
    _store_last_reference(session_id, {
        "type": "concept_list",
        "items": _concept_reference_items_for_course(memory, student_id, course_id, [chapter], db),
        "source": "chapter_focus_concept",
    })
    return f"The current chapter focus is: {chapter}."


def _overall_focus_concepts_for_student(
    memory: MemoryManager,
    student_id: str,
    db: DBSession,
    limit: int = 3,
) -> list[str]:
    """Return weakest discovered concepts across all courses for overall scope."""
    from models.db_models import Course

    rows = []
    for course in db.query(Course).all():
        for concept in _dynamic_concepts_for_course(memory, student_id, course):
            mastery = memory.get_mastery(student_id, course.id, concept)
            rows.append((concept, mastery))
    ordered = []
    for concept, _mastery in sorted(rows, key=lambda item: item[1]):
        if concept not in ordered:
            ordered.append(concept)
        if len(ordered) >= limit:
            break
    return ordered


def _concept_reference_items_by_real_course(
    memory: MemoryManager,
    student_id: str,
    concepts,
    db: DBSession,
    fallback_course_id: str | None = None,
) -> list[dict]:
    """Resolve concept metadata to the course that actually owns each concept.

    Overall remediation combines courses, so a concept list cannot inherit the
    currently open Moodle course. This resolver checks each synced course's
    material manifest/dynamic concept list, then falls back to mastery records,
    and only uses the current course when no better owner is known.
    """
    from models.db_models import Course

    requested = _ordered_concepts(concepts)
    if not requested:
        return []

    course_rows = db.query(Course).all()
    material_index: dict[str, list] = {}
    for course in course_rows:
        for concept in _dynamic_concepts_for_course(memory, student_id, course):
            key = _concept_key(concept)
            material_index.setdefault(key, []).append((course, concept))

    fallback_course = db.query(Course).filter_by(id=fallback_course_id).first() if fallback_course_id else None
    items = []
    for concept in requested:
        key = _concept_key(concept)
        owner = material_index.get(key, [None])[0]
        if owner:
            course, resolved_concept = owner
        else:
            course, resolved_concept = _find_course_for_mastery_concept(memory, student_id, concept, course_rows)
        if not course:
            course, resolved_concept = fallback_course, concept
        course_db_id = course.id if course else fallback_course_id
        items.append({
            "concept": resolved_concept or concept,
            "course_name": course.name if course else "Unknown course",
            "course_id": course.moodle_course_id if course else None,
            "mastery": memory.get_mastery(student_id, course_db_id, resolved_concept or concept) if course_db_id else None,
        })
    return items


def _find_course_for_mastery_concept(
    memory: MemoryManager,
    student_id: str,
    concept: str,
    courses,
):
    """Fallback concept owner lookup using persisted mastery records."""
    target_key = _concept_key(concept)
    for course in courses:
        for record in memory.get_all_mastery(student_id, course.id):
            if _concept_key(record.concept) == target_key:
                return course, record.concept
    return None, concept


def _concept_key(concept: str | None) -> str:
    label = _course_local_concept(concept) or str(concept or "")
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def _ordered_concepts(concepts) -> list[str]:
    """Deduplicate concepts while preserving a readable deterministic order."""
    ordered = []
    source = sorted(concepts) if isinstance(concepts, set) else (concepts or [])
    for concept in source:
        label = _course_local_concept(concept) or str(concept).strip()
        if label and label not in ordered:
            ordered.append(label)
    return ordered


def _format_focus_concepts_reply(intro: str, concepts: list[str]) -> str:
    if not concepts:
        return "I do not have synced remediation concepts for this scope yet."
    lines = "\n".join(f"{index}. {concept}" for index, concept in enumerate(concepts, start=1))
    return f"{intro}\n{lines}"


def _missing_internal_material_reply(
    concept: str,
    quoted: bool = False,
    available_concepts: set[str] | tuple[str, ...] | list[str] | None = None,
) -> str:
    label = f"'{concept}'" if quoted else concept
    available = ", ".join(available_concepts or [])
    return (
        f"I don't have course material for {label} available in ACRLA yet.\n\n"
        "Please ask the teacher to sync or upload the relevant Moodle PDF for this concept, "
        f"then I can help using the course materials.\n\nAvailable course concepts: {available}."
    )


def _is_reliable_internal_context(
    raw_message: str,
    scored_results: list[dict],
    available_concepts: set[str] | list[str] | tuple[str, ...] | None = None,
) -> bool:
    """Conservative relevance gate for internal RAG vs external fallback.

    The raw user message is checked against scored Chroma results. This avoids
    a previous topic or contextual rewrite making an unrelated question appear
    relevant to the current course. Returns True only when the retrieved chunks
    look close enough and overlap with the actual question/course vocabulary.

    Chroma scores from `similarity_search_with_score` are distance-like in this
    project: lower is better. Because embedding backends can use different
    numeric scales, the gate compares the nearest results relative to each
    other instead of hardcoding one global cutoff.
    """
    if not scored_results:
        return False

    query_tokens = _routing_tokens(raw_message)
    if not query_tokens:
        return False

    course_vocab = set()
    for concept in available_concepts or []:
        course_vocab.update(_routing_tokens(str(concept)))
    message_mentions_course = bool(query_tokens & course_vocab)

    sorted_results = sorted(scored_results, key=lambda result: float(result.get("score", 999.0)))
    best_score = float(sorted_results[0].get("score", 999.0))

    for index, item in enumerate(sorted_results[:3]):
        score = float(item.get("score", 999.0))
        text = " ".join(
            str(item.get(key) or "")
            for key in ("concept", "source", "source_file", "text")
        )
        overlap = query_tokens & _routing_tokens(text)
        # The nearest chunk alone is not enough: if the student just studied
        # Recursion and then asks about quantum computing, Chroma may still
        # return a recursion chunk. Requiring overlap with the raw question or
        # course vocabulary prevents that stale context from passing the gate.
        is_nearest = index == 0 or score == best_score
        if is_nearest and overlap and (message_mentions_course or len(overlap) >= 2):
            return True
    return False


def _routing_tokens(text: str) -> set[str]:
    stopwords = {
        "a", "an", "and", "are", "about", "can", "could", "do", "does", "explain",
        "for", "give", "how", "i", "in", "is", "it", "me", "my", "of", "on",
        "please", "tell", "the", "this", "to", "using", "what", "who", "why",
        "you", "your",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", str(text).lower())
        if len(token) > 2 and token not in stopwords
    }


def _dedupe_labels(labels: list[str | None]) -> list[str]:
    deduped: dict[str, str] = {}
    for label in labels:
        text = str(label or "").strip()
        if not text:
            continue
        key = text[:-4].lower() if text.lower().endswith(".pdf") else text.lower()
        deduped.setdefault(key, text)
    return list(deduped.values())


def _extract_unsupported_explicit_topic(message: str) -> str | None:
    if canonicalize_concept(message) or _is_vague_followup(message):
        return None
    text = re.sub(r"\s+", " ", message.strip())
    patterns = [
        r"^(?:please\s+)?(?:explain|teach|describe|show me|tell me about)\s+(.+)$",
        r"^(?:please\s+)?(?:test me|quiz me|ask me questions?)\s+(?:on|about|for)\s+(.+)$",
        r"^(?:please\s+)?(?:test me|quiz me)\s+(.+)$",
        r"^(?:practice|remediate|review)\s+(.+)$",
        r"^(?:give me|generate)\s+(?:a\s+)?(?:question|quiz|test)\s+(?:on|about|for)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        topic = _clean_requested_topic(match.group(1))
        if topic and not canonicalize_concept(topic):
            return topic
    return None


def _clean_requested_topic(raw_topic: str) -> str:
    topic = re.sub(r"\b(?:please|now|next|again)\b", " ", raw_topic, flags=re.IGNORECASE)
    topic = re.sub(r"\s+", " ", topic.strip(" ?.!'\"*"))
    topic = re.sub(r"^(?:the|a|an)\s+", "", topic, flags=re.IGNORECASE)
    if not topic or topic.lower() in {"this", "that", "it", "topic", "current topic"}:
        return ""
    return topic[:80]


def _extract_session_name_update(message: str) -> str | None:
    text = message.strip()
    patterns = [
        r"^my name is\s+([A-Za-z][A-Za-z .'-]{0,60})$",
        r"^call me\s+([A-Za-z][A-Za-z .'-]{0,60})$",
        r"^i am\s+([A-Za-z][A-Za-z .'-]{0,60})$",
        r"^i'm\s+([A-Za-z][A-Za-z .'-]{0,60})$",
    ]
    blocked = {
        "confused",
        "lost",
        "stuck",
        "ready",
        "fine",
        "ok",
        "okay",
        "not sure",
        "having trouble",
    }
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        name = _clean_session_name(match.group(1))
        if name and name.lower() not in blocked:
            return name
    return None


def _clean_session_name(raw_name: str) -> str:
    name = re.sub(r"\s+", " ", raw_name.strip(" .!?,;:"))
    if not name or len(name) > 60:
        return ""
    return " ".join(part.capitalize() if part.isalpha() else part for part in name.split())


def _is_name_recall_question(message: str) -> bool:
    msg = message.lower().strip(" ?.!") 
    return msg in {
        "what is my name",
        "what's my name",
        "do you remember my name",
        "can you remember my name",
        "tell me my name",
    }


def _is_preference_question(message: str) -> bool:
    msg = message.lower().strip(" ?.!") 
    preference_markers = [
        "what is my difficulty level",
        "what's my difficulty level",
        "difficulty level",
        "current difficulty",
        "what difficulty am i using",
        "what difficulty level am i using",
        "which difficulty am i using",
        "learning level",
        "current learning level",
        "what is my learning level",
        "what's my learning level",
        "hard moderate or easy",
        "what is my learning mode",
        "what's my learning mode",
        "what mode are you using",
        "which mode are you using",
        "are you using internal",
        "are you using external",
        "current learning mode",
        "am i in internal or external mode",
        "am i internal or external",
        "what are my preferences",
        "saved preferences",
    ]
    return any(marker in msg for marker in preference_markers)


def _handle_preference_question(
    student_id: str,
    session: SessionModel,
    memory: MemoryManager,
    message: str,
) -> str:
    prefs = memory.get_profile_preferences(student_id)
    difficulty = prefs.get("difficulty") or session.difficulty or "medium"
    name = prefs.get("name")
    msg = message.lower()

    if "difficulty" in msg or "learning level" in msg or "hard moderate or easy" in msg:
        return f"Your current difficulty level is {_label(difficulty)}."
    if "learning mode" in msg or "internal or external" in msg or "mode are you using" in msg or "are you using internal" in msg or "are you using external" in msg:
        return "ACRLA automatically searches course materials first. If relevant course material is found, I answer from it and show the source. If not, I use fallback support."

    parts = []
    if name:
        parts.append(f"name = {name}")
    parts.append(f"difficulty = {_label(difficulty)}")
    parts.append("response routing = Automatic")
    return "Your saved preferences are: " + ", ".join(parts) + "."


def _label(value: str) -> str:
    if not value:
        return ""
    return str(value).replace("_", " ").strip().capitalize()


def _handle_analytics(
    student_id: str,
    course_id: str,
    memory: MemoryManager,
    message: str = "",
    available_concepts: set[str] | None = None,
    db: DBSession | None = None,
) -> str:
    """Answer mastery/progress questions from stored profile data only.

    Analytics is intentionally deterministic and bypasses RAG because mastery
    values come from PostgreSQL/long-term memory, not PDFs. This keeps source
    citations hidden for profile-style questions.
    """
    if db is not None and _is_global_overall_mastery_question(message):
        return _format_global_overall_mastery(student_id, memory, db)

    available_concepts = available_concepts or set()
    all_mastery = [
        record for record in memory.get_all_mastery(student_id, course_id)
        if (_course_local_concept(record.concept) or record.concept) in available_concepts
    ]

    if not all_mastery:
        allowed = ", ".join(available_concepts)
        return f"We haven't tracked your course mastery yet. I can track these concepts: {allowed}."

    sorted_mastery = sorted(all_mastery, key=lambda r: r.mastery_level)
    avg = sum(r.mastery_level for r in all_mastery) / len(all_mastery)
    if _is_all_chapter_mastery_question(message):
        return _format_all_chapter_mastery(all_mastery)

    requested_concept = _course_local_concept(message)
    if requested_concept not in available_concepts:
        requested_concept = None

    if requested_concept:
        record = next((r for r in all_mastery if r.concept == requested_concept), None)
        if not record:
            return f"I don't have a mastery score for {requested_concept} yet."
        return f"{requested_concept} mastery: {record.mastery_level:.0%}."

    weak = [r for r in sorted_mastery if r.mastery_level < 0.6]
    strong = [r for r in sorted_mastery if r.mastery_level >= 0.75]

    if _is_strongest_concepts_question(message):
        top = sorted(all_mastery, key=lambda r: r.mastery_level, reverse=True)[:2]
        return _format_strongest_concepts(top, weak, avg)

    if _is_weakest_concepts_question(message):
        return _format_weakest_concepts(weak, strong, avg)

    if _is_overall_mastery_question(message):
        return _format_overall_mastery(avg, weak, strong)

    if _is_ambiguous_analytics_question(message):
        return (
            _format_all_chapter_mastery(all_mastery)
            + "\n\nIf you meant something else, let me know - I can also explain a new topic or test you on something specific."
        )

    reply = "You do not have any weak topics below 60% right now. Nice work."
    if weak:
        weakest = weak[:2]
        weak_lines = "\n".join(
            f"{index}. {record.concept} ({record.mastery_level:.0%})"
            for index, record in enumerate(weakest, start=1)
        )
        reply = f"Your weakest concepts are:\n\n{weak_lines}"

    lines = [reply + f"\n\nOverall course mastery: {avg:.0%}."]

    top = sorted(strong, key=lambda r: r.mastery_level, reverse=True)[:3]
    if top:
        lines.append("Strongest concepts: " + ", ".join(f"{r.concept} ({r.mastery_level:.0%})" for r in top))

    return "\n".join(lines)


def _is_ambiguous_analytics_question(message: str) -> bool:
    msg = message.lower()
    return any(
        re.search(pattern, msg)
        for pattern in [
            r"\brest\b",
            r"\bother\b.*\b(course|concept|topic)s?\b",
            r"\ball\b.*\b(course|concept|topic|mastery|progress)\b",
        ]
    )


def _is_weakest_concepts_question(message: str) -> bool:
    msg = message.lower()
    return any(marker in msg for marker in ["weak", "weakest", "struggling", "lowest"])


def _is_strongest_concepts_question(message: str) -> bool:
    msg = message.lower()
    return any(marker in msg for marker in ["strong", "strongest", "best", "highest"])


def _is_overall_mastery_question(message: str) -> bool:
    msg = message.lower()
    asks_mastery = any(marker in msg for marker in ["mastery", "progress", "level", "score", "performance"])
    excludes_specific = not (
        _is_weakest_concepts_question(message)
        or _is_strongest_concepts_question(message)
        or _is_all_chapter_mastery_question(message)
        or canonicalize_concept(message)
    )
    return asks_mastery and excludes_specific


def _is_global_overall_mastery_question(message: str) -> bool:
    msg = message.lower()
    if canonicalize_concept(message):
        return False
    asks_mastery = any(marker in msg for marker in ["mastery", "progress", "level", "score", "performance"])
    asks_overall = any(marker in msg for marker in [
        "overall",
        "all courses",
        "whole program",
        "everything",
        "across courses",
        "across all courses",
    ])
    if "course mastery" in msg and "overall" not in msg and "all courses" not in msg:
        return False
    return asks_mastery and asks_overall


def _format_records(records: list, suffix: str = "%") -> str:
    return "\n".join(
        f"{index}. {record.concept} ({record.mastery_level:.0%})"
        for index, record in enumerate(records, start=1)
    )


def _format_global_overall_mastery(student_id: str, memory: MemoryManager, db: DBSession) -> str:
    from models.db_models import Course

    courses = db.query(Course).all()
    course_rows = []
    concept_rows = []

    for course in courses:
        course_concepts = _dynamic_concepts_for_course(memory, student_id, course)
        if not course_concepts:
            continue
        scores = []
        for concept in course_concepts:
            mastery = memory.get_mastery(student_id, course.id, concept)
            scores.append(mastery)
            concept_rows.append({
                "concept": concept,
                "mastery": mastery,
                "course_name": course.name,
                "course_id": course.moodle_course_id,
            })
        avg = sum(scores) / len(scores) if scores else 0.0
        course_rows.append({
            "course_name": course.name,
            "course_id": course.moodle_course_id,
            "mastery": avg,
        })

    if not course_rows:
        return "I do not have enough course mastery data yet to calculate your overall mastery across all courses."

    overall = sum(row["mastery"] for row in course_rows) / len(course_rows)
    course_lines = "\n".join(
        f"- {row['course_name']}: {row['mastery']:.0%}"
        for row in sorted(course_rows, key=lambda item: item["course_name"])
    )
    weakest = sorted(concept_rows, key=lambda item: item["mastery"])[:3]
    strongest = sorted(concept_rows, key=lambda item: item["mastery"], reverse=True)[:3]
    weak_lines = "\n".join(
        f"{index}. {item['concept']} ({item['mastery']:.0%}) - {item['course_name']}"
        for index, item in enumerate(weakest, start=1)
    )
    strong_lines = "\n".join(
        f"{index}. {item['concept']} ({item['mastery']:.0%}) - {item['course_name']}"
        for index, item in enumerate(strongest, start=1)
    )

    return (
        f"Overall mastery across all courses: {overall:.0%}.\n\n"
        f"Course breakdown:\n{course_lines}\n\n"
        f"Weakest concepts across all courses:\n{weak_lines}\n\n"
        f"Strongest concepts across all courses:\n{strong_lines}"
    )


def _format_weakest_concepts(weak: list, strong: list, avg: float) -> str:
    if weak:
        weak_text = _format_records(weak[:2])
        response = f"Your weakest concepts are:\n\n{weak_text}"
    else:
        response = "You do not have any weak concepts below 60% right now."

    if strong:
        response += f"\n\nYour strongest concepts are:\n\n{_format_records(strong[:2])}"

    response += f"\n\nOverall course mastery: {avg:.0%}."
    return response


def _format_strongest_concepts(strong: list, weak: list, avg: float) -> str:
    if strong:
        response = f"Your strongest concepts are:\n\n{_format_records(strong[:2])}"
    else:
        response = "You do not have any concepts at or above 75% yet."

    if weak:
        response += f"\n\nYour weaker concepts are:\n\n{_format_records(weak[:2])}"

    response += f"\n\nOverall course mastery: {avg:.0%}."
    return response


def _format_overall_mastery(avg: float, weak: list, strong: list) -> str:
    response = f"Overall course mastery: {avg:.0%}."
    if strong:
        response += f"\n\nYour strongest concepts are:\n\n{_format_records(strong[:2])}"
    if weak:
        response += f"\n\nYour weakest concepts are:\n\n{_format_records(weak[:2])}"
    return response


def _is_all_chapter_mastery_question(message: str) -> bool:
    msg = message.lower()
    asks_mastery = any(
        marker in msg
        for marker in ["mastery", "progress", "score", "scores", "level", "levels", "performance"]
    )
    asks_all = any(
        marker in msg
        for marker in ["all chapters", "each chapter", "every chapter", "all topics", "each topic", "by chapter", "chapter scores"]
    )
    return asks_mastery and asks_all


def _format_all_chapter_mastery(records: list) -> str:
    record_by_concept = {record.concept: record for record in records}
    lines = []
    chapter_scores = []

    for index, concept in enumerate(record_by_concept.keys(), start=1):
        mastery = (record_by_concept.get(concept).mastery_level if record_by_concept.get(concept) else 0.0) or 0.0
        chapter_scores.append((concept, mastery))
        lines.append(f"{index}. {concept}: {mastery:.0%}")

    weak = sorted(
        [item for item in chapter_scores if item[1] < 0.6],
        key=lambda item: item[1],
    )[:2]

    response = "Your mastery levels are:\n\n" + "\n".join(lines)
    if weak:
        weak_lines = "\n".join(
            f"{index}. {concept}: {mastery:.0%}"
            for index, (concept, mastery) in enumerate(weak, start=1)
        )
        response += f"\n\nYour weakest chapters are:\n\n{weak_lines}"
    else:
        response += "\n\nYou do not have any chapters below 60% right now."
    return response


def _sanitize_reply(reply: str) -> str:
    """Remove orchestration/internal-state wording from LLM output.

    Prompts include backend-controlled fields such as selected concept and
    strategy. If the LLM echoes those implementation details, this guard keeps
    the student-facing answer clean.
    """
    blocked_patterns = [
        r"(?:the\s+)?selected course concept[^.\n]*[.\n]?",
        r"(?:the\s+)?focus topic[^.\n]*unresolved[^.\n]*[.\n]?",
        r"\bunresolved\b",
        r"\borchestration\b",
        r"\bbackend state\b",
    ]
    cleaned = reply
    for pattern in blocked_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or "I can help with the course topics we have loaded."


def _is_course_structure_question(message: str) -> bool:
    """Detect questions about the course outline rather than course content.

    These questions are answered from the active concept list so ACRLA can say
    exactly what Moodle/PDF sync has made available without invoking RAG.
    """
    msg = message.lower()
    if any(marker in msg for marker in ["weak", "struggling", "mastery", "score", "progress", "perform", "level"]):
        return False
    has_course_scope = any(
        marker in msg
        for marker in ["whole course", "entire course", "course contain", "course has", "all chapters"]
    )
    asks_chapters = bool(re.search(r"\b(how many|list|what|which|show).*\bchapters?\b", msg))
    asks_overview = bool(re.search(r"\b(course|syllabus|overview|structure)\b", msg)) and "chapter" in msg
    return has_course_scope or asks_chapters or asks_overview


def _course_scope_redirect(
    current_topic: str | None = None,
    available_concepts: set[str] | tuple[str, ...] | list[str] | None = None,
) -> str:
    concepts = list(available_concepts or [])
    if not concepts:
        return (
            "This course does not have synced ACRLA concepts yet.\n\n"
            "Please sync or upload the Moodle course PDFs first, then I can help with remediation."
        )
    topics = "\n".join(f"- {concept}" for concept in concepts)
    current = _course_local_concept(current_topic)
    followup = (
        f"Let's get back to {current}. Want to continue there?"
        if current
        else "Which topic would you like to study?"
    )
    return (
        "This question is outside the scope of the current course.\n\n"
        "I can help with:\n"
        f"{topics}\n\n"
        f"{followup}"
    )


def _is_off_topic_question(
    message: str,
    intent: Intent,
    selected_concept: str | None,
    session_id: str = "",
) -> bool:
    if selected_concept or canonicalize_concept(message):
        return False
    if intent in {Intent.ANALYTICS, Intent.NAVIGATION, Intent.PREFERENCE, Intent.ENGAGEMENT, Intent.GREETING}:
        return False
    if _is_course_structure_question(message) or _is_practice_request(message):
        return False
    if _awaiting_answer(session_id):
        return False

    msg = message.lower()
    course_markers = [
        "course",
        "chapter",
        "study",
        "learn",
        "concept",
        "algorithm",
        "code",
        "program",
        "memory",
        "pointer",
        "recursion",
        "recursive",
        "sorting",
        "sort",
        "binary tree",
        "bst",
        "tree",
        "logic",
        "truth table",
        "predicate",
        "set",
        "sets",
        "graph",
        "graphs",
        "vertex",
        "edge",
    ]
    if any(marker in msg for marker in course_markers):
        return False

    general_markers = [
        "recipe",
        "pancake",
        "grape",
        "grapes",
        "fruit",
        "food",
        "movie",
        "song",
        "game",
        "world cup",
        "weather",
        "news",
        "president",
        "capital of",
    ]
    if any(marker in msg for marker in general_markers):
        return True

    if re.search(r"\b(what|who|when|where|why|how|give me|tell me|i want)\b", msg):
        return True
    return False


GENERAL_EXTERNAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are ACRLA using automatic fallback support.
Do not claim the student selected internal or external mode.
If asked about routing, say: "ACRLA automatically searches course materials first. If relevant course material is found, I answer from it and show the source. If not, I use fallback support."
Answer general factual questions briefly and pedagogically without citing course PDFs."""),
    ("human", "{question}"),
])


def _generate_external_general_response(message: str, username: str = "Student") -> str:
    chain = GENERAL_EXTERNAL_PROMPT | get_llm(temperature=0.4, max_tokens=FINAL_ANSWER_MAX_TOKENS)
    response = chain.invoke({
        "student_name": username or "Student",
        "question": message,
    })
    return _sanitize_reply(response.content)


def _resolve_selected_concept(message: str, weak_concepts: list[str], session_id: str = "") -> str | None:
    direct = canonicalize_concept(message)
    if direct:
        return direct
    if _is_practice_request(message):
        return _extract_topic(message, weak_concepts, session_id)
    remembered_topic = _course_local_concept(_practice_state.get(session_id, {}).get("topic"))
    if remembered_topic and message.strip().lower() in {"yes", "yeah", "yep", "sure", "ok", "okay"}:
        return remembered_topic
    return None


def _current_topic_for_session(
    memory: MemoryManager,
    student_id: str,
    course_id: str,
    session_id: str,
) -> str | None:
    """Resolve the best current topic for vague follow-ups.

    Priority is in-memory current topic, then practice state, then persisted
    course memory. The result is used for "explain more" style messages, but it
    is still checked against `available_concepts` before being trusted.
    """
    current = _course_local_concept(memory.get_current_topic(session_id))
    if current:
        return current
    remembered = _course_local_concept(_practice_state.get(session_id, {}).get("topic"))
    if remembered:
        memory.set_current_topic(session_id, remembered)
        return remembered
    last_concept = _course_local_concept(memory.get_course_memory(student_id, course_id).get("last_concept"))
    if last_concept:
        memory.set_current_topic(session_id, last_concept)
        return last_concept
    return None


def _launch_concept_for_session(
    memory: MemoryManager,
    student_id: str,
    course_id: str,
    session_id: str,
) -> str | None:
    """Return the Moodle launch concept that should anchor remediation.

    Chapter launches use this to keep "test me" and "continue" on the clicked
    chapter until the student explicitly asks for a valid different concept.
    """
    course_memory = memory.get_course_memory(student_id, course_id)
    launch_concept = _course_local_concept(course_memory.get("launch_concept"))
    if not launch_concept:
        launch_context = dict(course_memory.get("launch_context") or {})
        launch_concept = _course_local_concept(launch_context.get("concept"))
    if launch_concept:
        memory.set_current_topic(session_id, launch_concept)
    return launch_concept


def _locked_chapter_concept_for_session(
    memory: MemoryManager,
    student_id: str,
    course_id: str,
) -> str | None:
    course_memory = memory.get_course_memory(student_id, course_id)
    if course_memory.get("locked_level_type") == "chapter":
        return _course_local_concept(course_memory.get("locked_concept"))
    launch_context = dict(course_memory.get("launch_context") or {})
    if launch_context.get("level_type") == "chapter":
        return _course_local_concept(launch_context.get("locked_concept") or launch_context.get("concept"))
    return None


def _locked_chapter_redirect(locked_concept: str) -> str:
    return policies.locked_chapter_redirect_response(locked_concept)


def _launch_level_for_session(
    memory: MemoryManager,
    student_id: str,
    course_id: str,
) -> str:
    course_memory = memory.get_course_memory(student_id, course_id)
    launch_context = dict(course_memory.get("launch_context") or {})
    level = str(
        launch_context.get("level_type")
        or course_memory.get("level_type")
        or "chapter"
    ).strip().lower()
    return level if level in {"chapter", "course", "overall"} else "chapter"


def _retrieval_course_ids_for_level(level: str, current_moodle_course_id: int, db: DBSession) -> list[int]:
    """Return the Moodle course IDs that retrieval is allowed to search.

    Chapter and course remediation search ONLY the current Moodle course --
    this branch returns a single-element list by construction, so a
    chapter/course-scoped turn can never fan out into more than one Chroma
    collection no matter what else is in the `courses` table.

    Overall remediation may search all synced Moodle courses, but "synced"
    means it actually has course material: a `Course` row can exist (e.g. a
    Moodle launch created it before any material was ever uploaded, or a
    leftover test/demo course_id) with no materials manifest and no
    discoverable concepts. Querying that row's Chroma collection is always an
    empty/stale no-op, so it is excluded here the same way
    `_canonical_synced_courses` already excludes it from mastery/analytics --
    this is the final course-level retrieval boundary before RAG is called,
    so it must apply the same exclusion, not just repeat the raw course list.
    """
    resolved_level = str(level or "chapter").lower()
    if resolved_level != "overall":
        print(
            "[ACRLA] retrieval_scope "
            f"level={resolved_level} "
            f"included_course_ids=[{current_moodle_course_id}]"
        )
        return [current_moodle_course_id]

    from models.db_models import Course
    ids: list[int] = []
    excluded: list[tuple[str, int | None]] = []
    for course in db.query(Course).all():
        if course.moodle_course_id is None:
            continue
        has_manifest = _manifest_moodle_id_for_course(course) is not None
        concepts = _material_concepts_for_course(course)
        if not has_manifest and not concepts:
            excluded.append((course.name, course.moodle_course_id))
            continue
        ids.append(course.moodle_course_id)
    resolved = sorted(set(ids)) or [current_moodle_course_id]
    print(
        "[ACRLA] retrieval_scope_overall "
        f"included_course_ids={resolved} "
        f"excluded_stale_course_ids={excluded}"
    )
    return resolved


def _concepts_for_level(
    level: str,
    current_moodle_course_id: int,
    db: DBSession,
    memory: MemoryManager,
    student_id: str,
) -> list[str]:
    if str(level or "chapter").lower() == "overall":
        from models.db_models import Course
        concepts: list[str] = []
        for course in db.query(Course).all():
            for concept in _dynamic_concepts_for_course(memory, student_id, course):
                if concept not in concepts:
                    concepts.append(concept)
        return concepts
    from models.db_models import Course
    course = db.query(Course).filter_by(moodle_course_id=current_moodle_course_id).first()
    if not course:
        return []
    course_concepts = _dynamic_concepts_for_course(memory, student_id, course)
    if str(level or "chapter").lower() == "chapter":
        # A chapter-scoped launch must lock remediation to the single concept
        # the student actually clicked into -- routers/api.py's Moodle-launch
        # handler already records this as course_memory["locked_concept"] /
        # launch_context["locked_concept"] specifically when level_type ==
        # "chapter" (never set for course/overall), and the REST assessment
        # path (_assessment_scope) already reads it. This chat-side function
        # previously ignored it entirely, so a chapter launch silently
        # behaved exactly like a course launch (RQ1 post-fix production bug,
        # fixed here without inventing any new chapter/concept mapping).
        course_memory = memory.get_course_memory(student_id, course.id)
        launch_context = course_memory.get("launch_context") or {}
        locked_concept = (
            course_memory.get("locked_concept")
            or launch_context.get("locked_concept")
            or launch_context.get("concept")
        )
        if locked_concept:
            normalized = _course_local_concept(locked_concept) or locked_concept
            if normalized in course_concepts:
                return [normalized]
            # Locked concept isn't part of this course's synced material
            # (stale/mismatched launch context) -- fall back to the full
            # course pool rather than returning an empty eligible list.
    return course_concepts


def _remediation_scope_instruction(level: str, available_concepts: set[str]) -> str:
    concepts = ", ".join(sorted(available_concepts))
    level = str(level or "chapter").lower()
    if level == "overall":
        return (
            "Overall-level launch: retrieval may use all synced Moodle courses for this student. "
            "Question generation may combine concepts from different courses and should encourage interdisciplinary reasoning."
        )
    if level == "course":
        return (
            "Course-level launch: retrieval must stay within the selected Moodle course. "
            f"Question generation may combine these course concepts: {concepts}. Encourage integration of chapter knowledge."
        )
    return (
        "Chapter-level launch: retrieval and question generation must stay inside the clicked chapter/concept only. "
        "Do not ask cross-chapter questions."
    )


def _should_stay_on_launch_concept(message: str) -> bool:
    msg = message.lower().strip(" ?.!") 
    if canonicalize_concept(msg):
        return False
    if _is_weakest_concepts_question(msg) or _is_strongest_concepts_question(msg) or _is_overall_remediation_request(msg):
        return False
    if _is_practice_request(msg) or _is_vague_followup(msg):
        return True
    return any(marker in msg for marker in [
        "another question",
        "give me another",
        "next question",
        "continue",
        "explain",
        "remediate",
        "practice",
    ])


def _is_overall_remediation_request(message: str) -> bool:
    msg = message.lower()
    return any(marker in msg for marker in [
        "overall remediation",
        "remediate everything",
        "whole course remediation",
        "all concepts remediation",
        "all topics remediation",
    ])


def _is_vague_followup(message: str) -> bool:
    msg = message.lower().strip(" ?.!") 
    if canonicalize_concept(msg):
        return False
    vague_exact = {
        "this",
        "it",
        "that",
        "the topic",
        "this topic",
        "that topic",
        "explain more",
        "more",
        "tell me more",
        "go deeper",
        "continue",
        "continue please",
        "can you explain more",
        "can you explain it more",
        "suggest videos",
        "suggest videos about it",
        "suggest videos about the topic",
    }
    if msg in vague_exact:
        return True
    if len(msg.split()) <= 5 and re.search(r"\b(it|this|that|topic|more)\b", msg):
        return True
    vague_patterns = [
        r"\b(explain|describe|tell me|show me|give me|suggest|recommend)\b.*\b(it|this|that|the topic|topic)\b",
        r"\b(more|again|deeper|details?|examples?|videos?|resources?)\b",
    ]
    return any(re.search(pattern, msg) for pattern in vague_patterns)


def _contextual_retrieval_query(message: str, current_topic: str | None) -> str:
    topic = canonicalize_concept(current_topic)
    if topic and _is_vague_followup(message):
        return f"{topic} {message}"
    return message


def _is_unsupported_course_topic_question(message: str) -> bool:
    if canonicalize_concept(message) or _is_course_structure_question(message):
        return False
    msg = message.lower()
    unsupported_markers = [
        "data structures",
        "variables",
        "variable",
        "loops",
        "loop",
        "current topic",
        "the current topic",
    ]
    asks_about_topic = any(
        marker in msg
        for marker in ["what is", "what are", "explain", "teach", "test me", "quiz me", "weak in", "topic"]
    )
    return asks_about_topic and any(marker in msg for marker in unsupported_markers)


def _is_practice_request(message: str) -> bool:
    msg = message.lower().strip()
    practice_phrases = [
        "test me",
        "quiz me",
        "ask me",
        "give me a question",
        "practice question",
        "try a question",
    ]
    return msg in {"yes", "yeah", "yep", "sure", "ok", "okay"} or any(
        phrase in msg for phrase in practice_phrases
    )


def _message_is_new_intent(message: str, intent: Intent) -> bool:
    msg = message.lower().strip()
    if _extract_session_name_update(message) or _is_name_recall_question(message) or _is_preference_question(message):
        return True
    if intent in {Intent.ANALYTICS, Intent.NAVIGATION, Intent.PREFERENCE, Intent.GREETING}:
        return True
    if _is_all_chapter_mastery_question(message) or _is_course_structure_question(message):
        return True
    if _is_practice_request(message) and canonicalize_concept(message):
        return True
    if any(msg.startswith(prefix) for prefix in ["explain ", "teach ", "switch topic", "change topic"]):
        return True
    if canonicalize_concept(message) and any(
        marker in msg
        for marker in ["explain", "teach", "test me", "quiz me", "switch", "change", "bst", "pointers", "recursion"]
    ):
        return True
    return False


def _extract_topic(message: str, weak_concepts: list[str], session_id: str = "") -> str | None:
    msg = message.lower()
    for marker in [" on ", " about ", " for "]:
        if marker in msg:
            topic = msg.split(marker, 1)[1].strip(" ?.!")
            canonical = canonicalize_concept(topic)
            if canonical:
                return canonical
    remembered_topic = _practice_state.get(session_id, {}).get("topic")
    remembered_topic = canonicalize_concept(remembered_topic)
    if remembered_topic:
        return remembered_topic
    buffer_topics = _topics_from_buffer(session_id)
    for topic in buffer_topics:
        canonical = canonicalize_concept(topic)
        if canonical:
            return canonical
    for concept in weak_concepts:
        canonical = canonicalize_concept(concept)
        if canonical:
            return canonical
    return None


def _topics_from_buffer(session_id: str) -> list[str]:
    if not session_id:
        return []
    text = "\n".join(m["content"] for m in get_buffer(session_id).messages[-6:])
    matches = re.findall(r"(?:focus on|focus areas:)\s*([^.\n]+)", text, re.IGNORECASE)
    matches.extend(re.findall(r"test me on\s+([^'\"*.\n]+)", text, re.IGNORECASE))
    topics = []
    for match in matches:
        cleaned = re.sub(r"<[^>]+>", "", match)
        cleaned = cleaned.replace("today and build it up step by step", "")
        for part in cleaned.split(","):
            topic = part.strip(" *'\".")
            canonical = canonicalize_concept(topic)
            if canonical:
                topics.append(canonical)
    return topics


def _is_mc_letter_answer(message: str) -> bool:
    """Returns True if the message is a bare multiple-choice letter (A–D)."""
    return bool(re.fullmatch(r"[a-dA-D]\.?", message.strip()))


def _is_invalid_mc_answer(message: str) -> bool:
    text = message.strip()
    if not text or _is_mc_letter_answer(text):
        return False
    return bool(re.fullmatch(r"[A-Za-z]\.?", text))


def _last_mc_question_from_buffer(session_id: str) -> str | None:
    """Returns the most recent assistant message that contains MC options (A./B. or A)/B))."""
    for msg in reversed(get_buffer(session_id).messages):
        if msg["role"] == "assistant":
            content = msg["content"]
            if re.search(r"\bA[.)]\s+\S", content) and re.search(r"\bB[.)]\s+\S", content):
                return content
    return None


def _last_assistant_question(session_id: str) -> str | None:
    state = _practice_state.get(session_id, {})
    if state.get("awaiting_answer") and state.get("current_question"):
        return state["current_question"]
    return None


def _last_easy_mc_question(session_id: str) -> str | None:
    question = _last_assistant_question(session_id)
    if not question:
        return None
    meta = _practice_question_meta(question)
    if meta and meta["difficulty"] == "easy":
        return question
    return None


def _practice_question_meta(content: str) -> dict | None:
    match = re.search(
        r"\b(Easy|Moderate|Hard)\s+(?:multiple-choice|open-ended|code/design)\s+question\s+on\s+([^:\n]+)",
        content,
        re.IGNORECASE,
    )
    if not match:
        return None
    difficulty = match.group(1).lower()
    if difficulty == "moderate":
        difficulty = "medium"
    return {
        "difficulty": difficulty,
        "concept": canonicalize_concept(match.group(2).strip()),
    }


def _evaluate_practice_answer(
    session_id: str,
    answer: str,
    memory: MemoryManager,
    student_id: str,
    course_id: str,
    session: SessionModel,
    db: DBSession,
) -> dict | None:
    question = _last_assistant_question(session_id)
    if not question:
        return None

    meta = _practice_question_meta(question)
    if not meta:
        return None

    normalized = answer.strip().lower()
    if not normalized:
        return None

    difficulty = meta["difficulty"]
    concept = meta["concept"]
    if not concept:
        return None
    memory.set_current_topic(session_id, concept)

    if difficulty == "easy":
        if not _is_mc_letter_answer(answer):
            return None
        correct = normalized[:1] == "b"
    elif difficulty == "hard":
        if _looks_like_clarification(normalized):
            return None
        correct = len(normalized.split()) >= 12 and any(
            token in normalized for token in ["def ", "for ", "while ", "{", "}", "return", "algorithm", "step"]
        )
    else:
        if _looks_like_clarification(normalized):
            return None
        correct = len(normalized.split()) >= 8

    memory.record_answer_history(
        student_id=student_id,
        course_id=course_id,
        concept=concept,
        question_text=_question_identity(question),
        correct=correct,
    )
    if session.analytics and correct:
        session.analytics.correct_answers = (session.analytics.correct_answers or 0) + 1
        db.commit()
    progress_note = "Mastery will update after you complete a quick progress check."
    next_level = _record_practice_result(session_id, difficulty, correct)
    current_mastery = memory.get_mastery(student_id, course_id, concept)
    answer_strategy = select_tutoring_strategy(current_mastery, session.difficulty, concept)

    if correct:
        reply = _adaptive_correct_feedback(concept, difficulty, answer_strategy.name, next_level, progress_note)
    else:
        reply = _adaptive_wrong_feedback(concept, difficulty, answer_strategy.name, progress_note)

    _clear_question_state(session_id)
    return {"reply": reply, "mastery_update": None}


def _adaptive_correct_feedback(
    concept: str,
    difficulty: str,
    strategy_name: str,
    next_level: str | None,
    mastery_tag: str,
) -> str:
    strategy = (strategy_name or "guided_practice").strip().lower()
    base = f"Good work. Your understanding of {concept} is improving!\n\n{mastery_tag}"

    if strategy == "advanced_challenge":
        stretch = ""
        if difficulty == "easy":
            stretch = "\n\nYou handled that easily. Want a harder challenge next?"
        elif next_level:
            label = _difficulty_label(next_level)
            stretch = f"\n\nYou're ready for {label} questions if you want to raise the challenge."
        else:
            stretch = "\n\nFor a stretch, try explaining an edge case or comparing this with another course concept."
        return f"{base}{stretch}\n\nWant to try another question on this?"

    if strategy == "simplified_remediation":
        return (
            f"{base}\n\n"
            "Nice progress. Keep the same idea in mind and we'll build it one step at a time.\n\n"
            "Want to try another question on this?"
        )

    if next_level:
        label = _difficulty_label(next_level)
        return f"{base}\n\nYou're getting steadier. You can stay here or try {label} next.\n\nWant another question?"

    return f"{base}\n\nWant to try another question on this?"


def _adaptive_wrong_feedback(
    concept: str,
    difficulty: str,
    strategy_name: str,
    mastery_tag: str,
) -> str:
    strategy = (strategy_name or "guided_practice").strip().lower()
    reminder = _quick_reminder_for_topic(concept)
    hint = _support_hint_for_topic(concept, difficulty)

    if strategy == "simplified_remediation":
        return (
            f"Not quite yet. Let's strengthen {concept} with one more step.\n\n"
            f"{mastery_tag}\n\n"
            f"Quick reminder: {reminder}\n\n"
            f"Hint for your next try: {hint}\n\n"
            "Want to retry with scaffolded steps?"
        )

    if strategy == "advanced_challenge":
        return (
            f"Not quite. The key issue is worth debugging carefully.\n\n"
            f"{mastery_tag}\n\n"
            f"Hint: {hint}\n\n"
            "Want to revise your answer, or try a different challenge on this concept?"
        )

    return (
        f"Not quite yet. Let's tighten the reasoning for {concept}.\n\n"
        f"{mastery_tag}\n\n"
        f"Hint: {hint}\n\n"
        "Want another practice question on this?"
    )


def _record_practice_result(session_id: str, difficulty: str, correct: bool) -> str | None:
    state = _get_practice_state(session_id)
    streaks = state.setdefault("streak", {})
    if not correct:
        streaks[difficulty] = 0
        return None

    streaks[difficulty] = streaks.get(difficulty, 0) + 1
    if difficulty == "easy" and streaks[difficulty] >= 3:
        streaks[difficulty] = 0
        return "medium"
    if difficulty == "medium" and streaks[difficulty] >= 2:
        streaks[difficulty] = 0
        return "hard"
    return None


def _looks_like_clarification(message: str) -> bool:
    clarification_markers = [
        "what topic",
        "what do you mean",
        "which topic",
        "clarify",
        "explain the question",
        "i don't understand",
        "dont understand",
    ]
    return any(marker in message for marker in clarification_markers)


def _build_practice_question(
    message: str,
    difficulty: str,
    weak_concepts: list[str],
    session_id: str = "",
    memory: MemoryManager = None,
    student_id: str = None,
    course_id: str = None,
    strategy_name: str = "guided_practice",
    strategy_reason: str = "",
    remediation_level: str = "chapter",
    available_concepts: list[str] = None,
    all_level_concepts: list[str] = None,
    requested_concepts: list[str] = None,
) -> str:
    remediation_level = str(remediation_level or "chapter").lower()
    explicit_requested = [
        concept for concept in (requested_concepts or [])
        if _course_local_concept(concept)
    ]
    if remediation_level in {"course", "overall"} and (len(explicit_requested) > 1 or not canonicalize_concept(message)):
        concepts = explicit_requested or all_level_concepts or available_concepts or weak_concepts
        if not concepts:
            return "I do not have synced course concepts to test yet. Please sync the Moodle PDFs first."
        return _build_integrated_practice_question(
            concepts=concepts,
            difficulty=difficulty,
            session_id=session_id,
            memory=memory,
            student_id=student_id,
            course_id=course_id,
            strategy_name=strategy_name,
            strategy_reason=strategy_reason,
            remediation_level=remediation_level,
        )

    topic = explicit_requested[0] if explicit_requested else _extract_topic(message, weak_concepts, session_id)
    if not topic:
        allowed = ", ".join(available_concepts or [])
        if not allowed:
            return "I do not have synced course concepts to test yet. Please sync the Moodle PDFs first."
        return f"Which course topic should I test you on? Choose one of: {allowed}."
    state = _get_practice_state(session_id)
    state["topic"] = topic
    sub_focus = _next_sub_concept_for_topic(topic, state)
    avoided_questions = (
        memory.get_recent_question_texts(student_id, course_id, topic)
        if memory and student_id and course_id else []
    )

    if difficulty == "easy":
        questions = _easy_questions_for_topic(topic)
        identities = [item[0] for item in questions]
        variant = _next_question_index(state, difficulty, topic, len(questions), identities, avoided_questions)
        prompt, a, b, c, d = questions[variant % len(questions)]
        question = _apply_adaptive_question_support(
            (
            f"Target sub-concept: {sub_focus}\n\n" if sub_focus else ""
            ) + (
            f"Easy multiple-choice question on {topic}:\n\n"
            f"{prompt}\n\n"
            f"{a}\n"
            f"{b}\n"
            f"{c}\n"
            f"{d}\n\n"
            "Reply with A, B, C, or D."
            ),
            topic,
            difficulty,
            strategy_name,
            strategy_reason,
        )
        _set_active_question(session_id, question)
        if memory and student_id and course_id:
            memory.record_asked_question(student_id, course_id, topic, _question_identity(question))
        return question
    if difficulty == "hard":
        questions = _hard_questions_for_topic(topic)
        variant = _next_question_index(state, difficulty, topic, len(questions), questions, avoided_questions)
        question = _apply_adaptive_question_support(
            f"{'Target sub-concept: ' + sub_focus + chr(10) + chr(10) if sub_focus else ''}"
            f"Hard code/design question on {topic}:\n\n{questions[variant % len(questions)]}",
            topic,
            difficulty,
            strategy_name,
            strategy_reason,
        )
        _set_active_question(session_id, question)
        if memory and student_id and course_id:
            memory.record_asked_question(student_id, course_id, topic, _question_identity(question))
        return question

    questions = _moderate_questions_for_topic(topic)
    variant = _next_question_index(state, difficulty, topic, len(questions), questions, avoided_questions)
    question = _apply_adaptive_question_support(
        f"{'Target sub-concept: ' + sub_focus + chr(10) + chr(10) if sub_focus else ''}"
        f"Moderate open-ended question on {topic}:\n\n{questions[variant % len(questions)]}",
        topic,
        difficulty,
        strategy_name,
        strategy_reason,
    )
    _set_active_question(session_id, question)
    if memory and student_id and course_id:
        memory.record_asked_question(student_id, course_id, topic, _question_identity(question))
    return question


def _build_integrated_practice_question(
    concepts: list[str],
    difficulty: str,
    session_id: str,
    memory: MemoryManager = None,
    student_id: str = None,
    course_id: str = None,
    strategy_name: str = "guided_practice",
    strategy_reason: str = "",
    remediation_level: str = "course",
) -> str:
    canonical_concepts = []
    for concept in concepts:
        canonical = _course_local_concept(concept)
        if canonical and canonical not in canonical_concepts:
            canonical_concepts.append(canonical)
    if not canonical_concepts:
        return "I do not have enough synced concepts to build an integrated question yet."

    state = _get_practice_state(session_id)
    counts = state.setdefault("counts", {})
    key = f"integrated:{remediation_level}:{difficulty}"
    index = counts.get(key, 0)
    counts[key] = index + 1

    rotated = canonical_concepts[index % len(canonical_concepts):] + canonical_concepts[:index % len(canonical_concepts)]
    primary = rotated[0]
    secondary = rotated[1] if len(rotated) > 1 else primary
    concept_list = _human_list(rotated)
    all_concepts_sentence = ", ".join(rotated)
    scope_label = "overall interdisciplinary" if remediation_level == "overall" else "course-integrated"
    scope_hint = (
        "Use ideas from different Moodle courses where helpful."
        if remediation_level == "overall"
        else "Connect ideas from multiple chapters in this Moodle course."
    )
    concrete = _concrete_synthesis_question(rotated, difficulty)
    if concrete:
        question = concrete
    elif difficulty == "easy":
        question = _fallback_easy_synthesis_question(rotated, concept_list, all_concepts_sentence)
    elif difficulty == "hard":
        question = _fallback_hard_synthesis_question(rotated, concept_list, all_concepts_sentence)
    else:
        question = _fallback_moderate_synthesis_question(rotated, concept_list, all_concepts_sentence)

    state["topic"] = primary
    question = _apply_adaptive_question_support(
        f"{scope_label.capitalize()} remediation. {scope_hint}\n\n{question}",
        primary,
        difficulty,
        strategy_name,
        strategy_reason,
    )
    _set_active_question(session_id, question)
    if memory and student_id and course_id:
        memory.record_asked_question(student_id, course_id, primary, _question_identity(question))
    return question


def _human_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _concrete_synthesis_question(concepts: list[str], difficulty: str) -> str | None:
    concept_set = set(concepts)
    if {"Logic", "Sets", "Graphs"}.issubset(concept_set):
        if difficulty == "hard":
            return (
                "Hard synthesis question using Logic, Sets, and Graphs:\n\n"
                "A graph has vertices V = {A, B, C, D} and edges E = {(A, B), (A, C), (C, D)}. "
                "Let S = {A, C}. Write a logical statement using set notation that means: "
                "\"every vertex in S has at least one neighbor in V.\" Then decide whether the statement is true for this graph."
            )
        if difficulty == "medium":
            return (
                "Moderate synthesis question using Logic, Sets, and Graphs:\n\n"
                "A graph has vertices V = {A, B, C} and edges E = {(A, B), (B, C)}. Let S = {A, C}. "
                "Explain whether the logical statement \"for every vertex x in S, there exists a vertex y in V such that (x, y) is an edge\" is true."
            )
        return (
            "Easy multiple-choice synthesis question using Logic, Sets, and Graphs:\n\n"
            "A graph has vertices V = {A, B, C}. Let S = {A, C}. Which logical statement correctly says that every vertex in S has at least one edge to another vertex?\n\n"
            "A. For every x in S, there exists y in V such that (x, y) is an edge.\n"
            "B. There exists x in S such that x is not in V.\n"
            "C. For every x in V, x must be equal to A.\n"
            "D. S is empty, so no vertices need edges.\n\n"
            "Reply with A, B, C, or D."
        )

    if {"Graphs", "Recursion"}.issubset(concept_set):
        if difficulty == "hard":
            return (
                "Hard synthesis question using Graphs and Recursion:\n\n"
                "Write pseudocode for recursive DFS on a graph represented by an adjacency list. "
                "Include the base case that prevents revisiting vertices, and explain what happens if that base case is missing."
            )
        if difficulty == "medium":
            return (
                "Moderate synthesis question using Graphs and Recursion:\n\n"
                "In depth-first search, explain how recursion helps explore neighboring vertices. "
                "Use a small graph with vertices {A, B, C} and describe the base case."
            )
        return (
            "Easy multiple-choice synthesis question using Graphs and Recursion:\n\n"
            "In depth-first search (DFS), recursion is used to:\n\n"
            "A. visit neighboring vertices until a base case is reached.\n"
            "B. sort vertices alphabetically.\n"
            "C. remove all edges from the graph.\n"
            "D. replace the graph with a set.\n\n"
            "Reply with A, B, C, or D."
        )

    if {"Sorting Algorithms", "Pointers and Memory Management"}.issubset(concept_set):
        if difficulty == "hard":
            return (
                "Hard synthesis question using Sorting Algorithms and Pointers and Memory Management:\n\n"
                "Design a sorting approach for a linked list of integers. Explain how pointer updates can rearrange nodes without copying every data value, "
                "and mention one memory-safety risk to avoid."
            )
        if difficulty == "medium":
            return (
                "Moderate synthesis question using Sorting Algorithms and Pointers and Memory Management:\n\n"
                "You are sorting a linked list. Explain why changing node pointers can be more efficient than copying each node's data during sorting."
            )
        return (
            "Easy multiple-choice synthesis question using Sorting Algorithms and Pointers and Memory Management:\n\n"
            "When sorting a linked list, why are pointers useful?\n\n"
            "A. They allow nodes to be rearranged without copying all data.\n"
            "B. They remove the need for comparisons.\n"
            "C. They automatically sort the list.\n"
            "D. They prevent memory allocation.\n\n"
            "Reply with A, B, C, or D."
        )

    return None


def _fallback_easy_synthesis_question(concepts: list[str], concept_list: str, all_concepts_sentence: str) -> str:
    first = concepts[0]
    second = concepts[1] if len(concepts) > 1 else concepts[0]
    return (
        f"Easy multiple-choice synthesis question using {all_concepts_sentence}:\n\n"
        f"A student is solving a small problem that requires {concept_list}. "
        f"Which option correctly uses {first} and {second} in the same concrete task?\n\n"
        f"A. Use {first} to solve the main step, then check the result with {second} in the example.\n"
        f"B. Ignore {second} because only one concept can be used at a time.\n"
        f"C. Replace the task with definitions only, without applying the concepts.\n"
        f"D. Use the concepts as labels but do not connect them to the problem.\n\n"
        "Reply with A, B, C, or D."
    )


def _fallback_moderate_synthesis_question(concepts: list[str], concept_list: str, all_concepts_sentence: str) -> str:
    return (
        f"Moderate synthesis question using {all_concepts_sentence}:\n\n"
        f"Create a small example problem where {concept_list} are all needed. "
        "Explain the example and state exactly what role each concept plays in solving it."
    )


def _fallback_hard_synthesis_question(concepts: list[str], concept_list: str, all_concepts_sentence: str) -> str:
    return (
        f"Hard synthesis question using {all_concepts_sentence}:\n\n"
        f"Design a short algorithm, proof, or structured solution for a concrete problem that requires {concept_list}. "
        "Your answer must include an edge case and explain why each requested concept is necessary."
    )


def _apply_adaptive_question_support(
    question: str,
    topic: str,
    difficulty: str,
    strategy_name: str,
    strategy_reason: str = "",
) -> str:
    strategy = (strategy_name or "guided_practice").strip().lower()
    reason = f" ({strategy_reason})" if strategy_reason else ""

    if strategy == "simplified_remediation":
        reminder = _quick_reminder_for_topic(topic)
        hint = _support_hint_for_topic(topic, difficulty)
        return (
            f"{_difficulty_label(difficulty)} question on {topic}. Since this is still a weak area{reason}, "
            f"here's a quick reminder: {reminder}\n\n"
            f"{question}\n\n"
            f"Hint: {hint}"
        )

    if strategy == "advanced_challenge":
        if difficulty == "easy":
            return (
                f"{question}\n\n"
                "Stretch: if this feels too easy, explain why the wrong options are wrong or ask me for a harder challenge."
            )
        return (
            f"{question}\n\n"
            "Challenge add-on: include one edge case, a complexity note, or a comparison with another course concept."
        )

    return (
        f"{question}\n\n"
        "Tip: answer in your own words first, then add a small example if you can."
    )


def _difficulty_label(difficulty: str) -> str:
    if difficulty == "medium":
        return "Moderate"
    return str(difficulty or "medium").capitalize()


def _next_sub_concept_for_topic(topic: str, state: dict) -> str:
    sub_concepts = sub_concepts_for(topic)
    if not sub_concepts:
        return ""
    key = f"sub:{canonicalize_concept(topic) or topic}"
    counts = state.setdefault("counts", {})
    index = counts.get(key, 0)
    counts[key] = index + 1
    return sub_concepts[index % len(sub_concepts)]


def _quick_reminder_for_topic(topic: str) -> str:
    topic_kind = _topic_kind(topic)
    if topic_kind == "pointers":
        return "a pointer stores a memory address; `&x` gets an address and `*p` uses the value at that address."
    if topic_kind == "recursion":
        return "recursion solves a problem by calling the same function on a smaller case until a base case stops it."
    if topic_kind == "sorting":
        return "sorting rearranges items into order; algorithms differ in how they compare, move, and split values."
    if topic_kind == "binary_trees":
        return "a binary tree node has at most two children, and a BST keeps smaller values left and larger values right."
    return f"focus on the core definition of {topic}, then apply it one step at a time."


def _support_hint_for_topic(topic: str, difficulty: str) -> str:
    topic_kind = _topic_kind(topic)
    if topic_kind == "pointers":
        return "track three things separately: the variable's value, its address, and what the pointer currently stores."
    if topic_kind == "recursion":
        return "write the base case first, then describe how each recursive call makes the problem smaller."
    if topic_kind == "sorting":
        return "name the input, the comparison rule, and how each pass or partition moves values closer to sorted order."
    if topic_kind == "binary_trees":
        return "think about the current node first, then decide whether to move left, move right, or stop."
    if difficulty == "hard":
        return "break your answer into setup, algorithm/code, and one edge case."
    return "start with the definition, then apply it to the example in the question."


def _next_question_index(
    state: dict,
    difficulty: str,
    topic: str,
    pool_size: int,
    question_texts: list[str] = None,
    avoided_questions: list[str] = None,
) -> int:
    key = f"{difficulty}:{topic.lower()}"
    asked = state.setdefault("asked", {})
    asked_for_pool = asked.setdefault(key, [])
    if len(asked_for_pool) >= pool_size:
        asked_for_pool.clear()

    normalized_avoided = {_normalize_question_text(q) for q in (avoided_questions or [])}
    question_texts = question_texts or [str(index) for index in range(pool_size)]

    for index in range(pool_size):
        if index not in asked_for_pool and _normalize_question_text(question_texts[index]) not in normalized_avoided:
            asked_for_pool.append(index)
            return index

    for index in range(pool_size):
        if index not in asked_for_pool:
            asked_for_pool.append(index)
            return index

    asked_for_pool.append(0)
    return 0


def _question_identity(question: str) -> str:
    parts = [part.strip() for part in question.split("\n\n") if part.strip()]
    for index, part in enumerate(parts):
        if re.search(r"\b(Easy|Moderate|Hard)\s+(?:multiple-choice|open-ended|code/design)\s+question\s+on\b", part, re.IGNORECASE):
            if index + 1 < len(parts):
                return parts[index + 1].splitlines()[0].strip()
            return part.splitlines()[0].strip()
    if len(parts) >= 2:
        return parts[1].splitlines()[0].strip()
    return question.strip()


def _normalize_question_text(question: str) -> str:
    return re.sub(r"\s+", " ", str(question).strip().lower())


def _easy_questions_for_topic(topic: str) -> list[tuple[str, str, str, str, str]]:
    topic_kind = _topic_kind(topic)
    if topic_kind == "recursion":
        return [
            (
                "Which statement best describes recursion?",
                "A. A loop that always runs forever.",
                "B. A function solving a problem by calling itself on a smaller version of the problem.",
                "C. A way to store a memory address.",
                "D. A method for sorting only arrays of size one.",
            ),
            (
                "What must a recursive function include to stop correctly?",
                "A. A pointer variable.",
                "B. A base case.",
                "C. A random number.",
                "D. A print statement after every line.",
            ),
            (
                "Why does recursion usually reduce the problem size?",
                "A. So the program ignores the input.",
                "B. So each call moves closer to the base case.",
                "C. So memory addresses are hidden.",
                "D. So no function ever returns.",
            ),
        ]
    if topic_kind == "pointers":
        return [
            (
                "In C-style code, what does `int *p = &x;` mean?",
                "A. `p` stores the value of `x` directly.",
                "B. `p` stores the memory address of `x`.",
                "C. `p` prints `x` to the screen.",
                "D. `p` creates a loop over `x`.",
            ),
            (
                "What does dereferencing a pointer with `*p` do?",
                "A. It deletes the pointer from memory.",
                "B. It accesses the value stored at the address inside `p`.",
                "C. It turns a loop into recursion.",
                "D. It sorts the value automatically.",
            ),
            (
                "What is a null pointer?",
                "A. A pointer that always points to the first variable.",
                "B. A pointer that does not currently point to a valid object.",
                "C. A pointer used only inside for loops.",
                "D. A pointer that stores a string length.",
            ),
            (
                "Which memory region usually stores local function variables?",
                "A. Code region.",
                "B. Stack.",
                "C. Heap.",
                "D. Data region only.",
            ),
            (
                "Which memory region is commonly used for dynamically allocated memory from `malloc`?",
                "A. Code region.",
                "B. Heap.",
                "C. Register names.",
                "D. Function names.",
            ),
            (
                "What does the address-of operator `&x` produce?",
                "A. The value stored inside `x` only.",
                "B. The memory address of `x`.",
                "C. A sorted copy of `x`.",
                "D. A new loop counter.",
            ),
            (
                "What can happen if dynamically allocated memory is never freed?",
                "A. The program always becomes faster.",
                "B. A memory leak can occur.",
                "C. The pointer becomes a base case.",
                "D. The stack automatically grows forever.",
            ),
            (
                "What is a dangling pointer?",
                "A. A pointer that stores the address of a valid live object.",
                "B. A pointer that still refers to memory that has already been freed or is no longer valid.",
                "C. A pointer that only appears in comments.",
                "D. A pointer used to choose a sorting pivot.",
            ),
            (
                "What is pointer arithmetic used for?",
                "A. Adding two unrelated strings.",
                "B. Moving a pointer through adjacent memory locations such as array elements.",
                "C. Preventing all memory leaks automatically.",
                "D. Calling a recursive function.",
            ),
            (
                "What can cause a buffer overflow?",
                "A. Freeing memory exactly once.",
                "B. Writing past the end of an allocated memory area.",
                "C. Reading a variable's address with `&`.",
                "D. Setting a pointer to `NULL` before use.",
            ),
            (
                "What should you do after calling `free(p)` if you want to reduce accidental reuse?",
                "A. Dereference `p` immediately.",
                "B. Set `p` to `NULL`.",
                "C. Add one to `p`.",
                "D. Store `p` in the code region.",
            ),
            (
                "What is the difference between `malloc` and `calloc` at a beginner level?",
                "A. `malloc` is only for the stack.",
                "B. `calloc` allocates memory and initializes it to zero.",
                "C. `calloc` frees memory automatically.",
                "D. `malloc` sorts memory addresses.",
            ),
        ]
    if topic_kind == "sorting":
        return [
            (
                "What is the main goal of sorting?",
                "A. To remove every duplicate variable.",
                "B. To arrange data into a chosen order.",
                "C. To store a memory address.",
                "D. To make recursion impossible.",
            ),
            (
                "In Quick Sort, what is the pivot used for?",
                "A. To delete the input list.",
                "B. To split values around a chosen reference value.",
                "C. To store the final answer in memory only.",
                "D. To stop all comparisons.",
            ),
            (
                f"Why do students analyze the efficiency of {topic}?",
                "A. To avoid writing any algorithm steps.",
                "B. To understand how runtime changes as input size grows.",
                "C. To choose random answers.",
                "D. To remove the need for testing.",
            ),
        ]
    if topic_kind == "binary_trees":
        return [
            (
                "What best describes a binary tree?",
                "A. A list where every value must be sorted.",
                "B. A tree structure where each node has at most two children.",
                "C. A pointer that always stores two addresses.",
                "D. A recursive function with no base case.",
            ),
            (
                "In a binary tree, what are the left and right children?",
                "A. The first two variables declared in a program.",
                "B. The two possible child nodes connected below a parent node.",
                "C. The two loops used to search an array.",
                "D. The two pivots used by Quick Sort.",
            ),
            (
                "What does an inorder traversal usually do in a binary search tree?",
                "A. It visits nodes in random order.",
                "B. It visits values in sorted order.",
                "C. It deletes every leaf node.",
                "D. It turns the tree into a pointer.",
            ),
        ]
    if topic_kind == "variables":
        return [
            (
                "What is a variable used for?",
                "A. To permanently remove data from a program.",
                "B. To store a value that a program can use or change.",
                "C. To stop all loops from running.",
                "D. To sort data automatically.",
            ),
            (
                "What happens when a variable is assigned a new value?",
                "A. The program deletes the variable name.",
                "B. The stored value associated with that variable changes.",
                "C. The computer creates a pointer automatically.",
                "D. The code becomes recursive.",
            ),
            (
                "Why should variable names be meaningful?",
                "A. They make the program run twice as fast.",
                "B. They help readers understand what data the variable represents.",
                "C. They remove the need for assignments.",
                "D. They prevent every syntax error.",
            ),
        ]
    if topic_kind == "loops":
        return [
            (
                "What is the main purpose of a loop?",
                "A. To store a memory address.",
                "B. To repeat a block of code while a condition or range applies.",
                "C. To make a function call itself.",
                "D. To erase variables after use.",
            ),
            (
                "What does a loop condition decide?",
                "A. The variable's memory address.",
                "B. Whether the loop should continue or stop.",
                "C. Which pointer is dereferenced.",
                "D. Whether sorting is forbidden.",
            ),
            (
                "What is an infinite loop?",
                "A. A loop with no code inside it.",
                "B. A loop that never reaches a stopping condition.",
                "C. A recursive function with a base case.",
                "D. A variable that changes once.",
            ),
        ]
    # Domain-neutral fallback for a concept this bank has no dedicated
    # (CS-course-specific) branch for -- e.g. a Data Science course's
    # "Linear Regression"/"Decision Trees". Deliberately does NOT invent
    # subject-specific facts (that would require either hardcoding this one
    # concept or an extra LLM call this deterministic path doesn't make);
    # it stays a genuine comprehension check about the concept's ROLE in
    # the course instead, and never assumes the course is programming/CS
    # (no "code"/"algorithm"/"program" wording, unlike the CS-specific
    # branches above, which correctly keep that language for their own
    # actual CS topics).
    return [
        (
            f"What is {topic} mainly about?",
            "A. It has no real meaning in this course.",
            f"B. {topic} is a course topic used to understand or work through a specific kind of problem.",
            "C. It is only ever mentioned in passing and never actually used.",
            "D. It applies equally to every subject and has no specific focus.",
        ),
        (
            f"Why would a student in this course study {topic}?",
            "A. It is only needed to answer one quiz question.",
            f"B. Understanding {topic} helps explain or work through problems covered in this course.",
            "C. It has no connection to anything else in the course.",
            "D. It matters only for memorizing a definition, never for applying it.",
        ),
        (
            f"When first learning {topic}, what should you focus on?",
            "A. Memorizing unrelated details first.",
            "B. The basic idea, then a simple example.",
            "C. Skipping the definition entirely.",
            "D. Avoiding any practice questions.",
        ),
    ]


def _topic_kind(topic: str) -> str:
    canonical = canonicalize_concept(topic)
    if canonical == "Recursion":
        return "recursion"
    if canonical == "Sorting Algorithms":
        return "sorting"
    if canonical == "Pointers and Memory Management":
        return "pointers"
    if canonical == "Binary Trees and BSTs":
        return "binary_trees"

    normalized = topic.lower()

    if (
        "chapter1" in normalized
        or "chapter 1" in normalized
        or "chapter1_recursion" in normalized
        or "recursion" in normalized
        or "recursive" in normalized
        or "base case" in normalized
    ):
        return "recursion"

    if (
        "chapter2" in normalized
        or "chapter 2" in normalized
        or "chapter2_sorting" in normalized
        or "sort" in normalized
        or "quick" in normalized
        or "merge" in normalized
        or "bubble" in normalized
        or "selection" in normalized
        or "pivot" in normalized
        or "partition" in normalized
    ):
        return "sorting"

    if (
        "chapter3" in normalized
        or "chapter 3" in normalized
        or "chapter3_pointers_memory" in normalized
        or "pointer" in normalized
        or "memory" in normalized
        or "address" in normalized
        or "dereference" in normalized
        or "null pointer" in normalized
    ):
        return "pointers"

    if (
        "chapter4" in normalized
        or "chapter 4" in normalized
        or "chapter4_binary_trees" in normalized
        or "binary tree" in normalized
        or "binary trees" in normalized
        or "tree traversal" in normalized
        or "inorder" in normalized
        or "preorder" in normalized
        or "postorder" in normalized
        or "leaf node" in normalized
        or "root node" in normalized
        or "binary search tree" in normalized
        or "bst" in normalized
    ):
        return "binary_trees"

    return "generic"


def _moderate_questions_for_topic(topic: str) -> list[str]:
    topic_kind = _topic_kind(topic)
    if topic_kind == "recursion":
        return [
            "Explain what a base case is and why a recursive function needs one.",
            "Describe how a recursive function moves from a larger problem to a smaller subproblem.",
            "Compare recursion with a loop for solving a repeated task. When might recursion be clearer?",
        ]
    if topic_kind == "pointers":
        return [
            "Given `int x = 7; int *p = &x;`, explain what `x`, `&x`, `p`, and `*p` each represent.",
            "Explain why changing `*p` can change the value of `x` when `p` stores `&x`.",
            "Describe what can go wrong if a pointer is uninitialized before it is dereferenced.",
            "Explain the difference between the stack and the heap in memory management.",
            "Describe the code, data, stack, and heap regions of a program's memory layout.",
            "Explain what `malloc` does and why the returned pointer should be checked before use.",
            "Compare `malloc`, `calloc`, and `realloc` at a high level.",
            "Explain why every successful dynamic allocation should eventually be paired with `free`.",
            "Describe a memory leak and give one simple example of how it can happen.",
            "Explain what a dangling pointer is and why dereferencing one is dangerous.",
            "Describe what a null pointer dereference means.",
            "Explain pointer arithmetic using an array example.",
            "Describe how a buffer overflow can happen when writing through a pointer.",
        ]
    if topic_kind == "sorting":
        return [
            "Explain how Quick Sort partitions a list around a pivot.",
            "Describe why Quick Sort can be fast on average but slow in an unlucky case.",
            "Compare Quick Sort with a simpler sorting method such as Bubble Sort or Selection Sort.",
        ]
    if topic_kind == "binary_trees":
        return [
            "Explain what a binary tree is and how parent, child, root, and leaf nodes relate to each other.",
            "Describe the difference between inorder, preorder, and postorder traversal in a binary tree.",
            "Explain why a balanced binary search tree can make searching faster than scanning a list.",
        ]
    if topic_kind == "variables":
        return [
            "Explain what a variable stores and how assigning a new value changes it.",
            "Describe the difference between declaring a variable and assigning a value to it.",
            "Give a short example of a variable whose value changes during a program.",
        ]
    if topic_kind == "loops":
        return [
            "Explain the difference between a for loop and a while loop.",
            "Describe how a loop condition controls when repetition stops.",
            "Give an example of a loop that processes every item in a list.",
        ]
    return [
        f"Explain {topic} in your own words, then give one short example of how it is used.",
        f"Compare {topic} with a related idea from the course. What makes it different?",
        f"Describe one common mistake students make with {topic}, and explain how to avoid it.",
    ]


def _hard_questions_for_topic(topic: str) -> list[str]:
    topic_kind = _topic_kind(topic)
    if topic_kind == "recursion":
        return [
            "Write recursive pseudocode for computing factorial(n). Include the base case and recursive case, then explain the call flow for n = 4.",
            "Design a recursive algorithm to sum all values in a list. Explain what happens when the list is empty.",
            "Write recursive pseudocode for finding the maximum value in a list. Mention one risk of deep recursion.",
        ]
    if topic_kind == "pointers":
        return [
            "Write a small C-style code example that declares an integer, stores its address in a pointer, dereferences it, and changes the original value.",
            "Design a function that swaps two integer values using pointers. Explain why passing addresses is necessary.",
            "Create a debugging scenario involving a null or uninitialized pointer. Explain the bug and show how to guard against it.",
            "Write C-style pseudocode that allocates an integer array with `malloc`, checks whether allocation succeeded, writes values, and then frees the memory.",
            "Given a function that returns the address of a local stack variable, explain why that creates a dangling pointer and rewrite it safely.",
            "Design a small example where forgetting `free` causes a memory leak. Explain exactly which allocation is leaked.",
            "Write pseudocode that grows a dynamic array with `realloc`. Include the safe pattern that avoids losing the original pointer if `realloc` fails.",
            "Analyze what happens when code writes one element past the end of an allocated array. Name the bug and explain the risk.",
            "Create a short example that uses pointer arithmetic to iterate through an integer array, then explain how the pointer advances.",
            "Compare stack allocation and heap allocation for a temporary array. Explain lifetime, ownership, and cleanup responsibilities.",
            "Write a guard pattern that prevents null pointer dereference before using `*p`.",
            "Debug this scenario: memory is freed, then the pointer is used again. Explain the error and show a safer version.",
        ]
    if topic_kind == "sorting":
        return [
            "Write pseudocode for Quick Sort, including partitioning around a pivot and the recursive calls.",
            "Design a partition function for Quick Sort and explain how it rearranges values less than and greater than the pivot.",
            "Analyze Quick Sort on an already sorted list when the first element is always chosen as pivot. Explain the time complexity.",
        ]
    if topic_kind == "binary_trees":
        return [
            "Write recursive pseudocode for inorder traversal of a binary tree. Explain what happens when the current node is null.",
            "Design an algorithm to search for a value in a binary search tree. Include the decisions made at each node.",
            "Write pseudocode to compute the height of a binary tree, then explain the base case and recursive case.",
        ]
    if topic_kind == "variables":
        return [
            "Write a short program trace showing how three variables change over five assignment statements.",
            "Design pseudocode that swaps two variables without losing either value. Explain each assignment.",
            "Create a bug caused by reusing a variable for two meanings, then rewrite the code more clearly.",
        ]
    if topic_kind == "loops":
        return [
            "Write pseudocode that uses a loop to count how many numbers in a list are even, then analyze its time complexity.",
            "Design a nested-loop algorithm for comparing every pair of items in a list. Explain why its runtime is O(n^2).",
            "Create a loop with an off-by-one bug, then correct it and explain the fix.",
        ]
    # Same domain-neutral rationale as _easy_questions_for_topic's own
    # fallback above -- no "code"/"algorithm" framing for a concept that
    # may not be a programming topic at all.
    return [
        f"Explain {topic} in depth: what problem does it address, and what are its key steps or components? Include one concrete example.",
        f"Walk through a scenario where {topic} would be applied. Cover the main steps, and explain one way it could go wrong if applied carelessly.",
        f"Compare {topic} with a related idea from the course. Describe a case where {topic} is clearly the better choice, and why.",
    ]


def _handle_preference(
    message: str,
    session: SessionModel,
    memory: MemoryManager,
    student_id: str,
    db: DBSession,
    session_id: str,
) -> tuple[str, SessionModel]:
    msg = message.lower()
    reply_parts = []

    requested_difficulty = _extract_requested_difficulty(msg)
    requested_mode = _extract_requested_mode(msg)

    if requested_difficulty == "easy":
        session.difficulty = "easy"
        reply_parts.append("Switching to easy difficulty.")
    elif requested_difficulty == "medium":
        session.difficulty = "medium"
        reply_parts.append("Switching to moderate difficulty.")
    elif requested_difficulty == "hard":
        session.difficulty = "hard"
        reply_parts.append("Switching to hard difficulty - I'll challenge you more!")

    if requested_mode == "external":
        reply_parts.append("ACRLA now chooses internal course material or external support automatically for each question.")
    elif requested_mode == "internal":
        reply_parts.append("ACRLA now chooses internal course material or external support automatically for each question.")

    db.commit()
    db.refresh(session)

    # Inject a clear difficulty reminder into conversation history
    buffer = get_buffer(session_id)
    buffer.add("system", f"IMPORTANT: The student just changed difficulty to {session.difficulty.upper()}. All future questions and explanations must strictly match {session.difficulty} level. Do not refer to any previous difficulty level.")

    memory.save_preferences(student_id, difficulty=session.difficulty)
    memory.set_profile_preferences(
        student_id,
        difficulty=session.difficulty,
    )

    return " ".join(reply_parts) or "Preference noted!", session


def _extract_requested_difficulty(message: str) -> str | None:
    msg = message.lower()
    if re.search(r"\b(easy|easier)\b", msg):
        return "easy"
    if re.search(r"\b(medium|moderate)\b", msg):
        return "medium"
    if re.search(r"\b(hard|harder|more difficult)\b", msg):
        return "hard"
    return None


def _extract_requested_mode(message: str) -> str | None:
    msg = message.lower()
    if re.search(r"\bexternal\b", msg):
        return "external"
    if re.search(r"\binternal\b", msg):
        return "internal"
    return None


def _get_moodle_course_id(session: SessionModel, db: DBSession) -> int:
    from models.db_models import Course
    course = db.query(Course).filter_by(id=session.course_id).first()
    return course.moodle_course_id if course else 1
