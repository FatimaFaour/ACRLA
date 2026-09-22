"""Deterministic dialogue policies: pure, canned-text answers.

Every function here is a pure function of already-known structured state --
no DB access, no LLM call, no side effects. They exist so the same wording
used by chat_orchestrator.py's legacy phrase-matched handlers is also
reachable as agent tools (see `tools.dialogue_tools`), instead of being
duplicated. Hard safety rules (mastery is assessment-gated, chat cannot
update mastery) stay expressed as plain deterministic text here -- nothing in
this module asks an LLM to decide safety-critical wording.
"""

from __future__ import annotations

from typing import Any


def mastery_modification_guard_response() -> str:
    """Refuse a request to change mastery from chat. Detection stays deterministic
    (see chat_orchestrator._is_mastery_modification_request); this is only the
    canned refusal text, reused by both the legacy guard and the agent tool.
    """
    return (
        "Chat explanations cannot update mastery.\n"
        "Only the Quick Progress Check assessment updates your mastery."
    )


def mastery_policy_response() -> str:
    """Explain ACRLA's mastery bands without changing any mastery value."""
    return (
        "ACRLA treats mastery below 50% as Weak, 50-79% as Moderate, "
        "and 80-100% as Strong.\n\n"
        "Chat explanations cannot update mastery. Only the Quick Progress Check assessment updates your mastery."
    )


def scoring_methodology_response() -> str:
    """Explain how ACRLA mastery scores are calculated in the MVP."""
    return (
        "ACRLA mastery uses this scoring methodology:\n\n"
        "1. Initial Moodle mastery is the baseline.\n"
        "2. Current ACRLA mastery starts from that baseline.\n"
        "3. Chat explanations and ordinary practice do not update mastery.\n"
        "4. Only the Quick Progress Check assessment updates mastery.\n"
        "5. The MVP formula is:\n"
        "   calculated_mastery = 0.7 * current_mastery + 0.3 * assessment_score\n"
        "   updated_mastery = max(current_mastery, calculated_mastery, initial_moodle_mastery)\n\n"
        "Mastery never decreases in the MVP."
    )


def tutoring_methodology_response() -> str:
    """Explain how ACRLA teaches and adapts, separate from score calculation."""
    return (
        "ACRLA's tutoring methodology works like this:\n\n"
        "1. It checks the active remediation level: chapter, course, or overall.\n"
        "2. It keeps the response inside the correct Moodle scope.\n"
        "3. It uses your mastery level and selected difficulty to choose the teaching strategy.\n"
        "4. It searches course materials first and uses internal RAG when relevant content is found.\n"
        "5. If course material is not relevant, it uses fallback support without showing PDF sources.\n"
        "6. It guides you through explanation, examples, practice, feedback, and then Quick Progress Check.\n\n"
        "Chat itself does not update mastery. Mastery changes only after Quick Progress Check."
    )


def methodology_meta_response() -> str:
    """Explain the difference between scoring and tutoring methodology."""
    return (
        "I can explain either one. Mastery scoring methodology explains how scores are calculated, "
        "while tutoring methodology explains how ACRLA teaches and adapts."
    )


def methodology_clarification_response() -> str:
    """Ask which of the two methodologies the student means, instead of guessing."""
    return "Do you mean the mastery scoring methodology or the tutoring methodology?"


def locked_chapter_redirect_response(locked_concept: str) -> str:
    """Redirect back to the chapter this remediation session is locked to."""
    return (
        f"This remediation session is currently focused on {locked_concept}. "
        "Please open that chapter grade to start remediation there."
    )


def course_structure_response(available_concepts: list[str]) -> str:
    """List the tracked concepts for the active course/scope."""
    concepts = sorted(available_concepts or [])
    if not concepts:
        return "I do not have synced remediation concepts for this scope yet."
    lines = "\n".join(f"{index}. {concept}" for index, concept in enumerate(concepts, start=1))
    return f"The course contains {len(concepts)} tracked concepts:\n\n{lines}"


def source_provenance_response(metadata: dict[str, Any] | None) -> str:
    """Answer "was that from the PDF?" from already-stored response metadata.

    Pure function of the metadata dict (see
    `tools.memory_tools.get_last_response_metadata_tool`) -- it does not look
    anything up itself.
    """
    metadata = metadata or {}
    pipeline = metadata.get("selected_pipeline")
    sources = metadata.get("sources") or []
    if pipeline == "internal_rag" and sources:
        return (
            "Yes. That answer was grounded in your Moodle course material.\n\n"
            "Sources: " + ", ".join(sources) + "."
        )
    if pipeline == "external_fallback":
        return "No. That answer used fallback general support because no reliable course material was found for that question."
    if pipeline:
        return "That answer came from ACRLA's deterministic course/session data, not from PDF retrieval."
    return "I do not have source metadata for the previous answer yet."


def recommendation_explanation_response(
    recommendation: dict[str, Any] | None,
    recommendation_reason: str | None,
) -> str:
    """Answer "why these?" / "why that one?" from a stored recommendation.

    Pure function of the structured turn's `recommendation`/`recommendation_reason`
    fields (see `agents.agent_models.ConversationTurn`) -- no re-computation.
    """
    if not recommendation or not recommendation_reason:
        return "I do not have a recommendation on record yet for this conversation."
    concept = recommendation.get("concept")
    if concept:
        return f"I recommended {concept} because {recommendation_reason}"
    return recommendation_reason


def clarification_question_response(missing_information: list[str], goal: str) -> str:
    """Compose a deterministic clarifying question when the request is ambiguous.

    Used when `next_action.type == "clarification"` and an LLM-phrased
    clarification is unavailable/undesired; a plain, safe question is always
    better than guessing.
    """
    if missing_information:
        needed = ", ".join(missing_information)
        return f"Could you clarify what you mean? I still need: {needed}."
    return "Could you clarify what you mean by that?"
