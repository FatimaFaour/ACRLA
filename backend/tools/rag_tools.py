"""RAG search tool: retrieves course material and validates it in one call.

This tool is not a pre-decided "internal mode" -- it is just a tool the agent
may try when course relevance is *possible*. Its result always embeds the
evidence verdict (see `agents.evidence_validator.validate_retrieved_evidence`)
so the agent brain and the coordinator both see the same reliability judgment
the tool itself produced, instead of the caller re-deriving it later from raw
chunks. An agent brain step that sees `evidence.reliable=false` here does not
retry RAG; it moves on to answer with external framing.
"""

from __future__ import annotations

from typing import Any

from pipelines.rag_pipeline import retrieve_context_for_scope
from services.course_concepts import concepts_in_text


def search_course_material_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Retrieve course material for one or more resolved concepts, pre-validated.

    Each requested concept is searched separately so a multi-concept question
    (e.g. a comparison) produces per-concept chunks instead of one blended
    result. The active/current chapter topic is only used as a stand-in when
    `context["is_followup"]` is true -- a judgment `agents.conversation_agent`
    derives from the agent brain's own `resolved_entities.references` for
    this step, never re-derived here from the message text. That is what
    prevents an unrelated question (e.g. "explain quantum computing") from
    silently retrieving the current chapter's material: the agent brain only
    populates `references` when the message genuinely draws on prior turns.
    """
    # Imported here (not at module scope) because agents.evidence_validator is
    # the one thing in `agents/` this tools/ module depends on; keeping it a
    # local import makes that one-directional dependency easy to spot in a diff.
    from agents.evidence_validator import validate_retrieved_evidence

    query = arguments.get("query") or context.get("message") or ""
    available_concepts = context.get("available_concepts") or []

    # Tool arguments come from the planner's own LLM output -- validate against
    # the real, per-course concept list rather than trusting them outright.
    raw_concepts = arguments.get("concepts") or []
    if isinstance(raw_concepts, str):
        raw_concepts = [raw_concepts]
    concepts = [c for c in raw_concepts if c in available_concepts]
    if not concepts:
        concepts = concepts_in_text(query, available_concepts)
    if not concepts and context.get("is_followup") and context.get("current_concept"):
        concepts = [context["current_concept"]]
    # agent_selected_concept is only ever set by an explicit selection tool
    # (e.g. select_lowest_mastery_among_previous_turn / run_study_recommendation)
    # run earlier in this same turn's action list -- using it here is
    # intentional multi-step reasoning, not stale-topic contamination.
    if not concepts and context.get("agent_selected_concept"):
        concepts = [context["agent_selected_concept"]]

    # course_ids/remediation_level must come from context only -- chat_orchestrator
    # already resolved the authorized chapter/course/overall scope for this turn.
    # Honoring a planner/tool-argument override here would let a prompt-injected or
    # hallucinated tool call read outside that scope, so `arguments` is never
    # consulted for either of these two values.
    course_ids = context.get("retrieval_course_ids") or []
    remediation_level = context.get("remediation_level") or "chapter"

    chunks: list[dict[str, Any]] = []
    all_sources: list[str] = []
    search_targets = concepts if concepts else [None]
    # canonical_courses is the same concept/material manifest tools.mastery_tools
    # already uses to resolve which real course teaches a concept for overall-
    # scope mastery lookups -- passing it through lets retrieve_context_for_scope
    # search only the course(s) that actually own each resolved concept instead
    # of every course in `course_ids`, never widening past the allowed scope
    # and falling back to the full scope automatically when ownership can't be
    # resolved (see pipelines.rag_pipeline._concept_owning_course_ids).
    canonical_courses = context.get("canonical_courses") or []
    # RQ2 institutional privacy: this tool's own result is what
    # agents.response_generator eventually phrases into the reply an
    # external LLM (currently Gemini) sees -- there is no other, purely-
    # local consumer of these chunks -- so retrieval always runs with
    # for_external=True here, which is what applies the RESTRICTED-content
    # exclusion and the external size budget (see
    # pipelines.rag_pipeline.retrieve_context /
    # services.privacy_context.filter_course_chunks_for_external).
    turn_audit: dict[str, Any] = {}
    for concept in search_targets:
        from services.privacy_context import merge_external_content_audit
        call_audit: dict[str, Any] = {}
        context_text, sources = retrieve_context_for_scope(
            course_ids,
            query,
            selected_concept=concept,
            requested_concepts=[concept] if concept else None,
            scope=remediation_level,
            canonical_courses=canonical_courses,
            for_external=True,
            audit=call_audit,
        )
        merge_external_content_audit(turn_audit, call_audit)
        if context_text:
            chunks.append({"concept": concept, "text": context_text[:3000], "sources": sources})
        for source in sources:
            if source not in all_sources:
                all_sources.append(source)

    evidence = validate_retrieved_evidence(
        question=query,
        resolved_concepts=concepts,
        retrieved_chunks=chunks,
        retrieved_sources=all_sources,
        available_concepts=available_concepts,
        require_multi_concept=context.get("goal") == "concept_comparison",
    )

    if turn_audit:
        from services.privacy_context import log_external_content_decision
        log_external_content_decision(course_id=context.get("course_db_id"), audit=turn_audit)

    return {
        "tool": "search_course_material",
        "success": True,
        "query": query,
        "requested_concepts": concepts,
        "chunks": chunks,
        "sources": all_sources,
        "concepts_found": evidence.get("supported_concepts") or [],
        "evidence": evidence,
        # True only when every candidate chunk this turn's retrieval found
        # was classified RESTRICTED -- i.e. relevant material existed but
        # institutional policy blocked all of it from the external prompt.
        # agents.simple_agent._deterministic_reply reads this to return a
        # controlled fallback instead of silently answering with unrelated
        # general knowledge. Never true just because nothing was found at
        # all (chunks_considered=0 -> False, the ordinary "no material"
        # case, unchanged from before this feature).
        "all_candidates_restricted": bool(turn_audit.get("all_candidates_restricted")),
        "privacy_audit": turn_audit,
    }
