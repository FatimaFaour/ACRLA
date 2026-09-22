"""Evidence validator: decides whether retrieved RAG chunks actually support
the question being asked, before anything is allowed to call itself
"internal_rag" and show PDF sources.

`validate_retrieved_evidence` is called from inside
`tools.rag_tools.search_course_material_tool` itself, so the RAG tool's own
observation already carries the evidence verdict (reliable/coverage/reason) --
the agent brain and the coordinator both read it from there rather than this
module being invoked a second time after the fact.

Stage A applies deterministic checks (raw-message token overlap, resolved
concept match, source/concept metadata match, multi-concept coverage). Stage
B (an LLM judge, temperature=0, strict JSON) only runs when Stage A cannot
reach a confident verdict. The judge never generates the final answer -- only
a reliability verdict -- and nearest-neighbor chunk existence alone is never
treated as sufficient evidence.
"""

from __future__ import annotations

from typing import Any

from langchain.prompts import ChatPromptTemplate

from agents.agent_json import compact_json, extract_message_text, model_dump, model_validate, parse_json_object
from agents.agent_models import EvidenceValidation
from agents.llm_errors import log_llm_provider_error
from services.llm_factory import get_json_llm
from tools.text_utils import normalize_key, routing_tokens


EVIDENCE_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict evidence judge for ACRLA. Return strict JSON only.
Do not answer the student's question. Do not invent facts that are not in the snippets.

Decide whether the retrieved course snippets actually contain content that supports answering
the question. Nearest-neighbor existence alone is not enough: the snippets must be topically
about what the question actually asks.

Return exactly:
{{
  "reliable": true/false,
  "coverage": "full|partial|none",
  "supported_concepts": [],
  "reason": "short string",
  "confidence": 0.0
}}"""),
    ("human", """QUESTION:
{question}

RESOLVED COURSE CONCEPTS:
{resolved_concepts}

RETRIEVED SNIPPETS:
{snippets}"""),
])


def validate_retrieved_evidence(
    question: str,
    resolved_concepts: list[str],
    retrieved_chunks: list[dict[str, Any]],
    retrieved_sources: list[str],
    available_concepts: list[str],
    require_multi_concept: bool = False,
) -> dict[str, Any]:
    """Decide whether retrieved course chunks actually support this question.

    `require_multi_concept` comes from the agent brain's own semantic
    classification (`goal == "concept_comparison"`), not from re-parsing the
    question text here. When true, evidence cannot be "full" from just one
    grounded concept: a comparison needs at least two concepts' worth of
    genuine course-material support, however well the one grounded side is
    covered -- this is what keeps "compare X and <thing with no course
    material>" from being reported as a fully-grounded comparison.
    """
    deterministic = _deterministic_evidence_check(
        question, resolved_concepts, retrieved_chunks, retrieved_sources, available_concepts, require_multi_concept,
    )
    if deterministic is not None:
        return deterministic
    return _llm_evidence_judge(question, resolved_concepts, retrieved_chunks)


def _deterministic_evidence_check(
    question: str,
    resolved_concepts: list[str],
    retrieved_chunks: list[dict[str, Any]],
    retrieved_sources: list[str],
    available_concepts: list[str],
    require_multi_concept: bool = False,
) -> dict[str, Any] | None:
    has_any_chunk_text = any((chunk.get("text") or "").strip() for chunk in retrieved_chunks)
    if not retrieved_sources or not has_any_chunk_text:
        return _evidence_dict(False, "none", [], "no_course_chunks_retrieved", 0.95)

    if resolved_concepts:
        supported: list[str] = []
        for concept in resolved_concepts:
            concept_key = normalize_key(concept)
            concept_tokens = routing_tokens(concept)
            for chunk in retrieved_chunks:
                chunk_concept_key = normalize_key(chunk.get("concept") or "")
                chunk_tokens = routing_tokens(
                    (chunk.get("text") or "") + " " + " ".join(chunk.get("sources") or [])
                )
                if chunk_concept_key and chunk_concept_key == concept_key:
                    supported.append(concept)
                    break
                if concept_tokens and len(concept_tokens & chunk_tokens) >= 1 and (chunk.get("text") or "").strip():
                    supported.append(concept)
                    break

        # A comparison goal (per the agent brain's own semantic classification,
        # not a regex here) needs at least two concepts' worth of evidence,
        # even if only one could be grounded to real course material.
        required = max(len(resolved_concepts), 2) if require_multi_concept else len(resolved_concepts)

        if not supported:
            return _evidence_dict(
                False, "none", [], "retrieved_chunks_do_not_match_resolved_concepts", 0.9,
            )
        if len(supported) >= required:
            return _evidence_dict(
                True, "full", supported, "all_requested_concepts_found_in_retrieved_chunks", 0.9,
            )
        return _evidence_dict(
            False, "partial", supported,
            "only_partial_concept_coverage_for_a_multi_concept_question", 0.85,
        )

    # No course concept was resolved from the raw message: nearest-neighbor
    # chunk existence alone is never enough evidence. Require real token
    # overlap between the question and either the retrieved content or the
    # course's own vocabulary before treating this as reliable.
    question_tokens = routing_tokens(question)
    course_vocab: set[str] = set()
    for concept in available_concepts or []:
        course_vocab.update(routing_tokens(str(concept)))
    chunk_tokens: set[str] = set()
    for chunk in retrieved_chunks:
        chunk_tokens.update(routing_tokens((chunk.get("text") or "")[:1200]))
        chunk_tokens.update(routing_tokens(" ".join(chunk.get("sources") or [])))

    overlap = question_tokens & chunk_tokens
    mentions_course_vocab = bool(question_tokens & course_vocab)
    if not overlap and not mentions_course_vocab:
        return _evidence_dict(
            False, "none", [], "no_token_overlap_between_question_and_retrieved_chunks", 0.9,
        )
    if mentions_course_vocab or len(overlap) >= 3:
        return _evidence_dict(
            True, "full", [], "question_vocabulary_matches_course_material", 0.75,
        )
    return None  # ambiguous nearest-neighbor case: defer to the LLM judge


def _llm_evidence_judge(
    question: str,
    resolved_concepts: list[str],
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    snippets = "\n\n".join(
        f"[{', '.join(chunk.get('sources') or []) or 'unknown source'}] {(chunk.get('text') or '')[:600]}"
        for chunk in retrieved_chunks[:4]
    ) or "none"
    prompt_values = {
        "question": question,
        "resolved_concepts": compact_json(resolved_concepts),
        "snippets": snippets[:3000],
    }
    llm = get_json_llm(temperature=0, max_tokens=300)
    try:
        chain = EVIDENCE_JUDGE_PROMPT | llm
        response = chain.invoke(prompt_values)
    except Exception as exc:
        # The call to the LLM provider itself never completed here -- kept
        # separate from the JSON-parse/schema try/except below so a provider
        # outage is never misreported as "the evidence judge returned bad
        # JSON" (it never returned anything at all).
        log_llm_provider_error(
            stage="evidence_judge", exc=exc, prompt_values=prompt_values,
            json_mode_enabled=True, llm=llm, response_length=0,
            final_fallback_reason="evidence_judge_provider_error",
        )
        return _evidence_dict(False, "none", [], "evidence_judge_provider_error", 0.0)

    raw = extract_message_text(response)
    try:
        data = parse_json_object(raw)
        validated = model_validate(EvidenceValidation, {
            "reliable": bool(data.get("reliable")),
            "coverage": data.get("coverage") if data.get("coverage") in {"full", "partial", "none"} else "none",
            "supported_concepts": [c for c in (data.get("supported_concepts") or []) if isinstance(c, str)],
            "reason": str(data.get("reason") or "llm_evidence_judge"),
            "confidence": max(0.0, min(1.0, float(data.get("confidence") or 0.0))),
        })
        return model_dump(validated)
    except Exception as exc:
        print(f"[ACRLA] conversation_agent_evidence_judge_failed error={exc} raw_output_len={len(raw)}")
        return _evidence_dict(False, "none", [], f"evidence_judge_failed:{exc}", 0.0)


def _evidence_dict(
    reliable: bool, coverage: str, supported_concepts: list[str], reason: str, confidence: float,
) -> dict[str, Any]:
    return model_dump(EvidenceValidation(
        reliable=reliable,
        coverage=coverage,
        supported_concepts=supported_concepts,
        reason=reason,
        confidence=confidence,
    ))


def empty_evidence(reason: str) -> dict[str, Any]:
    """Placeholder evidence status for turns that never ran RAG at all."""
    return {"reliable": None, "coverage": None, "supported_concepts": [], "reason": reason, "confidence": None}
