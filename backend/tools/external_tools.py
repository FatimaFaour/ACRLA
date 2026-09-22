"""External knowledge tool: a controlled general-knowledge fallback answer.

This is one tool the agent may call, not a permanently pre-selected "external
mode" -- the agent should only reach for it once RAG evidence has come back
irrelevant/unreliable, or the question is clearly general/casual and does not
need course material to begin with (see `agents.agent_brain`/
`agents.conversation_agent` for when that decision is made).

It deliberately reuses the exact same controlled prompt the legacy hybrid
pipeline uses (`pipelines.hybrid_pipeline.build_external_fallback_prompt`)
rather than inventing a new one, so this tool inherits, unchanged:
- no PDF/Moodle-source claims
- remediation-level-aware redirect rules (chapter/course/overall scope)
- the "chat/practice never updates mastery" rule
- the LLM provider configured in `services.llm_factory` (Groq by default)
"""

from __future__ import annotations

import re
from typing import Any

from langchain.prompts import ChatPromptTemplate

from agents.agent_json import extract_message_text
from agents.llm_errors import (
    FINAL_ANSWER_MAX_TOKENS,
    FINAL_ANSWER_RETRY_MAX_TOKENS,
    is_truncated_response,
    log_llm_provider_error,
)
from pipelines.hybrid_pipeline import build_external_fallback_prompt
from services.llm_factory import get_llm
from services.privacy_context import build_llm_safe_student_context


EXTERNAL_KNOWLEDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{fallback_prompt}"),
    ("human", "{question}"),
])


def _student_context_from_agent_context(context: dict[str, Any]) -> dict[str, Any]:
    """Adapt the agent's context packet into what build_external_fallback_prompt
    expects -- built ONLY from services.privacy_context's allow-listed safe
    context, never from the raw agent `context` directly, so no identity
    field (student_name/username/email/moodle_user_id/student_id) can reach
    this external-LLM prompt no matter what gets added to `context`
    elsewhere in the codebase later. `include_mastery=True` preserves this
    tool's pre-existing "mention weak concepts in the redirect" behavior
    (concept NAMES only, never a mastery percentage/row).
    """
    safe = build_llm_safe_student_context(context, include_mastery=True)
    tutoring_strategy = safe.get("tutoring_strategy") or {}
    return {
        "selected_concept": safe.get("current_concept"),
        "current_topic": safe.get("current_concept"),
        "available_concepts": safe.get("available_concepts") or [],
        "requested_concepts": safe.get("resolved_concepts") or [],
        "weak_concepts": safe.get("weak_concepts") or [],
        "remediation_level": safe.get("remediation_level") or "chapter",
        "course_name": safe.get("course_name"),
        "remediation_scope": safe.get("scope_rules") or "",
        "difficulty": safe.get("difficulty") or "medium",
        "strategy": tutoring_strategy.get("name") or "guided_practice",
        "strategy_instructions": tutoring_strategy.get("instructions") or "",
    }


def answer_with_external_knowledge_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Answer using controlled general LLM knowledge. No PDF sources, no mastery writes.

    Call this only after evidence has been judged irrelevant/unreliable, or
    the question is clearly general/casual. This tool never re-attempts RAG
    and always returns sources=[] and grounded_in_moodle=False.
    """
    question = arguments.get("query") or context.get("message") or ""
    student_context = _student_context_from_agent_context(context)
    student_context["skip_internal_context"] = True
    fallback_prompt = build_external_fallback_prompt(student_context, question, context="")
    prompt_values = {"fallback_prompt": fallback_prompt, "question": question}
    llm = get_llm(temperature=0.5, max_tokens=FINAL_ANSWER_MAX_TOKENS)
    try:
        chain = EXTERNAL_KNOWLEDGE_PROMPT | llm
        response = chain.invoke(prompt_values)
    except Exception as exc:
        # The call to the LLM provider itself never completed here -- kept
        # separate/classified rather than collapsed into a bare str(exc), so
        # a provider outage is never indistinguishable from "the model just
        # returned an empty reply".
        classification = log_llm_provider_error(
            stage="external_knowledge_answer", exc=exc, prompt_values=prompt_values,
            json_mode_enabled=False, llm=llm, response_length=0,
            final_fallback_reason="external_knowledge_provider_error",
        )
        return {
            "tool": "answer_with_external_knowledge",
            "success": False,
            "query": question,
            "reply": "",
            "sources": [],
            "grounded_in_moodle": False,
            "error": f"{classification['error_category']}:{type(exc).__name__}",
        }

    reply = extract_message_text(response).strip()
    if is_truncated_response(response):
        # Never silently returned mid-sentence -- one bounded retry at a
        # larger ceiling using the SAME prompt_values (same fallback_prompt/
        # question), never a second retry. This tool never claims Moodle
        # grounding/sources either way (sources=[], grounded_in_moodle=False
        # always), so there is no source-visibility concern to preserve here
        # unlike the RAG final-answer path in agents.response_generator.
        print(f"[ACRLA] external_knowledge_answer_truncated raw_output_len={len(reply)} max_tokens={FINAL_ANSWER_MAX_TOKENS}")
        try:
            retry_llm = get_llm(temperature=0.5, max_tokens=FINAL_ANSWER_RETRY_MAX_TOKENS)
            retry_response = (EXTERNAL_KNOWLEDGE_PROMPT | retry_llm).invoke(prompt_values)
            retry_reply = extract_message_text(retry_response).strip()
            if is_truncated_response(retry_response):
                print(f"[ACRLA] external_knowledge_answer_truncated_after_retry raw_output_len={len(retry_reply)} max_tokens={FINAL_ANSWER_RETRY_MAX_TOKENS}")
                reply = _trim_to_complete_sentence(retry_reply if len(retry_reply) > len(reply) else reply)
            else:
                reply = retry_reply
        except Exception as retry_exc:
            print(f"[ACRLA] external_knowledge_answer_retry_failed error={retry_exc}")
            reply = _trim_to_complete_sentence(reply)

    return {
        "tool": "answer_with_external_knowledge",
        "success": bool(reply),
        "query": question,
        "reply": reply,
        "sources": [],
        "grounded_in_moodle": False,
    }


def _trim_to_complete_sentence(text: str) -> str:
    """Last-resort cleanup for a completion STILL truncated after the one
    bounded retry -- same logic as
    agents.response_generator._trim_to_complete_sentence, kept as its own
    small copy here so this tool has no dependency on agents.response_generator."""
    text = text.rstrip()
    if not text or text[-1] in ".!?\"')]}":
        return text
    complete_sentence_ends = list(re.finditer(r"[.!?](?=\s|$)", text))
    if complete_sentence_ends:
        return text[: complete_sentence_ends[-1].end()].rstrip()
    return text
