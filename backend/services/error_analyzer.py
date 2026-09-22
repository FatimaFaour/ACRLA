"""Judges one student practice answer: correct/incorrect, confidence of a
correct answer's explanation, and (if wrong) an error type -- one LLM call,
the same shape/failure-handling pattern as `agents.simple_planner.plan`
(strict JSON via `invoke_with_json_mode_retry`, never retried on a real
provider error). Never writes mastery -- returns a judgment only; the caller
(`tools.tutor_state_tools.evaluate_practice_answer_tool`) decides what to do
with it.
"""

from __future__ import annotations

import time
from typing import Any

from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel

from agents.agent_json import AgentJSONError, extract_message_text, model_validate, parse_json_object
from agents.llm_errors import invoke_with_json_mode_retry, log_llm_provider_error
from services.llm_factory import get_json_llm


ERROR_TYPES = ("conceptual_misunderstanding", "logic_error", "missing_base_case", "algorithm_misuse")

_JUDGE_MAX_TOKENS = 300


class AnswerJudgment(BaseModel):
    correct: bool = False
    confident: bool = True
    error_type: str | None = None
    feedback_reason: str = ""


_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Judge one student practice answer. Return strict JSON only.
correct: true/false, by meaning against the concept, not exact wording.
confident: false if you are not sure your verdict is right -- either the \
answer is correct but the explanation is shallow/guessy/incomplete, OR no \
COURSE EVIDENCE was given below and you are relying on general knowledge \
rather than verified course content and are not clearly sure of your call.
error_type (only if correct=false), exactly one of: \
conceptual_misunderstanding, logic_error, missing_base_case, algorithm_misuse.
feedback_reason: one short sentence naming the specific mistake or gap \
(or, if correct, what was done well).
If COURSE EVIDENCE is provided below, judge primarily against that evidence \
-- it is the authoritative source for what counts as correct here, not \
general world knowledge.
Return exactly: {{"correct": false, "confident": true, "error_type": null, \
"feedback_reason": "..."}}"""),
    ("human", "CONCEPT: {concept}\nQUESTION: {question}\nSTUDENT ANSWER: {answer}{course_evidence_block}"),
])


def judge_answer(concept: str, question: str, answer: str, course_evidence: str | None = None) -> AnswerJudgment:
    """`course_evidence` (optional, default None = byte-identical prior
    behavior for every existing caller that doesn't pass it): relevant
    retrieved course material, when available, so this judgment can be
    grounded in it rather than resting solely on the model's pretrained
    knowledge of `concept`. Callers should already have applied the
    RQ2 institutional-privacy gateway (retrieve_context*'s `for_external=
    True`) before passing this in -- this function does not re-check
    sensitivity itself, it only renders whatever text it is given.
    """
    llm = get_json_llm(temperature=0, max_tokens=_JUDGE_MAX_TOKENS)
    evidence_block = ""
    if course_evidence and course_evidence.strip():
        evidence_block = f"\n\nCOURSE EVIDENCE (judge primarily against this, not general knowledge):\n{course_evidence.strip()[:1800]}"
    prompt_values = {
        "concept": concept or "", "question": question or "", "answer": answer or "",
        "course_evidence_block": evidence_block,
    }
    call_started = time.monotonic()
    try:
        response, _ = invoke_with_json_mode_retry(
            _JUDGE_PROMPT, prompt_values, llm=llm, temperature=0, max_tokens=_JUDGE_MAX_TOKENS,
            stage="error_analyzer_judge_answer",
        )
        raw = extract_message_text(response)
        data = parse_json_object(raw)
        judgment = model_validate(AnswerJudgment, data)
        if judgment.error_type not in ERROR_TYPES:
            judgment.error_type = None
        return judgment
    except (AgentJSONError, Exception) as exc:
        log_llm_provider_error(
            stage="error_analyzer_judge_answer", exc=exc, prompt_values=prompt_values,
            json_mode_enabled=True, llm=llm, response_length=0,
            final_fallback_reason="error_analyzer_provider_error",
        )
        return _heuristic_fallback(answer)
    finally:
        _ = time.monotonic() - call_started  # diagnostic only, not currently logged


def _heuristic_fallback(answer: str) -> AnswerJudgment:
    """Only reached on a provider error -- reuses the legacy engine's own
    length-based correctness heuristic (`services.chat_orchestrator.
    _evaluate_practice_answer`'s open-ended branch: at least 8 words counts
    as an attempted, plausible answer) so a provider outage degrades to the
    same behavior this project already ships today, not a new unproven one.
    Never asserts a specific error_type it cannot actually diagnose.
    """
    normalized = (answer or "").strip().lower()
    word_count = len(normalized.split())
    correct = word_count >= 8
    return AnswerJudgment(
        correct=correct,
        confident=False,
        error_type=None if correct else "conceptual_misunderstanding",
        feedback_reason="" if correct else "The answer looks incomplete or too short to confirm understanding.",
    )
