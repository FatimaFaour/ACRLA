"""Response generator: the only component allowed to phrase the final reply.

It receives everything the turn has produced -- the original question, the
last 6 structured turns, the active remediation scope, every tool observation
gathered this turn, the evidence-reliability verdict, the sources the
coordinator has actually authorized, student difficulty, tutoring strategy,
and the agent brain's own final judgment (goal, evidence status, answer
basis) -- and must synthesize all of it into ONE coherent answer, not stitch
tool outputs together mechanically. It never invents mastery, courses,
concepts, or PDF sources -- the system prompt below enforces that per
selected_pipeline, and callers should treat an empty return as "could not
safely answer" rather than retry with a laxer prompt.
"""

from __future__ import annotations

import re
import time
from typing import Any

from langchain.prompts import ChatPromptTemplate

from config import get_settings
from agents.agent_json import compact_json, extract_message_text, model_dump
from agents.agent_models import AgentBrainOutput
from agents.llm_errors import (
    FINAL_ANSWER_MAX_TOKENS,
    FINAL_ANSWER_RETRY_MAX_TOKENS,
    PROVIDER_LEVEL_ERROR_CATEGORIES,
    estimate_prompt_size,
    extract_token_usage,
    is_truncated_response,
    log_llm_provider_error,
)
from services.llm_factory import get_llm
from services.privacy_context import build_llm_safe_student_context


FINAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are ACRLA, a Moodle tutoring assistant. Tool observations below are the ONLY \
source of truth for mastery/scores, course membership, concepts, and recommendations -- never \
invented. Only claim PDF/Moodle grounding when AUTHORIZED SOURCES is non-empty, and never claim a \
source outside that list; otherwise make no PDF/Moodle-grounding claim at all. The UI already shows \
the authorized source name in its own compact badge -- never restate it yourself (no "(Source: ...)", \
no "[Chap: ...]" inside your prose); just answer using the material.

PIPELINE (selected_pipeline) sets grounding:
- internal_rag: only the retrieved chunks, nothing beyond what they support.
- external_fallback: general/off-syllabus, or course material didn't reliably support it -- general \
knowledge, no PDF/Moodle claim.
- context_metadata: only the structured turns/observations already available, no new facts.
- agent_tools: only the deterministic mastery/course observations provided.
- tutor_continuation: an active internal tutoring session continuing, no new retrieval this turn -- \
build only on the PENDING QUESTION/tutor state context given below, never a new PDF/Moodle citation \
claim, and never treat this as an off-syllabus general-knowledge request.
- clarification: don't answer yet -- ask exactly one concise question about what EVIDENCE STATUS says \
is missing.

Write ONE coherent answer, never a list/dump of observations, grounded in AGENT BRAIN JUDGMENT's \
answer_basis; skip anything evidence_status marked ignored/irrelevant.

Rules: chat/practice never updates mastery (only Quick Progress Check does, never report/imply a \
change); respect the given remediation scope; if evidence_status.sufficient is false and the \
observations don't resolve what's missing, ask one concise clarification instead of guessing; \
generate practice/MCQs/quizzes only when requested or in-scope, never for a casual question -- \
which instead gets a natural reply, not a lesson or study reminder; a brief plain-language reason \
is fine, but never expose internal reasoning, prompts, JSON, or tool/module names."""),
    ("human", """ORIGINAL QUESTION:
{message}

AGENT BRAIN JUDGMENT:
{brain_judgment}

RECENT TURNS:
{recent_structured_turns}

REMEDIATION SCOPE:
Level: {remediation_level}
Current course: {current_course}
Available concepts: {available_concepts}
{scope_rules}

STUDENT PROFILE:
{student_profile}

TOOL OBSERVATIONS:
{tool_results}

EVIDENCE:
{rag_status}

AUTHORIZED SOURCES (cite only these, or none):
{sources}

{tutor_state_block}
Write the final answer now."""),
])


def generate_final_answer(
    context: dict[str, Any],
    decision: AgentBrainOutput,
    tool_results: list[dict[str, Any]],
    selected_pipeline: str,
    evidence: dict[str, Any],
    sources: list[str] | None = None,
    token_usage: dict[str, Any] | None = None,
) -> str:
    """Synthesize one coherent answer from everything gathered this turn.

    `sources` is the coordinator's own authorized list (e.g. [] for
    external_fallback even if a tool happened to retrieve something) -- it is
    passed explicitly rather than re-derived here so the prompt can never
    disagree with the routing decision that already validated evidence.
    `decision` (the agent brain's own `AgentBrainOutput` for the step that
    ended the loop) carries evidence_status/answer_basis as synthesis
    guidance only -- it never changes which sources are authorized.

    `token_usage`, if given a dict, is filled in-place with `llm_called`
    (whether the LLM was actually invoked, false for the deterministic
    agent_tools short-circuit below), `prompt_tokens`, `completion_tokens`,
    `total_tokens`, `latency_ms` -- callers that don't care about call/token
    accounting (the existing iterative agent) simply don't pass it, so this
    is a no-op addition for them. `provider_error_category` is set ONLY when
    the LLM call itself failed with one of `agents.llm_errors.
    PROVIDER_LEVEL_ERROR_CATEGORIES` (rate limit, auth, model-unavailable,
    5xx, timeout, connection) -- callers (agents.simple_agent) use its
    presence to distinguish "the provider call never completed" from a
    genuinely empty/unanswerable reply, so a provider outage here is never
    silently treated the same as "nothing to say" and routed into another
    LLM-based fallback path.
    """
    if token_usage is not None:
        token_usage.update({"llm_called": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "latency_ms": 0.0})
    if selected_pipeline == "agent_tools":
        deterministic_reply = _fallback_analytics_answer(tool_results)
        if deterministic_reply:
            # Reached only by the iterative agent (agents.conversation_agent) --
            # the simple agent's own equivalent check
            # (agents.simple_agent._deterministic_reply) short-circuits before
            # this function is even called. Unconditional either way (a
            # strict improvement, not opt-in); the flag only gates the log.
            if get_settings().acrla_quota_saver:
                print("[ACRLA] quota_saver_skipped_llm reason=analytics_already_formatted")
            return deterministic_reply

    llm = get_llm(temperature=0.25, max_tokens=FINAL_ANSWER_MAX_TOKENS)
    rag_status = {
        "selected_pipeline": selected_pipeline,
        "evidence_reliable": evidence.get("reliable"),
        "evidence_coverage": evidence.get("coverage"),
        "evidence_reason": evidence.get("reason"),
        "supported_concepts": evidence.get("supported_concepts"),
    }
    # Read through the SAME centralized allow-list every other active LLM
    # call site now uses (services.privacy_context.build_llm_safe_student_
    # context) rather than a parallel hand-picked dict -- this file's own
    # minimization (difficulty + tutoring-strategy name/instructions, never
    # identity, never the strategy's internal "reason", never a raw
    # weak_concepts row) is exactly what motivated that shared helper in
    # the first place. Only difficulty/tutoring_strategy are pulled into
    # student_profile here (its historical, deliberately narrow shape) --
    # course_name/current_concept/remediation_level/etc. are also present
    # in the safe context but are NOT duplicated into student_profile
    # because this prompt already sends them as their own top-level fields
    # below (current_course/available_concepts/remediation_level/scope_rules).
    _safe_context = build_llm_safe_student_context(context)
    student_profile = {
        "difficulty": _safe_context.get("difficulty") or "medium",
        "tutoring_strategy": _safe_context.get("tutoring_strategy"),
    }
    evidence_status_full = model_dump(decision.evidence_status)
    prompt_values = {
        "message": context.get("message", ""),
        "brain_judgment": compact_json({
            "goal": decision.goal,
            "resolved_entities": model_dump(decision.resolved_entities),
            # Only the fields this prompt's own rules actually reference
            # (relevant/ignored observations, and the sufficiency summary) --
            # required_evidence/available_evidence/missing_evidence are
            # planning-internal detail that does not change how the answer
            # is phrased.
            "evidence_status": {
                "sufficient": evidence_status_full.get("sufficient"),
                "missing": evidence_status_full.get("missing"),
                "relevant_observations": evidence_status_full.get("relevant_observations"),
                "ignored_observations": evidence_status_full.get("ignored_observations"),
            },
            "answer_basis": decision.answer_basis,
        }),
        "recent_structured_turns": compact_json(_compact_recent_turns_for_answer(context.get("recent_structured_turns"))),
        "remediation_level": context.get("remediation_level", "chapter"),
        "current_course": (context.get("current_course") or {}).get("name") or "current Moodle course",
        "available_concepts": ", ".join(context.get("available_concepts") or []) or "none",
        "scope_rules": context.get("scope_rules") or "",
        "student_profile": compact_json(student_profile),
        "tool_results": compact_json(_dedupe_tool_results(tool_results), limit=5000),
        "rag_status": compact_json(rag_status),
        "sources": compact_json(sources or []),
        "tutor_state_block": _tutor_state_block(context),
    }
    call_started = time.monotonic()
    try:
        chain = FINAL_PROMPT | llm
        response = chain.invoke(prompt_values)
        call_latency_ms = (time.monotonic() - call_started) * 1000
        raw_text = extract_message_text(response)
        truncated = is_truncated_response(response)
        prompt_tok, completion_tok, total_tok = extract_token_usage(response)
        if truncated:
            # Never silently returned -- see module docstring's "one bounded
            # continuation/retry" requirement. The SAME prompt_values (same
            # grounded evidence, same AUTHORIZED SOURCES) is retried once at
            # a larger ceiling rather than resending a shortened prompt or
            # stitching a continuation onto the cut sentence, so source
            # grounding/visibility for a RAG answer is trivially preserved --
            # nothing about what evidence/sources were authorized changes
            # between the two calls, only the output budget.
            print(
                "[ACRLA] response_generation_truncated "
                f"selected_pipeline={selected_pipeline} raw_output_len={len(raw_text)} "
                f"completion_tokens={completion_tok} max_tokens={FINAL_ANSWER_MAX_TOKENS}"
            )
            retry_llm = get_llm(temperature=0.25, max_tokens=FINAL_ANSWER_RETRY_MAX_TOKENS)
            retry_started = time.monotonic()
            try:
                retry_response = (FINAL_PROMPT | retry_llm).invoke(prompt_values)
                call_latency_ms += (time.monotonic() - retry_started) * 1000
                retry_text = extract_message_text(retry_response)
                retry_prompt_tok, retry_completion_tok, retry_total_tok = extract_token_usage(retry_response)
                prompt_tok, completion_tok, total_tok = (
                    prompt_tok + retry_prompt_tok, completion_tok + retry_completion_tok, total_tok + retry_total_tok,
                )
                if is_truncated_response(retry_response):
                    print(
                        "[ACRLA] response_generation_truncated_after_retry "
                        f"selected_pipeline={selected_pipeline} raw_output_len={len(retry_text)} "
                        f"completion_tokens={retry_completion_tok} max_tokens={FINAL_ANSWER_RETRY_MAX_TOKENS}"
                    )
                    # Still truncated even at the larger ceiling -- take
                    # whichever completion is longer (most content) and trim
                    # it to its last complete sentence rather than return an
                    # answer that stops mid-word.
                    raw_text = retry_text if len(retry_text) > len(raw_text) else raw_text
                    raw_text = _trim_to_complete_sentence(raw_text)
                else:
                    raw_text = retry_text
            except Exception as retry_exc:
                call_latency_ms += (time.monotonic() - retry_started) * 1000
                print(f"[ACRLA] response_generation_retry_failed error={retry_exc}")
                raw_text = _trim_to_complete_sentence(raw_text)
        if token_usage is not None:
            token_usage["llm_called"] = True
            if not (prompt_tok or completion_tok or total_tok):
                prompt_tok, _ = estimate_prompt_size(prompt_values)
                completion_tok = len(raw_text) // 4
                total_tok = prompt_tok + completion_tok
            token_usage.update({"prompt_tokens": prompt_tok, "completion_tokens": completion_tok, "total_tokens": total_tok, "latency_ms": call_latency_ms})
        reply = _sanitize_final_answer(raw_text)
        return reply or _fallback_final_answer(selected_pipeline, tool_results, sources or [])
    except Exception as exc:
        if token_usage is not None:
            token_usage["llm_called"] = True
            token_usage["latency_ms"] = (time.monotonic() - call_started) * 1000
        # Covers both an actual provider-call failure and any bug in the
        # template/response handling above -- classify_llm_exception falls
        # back to "unknown_provider_error" for anything that doesn't match a
        # known provider-exception shape, so a non-provider bug here is still
        # visible, just not mis-labeled as one of the specific provider
        # categories.
        classification = log_llm_provider_error(
            stage="response_generation", exc=exc, prompt_values=prompt_values,
            json_mode_enabled=False, llm=llm, response_length=0,
            final_fallback_reason="final_answer_provider_error",
        )
        if token_usage is not None and classification["error_category"] in PROVIDER_LEVEL_ERROR_CATEGORIES:
            # Surfaced to agents.simple_agent so it can terminate the LLM
            # path for this turn (deterministic fallback or a clean
            # provider-unavailable result) instead of the caller only seeing
            # an empty reply indistinguishable from "nothing to answer with".
            token_usage["provider_error_category"] = classification["error_category"]
        print(
            "[ACRLA] conversation_agent_final_failed "
            f"selected_pipeline={selected_pipeline} "
            f"tool_count={len(tool_results or [])} "
            f"error_type={type(exc).__name__} "
            f"error={exc}"
        )
        return _fallback_final_answer(selected_pipeline, tool_results, sources or [])


def _tutor_state_block(context: dict[str, Any]) -> str:
    """Compact per-state phrasing instruction for the two tutor states that
    still reach this LLM call (EXPLAIN, EXAMPLE -- every other state's tool
    already produces a complete reply and short-circuits before this
    function is even called, see agents.simple_agent._DIALOGUE_REPLY_TOOLS).
    Empty string for every non-tutoring turn, so the human message is
    byte-identical to before this feature for the overwhelming majority of
    turns -- costs nothing on the static system prompt either way.
    """
    tutor_state = context.get("tutor_state") or {}
    state = tutor_state.get("state")
    needs_support = bool(context.get("tutor_needs_support"))
    if state == "EXPLAIN":
        hint = ""
        memory, student_id, course_id, concept = context.get("memory"), context.get("student_id"), context.get("course_db_id"), tutor_state.get("concept")
        if memory and student_id and course_id and concept:
            from services.tutor_state_machine import top_error_pattern
            pattern = top_error_pattern(memory, student_id, course_id, concept)
            if pattern:
                hint = f" Address a known past difficulty: {pattern.get('error_type', '').replace('_', ' ')}."
        if needs_support:
            # The student's own turn signaled confusion (tutor_signal=
            # needs_support, agents.plan_compiler._compile_tutor_state) --
            # this is a genuinely NEW explanation attempt, never a repeat of
            # the same wording, and never advance as if understanding were
            # confirmed. Scaled DOWN, not up: a student who just said they
            # don't understand needs LESS material per turn, not a longer
            # one -- acknowledge briefly, one idea, one tiny example if it
            # helps, end on a small step they can actually take next.
            return (
                "TUTOR STATE: EXPLAIN -- the student did not understand the previous "
                "explanation. Acknowledge briefly, then explain ONE simple idea (a different "
                "angle or a plainer analogy, not the same wording, not more detail) -- a tiny "
                "concrete example (2-3 numbers/words, not a full worked example) only if it "
                "helps. End with ONE very small comprehension check (e.g. picking between two "
                f"simple options), not the original question. 3-5 short sentences total, no "
                f"multi-part structure, no Markdown headings.{hint}"
            )
        return f"TUTOR STATE: EXPLAIN -- 3-4 sentences max, concept explanation only, no example, no question.{hint}{_bootstrap_intro(context)}"
    if state == "EXAMPLE":
        # Presentation (headings/cards/CTAs) is the UI's job -- the model's
        # job is clean, short content: no Markdown document formatting, since
        # that used to leak into the chat bubble as literal "####"/"---"
        # text instead of visual structure (see frontend/index.html's own
        # Markdown/math renderer for the display half of this fix).
        formatting_rule = (
            " Plain text only -- no Markdown headings (no ####/###), no bold/italic "
            "markup, no horizontal rules (no ---)."
        )
        if needs_support:
            return (
                "TUTOR STATE: EXAMPLE -- the student is still confused. Acknowledge briefly, "
                "then give ONE tiny, DIFFERENT worked example of the SAME concept -- 2-3 short "
                "data points/numbers and one sentence naming the pattern, NOT a multi-part "
                "textbook example, no re-explanation of the theory. End with ONE very small "
                "comprehension check (e.g. picking between two simple options), not the "
                f"original question.{formatting_rule}"
            )
        return (
            "TUTOR STATE: EXAMPLE -- exactly ONE concrete worked example, no re-explanation, "
            f"no question. At most 3 short parts (e.g. the data, the model, one worked "
            f"calculation), one calculation at a time, short sentences, no redundant summary/"
            f"conclusion paragraph.{formatting_rule}"
        )
    if state == "GUIDED_PRACTICE" and needs_support:
        # A question is pending and the student asked for help/expressed
        # inability to answer it rather than attempting it (tutor_signal=
        # needs_support with a pending question, agents.plan_compiler.
        # _compile_tutor_state) -- nothing has been submitted/graded this
        # turn, so this must never read like a verdict. One small hint or
        # simpler first sub-step toward THIS pending question, never the
        # final answer, never a new/different question.
        question = tutor_state.get("current_question") or ""
        return (
            "TUTOR STATE: GUIDED_PRACTICE -- the student is asking for help with the PENDING "
            "QUESTION below, not attempting it (nothing has been submitted or graded). "
            "Acknowledge briefly, then give ONE small hint or a simpler first sub-step toward "
            "answering it -- do not reveal the final answer, do not say anything is correct or "
            "incorrect, do not introduce a different question. End by inviting them to try just "
            f"that smaller step.\n\nPENDING QUESTION: {question}"
        )
    return ""


def _bootstrap_intro(context: dict[str, Any]) -> str:
    """Phrasing hint for a proactively bootstrapped remediation turn (see
    services.remediation_bootstrap / agents.simple_agent.
    _try_remediation_bootstrap_fast_path). Empty for every ordinary
    (non-bootstrapped) EXPLAIN turn, so this changes nothing about the
    existing tutor-loop phrasing otherwise.

    The UI's own one-time "today's focus" card (see services.chat_orchestrator.
    _proactive_bootstrap_payload) already shows the concept, its mastery, and
    why it was picked -- so this turn's own text must not restate any of
    that, must never open with a greeting/announcement, and must stay short:
    orientation and a first taste of the idea, never the whole chapter.
    """
    bootstrap = context.get("tutor_bootstrap")
    if not bootstrap:
        return ""
    return (
        " This is a proactively started lesson -- the screen already shows the concept, its "
        "mastery, and why it was picked, so do NOT restate a percentage, a selection reason, or "
        "open with a greeting/announcement (never \"Hello!\", \"I have selected...\", \"Based on "
        "your analytics...\", \"your lowest-mastery concept is...\"). Open with one short, natural "
        "sentence (e.g. \"Let's start with this.\" or \"Let's build this up step by step.\"), then "
        "explain in 2-4 short sentences total -- orientation and a first taste of the idea only, "
        "never the whole chapter."
    )


_RECENT_TURNS_FOR_ANSWER_LIMIT = 2


def _compact_recent_turns_for_answer(turns: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Trim recent structured turns to what synthesis actually needs: goal,
    concepts, tools used, selected pipeline, sources (for "where did that
    come from"-style continuity), and a recommendation if one was made.
    Drops raw message/reply text and full per-tool observation dumps -- this
    turn's own `tool_results` (already passed separately) is the source of
    truth for facts in the CURRENT answer; past turns are only needed for
    conversational continuity, not to re-derive today's content."""
    compact: list[dict[str, Any]] = []
    for turn in (turns or [])[-_RECENT_TURNS_FOR_ANSWER_LIMIT:]:
        entry: dict[str, Any] = {
            "goal": turn.get("goal"),
            "concepts": (turn.get("resolved_entities") or {}).get("concepts") or [],
            "tools_used": turn.get("tools_used") or [],
            "selected_pipeline": turn.get("selected_pipeline"),
            "sources": turn.get("sources") or [],
        }
        if turn.get("recommendation"):
            entry["recommendation"] = turn["recommendation"]
        compact.append(entry)
    return compact


def _dedupe_tool_results(tool_results: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Keep only the last occurrence of each (tool, arguments) pair -- a
    defensive de-duplication so a tool re-attempted across a bounded replan
    (or an iterative-agent step) never sends the same RAG chunk text or
    mastery rows to the phrasing model twice. Order is otherwise preserved.
    """
    if not tool_results:
        return []
    last_index_for_key: dict[tuple[str, str], int] = {}
    for index, observation in enumerate(tool_results):
        args = {k: v for k, v in (observation.get("arguments") or {}).items() if k != "analytics_request"}
        key = (observation.get("tool"), compact_json(args))
        last_index_for_key[key] = index
    keep_indices = sorted(set(last_index_for_key.values()))
    return [tool_results[i] for i in keep_indices]


def _sanitize_final_answer(reply: str) -> str:
    cleaned = re.sub(r"```.*?```", "", reply, flags=re.DOTALL).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _trim_to_complete_sentence(text: str) -> str:
    """Last-resort cleanup for a completion that is STILL truncated even
    after the one bounded retry (see generate_final_answer) -- trims a
    trailing partial sentence so the student never sees an answer stopped
    mid-word ("...by calling"), at the cost of losing that last incomplete
    clause. Only reached when both the original and the retry call hit the
    output-token ceiling (rare at FINAL_ANSWER_RETRY_MAX_TOKENS)."""
    text = text.rstrip()
    if not text or text[-1] in ".!?\"')]}":
        return text
    complete_sentence_ends = list(re.finditer(r"[.!?](?=\s|$)", text))
    if complete_sentence_ends:
        return text[: complete_sentence_ends[-1].end()].rstrip()
    return text


def _fallback_final_answer(
    selected_pipeline: str,
    tool_results: list[dict[str, Any]],
    sources: list[str],
) -> str:
    """Return a safe, grounded answer if the final phrasing LLM fails.

    This is deliberately narrow. It prevents a reliable RAG retrieval from
    falling through to the legacy analytics router just because the final
    natural-language synthesis call returned empty/errored. It only uses
    already-authorized tool observations and never creates new PDF/source
    claims.
    """
    if selected_pipeline == "internal_rag":
        chunks = []
        for observation in tool_results or []:
            result = observation.get("result") or {}
            for chunk in result.get("chunks") or []:
                if isinstance(chunk, dict) and (chunk.get("text") or "").strip():
                    chunks.append(chunk)
        if not chunks:
            return ""
        concept = chunks[0].get("concept") or "this concept"
        excerpt = _first_readable_sentences(" ".join(str(chunk.get("text") or "") for chunk in chunks), limit=650)
        source_line = ""
        if sources:
            source_line = "\n\nSource: " + ", ".join(dict.fromkeys(str(source) for source in sources if source))
        return f"Based on the course material for {concept}, {excerpt}{source_line}".strip()

    analytics_reply = _fallback_analytics_answer(tool_results)
    if analytics_reply:
        return analytics_reply

    for observation in tool_results or []:
        result = observation.get("result") or {}
        reply = str(result.get("reply") or "").strip()
        if reply:
            return reply
    return ""


def _fallback_analytics_answer(tool_results: list[dict[str, Any]]) -> str:
    """Format successful analytics/mastery/policy tool results without
    another LLM call.

    Agent tools already return structured, authoritative data. If final
    synthesis fails, these deterministic fields are safer than falling back to
    the legacy router, which may ask for clarification even though the data was
    already gathered. This is also `agents.simple_agent._deterministic_reply`'s
    short-circuit for the "agent_tools" pipeline -- every branch here is a
    turn that never needs to reach the final-answer LLM call at all.
    """
    for observation in tool_results or []:
        result = observation.get("result") or {}
        formatted = str(result.get("formatted") or "").strip()
        if formatted:
            return formatted

    for observation in tool_results or []:
        result = observation.get("result") or {}
        # tools.profile_tools.run_study_recommendation_tool's own shape
        # (recommended_concept/current_mastery/basis/candidates) -- checked
        # BEFORE the generic `candidates` fallback below, which would
        # otherwise match its `candidates` field too and silently produce a
        # raw ranked list instead of the actual recommendation (a real gap:
        # a study_recommendation turn's only tool is run_study_recommendation,
        # so this was reachable on the primary path, not just a fallback).
        if result.get("recommended_concept"):
            concept = result["recommended_concept"]
            score = _format_mastery_value(result.get("current_mastery"))
            return f"I recommend focusing on {concept} next -- your current mastery there is {score}."
        # tools.mastery_tools.get_mastery_policy_tool / get_mastery_scoring_methodology_tool
        # -- fixed, single-purpose factual lookups (mastery band thresholds,
        # scoring formula rules) with no student-specific nuance to phrase,
        # so a template is exactly as accurate as an LLM paraphrase and was
        # previously reaching the final-answer LLM call for no benefit.
        bands = result.get("bands")
        if isinstance(bands, list) and bands:
            lines = [f"- {b.get('label')}: {b.get('range')}" for b in bands if isinstance(b, dict)]
            return "Mastery bands:\n\n" + "\n".join(lines)
        rules = result.get("rules")
        if isinstance(rules, list) and rules:
            lines = [f"- {r}" for r in rules]
            return "Mastery scoring works like this:\n\n" + "\n".join(lines)
        if isinstance(result.get("selected_concept"), dict) and result["selected_concept"].get("concept"):
            item = result["selected_concept"]
            return _format_mastery_lines("The lowest-mastery concept is:", [item])
        items = result.get("items")
        if isinstance(items, list) and items:
            return _format_mastery_lines("Here is the mastery data I found:", items)
        mastery = result.get("mastery")
        if isinstance(mastery, list) and mastery:
            ordered = sorted(mastery, key=lambda item: float(item.get("current_mastery") or item.get("mastery") or 0.0))
            return _format_mastery_lines("Here are your concept mastery levels, lowest first:", ordered)
        candidates = result.get("candidates")
        if isinstance(candidates, list) and candidates:
            ordered = sorted(candidates, key=lambda item: float(item.get("current_mastery") or item.get("mastery") or 0.0))
            return _format_mastery_lines("Here are the candidate mastery levels, lowest first:", ordered)
    return ""


def _format_mastery_lines(title: str, rows: list[dict[str, Any]]) -> str:
    lines = []
    for index, item in enumerate(rows, start=1):
        concept = item.get("concept") or item.get("course_name") or item.get("name") or "Item"
        course = item.get("course_name")
        value = (
            item.get("current_mastery")
            if item.get("current_mastery") is not None
            else item.get("mastery")
        )
        if value is None:
            value = item.get("course_average") if item.get("course_average") is not None else item.get("overall_average")
        score = _format_mastery_value(value)
        course_part = f" - {course}" if course and course != concept else ""
        lines.append(f"{index}. {concept}{course_part}: {score}")
    return title + "\n\n" + "\n".join(lines)


def _format_mastery_value(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "not available"
    if numeric <= 1.0:
        numeric *= 100
    return f"{numeric:.1f}%"


def _first_readable_sentences(text: str, limit: int = 650) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = re.sub(r"(?i)\b(source|page|chapter)\s*[:#]?\s*\S+", "", cleaned).strip()
    if not cleaned:
        return "the retrieved course material is available, but I could not synthesize it cleanly."
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    selected = []
    total = 0
    for sentence in sentences:
        if not sentence:
            continue
        if total + len(sentence) > limit and selected:
            break
        selected.append(sentence)
        total += len(sentence)
        if total >= limit:
            break
    summary = " ".join(selected).strip() or cleaned[:limit].strip()
    if summary and summary[-1] not in ".!?":
        summary += "."
    return summary
