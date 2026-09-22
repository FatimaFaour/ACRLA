"""Semantic planner: one LLM call that returns ONLY what genuinely requires
language understanding for this turn.

    User message
    -> one semantic planner call (this module)          [agents.simple_planner]
    -> deterministic goal -> tool compilation            [agents.plan_compiler]
    -> execute every required tool, in dependency order  [agents.simple_agent]
    -> one final answer call (or a deterministic reply)  [agents.response_generator]
    -> save structured memory                            [chat_orchestrator]

This module used to also ask the LLM to choose tools, order their
dependencies, and pick evidence -- all deterministic responsibilities that
depend only on which goal was classified, not on anything requiring language
understanding. Those now live in `agents.plan_compiler`, which maps this
module's `SemanticPlan` output onto a concrete `SimplePlan` (with `tools`)
purely by code: the same goal always compiles to the same tool shape. That
split is what keeps this prompt small -- there is no tool registry, no tool
capability table, and no JSON schema field for `tools`/`depends_on` here at
all, because none of that is a language-understanding decision.

The SAME prompt template is used both for the first call and the one bounded
replan `agents.simple_agent` may issue: `turn_observations` is empty on the
first call and populated with a short corrective note (plus `context`'s own
compact recent-turn memory) on the second, so the model reasons the same way
both times without a second prompt to maintain, and without ever resending a
tool registry that was never in this prompt to begin with.
"""

from __future__ import annotations

import time
from typing import Any

from langchain.prompts import ChatPromptTemplate

from agents.agent_json import AgentJSONError, compact_json, compact_observations, extract_message_text, model_validate, parse_json_object
from agents.agent_models import SemanticPlan
from agents.llm_errors import estimate_prompt_size, extract_token_usage, invoke_with_json_mode_retry, log_llm_provider_error
from services.llm_factory import get_json_llm


# Output-token ceiling for the planner call. Raising a ceiling never costs
# more for a provider that finishes well under it (Groq/Cerebras/OpenAI
# routinely complete this plan in a few hundred tokens) -- it only matters
# for a provider that needs more headroom to avoid truncating mid-completion.
# Gemini's "thinking"-capable models were observed consuming several hundred
# tokens of internal reasoning that count against this same ceiling but
# produce no visible text (confirmed live: a request with prompt_tokens=1243
# and a fully visible, well-formed JSON completion still reported
# total_tokens=2136 and finish_reason="MAX_TOKENS" at the old ceiling of
# 900) -- kept at 2500 even though this plan is now much smaller to answer
# with, since the same hidden-thinking-token risk applies regardless of
# input size.
_PLANNER_MAX_TOKENS = 2500

# finish_reason values (case-insensitive) that mean the provider stopped
# because it hit the output-token ceiling, not because it was done --
# Gemini reports "MAX_TOKENS", OpenAI/Groq-style providers report "length".
_TRUNCATION_FINISH_REASONS = {"MAX_TOKENS", "LENGTH"}

# How many of the most recent structured turns to give the planner. Only the
# fields a planning decision actually needs are kept per turn (see
# `_compact_recent_turns`). 2 is enough for every tested follow-up chain (a
# follow-up only ever needs the immediately preceding turn(s), never a
# longer history).
_RECENT_TURNS_LIMIT = 2


SIMPLE_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the ACRLA semantic planner. Return strict JSON \
only: your understanding of this message -- not a tool plan, not the final \
answer.

GOAL (exactly one, by meaning not wording): concept_explanation, \
concept_comparison, study_recommendation, analytics_query, \
personal_profile_query, personalized_tutoring, reference_followup, \
source_provenance, mastery_policy, mastery_modification_request, \
tutoring_methodology, clarification, casual_conversation, external_knowledge, \
assessment, navigation, unclear.
Own scores/mastery/progress -> analytics_query (or reference_followup if \
continuing) only for real intent to query/compare/inspect/rank stored data, \
however brief. A subjective statement about \
feelings, difficulty, confusion, or confidence ("I feel weak", "I'm \
struggling"), and likewise a request to move forward in the lesson itself \
("show me an example", "let's practice", "next question"), is \
casual_conversation/concept_explanation respectively -- UNLESS TUTOR STATE \
below is active, where EITHER kind of message is personalized_tutoring \
instead (see TUTOR_SIGNAL) -- mastery-band words alone never imply \
analytics. A request \
to \
launch a check ("check my progress", "take a \
quiz", "assess me") is assessment, never analytics_query. RECENT TURNS' \
last entry has \
pending_clarification_question -> continue that request (refine its \
analytics_request/entities) unless clearly unrelated. A request that \
explicitly names an external resource TYPE ("YouTube video", "a video", \
"website", "online tutorial", "external link/resource") is \
external_knowledge even if it also names an in-course concept ("a YouTube \
video about recursion") -- the concept is just entity context for that \
external answer, never a request to teach/explain from course material, so \
this is never concept_explanation/personalized_tutoring.

ENTITIES: concepts/courses = real data only. references = short labels for \
reused state, else empty. needs_clarification=true \
(+clarification_question) only if genuinely ambiguous/unresolvable.

TUTOR_SIGNAL (only if TUTOR STATE is set, else null): continue (ready to \
move forward -- either acknowledges understanding, e.g. "ok"/"got it"/"I \
understand", OR explicitly asks for the next step itself, e.g. "show me an \
example"/"let's try a question"/"next question" -- both mean the same \
thing: proceed), needs_support (confusion about the material, OR inability \
to answer the pending question -- "I don't understand", "I'm confused", "I \
don't know how to answer", "I need help", "can you give me a hint?", "I'm \
stuck" -- NOT an attempt at the question, even a wrong one), practice_answer \
(an actual attempt at the pending question's content, even briefly/wrong), \
new_topic (explicit different concept/course) -- by meaning, never wording; \
a clear analytics/external/casual intent still wins.

ANALYTICS REQUEST (analytics_query/refinement, else null): \
operation(list/rank/compare/get_value/summarize/recommend), \
entity(concept/course/overall), scope(current_chapter/current_course/\
all_courses), metric(current_mastery default/course_average/\
overall_average), direction(lowest/highest), threshold_band(weak<50%/\
moderate 50-79%/strong>=80%), compare_entities. A follow-up REFINES, not \
repeats: switch entity AND/OR scope to match THIS message's own dimension \
-- "in this course" narrows scope, "overall" widens it; else keep both. \
New concepts/courses named in THIS message \
override the prior turn's -- resolve to the real teaching course, not the \
active one by default.

Return exactly: {{"goal": "...", "concepts": [], "courses": [], \
"references": [], "analytics_request": null, "needs_clarification": false, \
"clarification_question": null, "confidence": 0.0, "tutor_signal": null}}"""),
    ("human", """MESSAGE: {message}

RECENT TURNS (most recent last): {recent_structured_turns}
{turn_observations_block}{recent_conversation_block}
SCOPE: remediation={remediation_level}; course={current_course}; \
concept={current_concept}; concepts={available_concepts}; {scope_rules}
TUTOR STATE: {tutor_state_block}"""),
])


def _compact_recent_turns(turns: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Keep only the fields a planning decision needs from each recent
    structured turn, most recent last, trimmed to `_RECENT_TURNS_LIMIT`.

    Raw message/reply text, per-tool observation summaries, sources,
    evidence, comparison detail, tools_used, and selected_pipeline are what
    `agents.response_generator`/`agents.entity_validator`/
    `agents.simple_agent._pending_context_issue` need downstream (the latter
    reads `context["recent_structured_turns"]` directly, uncompacted, so
    dropping fields here never affects it) -- none of that changes what this
    one-shot semantic judgment should be, so it is dropped here rather than
    sent (and re-sent) on every planner call.
    """
    compact: list[dict[str, Any]] = []
    for turn in (turns or [])[-_RECENT_TURNS_LIMIT:]:
        entry: dict[str, Any] = {
            "goal": turn.get("goal"),
            "concepts": (turn.get("resolved_entities") or {}).get("concepts") or [],
        }
        if turn.get("analytics_request"):
            entry["analytics_request"] = turn["analytics_request"]
        if turn.get("recommendation"):
            entry["recommendation"] = turn["recommendation"]
        if turn.get("selected_pipeline") == "clarification" and turn.get("assistant_reply"):
            # The one targeted exception to "no raw reply text": if the
            # previous turn's answer WAS a clarification question, the
            # planner needs to see what was actually asked in order to
            # recognize a short reply (e.g. naming a scope/entity/view) as
            # answering it, rather than reclassifying the reply as a fresh,
            # ambiguous, possibly-external message. Every other turn keeps
            # zero raw text, so this only adds tokens on the rare clarifying
            # turn, not every turn.
            entry["pending_clarification_question"] = turn["assistant_reply"]
        compact.append(entry)
    return compact


def _tutor_state_block(context: dict[str, Any]) -> str:
    """Compact one-line summary of the active tutor state, if any -- lets the
    planner classify `tutor_signal` without ever seeing the actual practice
    question text (that stays server-side, in agents.plan_compiler/the
    tools). Human-message only, not system-prompt-budget-constrained."""
    tutor_state = context.get("tutor_state") or {}
    state = tutor_state.get("state")
    if not state:
        return "none"
    return f"state={state} concept={tutor_state.get('concept')} difficulty={tutor_state.get('difficulty') or 'medium'}"


def _expand_flat_semantic_plan(data: Any) -> Any:
    """Repackage the LLM's flat `{goal, concepts, courses, references, ...}`
    JSON into `SemanticPlan.resolved_entities`'s nested shape.

    The prompt shows the model the smallest possible schema (no nested
    `resolved_entities` object to name) -- every downstream consumer
    (grounding, `agents.entity_validator`, `agents.plan_compiler`) still sees
    the same `ResolvedEntities` structure `agents.agent_models` already
    defines, this just does the repackaging once, right after parsing.
    """
    if not isinstance(data, dict) or "resolved_entities" in data:
        return data
    expanded = dict(data)
    expanded["resolved_entities"] = {
        "concepts": data.get("concepts") or [],
        "courses": data.get("courses") or [],
        "references": data.get("references") or [],
    }
    for key in ("concepts", "courses", "references"):
        expanded.pop(key, None)
    return expanded


def plan(context: dict[str, Any]) -> tuple[SemanticPlan, str, str | None, bool, dict[str, int]]:
    """Ask the LLM for one minimal semantic judgment of this turn.

    Returns (plan, raw_output, error, json_mode_fallback_used, token_usage).
    `token_usage` is `{"prompt_tokens", "completion_tokens", "total_tokens",
    "latency_ms"}` -- all zero if the provider reported no usage metadata and
    the call itself never happened (i.e. `error` is set). `latency_ms` is
    wall-clock time for the LLM call itself (including any internal
    json-mode-fallback retry), added so a slow turn's time can be attributed
    to this call specifically rather than only seen in the whole-turn total.
    See `agents.plan_compiler` for how this becomes a concrete tool plan.
    """
    raw = ""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "latency_ms": 0.0}
    llm = get_json_llm(temperature=0, max_tokens=_PLANNER_MAX_TOKENS)

    compact_turns = _compact_recent_turns(context.get("recent_structured_turns"))
    if compact_turns:
        # Structured memory already covers what a follow-up needs (goal,
        # entities, analytics dimension) -- sending the raw message/reply
        # history on top of it would just repeat the same information in a
        # bulkier, less structured form.
        recent_conversation_block = ""
    else:
        # No structured memory yet (e.g. the first turn of a session) --
        # raw recent messages are the only continuity signal available.
        recent_conversation_block = (
            "RECENT CONVERSATION (raw text, no structured memory yet):\n"
            f"{compact_json(context.get('recent_messages', []))}\n"
        )

    turn_observations = context.get("turn_observations") or []
    if turn_observations:
        # Only shown at all on the one bounded replan -- omitted entirely on
        # the (vast majority) first call, where it would otherwise just be
        # an empty-array label costing tokens for no information.
        turn_observations_block = f"\nOBSERVATIONS SO FAR THIS TURN: {compact_observations(turn_observations)}\n"
    else:
        turn_observations_block = ""

    prompt_values = {
        "message": context.get("message", ""),
        "recent_structured_turns": compact_json(compact_turns),
        "turn_observations_block": turn_observations_block,
        "recent_conversation_block": recent_conversation_block,
        "remediation_level": context.get("remediation_level", "chapter"),
        "current_course": (context.get("current_course") or {}).get("name") or "current Moodle course",
        "current_concept": context.get("current_concept") or "not set",
        "available_concepts": ", ".join(context.get("available_concepts") or []) or "none",
        "scope_rules": context.get("scope_rules") or "",
        "tutor_state_block": _tutor_state_block(context),
    }
    call_started = time.monotonic()
    try:
        response, json_mode_fallback_used = invoke_with_json_mode_retry(
            SIMPLE_PLANNER_PROMPT, prompt_values, llm=llm, temperature=0, max_tokens=_PLANNER_MAX_TOKENS,
            stage="simple_planner_plan",
        )
    except Exception as exc:
        # Never retried here -- a 429/auth/timeout/connection/5xx means the
        # call itself never completed, and agents.simple_agent must not spend
        # its bounded replan budget re-asking the same failing call.
        usage["latency_ms"] = (time.monotonic() - call_started) * 1000
        log_llm_provider_error(
            stage="simple_planner_plan", exc=exc, prompt_values=prompt_values,
            json_mode_enabled=True, llm=llm, response_length=0,
            final_fallback_reason="plan_provider_error",
        )
        return SemanticPlan(), raw, f"provider_error:{type(exc).__name__}:{exc}", False, usage
    call_latency_ms = (time.monotonic() - call_started) * 1000

    raw = extract_message_text(response)
    prompt_tok, completion_tok, total_tok = extract_token_usage(response)
    if not (prompt_tok or completion_tok or total_tok):
        prompt_tok, _ = estimate_prompt_size(prompt_values)
        completion_tok = len(raw) // 4
        total_tok = prompt_tok + completion_tok
    usage = {"prompt_tokens": prompt_tok, "completion_tokens": completion_tok, "total_tokens": total_tok, "latency_ms": call_latency_ms}

    response_metadata = getattr(response, "response_metadata", None) or {}
    finish_reason = str(response_metadata.get("finish_reason") or "")
    truncated = finish_reason.upper() in _TRUNCATION_FINISH_REASONS
    if truncated:
        # Distinct from a genuinely malformed completion -- the provider
        # stopped because it hit max_tokens, not because it produced bad
        # JSON. Logged separately so the two failure modes are never
        # conflated (a truncation needs a bigger ceiling; a malformed
        # completion at a normal length needs prompt/schema investigation).
        print(
            "[ACRLA] simple_planner_output_truncated "
            f"finish_reason={finish_reason} raw_output_len={len(raw)} "
            f"completion_tokens={completion_tok} max_tokens={_PLANNER_MAX_TOKENS}"
        )

    try:
        data = parse_json_object(raw)
    except AgentJSONError as exc:
        print(
            "[ACRLA] simple_planner_json_parse_error "
            f"raw_output_len={len(raw)} raw_output_preview={raw[:240]!r} error={exc} "
            f"finish_reason={finish_reason or 'unknown'}"
        )
        reason = "output_truncated" if truncated else f"json_parse_error:{exc}"
        return SemanticPlan(), raw, reason, json_mode_fallback_used, usage
    try:
        data = _expand_flat_semantic_plan(data)
        return model_validate(SemanticPlan, data), raw, None, json_mode_fallback_used, usage
    except Exception as exc:
        print(
            "[ACRLA] simple_planner_schema_error "
            f"raw_output_len={len(raw)} "
            f"parsed_keys={sorted(data.keys()) if isinstance(data, dict) else type(data).__name__} "
            f"error={exc} finish_reason={finish_reason or 'unknown'}"
        )
        # A truncated completion can be lenient-repaired into *parseable but
        # incomplete* JSON (e.g. a cut-off field) -- schema validation is
        # where that surfaces, not the JSON parse step, so the truncation
        # signal must be checked here too, not just in the parse-failure
        # branch above.
        reason = "output_truncated" if truncated else f"schema_validation_error:{type(exc).__name__}:{exc}"
        return SemanticPlan(), raw, reason, json_mode_fallback_used, usage
