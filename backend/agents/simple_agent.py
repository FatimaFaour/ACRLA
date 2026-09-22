"""Simple conversation agent: a low-token-usage alternative to the iterative
`agents.conversation_agent` loop, selected via `ACRLA_AGENT_MODE=simple`
(see `services.chat_orchestrator._try_conversation_agent`).

    User message
    -> one semantic planner call                         [agents.simple_planner]
    -> execute every required tool, in dependency order   [agents.agent_tools]
    -> (at most one bounded replan if genuinely needed)   [agents.simple_planner]
    -> one final answer call, or a deterministic reply    [agents.response_generator]
    -> save structured memory                             [chat_orchestrator, caller]

Hard budget for one turn: at most MAX_PLANNER_CALLS (2) planner calls and at
most MAX_RESPONSE_CALLS (1) final-answer call -- both enforced structurally
(this module never loops back to call either again once used), not just by a
counter check. A provider error (429/auth/timeout/etc.) on either call is
never retried -- see `agents.simple_planner.plan`, which re-raises any
non-json-mode provider error immediately.

Goal classification, entity resolution, tool selection, and the
tool-dependency plan are all decided semantically by the LLM in one shot
(`agents.simple_planner`). Everything in *this* module is deterministic
verification/orchestration against real data -- grounding hallucinated
concepts/tools, executing tools in dependency order, checking whether the
plan's own goal is actually backed by real evidence
(`agents.entity_validator`, already used identically by the iterative agent),
and picking the pipeline/sources the same way `agents.conversation_agent`
does. Nothing here keys off message wording.
"""

from __future__ import annotations

import re
import time
from typing import Any

from config import get_settings
from agents import entity_validator, evidence_validator, response_generator
from agents.agent_brain import _grounded_concepts_in_message
from agents.debug_log import vprint
from agents.agent_json import compact_json, model_dump, model_validate
from agents.agent_models import (
    AgentBrainOutput,
    AgentResult,
    Decision,
    EvidenceStatus,
    PlannerToolCall,
    ResolvedEntities,
    SemanticPlan,
    SimplePlan,
)
from agents.agent_tools import TOOL_REGISTRY, execute_agent_tools
from agents.conversation_agent import (
    _CONTEXT_TOOLS,
    _DETERMINISTIC_DATA_TOOLS,
    _analytics_reference_from_observations,
    _get_or_create_analytics_request,
    _last_observation,
    _recommendation_from_tool_results,
    _redact_arguments,
    _summarize_one_result,
)
from agents import plan_compiler, simple_planner
from services import course_concepts, remediation_bootstrap


MAX_PLANNER_CALLS = 2
MAX_RESPONSE_CALLS = 1

# Dialogue tools whose own result already carries a complete, deterministic
# "reply" string (agents.policies via tools.dialogue_tools) -- no separate
# final-answer LLM call is needed or wanted for these (see requirement to
# answer source-provenance/mastery-policy/course-structure-shaped questions
# without an LLM when possible).
_DIALOGUE_REPLY_TOOLS = {
    "get_source_provenance", "explain_methodology", "explain_recommendation",
    "get_mastery_guard_response", "get_course_structure", "get_preferences", "switch_focus",
    "generate_practice_question", "evaluate_practice_answer", "run_quick_progress_check",
}

# RQ2 institutional-privacy step: shown ONLY when search_course_material's
# own privacy gateway (services.privacy_context.filter_course_chunks_for_
# external, via pipelines.rag_pipeline.retrieve_context's for_external=True
# path) found relevant material but every candidate chunk was classified
# RESTRICTED -- never for "no material was found at all" (that already has
# its own, different external_fallback/clarification handling, unchanged).
# Deterministic and fixed -- never phrased by an LLM, so there is no risk
# of the wording itself echoing any blocked text, and no student-facing
# detail about *why* (sensitivity levels, chunk counts) leaks into the reply.
_RESTRICTED_CONTENT_REPLY = (
    "I have course material on this, but it isn't available for this kind of explanation here. "
    "Let's try a different angle, or check with your instructor for that specific material."
)

_PROVIDER_ERROR_MARKERS = (
    "llm_rate_limit", "llm_auth_error", "llm_timeout", "llm_connection_error",
    "llm_provider_5xx", "evidence_judge_provider_error", "external_knowledge_provider_error",
)


def _try_quick_progress_check_fast_path(
    context: dict[str, Any],
    observations: list[dict[str, Any]],
    requested_all: list[str],
    metrics: dict[str, Any],
) -> AgentResult | None:
    """Skip the planner LLM call entirely while a Quick Progress Check
    question is actively pending.

    `agents.plan_compiler.compile_plan` already ignores whatever goal a
    semantic classification would produce for this turn once a check is
    pending (its own session-state override -- see that module) and always
    routes to `run_quick_progress_check` instead. Since that result is
    discarded either way, spending a planner call to produce it is pure
    waste: build the same fixed plan directly from a default `SemanticPlan`
    and go straight to tool execution. Returns None (no fast path taken) the
    very first "start a check" turn, when nothing is pending yet and a real
    semantic classification is still needed to recognize that intent -- see
    tools.assessment_tools for the tool itself, which is what actually
    starts a check once this turn's goal=assessment classification runs.
    """
    quick_progress_check = context.get("quick_progress_check") or {}
    if not (quick_progress_check.get("concepts") and quick_progress_check.get("current_question")):
        return None

    plan = _compile_and_ground(context, SemanticPlan())
    context["goal"] = plan.goal
    context["is_followup"] = False
    requested_all.extend(t.name for t in plan.tools)

    print("[ACRLA] quick_progress_check_planner_skipped reason=assessment_answer_pending")
    tools_started = time.monotonic()
    obs, executed, rejected = _execute_tools(context, plan.tools, plan, set())
    metrics["tool_call_latency_ms"] += (time.monotonic() - tools_started) * 1000
    observations.extend(obs)
    _log_tools(1, obs)

    return _finalize_answer(context, plan, observations, requested_all, list(executed), metrics)


def _try_remediation_bootstrap_fast_path(
    context: dict[str, Any],
    observations: list[dict[str, Any]],
    requested_all: list[str],
    metrics: dict[str, Any],
) -> AgentResult | None:
    """Proactively start remediation on a fresh Moodle launch, before the
    student has to type "explain X"/"help me learn"/"where should I start?"
    -- from structured launch/session state (services.remediation_bootstrap).
    Returns None (no fast path taken, falls through to the ordinary
    planner-driven flow) whenever bootstrap should not fire this turn:
    already bootstrapped this session, a tutor/practice/assessment flow is
    already active, there is no usable launch concept, or -- most
    commonly -- a genuine, non-empty message is actually present this turn
    (`services.remediation_bootstrap.get_bootstrap_target`'s content check),
    in which case it must reach real classification instead of being
    silently overridden.

    Zero planner calls: builds a `SemanticPlan` naming the deterministically
    selected concept directly, then lets the EXISTING tutor-state fresh-start
    compilation (`agents.plan_compiler._compile_tutor_state`) and its
    `search_course_material`/`advance_tutor_state` tools do the rest --
    exactly the same tool sequence an ordinary "explain <concept>" turn
    already produces, so EXPLAIN phrasing/RAG grounding/evidence validation
    are all completely unchanged. One response-generator call is still spent
    to phrase the EXPLAIN turn (see `agents.response_generator._tutor_state_block`,
    which adds a short "why this concept was picked" hint via
    `context["tutor_bootstrap"]`).
    """
    bootstrap_target = remediation_bootstrap.get_bootstrap_target(context)
    if bootstrap_target is None:
        return None

    remediation_bootstrap.mark_bootstrapped(context)
    context["tutor_bootstrap"] = bootstrap_target
    print(
        "[ACRLA] proactive_remediation_bootstrap "
        f"level={bootstrap_target['level']} course_id={bootstrap_target.get('course_id')} "
        f"concept={bootstrap_target['concept']} mastery={bootstrap_target.get('mastery')} "
        f"reason={bootstrap_target['selection_reason']} planner_skipped=true"
    )

    semantic_plan = SemanticPlan(
        goal="concept_explanation",
        resolved_entities=ResolvedEntities(concepts=[bootstrap_target["concept"]]),
        confidence=1.0,
    )
    plan = _compile_and_ground(context, semantic_plan)
    context["goal"] = plan.goal
    context["is_followup"] = False
    requested_all.extend(t.name for t in plan.tools)

    tools_started = time.monotonic()
    obs, executed, rejected = _execute_tools(context, plan.tools, plan, set())
    metrics["tool_call_latency_ms"] += (time.monotonic() - tools_started) * 1000
    observations.extend(obs)
    _log_tools(1, obs)

    return _finalize_answer(context, plan, observations, requested_all, list(executed), metrics)


def run_simple_conversation_agent(context: dict[str, Any]) -> AgentResult:
    """Run the simple plan-once -> execute-all -> answer-once flow for one turn."""
    metrics = _new_metrics()
    observations: list[dict[str, Any]] = []
    context["turn_observations"] = observations
    executed_all: list[str] = []
    rejected_all: list[str] = []
    requested_all: list[str] = []

    fast_path_result = _try_quick_progress_check_fast_path(context, observations, requested_all, metrics)
    if fast_path_result is not None:
        return fast_path_result

    fast_path_result = _try_remediation_bootstrap_fast_path(context, observations, requested_all, metrics)
    if fast_path_result is not None:
        return fast_path_result

    semantic_plan, raw, error, json_mode_fallback_used, usage = simple_planner.plan(context)
    metrics["planner_call_count"] += 1
    _accumulate(metrics, "planner", usage)
    if error:
        _log_plan(1, semantic_plan, None, error, usage, json_mode_fallback_used)
        reason = "decision_provider_error" if str(error).startswith("provider_error:") else "decision_json_or_schema_invalid"
        _log_terminal(agent_success=False, goal="unclear", selected_pipeline=None, fallback_reason=reason, metrics=metrics)
        return AgentResult(fallback_reason=reason, **_metrics_kwargs(metrics))

    plan = _compile_and_ground(context, semantic_plan)
    _log_plan(1, semantic_plan, plan.tools, error, usage, json_mode_fallback_used)
    context["goal"] = plan.goal
    context["is_followup"] = bool(plan.resolved_entities.references)
    requested_all.extend(t.name for t in plan.tools)

    if plan.needs_clarification:
        return _clarification_agent_result(context, plan, requested_all, executed_all, metrics)

    # Pre-execution validation: a plan that proposes no tools for a goal that
    # structurally requires real evidence, or that quietly classifies an
    # ACRLA-internal-looking message as external/unclear right after a
    # pending analytics/clarification turn, is rejected BEFORE spending a
    # (no-op) execution pass on it -- the bounded replan gets a specific,
    # directive correction instead of rediscovering the same gap from zero
    # observations after the fact.
    pre_execution_issue = _structural_plan_issue(plan) or _pending_context_issue(context, plan)
    if pre_execution_issue and metrics["planner_call_count"] < MAX_PLANNER_CALLS:
        observations.append(_corrective_observation(pre_execution_issue))
        semantic_plan2, raw2, error2, json_mode_fallback_used2, usage2 = simple_planner.plan(context)
        metrics["planner_call_count"] += 1
        metrics["optional_replan_used"] = True
        _accumulate(metrics, "planner", usage2)
        if error2:
            _log_plan(2, semantic_plan2, None, error2, usage2, json_mode_fallback_used2)
        else:
            plan2 = _compile_and_ground(context, semantic_plan2)
            _log_plan(2, semantic_plan2, plan2.tools, error2, usage2, json_mode_fallback_used2)
            context["goal"] = plan2.goal
            context["is_followup"] = bool(plan2.resolved_entities.references)
            requested_all.extend(t.name for t in plan2.tools)
            if plan2.needs_clarification:
                return _clarification_agent_result(context, plan2, requested_all, executed_all, metrics)
            plan = plan2
        # else: the replan itself failed (provider/parse error) -- never
        # retried a third time; proceed with the original plan. If it is
        # still structurally incomplete, the post-execution gap check below
        # (budget now exhausted) safely falls to clarification/fallback
        # instead of hallucinating.

    tools_started = time.monotonic()
    obs1, executed1, rejected1 = _execute_tools(context, plan.tools, plan, set())
    metrics["tool_call_latency_ms"] += (time.monotonic() - tools_started) * 1000
    observations.extend(obs1)
    executed_all.extend(executed1)
    rejected_all.extend(rejected1)
    _log_tools(1, obs1)

    current_plan = plan
    gap_status = entity_validator.validate_answer_readiness(context, _plan_to_brain_output(current_plan), observations)
    replan_reason = _replan_reason(gap_status, observations)

    if replan_reason and metrics["planner_call_count"] < MAX_PLANNER_CALLS:
        observations.append(entity_validator.entity_gap_observation(gap_status))
        semantic_plan2, raw2, error2, json_mode_fallback_used2, usage2 = simple_planner.plan(context)
        metrics["planner_call_count"] += 1
        metrics["optional_replan_used"] = True
        _accumulate(metrics, "planner", usage2)
        if error2:
            _log_plan(2, semantic_plan2, None, error2, usage2, json_mode_fallback_used2)
        if not error2:
            plan2 = _compile_and_ground(context, semantic_plan2)
            _log_plan(2, semantic_plan2, plan2.tools, error2, usage2, json_mode_fallback_used2)
            requested_all.extend(t.name for t in plan2.tools)
            if plan2.needs_clarification:
                return _clarification_agent_result(context, plan2, requested_all, executed_all, metrics)
            already_done = _already_succeeded_keys(observations)
            tools2_started = time.monotonic()
            obs2, executed2, rejected2 = _execute_tools(context, plan2.tools, plan2, already_done)
            metrics["tool_call_latency_ms"] += (time.monotonic() - tools2_started) * 1000
            observations.extend(obs2)
            executed_all.extend(executed2)
            rejected_all.extend(rejected2)
            _log_tools(2, obs2)
            current_plan = plan2
        # else: provider/parse error on the replan itself -- never retried a
        # third time; fall through to finalize with plan1 + whatever ran.
        gap_status = entity_validator.validate_answer_readiness(context, _plan_to_brain_output(current_plan), observations)

    if _effective_gaps(gap_status, observations):
        # `observations` may already contain the synthetic gap-note appended
        # before the replan even when zero real tools ever ran -- check
        # `executed_all`/`rejected_all` (real tool attempts only) instead, so
        # a plan that named no tools at all correctly falls back here rather
        # than spending the one allowed response call on a clarification with
        # nothing behind it.
        if not executed_all and not rejected_all:
            _log_terminal(agent_success=False, goal=current_plan.goal, selected_pipeline=None, fallback_reason="simple_agent_no_evidence_gathered", metrics=metrics)
            return AgentResult(
                goal=current_plan.goal, confidence=current_plan.confidence, tools_requested=requested_all,
                resolved_concepts=current_plan.resolved_entities.concepts,
                analytics_request=model_dump(current_plan.analytics_request) if current_plan.analytics_request else None,
                fallback_reason="simple_agent_no_evidence_gathered", **_metrics_kwargs(metrics),
            )
        return _finalize_clarification(context, current_plan, observations, requested_all, executed_all, gap_status, metrics)

    return _finalize_answer(context, current_plan, observations, requested_all, executed_all, metrics)


# ---------------------------------------------------------------------------
# Grounding (mirrors agents.agent_brain._ground_decision, adapted to a plan)
# ---------------------------------------------------------------------------


def _ground_plan(context: dict[str, Any], plan: SimplePlan) -> SimplePlan:
    """Validate the compiled plan against real data -- never re-classify it.

    Same three grounding checks as `agent_brain._ground_decision`: drop any
    concept that is not real (recording it in `unresolved` instead of
    silently vanishing), drop any tool name that is not allowlisted (a
    defense-in-depth safety net now -- `agents.plan_compiler` only ever emits
    real, allowlisted tool names, so this should never actually drop
    anything), and force a clarification (with an empty tool list) when a
    goal that structurally requires multiple real concepts (concept
    comparison) does not have enough of them -- so a request like "compare
    recursion and charts" cannot silently answer for Recursion alone once
    "charts" fails to resolve.

    RQ1.B root-cause fix (see rq1b_root_cause_analysis.md, Root cause 2): a
    concept name is matched against `available` both exactly and via
    `course_concepts.resolve_concept_identity` -- a deterministic,
    exact-after-normalization match that tolerates only a superficial
    ingestion-time display prefix (e.g. "Chap: Linear Regression" vs.
    "Linear Regression"), never a fuzzy/token-overlap one. This resolves a
    planner's natural clean-form proposal to the exact raw storage key so
    it is never spuriously dropped into `unresolved` merely because a
    course's material was never registered in `COURSE_CONCEPTS`.
    """
    available = context.get("available_concepts") or []
    available_set = set(available)

    data = model_dump(plan)
    proposed_concepts = data["resolved_entities"]["concepts"]
    llm_concepts: list[str] = []
    dropped_concepts: list[str] = []
    concept_identity_rewrites: dict[str, str] = {}
    for concept in proposed_concepts:
        resolved = course_concepts.resolve_concept_identity(concept, available)
        if resolved is not None:
            if resolved not in llm_concepts:
                llm_concepts.append(resolved)
            if resolved != concept:
                concept_identity_rewrites[concept] = resolved
        else:
            dropped_concepts.append(concept)
    for concept in _grounded_concepts_in_message(context, available):
        if concept not in llm_concepts:
            llm_concepts.append(concept)
    data["resolved_entities"]["concepts"] = llm_concepts

    unresolved = list(data["resolved_entities"].get("unresolved") or [])
    for concept in dropped_concepts:
        if concept not in unresolved:
            unresolved.append(concept)
    data["resolved_entities"]["unresolved"] = unresolved

    if concept_identity_rewrites:
        # `plan_compiler` already baked the planner's ORIGINAL (pre-grounding)
        # concept strings into each compiled tool call's arguments (e.g.
        # search_course_material's "concepts" list, advance_tutor_state's
        # "concept" string) before this function ever ran. Without this
        # rewrite, a concept resolved above via `resolve_concept_identity`
        # (display-form -> raw storage key) would still reach the tool under
        # its pre-resolution spelling, so the tool -- and every downstream
        # consumer that expects the canonical raw key (evidence/tool-result
        # matching, tutor_state persistence) -- would not see the same
        # identity `resolved_entities.concepts` now reports.
        for tool_call in data.get("tools") or []:
            arguments = tool_call.get("arguments")
            if not isinstance(arguments, dict):
                continue
            if isinstance(arguments.get("concepts"), list):
                arguments["concepts"] = [
                    concept_identity_rewrites.get(c, c) for c in arguments["concepts"]
                ]
            if isinstance(arguments.get("concept"), str):
                arguments["concept"] = concept_identity_rewrites.get(
                    arguments["concept"], arguments["concept"]
                )

    valid_tools = []
    dropped_tools = []
    for tool_call in data.get("tools") or []:
        name = tool_call.get("name")
        if name in TOOL_REGISTRY:
            valid_tools.append(tool_call)
        else:
            dropped_tools.append(name or "unknown")
    data["tools"] = valid_tools
    if dropped_tools:
        print(f"[ACRLA] simple_agent_dropped_hallucinated_tools tools={dropped_tools}")

    # RQ1.B root-cause fix (see rq1b_root_cause_analysis.md, Root cause 1):
    # evaluated directly against how many real concepts resolved for a goal
    # that structurally requires them -- never gated on whether `unresolved`
    # happens to be non-empty. `unresolved` only reflects whether the
    # planner happened to NAME a concept that then failed resolution; a
    # planner that instead silently omits an unrecognized entity produces an
    # empty `unresolved` list, which must not exempt a `concept_comparison`
    # request from its own >=2-real-concepts requirement.
    #
    # A single-concept `concept_explanation` request with zero resolved
    # concepts is deliberately NOT forced to clarification here: plan_compiler
    # already compiles `search_course_material` for this goal regardless of
    # how many concepts resolved (the tool resolves a missing concept itself
    # via free-text query fallback -- see tools/rag_tools.py), so a
    # plausible but out-of-corpus topic is allowed to reach real retrieval
    # and let the evidence-reliability gate (agents.evidence_validator)
    # decide internal_rag vs. external_fallback, instead of being pre-empted
    # before retrieval ever runs.
    if (
        data.get("goal") == "concept_comparison"
        and len(llm_concepts) < 2
        and not data.get("needs_clarification")
    ):
        data["needs_clarification"] = True
        data["tools"] = []

    if data.get("needs_clarification") and not data.get("clarification_question"):
        data["clarification_question"] = _default_clarification_text(unresolved, llm_concepts)

    return model_validate(SimplePlan, data)


def _compile_and_ground(context: dict[str, Any], semantic_plan: SemanticPlan) -> SimplePlan:
    """Deterministically compile a semantic plan into concrete tools, then
    ground it against real data -- the one place `agents.plan_compiler` and
    `_ground_plan` are chained, used at all three points this module ever
    turns a planner call into an executable plan (the first call and both
    bounded-replan call sites)."""
    return _ground_plan(context, plan_compiler.compile_plan(semantic_plan, context))


def _default_clarification_text(unresolved: list[str], resolved_concepts: list[str]) -> str:
    if unresolved:
        names = ", ".join(dict.fromkeys(unresolved))
        return f"I couldn't find '{names}' in this course. Could you clarify what you'd like me to work with instead?"
    return "Could you clarify what you'd like help with?"


def _default_clarification_question(resolved_entities) -> str:
    return _default_clarification_text(resolved_entities.unresolved, resolved_entities.concepts)


def _clarification_agent_result(
    context: dict[str, Any],
    plan: SimplePlan,
    requested_all: list[str],
    executed_all: list[str],
    metrics: dict[str, Any],
) -> AgentResult:
    """Build the AgentResult for a needs_clarification plan -- the planner's
    own clarification_question is the reply directly, no response-generator
    call needed. Shared by all three points a plan can resolve to a
    clarification (up front, after the pre-execution structural/context
    replan, after the post-execution entity-gap replan) so the exact same
    fields are always populated, including carrying forward whatever
    analytics_request already exists (this plan's own, or the turn's
    already-persisted one) so a FOLLOW-UP answering this clarification has
    something concrete to refine instead of starting over."""
    reply = plan.clarification_question or _default_clarification_question(plan.resolved_entities)
    _log_terminal(agent_success=True, goal=plan.goal, selected_pipeline="clarification", fallback_reason=None, metrics=metrics)
    return AgentResult(
        agent_used=True, reply=reply, goal=plan.goal, confidence=plan.confidence,
        tools_requested=requested_all, tools_executed=executed_all,
        resolved_concepts=plan.resolved_entities.concepts, selected_pipeline="clarification",
        reasoning_basis=plan.answer_basis,
        analytics_request=model_dump(plan.analytics_request) if plan.analytics_request else context.get("analytics_request"),
        **_metrics_kwargs(metrics),
    )


def _plan_to_brain_output(plan: SimplePlan) -> AgentBrainOutput:
    """Adapt a SimplePlan into the AgentBrainOutput shape
    `agents.entity_validator`/`agents.response_generator` already consume,
    so this module reuses their exact, already-tested logic instead of
    duplicating it."""
    return AgentBrainOutput(
        goal=plan.goal,
        resolved_entities=plan.resolved_entities,
        decision=Decision(type="answer", tool=None, arguments={}, reason="simple_plan_finalized"),
        evidence_status=EvidenceStatus(sufficient=True),
        answer_basis=plan.answer_basis,
        confidence=plan.confidence,
        analytics_request=plan.analytics_request,
    )


def _with_missing_from_gaps(brain_like: AgentBrainOutput, gap_status) -> AgentBrainOutput:
    data = model_dump(brain_like)
    data["evidence_status"]["sufficient"] = False
    data["evidence_status"]["missing"] = entity_validator.entity_gap_descriptions(gap_status)
    return model_validate(AgentBrainOutput, data)


# ---------------------------------------------------------------------------
# Pre-execution plan validation (checked before any tool runs, so a
# structurally incomplete plan gets a specific, directive replan correction
# instead of rediscovering the identical gap from zero observations after a
# no-op execution pass)
# ---------------------------------------------------------------------------


def _structural_plan_issue(plan: SimplePlan) -> str | None:
    """A plan that proposes NO tools and does not ask for clarification for
    a goal that structurally requires real tool evidence
    (`entity_validator.EVIDENCE_REQUIRING_GOALS` -- the same goal/tool
    mapping the post-execution entity-completeness gate already enforces)
    is incomplete before it is even executed. Purely goal-enum-keyed, never
    message wording."""
    if plan.tools or plan.needs_clarification:
        return None
    required_tools = entity_validator.EVIDENCE_REQUIRING_GOALS.get(plan.goal)
    if not required_tools:
        return None
    return (
        f"goal={plan.goal} requires real evidence from one of {sorted(required_tools)} "
        "before it can be answered, but this plan proposed tools=[] and did not set "
        "needs_clarification=true. Include one of those tools, or set "
        "needs_clarification=true with a specific clarification_question if the "
        "request is genuinely ambiguous."
    )


def _pending_context_issue(context: dict[str, Any], plan: SimplePlan) -> str | None:
    """A plan that classifies the turn as unclear/external_knowledge with
    low confidence and no internal-data tool, right after a turn that was
    itself analytics_query or ended in a clarification, is suspicious -- the
    planner likely failed to recognize this message as a continuation of
    that pending request rather than a genuinely new external question.
    Purely state-keyed (the previous turn's own goal/selected_pipeline,
    already in structured memory) -- never message wording, and never
    triggered for a genuinely confident/tool-backed external classification.
    """
    if plan.goal not in {"unclear", "external_knowledge"}:
        return None
    if plan.confidence >= 0.80:
        return None
    if any(t.name == _ANALYTICS_SATISFYING_TOOL for t in plan.tools):
        return None
    recent_turns = context.get("recent_structured_turns") or []
    if not recent_turns:
        return None
    last_turn = recent_turns[-1]
    last_goal = last_turn.get("goal")
    last_pipeline = last_turn.get("selected_pipeline")
    if last_goal != "analytics_query" and last_pipeline != "clarification":
        return None
    return (
        f"the previous turn was goal={last_goal!r} selected_pipeline={last_pipeline!r}, "
        f"but this plan classified the CURRENT message as goal={plan.goal!r} with low "
        "confidence and no internal-data tool. Reconsider whether this message answers "
        "or continues that pending request (e.g. selecting an analytics view/scope) "
        "before treating it as external knowledge or unclear -- an ACRLA/Moodle/course/"
        "student question must never become external_knowledge merely because "
        "confidence is low."
    )


def _corrective_observation(reason: str) -> dict[str, Any]:
    """Synthetic observation surfacing a pre-execution structural/context
    violation to the next planner call -- the same mechanism
    `entity_validator.entity_gap_observation` uses for post-execution gaps,
    with a specific, directive note instead of a generic one."""
    return {
        "tool": "plan_structure_check", "arguments": {}, "rejected": False,
        "result": {
            "accepted": False, "violation": reason,
            "note": f"Your previous plan was rejected before execution: {reason}",
        },
    }


# ---------------------------------------------------------------------------
# Replan trigger
# ---------------------------------------------------------------------------


_ANALYTICS_SATISFYING_TOOL = "run_analytics_query"
_EXTERNAL_SATISFYING_TOOL = "answer_with_external_knowledge"
_PROFILE_SATISFYING_TOOL = "get_student_learning_profile"


def _tool_succeeded_with_usable_output(observations: list[dict[str, Any]], tool_name: str) -> bool:
    """Did `tool_name` already run this turn and produce output a final
    answer can actually be built from -- an authoritative analytics
    `formatted` string, a successful external-knowledge reply, or any
    error-free profile read? Used to decide whether a remaining
    entity/goal-evidence gap is a genuine missing requirement or just a
    bookkeeping mismatch on top of an already-usable result (see
    `_replan_reason`)."""
    for obs in observations:
        if obs.get("tool") != tool_name or obs.get("rejected"):
            continue
        result = obs.get("result") or {}
        if result.get("error"):
            continue
        if tool_name == _ANALYTICS_SATISFYING_TOOL and str(result.get("formatted") or "").strip():
            return True
        if tool_name == _EXTERNAL_SATISFYING_TOOL and result.get("success") and str(result.get("reply") or "").strip():
            return True
        if tool_name == _PROFILE_SATISFYING_TOOL:
            return True
    return False


def _content_goal_evidence_rendered(observations: list[dict[str, Any]]) -> bool:
    """Did `search_course_material` already run this turn and come back with
    a real evidence verdict (reliable OR unreliable -- either is a genuine
    decision, not a gap)?

    RQ1.B root-cause fix (see rq1b_root_cause_analysis.md, Root cause 1):
    once a content goal's own authoritative evidence source -- the
    reliability gate inside `tools.rag_tools.search_course_material_tool` --
    has actually rendered a verdict this turn, a concept name that failed to
    resolve (`entity_type == "unresolved"`) is no longer a reason to force a
    replan/clarification on top of it: the turn already has a real basis to
    finalize on (`internal_rag` if reliable, `external_fallback` if not --
    see `agents.simple_agent._finalize_answer`). Without this, `_ground_plan`
    now correctly allows retrieval to proceed for a single-concept
    `concept_explanation` naming an out-of-corpus topic (Root cause 1's
    other half), but the unconditional `unresolved` gap below would still
    silently override that real verdict and force clarification anyway --
    defeating the very requirement this fix exists to satisfy ("if evidence
    is unreliable, the reliability gate must be able to produce external
    fallback")."""
    for obs in observations:
        if obs.get("tool") != "search_course_material" or obs.get("rejected"):
            continue
        result = obs.get("result") or {}
        if result.get("error"):
            continue
        if isinstance(result.get("evidence"), dict):
            return True
    return False


def _effective_gaps(gap_status, observations: list[dict[str, Any]]) -> list:
    """`gap_status.gaps` with any moot `unresolved` entry dropped once the
    goal's own content-evidence source already rendered a real verdict this
    turn (see `_content_goal_evidence_rendered`).

    RQ1.B root-cause fix (Root cause 1): `_replan_reason` already stops
    spending a second planner call on a moot unresolved-concept gap once
    real evidence exists; this is the matching fix for the FINAL
    answer-vs-clarification decision in `run_simple_conversation_agent` --
    without it, the unfiltered `gap_status.gaps` (entity_validator's own
    unconditional per-turn check, independent of any replan) would still
    force `_finalize_clarification` even after a replan was correctly
    skipped, silently overriding a real `internal_rag`/`external_fallback`
    verdict the reliability gate already rendered.
    """
    if not gap_status.gaps:
        return []
    if not _content_goal_evidence_rendered(observations):
        return list(gap_status.gaps)
    return [gap for gap in gap_status.gaps if gap.entity_type != "unresolved"]


def _replan_reason(gap_status, observations: list[dict[str, Any]]) -> str | None:
    """One of the bounded-replan triggers, or None if nothing needs a second
    planner call.

    Two categories of gap always trigger a replan, since nothing already
    gathered this turn can resolve them on their own: a genuinely unresolved
    entity (`entity_type == "unresolved"`, unless the goal's own content
    evidence source already rendered a real verdict this turn -- see
    `_content_goal_evidence_rendered`), and a real (non-provider) tool
    failure (`entity_type == "tool"`, unless every failed tool's own error
    was itself a provider-connectivity issue -- replanning cannot fix a rate
    limit/timeout/connection error, see `_is_provider_error_gap`).

    Every other gap is an entity-completeness/goal-evidence bookkeeping
    check -- e.g. a resolved concept the LLM listed but the tool result
    happens not to name verbatim, or a goal label that does not exactly
    match which tool ran. Once the turn's own authoritative tool already
    produced a usable result (analytics formatted output, a successful
    external-knowledge reply, an error-free profile read, or a rendered
    content-evidence verdict), a bookkeeping mismatch on top of that is not
    a reason to spend a second planner call -- the deterministic tool result
    already is the answer's basis.
    """
    if not gap_status.gaps:
        return None

    analytics_satisfied = _tool_succeeded_with_usable_output(observations, _ANALYTICS_SATISFYING_TOOL)
    external_satisfied = _tool_succeeded_with_usable_output(observations, _EXTERNAL_SATISFYING_TOOL)
    profile_satisfied = _tool_succeeded_with_usable_output(observations, _PROFILE_SATISFYING_TOOL)
    content_evidence_rendered = _content_goal_evidence_rendered(observations)
    already_satisfied = analytics_satisfied or external_satisfied or profile_satisfied

    for gap in gap_status.gaps:
        if gap.entity_type == "unresolved":
            if content_evidence_rendered:
                continue
            return gap.reason
        if _is_provider_error_gap(gap, observations):
            continue
        if gap.entity_type == "tool":
            return gap.reason
        if already_satisfied:
            continue
        return gap.reason
    return None


def _is_provider_error_gap(gap, observations: list[dict[str, Any]]) -> bool:
    if gap.entity_type != "tool":
        return False
    failed_tool_names = {name.strip() for name in (gap.value or "").split(",") if name.strip()}
    if not failed_tool_names:
        return False
    for obs in observations:
        if obs.get("tool") not in failed_tool_names:
            continue
        result = obs.get("result") or {}
        error_text = str(result.get("error") or result.get("reason") or "")
        if not any(marker in error_text for marker in _PROVIDER_ERROR_MARKERS):
            return False
    return True


# ---------------------------------------------------------------------------
# Tool dependency ordering + execution (reuses agents.agent_tools.execute_agent_tools)
# ---------------------------------------------------------------------------


def _topological_order(tools: list[PlannerToolCall]) -> list[PlannerToolCall]:
    """Order tools so a tool naming `depends_on` runs after the tool it
    depends on. Purely structural (by tool `name`, a fixed allowlisted
    identifier) -- never message-derived. A `depends_on` that names a tool
    not present in this same plan, or a cycle, is treated as no dependency
    (the tool is placed as soon as nothing blocks it) rather than dropped."""
    names_in_plan = {t.name for t in tools}
    ordered: list[PlannerToolCall] = []
    placed_names: set[str] = set()
    remaining = list(tools)
    for _ in range(len(tools) + 1):
        if not remaining:
            break
        still_remaining = []
        progressed = False
        for t in remaining:
            dep = (t.depends_on or "").strip()
            if not dep or dep not in names_in_plan or dep in placed_names:
                ordered.append(t)
                placed_names.add(t.name)
                progressed = True
            else:
                still_remaining.append(t)
        remaining = still_remaining
        if not progressed:
            break
    ordered.extend(remaining)
    return ordered


def _already_succeeded_keys(observations: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """(tool, arguments) pairs that already succeeded this turn -- excludes
    the code-injected `analytics_request` key (same exclusion
    `agents.decision_acceptance._is_duplicate_successful_call` uses) so a
    persisted analytics request never defeats duplicate detection."""
    keys: set[tuple[str, str]] = set()
    for obs in observations:
        if obs.get("rejected"):
            continue
        result = obs.get("result") or {}
        if result.get("error") or result.get("success") is False:
            continue
        args = {k: v for k, v in (obs.get("arguments") or {}).items() if k != "analytics_request"}
        keys.add((obs.get("tool"), compact_json(args)))
    return keys


def _execute_tools(
    context: dict[str, Any],
    tool_calls: list[PlannerToolCall],
    plan: SimplePlan,
    already_done: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Execute every tool in the plan, in dependency order, through the SAME
    `execute_agent_tools` the iterative agent uses -- so a dependent tool
    (e.g. search_course_material after a selection tool) automatically sees
    whatever side-effect (`context["agent_selected_concept"]`) the tool it
    depends on already set, with no bespoke templating engine needed here."""
    ordered = _topological_order(tool_calls)
    actions: list[dict[str, Any]] = []
    for t in ordered:
        arguments = dict(t.arguments or {})
        if t.name == "run_analytics_query":
            persisted = _get_or_create_analytics_request(context, plan)
            if persisted is not None:
                arguments = dict(arguments)
                arguments["analytics_request"] = persisted
        key_args = {k: v for k, v in arguments.items() if k != "analytics_request"}
        if (t.name, compact_json(key_args)) in already_done:
            continue
        actions.append({"tool": t.name, "arguments": arguments})

    results, executed, rejected_names = execute_agent_tools(actions, context)
    observations = [{**result, "rejected": False} for result in results]
    for name in rejected_names:
        observations.append({
            "tool": name, "arguments": {}, "rejected": True,
            "result": {"error": f"tool_not_allowed:{name}"},
        })
    return observations, executed, rejected_names


# ---------------------------------------------------------------------------
# Finalize: pipeline selection, deterministic-reply shortcut, one answer call
# ---------------------------------------------------------------------------


# Exact, normalized (lowercased, punctuation-stripped) short phrases only --
# never a substring/keyword match -- so a genuinely subjective or nuanced
# casual_conversation message ("I feel weak", "recursion is confusing")
# always still gets a real, context-aware LLM-phrased reply. This only
# changes response PHRASING for an already-classified casual_conversation
# turn; it never participates in goal classification/routing.
_QUOTA_SAVER_GREETING_REPLIES = {
    "hi": "Hi! What would you like to work on?",
    "hello": "Hello! What would you like to work on?",
    "hey": "Hey! What can I help you with?",
    "thanks": "You're welcome! Let me know if you'd like to keep practicing.",
    "thank you": "You're welcome! Let me know if you'd like to keep practicing.",
    "ok": "Got it -- let me know what you'd like to do next.",
    "okay": "Got it -- let me know what you'd like to do next.",
    "bye": "See you next time -- good luck with your studies!",
    "goodbye": "See you next time -- good luck with your studies!",
}


def _quota_saver_greeting_reply(message: str) -> str | None:
    normalized = re.sub(r"[^a-z ]", "", str(message or "").strip().lower())
    return _QUOTA_SAVER_GREETING_REPLIES.get(normalized)


def _deterministic_reply(context: dict[str, Any], plan: SimplePlan, observations: list[dict[str, Any]], selected_pipeline: str) -> str | None:
    """A reply already fully formed by a tool result -- no final-answer LLM
    call needed. `agent_tools` reuses the exact same check
    `response_generator.generate_final_answer` applies internally (kept here
    too so the caller can skip the call entirely and report an accurate
    response_call_count instead of calling in and discovering the
    short-circuit only after the fact)."""
    if selected_pipeline == "agent_tools":
        reply = response_generator._fallback_analytics_answer(observations)
        if reply:
            if get_settings().acrla_quota_saver:
                print("[ACRLA] quota_saver_skipped_llm reason=analytics_already_formatted")
            return reply
    if selected_pipeline == "deterministic_reply":
        for obs in reversed(observations):
            if obs.get("tool") in _DIALOGUE_REPLY_TOOLS:
                result = obs.get("result") or {}
                reply = str(result.get("reply") or "").strip()
                if reply:
                    return reply
    if selected_pipeline == "external_fallback":
        # RQ2 institutional-privacy step: search_course_material found
        # relevant chunks but the privacy gateway blocked every one of them
        # as RESTRICTED (evidence.reliable=False in that case, exactly like
        # "nothing found" -- which is why this lands on external_fallback
        # the same way an ordinary empty-retrieval turn does) -- return a
        # fixed, deterministic reply instead of silently letting this fall
        # through to a real external_fallback LLM call, which would answer
        # from general knowledge without ever telling the student that
        # institutional material existed but was withheld. Checked BEFORE
        # the quota-saver greeting check below (a distinct, unrelated case).
        for obs in observations:
            if obs.get("tool") == "search_course_material":
                result = obs.get("result") or {}
                if result.get("all_candidates_restricted"):
                    return _RESTRICTED_CONTENT_REPLY
    if selected_pipeline == "external_fallback" and plan.goal == "casual_conversation" and get_settings().acrla_quota_saver:
        greeting_reply = _quota_saver_greeting_reply(context.get("message", ""))
        if greeting_reply:
            print("[ACRLA] quota_saver_skipped_llm reason=greeting_template")
            return greeting_reply
    return None


def _finalize_answer(
    context: dict[str, Any],
    plan: SimplePlan,
    observations: list[dict[str, Any]],
    requested_all: list[str],
    executed_all: list[str],
    metrics: dict[str, Any],
) -> AgentResult:
    resolved_concepts = plan.resolved_entities.concepts
    recommendation, recommendation_reason = _recommendation_from_tool_results(observations)

    if "search_course_material" in executed_all:
        search_result = (_last_observation(observations, "search_course_material") or {}).get("result") or {}
        evidence = search_result.get("evidence")
        retrieved_sources = search_result.get("sources") or []
        if not isinstance(evidence, dict):
            evidence = {"reliable": False, "coverage": "none", "supported_concepts": [], "reason": "search_tool_returned_no_evidence", "confidence": 0.0}
        if evidence["reliable"] and not retrieved_sources:
            _log_terminal(agent_success=False, goal=plan.goal, selected_pipeline=None, fallback_reason="rag_reliable_but_sources_missing", metrics=metrics)
            return AgentResult(
                goal=plan.goal, confidence=plan.confidence, tools_requested=requested_all, tools_executed=executed_all,
                resolved_concepts=resolved_concepts, evidence_reliable=evidence["reliable"], evidence_coverage=evidence["coverage"],
                evidence_supported_concepts=evidence["supported_concepts"], evidence_reason=evidence["reason"],
                evidence_confidence=evidence["confidence"], recommendation=recommendation, recommendation_reason=recommendation_reason,
                analytics_request=context.get("analytics_request"),
                fallback_reason="rag_reliable_but_sources_missing", **_metrics_kwargs(metrics),
            )
        selected_pipeline = "internal_rag" if evidence["reliable"] else "external_fallback"
        sources = retrieved_sources if evidence["reliable"] else []
        agent_fallback_reason = None if evidence["reliable"] else f"evidence_not_reliable:{evidence['reason']}"
    elif set(executed_all) & _DETERMINISTIC_DATA_TOOLS:
        selected_pipeline, sources, agent_fallback_reason = "agent_tools", [], None
        evidence = evidence_validator.empty_evidence("analytics_no_rag")
    elif set(executed_all) & _DIALOGUE_REPLY_TOOLS:
        selected_pipeline, sources, agent_fallback_reason = "deterministic_reply", [], None
        evidence = evidence_validator.empty_evidence("deterministic_tool_reply")
    elif set(executed_all) & _CONTEXT_TOOLS:
        selected_pipeline, sources, agent_fallback_reason = "context_metadata", [], None
        evidence = evidence_validator.empty_evidence("context_only_no_rag")
    elif "advance_tutor_state" in executed_all and context.get("tutor_needs_support"):
        # First-time needs_support continuation of an active GUIDED_PRACTICE
        # question (agents.plan_compiler._compile_tutor_state's
        # consecutive_confusion < 2 branch): advance_tutor_state is the only
        # tool this turn BY DESIGN (the same already-active concept's course
        # material was already retrieved earlier this session; re-searching
        # for a hint on the same pending question would be redundant). This
        # is still a live, internal, course-grounded tutoring continuation,
        # never an off-syllabus/general-knowledge request -- must not fall
        # into the generic "nothing tool-related happened" external_fallback
        # bucket below, which mislabels it and tells the final-answer LLM to
        # answer from general knowledge with no course claim. No new
        # retrieval is performed here (that would be a behavior change, not
        # a classification fix); `sources` stays [] because no source was
        # re-verified this turn, matching the honest "no fresh evidence"
        # state -- see FINAL_PROMPT's own tutor_continuation grounding rule.
        selected_pipeline, sources, agent_fallback_reason = "tutor_continuation", [], None
        evidence = evidence_validator.empty_evidence("tutor_state_continuation_no_fresh_retrieval")
    else:
        selected_pipeline, sources, agent_fallback_reason = "external_fallback", [], None
        evidence = evidence_validator.empty_evidence("no_tool_executed_external")

    reply = _deterministic_reply(context, plan, observations, selected_pipeline)
    provider_error_category = None
    if reply is None:
        brain_like = _plan_to_brain_output(plan)
        token_usage: dict[str, Any] = {}
        reply = response_generator.generate_final_answer(
            context, brain_like, observations, selected_pipeline, evidence, sources=sources, token_usage=token_usage,
        )
        if token_usage.get("llm_called"):
            metrics["response_call_count"] += 1
            _accumulate(metrics, "response", token_usage)
        provider_error_category = token_usage.get("provider_error_category")

    if not reply:
        # A provider-level failure on THIS call (rate limit/auth/model-
        # unavailable/5xx/timeout/connection) gets its own fallback_reason,
        # distinct from a genuinely empty/unanswerable reply -- so
        # services.chat_orchestrator can terminate the LLM path for this
        # turn (deterministic provider-unavailable reply) instead of falling
        # through to legacy LLM-based routing, which would just re-hit the
        # same failing provider with a different call. When a deterministic
        # grounded fallback WAS possible (e.g. internal_rag with reliable
        # retrieved chunks -- see response_generator._fallback_final_answer),
        # `reply` is non-empty and this branch is never reached at all; the
        # grounded text is returned as a normal successful answer below.
        fallback_reason = "final_answer_provider_error" if provider_error_category else "empty_final_answer"
        _log_terminal(agent_success=False, goal=plan.goal, selected_pipeline=selected_pipeline, fallback_reason=fallback_reason, metrics=metrics)
        return AgentResult(
            goal=plan.goal, confidence=plan.confidence, tools_requested=requested_all, tools_executed=executed_all,
            resolved_concepts=resolved_concepts, evidence_reliable=evidence.get("reliable"), evidence_coverage=evidence.get("coverage"),
            evidence_supported_concepts=evidence.get("supported_concepts") or [], evidence_reason=evidence.get("reason"),
            evidence_confidence=evidence.get("confidence"), recommendation=recommendation, recommendation_reason=recommendation_reason,
            analytics_request=context.get("analytics_request"),
            fallback_reason=fallback_reason, **_metrics_kwargs(metrics),
        )

    concepts_out = evidence.get("supported_concepts") or resolved_concepts
    analytics_operation, analytics_items = _analytics_reference_from_observations(observations)
    _log_terminal(agent_success=True, goal=plan.goal, selected_pipeline=selected_pipeline, fallback_reason=agent_fallback_reason, metrics=metrics)
    return AgentResult(
        agent_used=True, reply=reply, goal=plan.goal, confidence=plan.confidence,
        tools_requested=requested_all, tools_executed=executed_all, resolved_concepts=resolved_concepts,
        selected_pipeline=selected_pipeline, sources=sources, concepts=concepts_out,
        evidence_reliable=evidence.get("reliable"), evidence_coverage=evidence.get("coverage"),
        evidence_supported_concepts=evidence.get("supported_concepts") or [], evidence_reason=evidence.get("reason"),
        evidence_confidence=evidence.get("confidence"), recommendation=recommendation, recommendation_reason=recommendation_reason,
        analytics_operation=analytics_operation, analytics_items=analytics_items,
        analytics_request=context.get("analytics_request"), reasoning_basis=plan.answer_basis,
        reasoning_summary="simple_plan_finalized", fallback_reason=agent_fallback_reason,
        tutor_state=(context.get("tutor_state") or {}).get("state"),
        tutor_bootstrap=context.get("tutor_bootstrap"),
        **_metrics_kwargs(metrics),
    )


def _finalize_clarification(
    context: dict[str, Any],
    plan: SimplePlan,
    observations: list[dict[str, Any]],
    requested_all: list[str],
    executed_all: list[str],
    gap_status,
    metrics: dict[str, Any],
) -> AgentResult:
    """Reached only after the bounded replan budget is used and a real gap
    (unresolved entity, missing goal-required evidence, or a tool failure)
    still remains -- ask one concise clarification rather than hallucinate.
    Spends the one allowed response call to phrase it (same
    `response_generator` clarification path the iterative agent uses), so
    the worst case for a turn that needed a replan is 2 planner + 1 response
    call, still within the hard per-turn budgets."""
    brain_like = _with_missing_from_gaps(_plan_to_brain_output(plan), gap_status)
    token_usage: dict[str, Any] = {}
    reply = response_generator.generate_final_answer(
        context, brain_like, observations, "clarification",
        evidence_validator.empty_evidence("clarification_requested"), sources=[], token_usage=token_usage,
    )
    if token_usage.get("llm_called"):
        metrics["response_call_count"] += 1
        _accumulate(metrics, "response", token_usage)
    if not reply:
        # Same distinction as _finalize_answer: a provider-level failure on
        # this call must not collapse into the generic "empty_final_answer"
        # reason, or services.chat_orchestrator would fall through to legacy
        # LLM-based routing and re-hit the same failing provider.
        fallback_reason = "final_answer_provider_error" if token_usage.get("provider_error_category") else "empty_final_answer"
        _log_terminal(agent_success=False, goal=plan.goal, selected_pipeline=None, fallback_reason=fallback_reason, metrics=metrics)
        return AgentResult(
            goal=plan.goal, confidence=plan.confidence, resolved_concepts=plan.resolved_entities.concepts,
            analytics_request=context.get("analytics_request"), fallback_reason=fallback_reason, **_metrics_kwargs(metrics),
        )
    _log_terminal(agent_success=True, goal=plan.goal, selected_pipeline="clarification", fallback_reason=None, metrics=metrics)
    return AgentResult(
        agent_used=True, reply=reply, goal=plan.goal, confidence=plan.confidence,
        tools_requested=requested_all, tools_executed=executed_all,
        resolved_concepts=plan.resolved_entities.concepts, selected_pipeline="clarification",
        reasoning_basis=plan.answer_basis, tutor_state=(context.get("tutor_state") or {}).get("state"),
        **_metrics_kwargs(metrics),
    )


# ---------------------------------------------------------------------------
# Metrics + logging
# ---------------------------------------------------------------------------


def _new_metrics() -> dict[str, Any]:
    return {
        "planner_call_count": 0, "response_call_count": 0,
        "planner_prompt_tokens": 0, "planner_completion_tokens": 0,
        "response_prompt_tokens": 0, "response_completion_tokens": 0,
        # Per-phase wall-clock time this turn, in ms -- diagnostic only (see
        # _log_terminal), never used for any routing/retry/fallback decision.
        # Added to attribute a slow turn to a specific phase (planner call,
        # tool execution, or final-answer call) instead of only seeing the
        # whole-turn total in services.chat_orchestrator's turn_usage log.
        "planner_latency_ms": 0.0, "tool_call_latency_ms": 0.0, "response_latency_ms": 0.0,
        "optional_replan_used": False,
    }


def _accumulate(metrics: dict[str, Any], stage: str, usage: dict[str, Any]) -> None:
    metrics[f"{stage}_prompt_tokens"] += usage.get("prompt_tokens", 0)
    metrics[f"{stage}_completion_tokens"] += usage.get("completion_tokens", 0)
    metrics[f"{stage}_latency_ms"] += usage.get("latency_ms", 0) or 0


def _metrics_kwargs(metrics: dict[str, Any]) -> dict[str, Any]:
    total_tokens = (
        metrics["planner_prompt_tokens"] + metrics["planner_completion_tokens"]
        + metrics["response_prompt_tokens"] + metrics["response_completion_tokens"]
    )
    return {
        "agent_mode": "simple",
        "planner_call_count": metrics["planner_call_count"],
        "response_call_count": metrics["response_call_count"],
        "total_llm_call_count": metrics["planner_call_count"] + metrics["response_call_count"],
        "planner_prompt_tokens": metrics["planner_prompt_tokens"],
        "planner_completion_tokens": metrics["planner_completion_tokens"],
        "response_prompt_tokens": metrics["response_prompt_tokens"],
        "response_completion_tokens": metrics["response_completion_tokens"],
        "total_tokens": total_tokens,
        "optional_replan_used": metrics["optional_replan_used"],
    }


def _log_plan(
    call_number: int,
    semantic_plan: SemanticPlan,
    compiled_tools: list[PlannerToolCall] | None,
    error: str | None,
    usage: dict[str, Any],
    json_mode_fallback_used: bool,
) -> None:
    """Log one planner call's semantic decision plus the tools
    `agents.plan_compiler` deterministically derived from it (None on a
    provider/parse error, since nothing was compiled)."""
    if error:
        print(f"[ACRLA] simple_agent_plan_error planner_call={call_number} error={str(error)[:300]!r}")
        return
    print(
        "[ACRLA] simple_agent_plan "
        f"planner_call={call_number} goal={semantic_plan.goal} "
        f"tools={[t.name for t in (compiled_tools or [])]} "
        f"needs_clarification={semantic_plan.needs_clarification} "
        f"confidence={semantic_plan.confidence:.2f} "
        f"prompt_tokens={usage.get('prompt_tokens', 0)} completion_tokens={usage.get('completion_tokens', 0)} "
        f"latency_ms={usage.get('latency_ms', 0):.1f} "
        f"json_mode_fallback_used={json_mode_fallback_used}"
    )


def _log_tools(pass_number: int, observations: list[dict[str, Any]]) -> None:
    """Full per-tool arguments/results -- verbose-only (see agents.debug_log).
    Which tools executed at all is already part of the always-on
    `simple_agent_result` summary line."""
    for obs in observations:
        result = obs.get("result") or {}
        vprint(
            "[ACRLA] simple_agent_tool "
            f"pass={pass_number} tool={obs.get('tool')} "
            f"arguments={_redact_arguments(obs.get('arguments') or {})} "
            f"success={not obs.get('rejected') and not result.get('error')} "
            f"result_summary={_summarize_one_result(result, bool(obs.get('rejected')))!r}"
        )


def _log_terminal(*, agent_success: bool, goal: str, selected_pipeline: str | None, fallback_reason: str | None, metrics: dict[str, Any]) -> None:
    kwargs = _metrics_kwargs(metrics)
    # Per-phase latency breakdown -- diagnostic only, read from `metrics`
    # directly (not part of _metrics_kwargs/AgentResult, so this never
    # changes what chat_orchestrator or any other caller receives). Lets a
    # slow turn be attributed to the planner call(s), tool execution, or the
    # final-answer call specifically, instead of only the whole-turn total
    # already logged by chat_orchestrator's own turn_usage/agent_latency_ms.
    total_latency_ms = metrics["planner_latency_ms"] + metrics["tool_call_latency_ms"] + metrics["response_latency_ms"]
    print(
        "[ACRLA] simple_agent_result "
        f"agent_goal={goal} agent_success={agent_success} selected_pipeline={selected_pipeline} "
        f"fallback_reason={fallback_reason or 'none'} "
        f"planner_call_count={kwargs['planner_call_count']} response_call_count={kwargs['response_call_count']} "
        f"total_llm_call_count={kwargs['total_llm_call_count']} total_tokens={kwargs['total_tokens']} "
        f"optional_replan_used={kwargs['optional_replan_used']} "
        f"planner_latency_ms={metrics['planner_latency_ms']:.1f} "
        f"tool_call_latency_ms={metrics['tool_call_latency_ms']:.1f} "
        f"response_latency_ms={metrics['response_latency_ms']:.1f} "
        f"accounted_latency_ms={total_latency_ms:.1f}"
    )
