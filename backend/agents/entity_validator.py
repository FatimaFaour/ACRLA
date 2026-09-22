"""Answer-readiness validator: a deterministic re-check of whether an
"answer" (or "clarification") decision is actually backed by real evidence,
before it is ever allowed to reach the student. Several independent, generic
checks -- entity completeness, per-goal evidence-type requirements, and tool
health -- none keyed to message wording. `agents.conversation_agent` calls
`validate_answer_readiness` (the combined gate); `validate_entity_completeness`
alone remains available for just the entity-level check.

Entity completeness: every entity the agent brain says this turn is about
(`resolved_entities` -- any list field, not just concepts) must actually be
backed by real evidence.

This is intentionally generic across entity *types*. It does not know what a
"concept" or a "course" is any more than what a "metric" or some future
entity category is -- it iterates whatever list fields exist on
`ResolvedEntities` (via `agent_json.model_dump`) and applies the same two
deterministic checks to each value:

1. Known-vocabulary check (only when a real vocabulary source exists in
   `context` for that field -- e.g. `available_concepts` for "concepts",
   `canonical_courses` for "courses"; a field with no known vocabulary source
   just skips this check rather than being hardcoded as unsupported).
2. Tool-evidence check: the value's normalized text must actually appear
   somewhere in a successful tool observation's result this turn (or be one
   of a narrow set of values already known from static session context, e.g.
   the current concept/course, which need no fresh tool call to justify).

`references` (conversational-continuity labels like "previous_comparison",
not literal searchable strings) get a different, equally generic check: they
just require *some* memory to resolve from (this turn's observations or
recent structured turns), not a literal text match.

A value appearing under two different entity-type fields is flagged as
ambiguous -- this is a structural check (same key resolved two ways), not a
sentence-pattern rule, so it costs nothing to extend to new entity types.
"""

from __future__ import annotations

import json
from typing import Any

from agents.agent_json import model_dump
from agents.agent_models import AgentBrainOutput, EntityCompletenessStatus, EntityGap
from tools.text_utils import normalize_key


def validate_entity_completeness(
    context: dict[str, Any],
    brain_output: AgentBrainOutput,
    observations: list[dict[str, Any]],
) -> EntityCompletenessStatus:
    """Deterministically verify every resolved entity this turn has real evidence.

    Called right before an "answer" decision is honored (see
    `agents.conversation_agent`). Independent of the agent brain's own
    `evidence_status.sufficient` -- this never trusts that self-assessment,
    it re-derives completeness from what tools actually returned.
    """
    resolved = model_dump(brain_output.resolved_entities)
    gaps: list[EntityGap] = []

    gaps.extend(_ambiguous_cross_field_gaps(resolved))

    known_vocab = _known_vocabularies(context)
    static_keys = _statically_known_keys(context)
    observation_haystacks = _flatten_observation_texts(observations)

    for field_name, values in resolved.items():
        if field_name == "references" or not isinstance(values, list):
            continue
        vocab = known_vocab.get(field_name)
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            key = normalize_key(value)
            if vocab is not None and key not in vocab:
                gaps.append(EntityGap(entity_type=field_name, value=value, reason="not_a_known_value"))
                continue
            if key in static_keys:
                continue
            if not observation_haystacks:
                gaps.append(EntityGap(entity_type=field_name, value=value, reason="no_tool_evidence_gathered_this_turn"))
                continue
            if not any(key in haystack for haystack in observation_haystacks):
                gaps.append(EntityGap(entity_type=field_name, value=value, reason="not_confirmed_by_any_tool_result"))

    references = resolved.get("references") or []
    if references and not observation_haystacks and not context.get("recent_structured_turns"):
        for reference in references:
            if isinstance(reference, str) and reference.strip():
                gaps.append(EntityGap(entity_type="references", value=reference, reason="no_supporting_memory_found"))

    # `unresolved` is populated deterministically by agent_brain._ground_decision
    # for any entity mention the LLM proposed that failed a known-vocabulary
    # check (e.g. "charts" when no such course concept exists) -- it is never
    # silently dropped, it lands here so it is always surfaced as a gap
    # instead of just vanishing from the turn.
    for value in resolved.get("unresolved") or []:
        if isinstance(value, str) and value.strip():
            gaps.append(EntityGap(entity_type="unresolved", value=value, reason="entity_mentioned_but_not_found"))

    return EntityCompletenessStatus(complete=not gaps, gaps=gaps)


def validate_answer_readiness(
    context: dict[str, Any],
    brain_output: AgentBrainOutput,
    observations: list[dict[str, Any]],
) -> EntityCompletenessStatus:
    """Full deterministic pre-answer gate: entity completeness plus more
    structural (never phrase-based) consistency checks:

    1. Does the evidence actually match the *kind* the goal itself requires?
       A goal whose own definition needs course material (concept_explanation,
       concept_comparison) is not satisfied by a mastery/analytics tool just
       because it happened to mention the concept's name -- that is a
       goal/evidence-type mismatch, not real content evidence.
    2. analytics_query and personal_profile_query each have exactly one
       deterministic data source (mastery/analytics tools; the student
       profile tool). Profile/session context lying around is never
       sufficient on its own -- a real observation from that tool must exist
       this turn.
    3. Did every tool call this turn actually succeed? A provider/tool
       failure (a truthy "error" value, or an explicit success=False) is not
       silently treated as usable evidence just because a tool ran.

    All checks are "trust but verify" against real structure (the fixed goal
    enum, the observation's own success/error shape), never against message
    wording -- the same pattern `agent_brain._ground_decision` already
    applies to tool names and concept names. Called for both "answer" and
    "clarification" decisions (see agents.conversation_agent) -- a goal with
    a real deterministic data source may not fall back to asking the student
    something ACRLA already has an authoritative tool for, either.
    """
    status = validate_entity_completeness(context, brain_output, observations)
    gaps = list(status.gaps)
    gaps.extend(_content_goal_evidence_gaps(brain_output, observations))
    gaps.extend(_analytics_goal_evidence_gaps(brain_output, observations))
    gaps.extend(_personal_profile_goal_evidence_gaps(brain_output, observations))
    gaps.extend(_tool_health_gaps(brain_output, observations))
    return EntityCompletenessStatus(complete=not gaps, gaps=gaps)


# Goals whose own definition (see the GOAL UNDERSTANDING section of
# agents/agent_brain.py's prompt) requires real course material, not mastery,
# analytics, or profile data. Derived from the fixed, closed SemanticGoal
# enum -- not from message wording -- so this is a structural mapping, not a
# phrase-specific rule, and applies identically no matter how the request was
# worded.
_CONTENT_GOALS = {"concept_explanation", "concept_comparison"}

# Analytics questions are about the student's actual stored performance. The
# profile snapshot is useful context, but it is not authoritative enough to
# answer weak/strong concepts, rankings, averages, progress, or scores. This
# closed set is tool-contract based, not wording based: any turn classified as
# `analytics_query` must gather one of these deterministic mastery/analytics
# observations before an "answer" decision is honored.
_ANALYTICS_EVIDENCE_TOOLS = {
    "get_all_mastery",
    "get_mastery_for_concepts",
    "run_analytics_query",
    "select_lowest_mastery_concept",
    "select_lowest_mastery_among_previous_turn",
}

# Exported so agents.conversation_agent can recognize this specific gap and
# deterministically force run_analytics_query next, instead of relying on
# another (unreliable) LLM decide() call to act on the gap correctly -- see
# the note at that call site.
BROAD_ANALYTICS_GAP_REASON = "broad_analytics_requires_run_analytics_query"


def _content_goal_evidence_gaps(brain_output: AgentBrainOutput, observations: list[dict[str, Any]]) -> list[EntityGap]:
    if brain_output.goal not in _CONTENT_GOALS:
        return []
    has_material_search = any(
        obs.get("tool") == "search_course_material" and not obs.get("rejected")
        and isinstance(obs.get("result"), dict) and not obs["result"].get("error")
        for obs in observations
    )
    if has_material_search:
        return []
    return [EntityGap(
        entity_type="goal_evidence",
        value=brain_output.goal,
        reason="content_goal_requires_course_material_search",
    )]


def _analytics_goal_evidence_gaps(brain_output: AgentBrainOutput, observations: list[dict[str, Any]]) -> list[EntityGap]:
    """Require authoritative mastery evidence for student analytics answers.

    This is the analytics counterpart to `_content_goal_evidence_gaps`: once
    the semantic agent brain classifies the turn as `analytics_query`, profile
    context or a bare answer with no observations is not sufficient. The turn
    must include a successful read from a deterministic mastery/analytics
    tool, or else the brain gets a synthetic gap observation and has to call
    the appropriate tool on the next step.
    """
    if brain_output.goal != "analytics_query":
        return []
    resolved = model_dump(brain_output.resolved_entities)
    resolved_concepts = [
        value for value in (resolved.get("concepts") or [])
        if isinstance(value, str) and value.strip()
    ]
    has_query_plan_observation = any(
        obs.get("tool") == "run_analytics_query"
        and not obs.get("rejected")
        and isinstance(obs.get("result"), dict)
        and not obs["result"].get("error")
        and obs["result"].get("success") is not False
        for obs in observations
    )
    # Broad analytics ("weak concepts", rankings, averages, all-course views)
    # need the analytics executor's plan, because raw get_all_mastery rows do
    # not encode which operation the student requested. Concept-specific
    # lookups can still be answered by direct concept mastery tools.
    if not resolved_concepts and not has_query_plan_observation:
        return [EntityGap(
            entity_type="goal_evidence",
            value=brain_output.goal,
            reason=BROAD_ANALYTICS_GAP_REASON,
        )]
    has_authoritative_mastery_observation = any(
        obs.get("tool") in _ANALYTICS_EVIDENCE_TOOLS
        and not obs.get("rejected")
        and isinstance(obs.get("result"), dict)
        and not obs["result"].get("error")
        and obs["result"].get("success") is not False
        for obs in observations
    )
    if has_authoritative_mastery_observation:
        return []
    return [EntityGap(
        entity_type="goal_evidence",
        value=brain_output.goal,
        reason="analytics_goal_requires_authoritative_mastery_tool",
    )]


# The only tool that can authoritatively answer a question about the
# student's own profile identity (name, stored preferences) -- see
# tools.profile_tools.get_student_learning_profile_tool.
_PERSONAL_PROFILE_EVIDENCE_TOOLS = {"get_student_learning_profile"}


def _personal_profile_goal_evidence_gaps(brain_output: AgentBrainOutput, observations: list[dict[str, Any]]) -> list[EntityGap]:
    """Require authoritative profile evidence for personal_profile_query answers.

    Same pattern as `_analytics_goal_evidence_gaps`: once the agent brain
    classifies a turn as `personal_profile_query` (its own name, a stored
    preference), a bare answer or a clarification with no tool call is not
    valid -- ACRLA has a real, deterministic source for this
    (get_student_learning_profile) and must read it, not guess or ask the
    student to repeat information ACRLA already stores.
    """
    if brain_output.goal != "personal_profile_query":
        return []
    has_profile_observation = any(
        obs.get("tool") in _PERSONAL_PROFILE_EVIDENCE_TOOLS
        and not obs.get("rejected")
        and isinstance(obs.get("result"), dict)
        and not obs["result"].get("error")
        and obs["result"].get("success") is not False
        for obs in observations
    )
    if has_profile_observation:
        return []
    return [EntityGap(
        entity_type="goal_evidence",
        value=brain_output.goal,
        reason="personal_profile_goal_requires_student_profile_tool",
    )]


# Public, reusable view of which goals structurally require at least one
# tool call (or an explicit clarification) before they can be answered, and
# which tools would satisfy that requirement -- the same goal/tool mapping
# `_content_goal_evidence_gaps`/`_analytics_goal_evidence_gaps`/
# `_personal_profile_goal_evidence_gaps` already enforce *after* a tool
# executes. Exported so `agents.simple_agent` can run the identical check
# *before* executing an empty/insufficient tool list -- a plan can be
# rejected as structurally incomplete without spending a wasted execution
# pass, and the single goal/tool mapping never has to be duplicated or
# drift between the two call sites.
EVIDENCE_REQUIRING_GOALS: dict[str, frozenset[str]] = {
    "concept_explanation": frozenset({"search_course_material"}),
    "concept_comparison": frozenset({"search_course_material"}),
    "analytics_query": frozenset(_ANALYTICS_EVIDENCE_TOOLS),
    "personal_profile_query": frozenset(_PERSONAL_PROFILE_EVIDENCE_TOOLS),
}


def _tool_health_gaps(brain_output: AgentBrainOutput, observations: list[dict[str, Any]]) -> list[EntityGap]:
    """Flag a turn where a tool call was actually attempted but every single
    one failed (provider error, tool exception, or explicit success=False) --
    schema-agnostic, so it covers any current or future tool the same way."""
    attempted = [obs for obs in observations if not obs.get("rejected") and obs.get("tool") != "entity_completeness_check"]
    if not attempted:
        return []
    if not all(_observation_failed(obs) for obs in attempted):
        return []
    failed_tools = sorted({str(obs.get("tool")) for obs in attempted})
    return [EntityGap(
        entity_type="tool",
        value=", ".join(failed_tools),
        reason="all_attempted_tool_calls_failed_this_turn",
    )]


def _observation_failed(observation: dict[str, Any]) -> bool:
    result = observation.get("result")
    if not isinstance(result, dict):
        return True
    # Truthy check, not key presence: some tools (e.g. run_analytics_query)
    # always include an "error" key, set to None when there was no error --
    # "error" in result would then treat every successful call as a failure.
    if result.get("error"):
        return True
    if result.get("success") is False:
        return True
    return False


def _ambiguous_cross_field_gaps(resolved: dict[str, Any]) -> list[EntityGap]:
    seen: dict[str, str] = {}
    gaps: list[EntityGap] = []
    for field_name, values in resolved.items():
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            key = normalize_key(value)
            if key in seen and seen[key] != field_name:
                gaps.append(EntityGap(
                    entity_type=field_name, value=value,
                    reason=f"ambiguous_also_resolved_as_{seen[key]}",
                ))
            else:
                seen.setdefault(key, field_name)
    return gaps


def _known_vocabularies(context: dict[str, Any]) -> dict[str, set[str]]:
    """Real, per-turn vocabularies to validate entity values against.

    Only defined for entity types this codebase actually has an authoritative
    source for. A field with no entry here simply skips the vocabulary check
    (falls through to the tool-evidence check only) -- adding a new entity
    type never requires touching this function to stay generic.
    """
    vocab: dict[str, set[str]] = {}
    available_concepts = context.get("available_concepts") or []
    if available_concepts:
        vocab["concepts"] = {normalize_key(c) for c in available_concepts}
    course_names = {
        normalize_key(course.get("name"))
        for course in (context.get("canonical_courses") or [])
        if course.get("name")
    }
    if course_names:
        vocab["courses"] = course_names
    return vocab


def _statically_known_keys(context: dict[str, Any]) -> set[str]:
    """Values already settled from session context, needing no fresh tool call
    this turn to justify (e.g. the concept/course already being discussed)."""
    keys: set[str] = set()
    current_concept = context.get("current_concept")
    if current_concept:
        keys.add(normalize_key(current_concept))
    current_course_name = (context.get("current_course") or {}).get("name")
    if current_course_name:
        keys.add(normalize_key(current_course_name))
    return keys


def _flatten_observation_texts(observations: list[dict[str, Any]]) -> list[str]:
    """Normalized text of every successful tool result this turn.

    Deliberately schema-agnostic: it does not know or care what shape any
    given tool's result takes, it just serializes and normalizes it so a
    substring check works the same for a concept name, a course name, or any
    future entity's value.
    """
    texts: list[str] = []
    for obs in observations:
        if obs.get("rejected"):
            continue
        result = obs.get("result")
        if not isinstance(result, dict) or result.get("error"):
            continue
        try:
            blob = json.dumps(result, default=str)
        except (TypeError, ValueError):
            blob = str(result)
        texts.append(normalize_key(blob))
    return texts


def entity_gap_observation(entity_status: EntityCompletenessStatus) -> dict[str, Any]:
    """Synthetic observation surfacing the gap to the NEXT agent-brain call.

    Appended to the same `observations`/`turn_observations` list real tool
    calls populate, so the existing "OBSERVATIONS SO FAR THIS TURN" prompt
    section (agents/agent_brain.py) shows it with no prompt-template change --
    the brain decides how to react (another tool, clarification, or reporting
    the entity was not found), this only makes the gap visible.
    """
    return {
        "tool": "entity_completeness_check",
        "arguments": {},
        "rejected": False,
        "result": {
            "complete": False,
            "gaps": [model_dump(gap) for gap in entity_status.gaps],
            "note": (
                "This turn is not ready to answer yet -- see gaps above (an unresolved or "
                "not-yet-confirmed entity, a goal that needs course material you have not "
                "retrieved, or a tool call that failed). Do not answer yet -- call another "
                "tool to resolve it, ask the student one concise clarifying question, or "
                "explain that the requested entity/information could not be found."
            ),
        },
    }


def entity_gap_descriptions(entity_status: EntityCompletenessStatus) -> list[str]:
    """Human-readable reasons for `evidence_status.missing`, used when the step
    limit is reached with an unresolved gap and there is no step left to
    replan -- see `agents.conversation_agent._handle_step_limit_exhausted`."""
    return [f"{gap.value} ({gap.entity_type}): {gap.reason}" for gap in entity_status.gaps]
