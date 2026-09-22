"""
Structured contracts for the ACRLA conversational agent layer.

The agent is intentionally narrow: an LLM may decide which deterministic ACRLA
tools to call and may phrase the final answer, but it cannot mutate mastery or
invent course/mastery/source facts outside these schemas.
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _none_to_empty_list(value: Any) -> Any:
    """Some providers emit an explicit JSON `null` for "nothing here" instead
    of an empty list (e.g. `"compare_entities": null`) -- Pydantic v2 only
    applies `default_factory` when the key is *absent*, not when it is
    present but null, so that raises a validation error otherwise. Coerce
    None to [] before type validation; any other value passes through
    unchanged so a real (even if malformed) list still gets the normal
    type-checking error instead of being silently swallowed here.
    """
    return [] if value is None else value


AnswerStyle = Literal["concise", "explanatory", "comparison", "list", "clarification"]

EvidenceCoverage = Literal["full", "partial", "none"]

# tool: run exactly one more deterministic tool before deciding anything else.
# answer: enough has been gathered (or nothing needs gathering); generate the final answer.
# clarification: the request is ambiguous; ask the student one concise question instead of guessing.
# fallback: the agent cannot safely handle this turn; defer to the legacy orchestrator.
NextActionType = Literal["tool", "answer", "clarification", "fallback"]

# Semantic goals the agent brain classifies into by *meaning*, not by matching
# wording. A message is mapped to one of these because an LLM understood what
# the student wants, not because code recognized a phrase -- see
# agents/agent_brain.py.
SemanticGoal = Literal[
    "concept_explanation",
    "concept_comparison",
    "study_recommendation",
    "analytics_query",
    "personal_profile_query",
    "personalized_tutoring",
    "reference_followup",
    "source_provenance",
    "mastery_policy",
    "mastery_modification_request",
    "tutoring_methodology",
    "clarification",
    "casual_conversation",
    "external_knowledge",
    "assessment",
    "navigation",
    "unclear",
]

# What the final answer is actually grounded in, as judged by the agent brain
# after looking at everything gathered this turn. This is informational/
# synthesis guidance only -- it never overrides the deterministic evidence-
# reliability gate that controls whether PDF sources may be shown.
AnswerBasis = Literal["rag", "analytics", "profile", "conversation", "external", "mixed"]


class ResolvedEntities(BaseModel):
    """Entities the agent brain has identified for this turn so far.

    Populated fresh each step from the raw message plus this turn's tool
    observations -- never permanently fixed at the start of the turn.
    `references` holds short labels for what prior structured-turn state (if
    any) this message draws on, e.g. "previous_comparison" or
    "previous_recommendation" -- an empty list means the message stands on
    its own with no conversational continuity needed.

    `unresolved` is never set by the LLM itself -- it is populated
    deterministically by `agent_brain._ground_decision` for any entity
    mention the LLM proposed that failed a real-vocabulary check (e.g. a
    course concept name that does not exist). Grounding used to just silently
    drop these; now they land here instead, so
    `agents.entity_validator.validate_entity_completeness` always surfaces
    them as a gap rather than the entity quietly vanishing from the turn.
    """

    concepts: list[str] = Field(default_factory=list)
    courses: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    """The single next step the agent brain has chosen, fresh each step."""

    type: NextActionType = "fallback"
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


EvidenceItemStatus = Literal["available", "missing", "unreliable", "ambiguous"]


class EvidenceRequirement(BaseModel):
    """One piece of evidence the current goal requires, and whether it has
    actually been obtained yet this turn.

    `type` is a free-form semantic label (e.g. "student_mastery",
    "course_material", "student_profile", "analytics_result") matched against
    `agents.tool_capabilities.TOOL_CAPABILITIES`'s `provides` entries -- not a
    fixed enum, so a new evidence type needs no schema change here, the same
    way `ResolvedEntities`/`EntityGap` stay open to new categories.
    `supported_by` names which tool(s) actually supplied it (once available)
    or could supply it (while still missing), so the planner's tool choice is
    traceable to a specific requirement instead of a memorized example.
    """

    type: str
    status: EvidenceItemStatus = "missing"
    supported_by: list[str] = Field(default_factory=list)


class EvidenceStatus(BaseModel):
    """The agent brain's judgment of everything gathered so far this turn.

    Decided in the *same* step as `Decision` -- "do I have enough evidence"
    and "what should I do next" are one judgment from the same context, not
    two separate stages. `sufficient=false` is what makes `decision.type`
    "tool" rather than "answer"; it never overrides the deterministic RAG
    evidence-reliability gate downstream (see `agents.evidence_validator`),
    which still decides whether PDF sources may actually be shown.

    `required_evidence`/`available_evidence`/`missing_evidence` are the
    explicit structured evidence-state representation: what this goal needs,
    what has actually been confirmed, and what is still outstanding, by
    semantic evidence *type* rather than free text. These are additive to the
    original `sufficient`/`missing` fields (kept for backward compatibility
    with existing gates/logging) -- the LLM populates both, but only
    `required_evidence` lets `agents.decision_acceptance` reason about
    exactly which tool would resolve a specific gap.
    """

    sufficient: bool = False
    missing: list[str] = Field(default_factory=list)
    required_evidence: list[EvidenceRequirement] = Field(default_factory=list)
    available_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    # Annotated permissively (str or dict) because some providers (e.g.
    # openai/gpt-oss-20b) return a list of observation objects/dicts here
    # instead of the plain tool-name strings the prompt asks for -- a strict
    # list[str] annotation alone raised a validation error on that shape
    # (evidence_status.relevant_observations.0: "Input should be a valid
    # string"). The field_validator below coerces every entry to a string
    # before this annotation is even checked, so in practice the rest of the
    # pipeline (agents.conversation_agent._filter_relevant, which matches
    # these against real tool names) always receives list[str] regardless of
    # which shape the LLM actually returned; the permissive annotation is
    # just a safety net if that coercion ever misses a shape.
    relevant_observations: list[str | dict[str, Any]] = Field(default_factory=list)
    ignored_observations: list[str | dict[str, Any]] = Field(default_factory=list)

    @field_validator("missing", "required_evidence", "available_evidence", "missing_evidence", mode="before")
    @classmethod
    def _coerce_none_lists(cls, value: Any) -> Any:
        return _none_to_empty_list(value)

    @field_validator("relevant_observations", "ignored_observations", mode="before")
    @classmethod
    def _coerce_observation_entries_to_strings(cls, value: Any) -> Any:
        """Normalize each entry to a string regardless of what the LLM sent.

        A dict entry becomes its own "tool" value when present (the common
        case: the LLM echoed back an observation object instead of just its
        tool name), otherwise a compact JSON string so no information is
        silently dropped. Runs before type validation, so downstream code
        never has to handle anything but plain strings.
        """
        if not isinstance(value, list):
            return value
        coerced: list[str] = []
        for item in value:
            if isinstance(item, str):
                coerced.append(item)
            elif isinstance(item, dict):
                tool_name = item.get("tool")
                if isinstance(tool_name, str) and tool_name:
                    coerced.append(tool_name)
                else:
                    try:
                        coerced.append(json.dumps(item, default=str))
                    except (TypeError, ValueError):
                        coerced.append(str(item))
            else:
                coerced.append(str(item))
        return coerced


class AnalyticsRequest(BaseModel):
    """Structured representation of one analytics question.

    Created once (agents.conversation_agent, the first step where
    goal == "analytics_query" and no request exists yet in context),
    persisted across the rest of this turn's agent-loop iterations, passed
    directly to tools.analytics_tools.run_analytics_query_tool, and carried
    into structured conversation memory (ConversationTurn) so a follow-up
    ("what about strongest instead?") can refine it instead of re-inferring
    the whole request from scratch. Every field is a semantic category, not a
    sentence pattern -- the LLM classifies meaning into these fields the same
    way it classifies `goal`.
    """

    operation: str = "list"  # list | rank | compare | get_value | summarize | recommend
    entity: str = "concept"  # concept | course | overall
    scope: str = "current_course"  # current_chapter | current_course | all_courses
    metric: str = "current_mastery"
    direction: str | None = None  # "lowest" | "highest", for rank
    threshold_band: str | None = None  # "weak" | "moderate" | "strong"
    group_by: str | None = None
    limit: int | None = None
    compare_entities: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @field_validator("compare_entities", mode="before")
    @classmethod
    def _coerce_none_compare_entities(cls, value: Any) -> Any:
        return _none_to_empty_list(value)


class AgentBrainOutput(BaseModel):
    """One step's combined judgment from `agents.agent_brain`.

    This is deliberately re-produced every iteration of the conversation
    agent's loop: each step decides goal + entities + next action + evidence
    sufficiency together, from the message and whatever has been observed so
    far this turn. It never decides "internal vs external" once and for all,
    and it never generates the final answer -- that split is an emergent
    result of which tool(s) got called and whether their evidence validated.

    `goal` is classified by meaning (see `SemanticGoal`), not by matching
    wording -- there is deliberately no code path that forces `goal` or
    `decision` from a regex on the message.
    """

    goal: SemanticGoal = "unclear"
    resolved_entities: ResolvedEntities = Field(default_factory=ResolvedEntities)
    decision: Decision = Field(default_factory=Decision)
    evidence_status: EvidenceStatus = Field(default_factory=EvidenceStatus)
    answer_basis: AnswerBasis = "external"
    confidence: float = 0.0
    # Populated only when goal == "analytics_query" (or a follow-up refining
    # one). Created once and then persisted by agents.conversation_agent
    # (context["analytics_request"]) rather than re-derived by the brain on
    # every step.
    analytics_request: AnalyticsRequest | None = None


class PlannerToolCall(BaseModel):
    """One tool call inside a `SimplePlan.tools` list.

    `depends_on` is the *name* of another tool earlier in the same plan whose
    result/side-effect this call needs before it can run meaningfully (e.g.
    `search_course_material` depending on a selection tool that first narrows
    down which concept is weakest). It is a structural ordering hint only --
    `agents.simple_agent` topologically sorts by this field and then executes
    every tool through the same `agents.agent_tools.execute_agent_tools` the
    iterative agent already uses, so a dependent tool automatically sees
    whatever side-effect (e.g. `context["agent_selected_concept"]`) the tool
    it depends on already sets. Never a template/argument-substitution engine
    -- just execution order.
    """

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: str | None = None

    @field_validator("arguments", mode="before")
    @classmethod
    def _coerce_none_arguments(cls, value: Any) -> Any:
        return {} if value is None else value


class SemanticPlan(BaseModel):
    """Minimal semantic judgment from the lightweight planner (`agents.simple_planner`).

    Contains ONLY what genuinely requires language understanding: goal
    classification, entity resolution, and analytics-dimension
    classification. Tool selection, dependency ordering, and evidence
    requirements are NOT here -- they are deterministic and computed by
    `agents.plan_compiler.compile_plan` from this output, never asked of the
    LLM. `agents.simple_agent` compiles this into a `SimplePlan` (which still
    carries `tools`) before grounding/executing it, so every downstream
    consumer of `SimplePlan` is unaffected by this split.
    """

    goal: SemanticGoal = "unclear"
    resolved_entities: ResolvedEntities = Field(default_factory=ResolvedEntities)
    needs_clarification: bool = False
    clarification_question: str | None = None
    confidence: float = 0.0
    analytics_request: AnalyticsRequest | None = None
    # Only meaningful when a tutor-state session is active (context["tutor_state"]
    # is non-empty) -- classified by meaning in the SAME call as `goal`, never
    # keyword-matched. See agents.plan_compiler._compile_tutor_state.
    tutor_signal: Literal["continue", "practice_answer", "new_topic", "needs_support"] | None = None


class SimplePlan(BaseModel):
    """One complete turn plan: a `SemanticPlan` compiled deterministically
    into concrete tool calls by `agents.plan_compiler.compile_plan` (never
    produced directly by the LLM -- see `SemanticPlan` for what the LLM
    itself decides). Fully describes every tool the turn needs, in order --
    see `agents.simple_agent` for how it is grounded and executed.
    """

    goal: SemanticGoal = "unclear"
    resolved_entities: ResolvedEntities = Field(default_factory=ResolvedEntities)
    tools: list[PlannerToolCall] = Field(default_factory=list)
    answer_basis: AnswerBasis = "external"
    needs_clarification: bool = False
    clarification_question: str | None = None
    confidence: float = 0.0
    analytics_request: AnalyticsRequest | None = None

    @field_validator("tools", mode="before")
    @classmethod
    def _coerce_none_tools(cls, value: Any) -> Any:
        return _none_to_empty_list(value)


class ToolObservation(BaseModel):
    """One tool call's outcome, as recorded in the agent's step trace.

    This is the typed shape behind the plain dicts `agents.conversation_agent`
    appends to `observations`/`AgentResult.steps` -- used for logging (see
    Part 11) and for anything downstream that wants a structured record of
    what actually happened this turn, rather than re-parsing a raw dict.
    """

    step: int = 0
    tool: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool = False
    result: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


class AgentRunResult(BaseModel):
    """Target shape for the agent's overall run outcome.

    Defined now as the model `agents.conversation_agent` steps/observations
    are recorded against; `AgentResult` (below) remains the actual return
    contract `chat_orchestrator.py` consumes for this migration, since its
    specific field names are already threaded through many call sites there.
    Consolidating onto `AgentRunResult` as the live contract is future work,
    not done in this pass.
    """

    success: bool = False
    goal: str = "unclear"
    resolved_entities: dict[str, Any] = Field(default_factory=dict)
    observations: list[ToolObservation] = Field(default_factory=list)
    reply: str = ""
    selected_pipeline: str | None = None
    sources: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None


class EvidenceValidation(BaseModel):
    """Result of validating retrieved RAG chunks against the actual question."""

    reliable: bool = False
    coverage: EvidenceCoverage = "none"
    supported_concepts: list[str] = Field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0


class EntityGap(BaseModel):
    """One requested entity that is missing, unresolved, or ambiguous.

    `entity_type` is whatever field name it came from on `ResolvedEntities`
    (e.g. "concepts", "courses", "metrics") -- not a fixed enum -- so a new
    entity category added to `ResolvedEntities` in the future is covered by
    `agents.entity_validator` automatically, with no code change here.
    """

    entity_type: str
    value: str
    reason: str


class EntityCompletenessStatus(BaseModel):
    """Deterministic verdict: does every entity the agent brain says this turn
    is about actually have real, tool-confirmed evidence behind it?

    Produced by `agents.entity_validator.validate_entity_completeness`, called
    from `agents.conversation_agent` right before any "answer" decision is
    honored. This is independent of (and does not trust) the agent brain's own
    `evidence_status.sufficient` -- that is the LLM's self-assessment; this is
    a deterministic re-check against what tools actually returned, the same
    "trust but verify" pattern `agent_brain._ground_decision` already applies
    to tool names and concept names.
    """

    complete: bool = True
    gaps: list[EntityGap] = Field(default_factory=list)


class DecisionAcceptance(BaseModel):
    """Verdict from `agents.decision_acceptance.evaluate_decision_acceptance`:
    should THIS step's decision (tool/answer/clarification/fallback) actually
    be acted on, given raw confidence plus structural signals -- not raw
    confidence against one fixed threshold. See that module for the full
    calibration policy (tool actions may be accepted at a lower confidence
    than a final answer; a clarification is only accepted once it is shown no
    allowed tool addresses the missing evidence).
    """

    accepted: bool = False
    calibrated_confidence: float = 0.0
    acceptance_reason: str | None = None
    rejection_reason: str | None = None
    confidence_adjustments: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    """Return value consumed by chat_orchestrator."""

    agent_used: bool = False
    reply: str = ""
    goal: str = "unclear"
    knowledge_strategy: str | None = None
    confidence: float = 0.0
    tools_requested: list[str] = Field(default_factory=list)
    tools_executed: list[str] = Field(default_factory=list)
    resolved_reference: str | None = None
    resolved_concepts: list[str] = Field(default_factory=list)
    selected_pipeline: str | None = None
    sources: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    evidence_reliable: bool | None = None
    evidence_coverage: str | None = None
    evidence_supported_concepts: list[str] = Field(default_factory=list)
    evidence_reason: str | None = None
    evidence_confidence: float | None = None
    recommendation: dict[str, Any] | None = None
    recommendation_reason: str | None = None
    analytics_operation: str | None = None
    analytics_items: list[dict[str, Any]] = Field(default_factory=list)
    analytics_request: dict[str, Any] | None = None
    reasoning_basis: str | None = None
    reasoning_summary: str | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)
    fallback_reason: str | None = None
    # Per-turn LLM call/token accounting -- populated by both agents.conversation_agent
    # (iterative) and agents.simple_agent (simple) so the two can be compared directly.
    agent_mode: str = "iterative"
    planner_call_count: int = 0
    response_call_count: int = 0
    total_llm_call_count: int = 0
    planner_prompt_tokens: int = 0
    planner_completion_tokens: int = 0
    response_prompt_tokens: int = 0
    response_completion_tokens: int = 0
    total_tokens: int = 0
    optional_replan_used: bool = False
    # Adaptive tutor state machine presentation metadata (services.tutor_state_machine /
    # services.remediation_bootstrap) -- structured so the frontend can drive UI (a
    # subtle stage label, a one-time "focus card" on a proactive bootstrap turn)
    # from real backend state instead of parsing the reply text. `tutor_state` is
    # the current TutorState value (e.g. "EXPLAIN") whenever a tutor-state session
    # is active this turn, else None. `tutor_bootstrap` is populated ONLY on the
    # one turn services.remediation_bootstrap actually started a proactive
    # remediation session -- None on every other turn (including every later
    # turn of that same session), so a UI reading it can show a "today's focus"
    # card exactly once per bootstrap.
    tutor_state: str | None = None
    tutor_bootstrap: dict[str, Any] | None = None


class ConversationTurn(BaseModel):
    """One structured, storable turn of the conversation -- the PRIMARY memory.

    Recent structured turns (see `tools.memory_tools.get_recent_structured_turns`)
    are what the agent brain is given each step -- NOT raw message text and
    NOT `last_reference`/`last_answer_type`/`last_discussed_metric` (those
    remain only for backward compatibility during migration -- see
    chat_orchestrator.py's dialogue_state helpers and the `references` field
    below, a write-only legacy snapshot). This is what lets a follow-up like
    "why these?", "which one?", "what do you mean?", or "where are they
    from?" be answered from real structured state -- resolved entities, every
    tool observation, and any recommendation + its reason -- instead of being
    re-derived from free text each time.
    """

    user_message: str = ""
    assistant_reply: str = ""
    goal: str = "unclear"
    resolved_entities: ResolvedEntities = Field(default_factory=ResolvedEntities)
    references: dict[str, Any] = Field(default_factory=dict)
    tools_used: list[str] = Field(default_factory=list)
    # One entry per tool call this turn: {"tool": str, "result_summary": dict, "success": bool}.
    # Only key/shape summaries are stored, never full tool result content (e.g.
    # RAG chunk text) -- see agents.conversation_agent._summarize_one_result.
    observations: list[dict[str, Any]] = Field(default_factory=list)
    answer_basis: str | None = None
    selected_pipeline: str | None = None
    sources: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    comparison: dict[str, Any] | None = None
    recommendation: dict[str, Any] | None = None
    recommendation_reason: str | None = None
    selected_courses: list[str] = Field(default_factory=list)
    # The structured AnalyticsRequest (as a dict) this turn was answered with,
    # if any -- carried forward so a follow-up ("what about strongest
    # instead?") can refine the prior request instead of re-inferring the
    # whole thing from scratch. See agents.conversation_agent's
    # _get_or_create_analytics_request.
    analytics_request: dict[str, Any] | None = None
