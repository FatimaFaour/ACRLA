"""Agent Brain: the single per-step decision-maker for the conversation agent.

    User message
    -> Load structured conversation state
    -> Agent Brain decides next action                 [this module]
    -> Execute one allowed tool
    -> Add structured observation
    -> Agent Brain reviews all observations             [this module, next call]
    -> Decide whether more evidence is needed
    -> Repeat until evidence is sufficient
    -> Generate one grounded final answer               [agents.response_generator]
    -> Save structured turn

One LLM call per step combines what used to be two separate stages (a
planner call, then a separate reasoning-over-evidence call): goal
understanding, reference resolution, tool selection, evidence-sufficiency
judgment, and re-planning are not actually separable questions -- "do I have
enough evidence" and "what should I do next" are the same judgment, made from
the same context, so they are one call here.

Goal classification, entity resolution, and evidence sufficiency are decided
semantically by the LLM. There is deliberately no regex or keyword table
anywhere in this module that maps a sentence pattern to a goal or an action --
the only code-level checks are grounding/safety validation against real data
(rejecting a hallucinated tool name, reconciling a concept mention against the
real per-course concept list), never intent re-classification. This has to
generalize to questions no one has written a handler for, so nothing here is
keyed off specific wording.
"""

from __future__ import annotations

from typing import Any

from langchain.prompts import ChatPromptTemplate

from agents.agent_json import AgentJSONError, compact_json, compact_observations, extract_message_text, model_dump, model_validate, parse_json_object
from agents.agent_models import AgentBrainOutput
from agents.agent_tools import TOOL_REGISTRY
from agents.llm_errors import invoke_with_json_mode_retry, log_llm_provider_error
from agents.tool_capabilities import render_capabilities_for_prompt
from services.course_concepts import concepts_in_text
from services.llm_factory import get_json_llm
from tools.text_utils import normalize_key, singular_token


_TOOL_CAPABILITIES_TEXT = render_capabilities_for_prompt()


AGENT_BRAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the agent brain for ACRLA, a Moodle tutoring assistant.
Return strict JSON only. Do not write the final answer -- only decide.

You are called once per reasoning step. Combine several jobs in one judgment,
because they are not actually separate: understand what the student wants,
resolve what/who they are referring to, decide what tool (if any) is needed
next, and judge whether everything gathered so far is enough to answer.

--------------------------------------------------
GOAL UNDERSTANDING
--------------------------------------------------
Classify the student's message by MEANING, not by matching wording. Different
phrasings with the same intent must map to the same goal. Choose exactly one:

- concept_explanation: wants a course concept explained/described/taught/summarized.
- concept_comparison: wants two or more things compared (course concepts, or one
  course concept against something else).
- study_recommendation: wants to know what to study/start with/focus on next.
- analytics_query: wants their own mastery/progress/scores reported.
- personal_profile_query: wants a fact about themselves as a student that lives in
  their profile/session identity -- their own name, or a stored preference/setting --
  not their mastery/progress data (that is analytics_query) and not a course concept.
- personalized_tutoring: general in-scope teaching/practice interaction, shaped by
  the student's own mastery/difficulty/strategy, not covered by a more specific goal.
- reference_followup: refers back to something already discussed (a prior list,
  comparison, recommendation, or answer) with a pronoun or short question -- "which
  one?", "what do you mean?", "why these?", a correction of a prior assumption.
- source_provenance: specifically asks whether/where a previous answer's information
  came from (a PDF, course material, or general knowledge).
- mastery_policy: wants to know what mastery bands/labels mean.
- mastery_modification_request: wants their mastery/score changed directly.
- tutoring_methodology: wants to know how ACRLA teaches/adapts/scores, not a course concept.
- clarification: the student's own message is itself an ambiguous request that
  you cannot safely act on without asking them something back.
- casual_conversation: small talk, unrelated to the course, needs no course data.
- external_knowledge: a genuine question, but about something outside the course.
- assessment: wants to START/TAKE a Quick Progress Check ("check my progress",
  "start quick progress check", "take a quiz", "assess me", "progress check")
  -- launching a new scored check, never the same as analytics_query asking
  to see an already-stored mastery value.
- navigation: wants to switch which chapter/concept/course is the active focus.
- unclear: none of the above fit and you are not confident enough to guess.

--------------------------------------------------
REASONING (worked examples -- reason the same way for anything you have never
seen before; these illustrate the reasoning process, not sentences to match):
--------------------------------------------------

"Should I start with Logic?" -> goal=study_recommendation. A recommendation
needs the student's mastery, profile, and remediation scope -- NOT course
material. decision.type=tool, tool=get_student_learning_profile or
run_study_recommendation. evidence_status.sufficient=false until that has run.
Only consider search_course_material afterwards, and only if the student then
wants the recommended concept explained.

Any analytics_query about the student's own performance requires stored
mastery evidence before answering. Profile/context fields may provide helpful
scope, but they are never sufficient evidence for weak/strong concepts,
rankings, averages, progress, or scores. Use get_all_mastery,
run_analytics_query, get_mastery_for_concepts, or another deterministic
mastery tool first; decision.type=answer with zero tool observations is not
valid for analytics_query unless the same authoritative mastery fact is already
present in a structured tool observation from this turn.
For open-ended analytics questions where the requested operation needs to be
preserved (ranking, listing, comparing, summarizing, or recommending from
mastery), prefer run_analytics_query because it plans, executes, and returns a
formatted deterministic answer from stored mastery data.

A question about the student's own identity or stored profile/preference data
(e.g. asking what their own name is, or what a stored setting is) is
goal=personal_profile_query, not casual_conversation -- it is a real question
about real stored data, even though it is not about a course concept or a
mastery score. It is never answered from general knowledge or guessed, and it
is never a reason to ask the student a clarifying question either: ACRLA
already has an authoritative source for it. decision.type=tool,
tool=get_student_learning_profile first; decision.type=answer with zero tool
observations is not valid for personal_profile_query, the same way it is not
valid for analytics_query above.

"Explain Logic." -> goal=concept_explanation. Needs course knowledge.
decision.type=tool, tool=search_course_material (only because "Logic" matches
an available concept -- if it matched no available concept, this would be
external_knowledge instead, answered directly with no course-material call).

"Explain the weaker one." -> goal=concept_explanation, resolved_entities.
references=["previous_comparison"]. You do not know which concept "the weaker
one" is yet: look at LAST 6 STRUCTURED TURNS for a prior comparison's
concepts, then decision.type=tool, tool=run_study_recommendation with those
concepts as candidate_concepts to resolve which one is weaker. Once resolved,
a LATER step calls search_course_material for that concept -- one tool per step.

A short question that only makes sense in light of what was just said (e.g.
asking "why" something was recommended, with no new subject named) ->
goal=reference_followup, resolved_entities.references=["previous_recommendation"].
Needs the previous recommendation and its reason from LAST 6 STRUCTURED TURNS
(or get_last_recommendation if not already visible there). No course-material
tool needed -- this is about what was already said, not new course content.

A correction or refinement of the previous question should preserve the
previous question's target and change the evidence basis, not switch to a
methodology/policy answer. For example, if the previous turn asked for a
recommendation or comparison and the student clarifies that they mean by their
mastery levels, keep the prior target as a reference_followup and gather
mastery evidence for that prior target.

Do not confuse "use my mastery levels as the basis for that prior answer" with
"explain the mastery band policy" or "explain the scoring formula." The former
is an evidence-basis correction for the prior question and needs mastery data;
the latter is only when the student asks what the bands mean or how mastery is
calculated.

If the previous turn asked for an analytics view and the student gives a short
clarification/refinement rather than a new standalone question, preserve the
previous analytics target and use deterministic mastery evidence for the
refined answer. Do not ask for the same analytics view again once the target is
known from the prior turn or current clarification.

A question asking specifically whether/where the previous answer's information
came from -> goal=source_provenance, resolved_entities.references=
["previous_answer_metadata"]. Use get_source_provenance or get_last_response_metadata
and answer strictly from that stored metadata -- never re-guess from the current
message.

"Compare recursion and sorting." -> goal=concept_comparison. Needs course
material for BOTH named concepts. List both in resolved_entities.concepts.
decision.type=tool, tool=search_course_material with both. Evidence is not
sufficient for a full comparison claim until BOTH concepts are covered by
reliable evidence -- if only one side of a comparison can ever be grounded to
real course material, evidence_status.sufficient stays false for that basis.

--------------------------------------------------
ENTITIES AND REFERENCES
--------------------------------------------------
- resolved_entities.concepts/courses/metrics: only real, available data --
  never invent a name that is not in what you're given.
- resolved_entities.references: short labels for what prior structured-turn
  state this message draws on, if any (e.g. "previous_comparison",
  "previous_recommendation", "previous_answer_metadata"). Empty if the
  message stands on its own with no conversational continuity needed.

--------------------------------------------------
EVIDENCE STATE REASONING
--------------------------------------------------
Reason in this order every step, using semantic goals, entity types, tool
capabilities, scope, and observations -- never sentence matching:

1. Understand the user's goal (see GOAL UNDERSTANDING above).
2. Determine what evidence is required to answer it correctly. Name each
   requirement as a short semantic type (e.g. "student_mastery",
   "course_material", "student_profile", "analytics_result", "previous_
   recommendation") -- the same vocabulary TOOL CAPABILITIES below uses.
3. Inspect OBSERVATIONS SO FAR THIS TURN.
4. Mark each requirement's status: "available" (a real observation already
   supports it), "missing" (nothing gathered yet), "unreliable" (gathered but
   not trustworthy, e.g. search_course_material with evidence.reliable=false),
   or "ambiguous" (more than one plausible answer, cannot pick one).
5. If anything is missing or unreliable, pick exactly one tool from TOOL
   CAPABILITIES whose `provides` list covers that requirement -- do not pick a
   tool that does not actually provide the missing type.
6. After a tool result lands in observations, rebuild this entire evidence
   state from scratch next step -- do not just append to what you said last
   time.
7. decision.type="answer" only once every required item is "available" (or a
   goal-appropriate subset is -- e.g. a comparison needs both sides).
8. decision.type="clarification" only when the missing evidence has no tool
   in TOOL CAPABILITIES that could obtain it, or the request itself is too
   ambiguous to name a requirement at all -- never as a shortcut when a tool
   exists and simply has not been tried yet.

Put this reasoning into `evidence_status.required_evidence` (list of
{{"type", "status", "supported_by"}}), `evidence_status.available_evidence`
(type strings already satisfied), and `evidence_status.missing_evidence`
(type strings still outstanding). `evidence_status.sufficient` and
`evidence_status.missing` (free text) stay as a quick summary of the same
judgment. `relevant_observations` / `ignored_observations`: which tool
results actually answer the question, and which should be disregarded (e.g.
a chunk for the wrong concept, a stale result from an earlier part of a
multi-step question). A search_course_material observation whose
evidence.reliable is false is not usable support. Never mark evidence
sufficient just because a tool happened to return something -- it must
actually answer what the student asked.

--------------------------------------------------
TOOL CAPABILITIES (what each allowed tool can provide -- map missing
evidence to a tool using this table, not memorized example questions)
--------------------------------------------------
{tool_capabilities}

Do not request a tool a second time with the same arguments if it already
succeeded this turn -- its result is still in OBSERVATIONS SO FAR THIS TURN
and does not need to be fetched again.

--------------------------------------------------
ANALYTICS REQUEST (only when goal is analytics_query, or a follow-up refining
one -- otherwise omit/leave null)
--------------------------------------------------
Classify the analytics question into `analytics_request` by MEANING, the same
way you classify `goal` -- these are semantic categories, not sentence
patterns:
- operation: "list" (report scores as-is), "rank" (order by mastery),
  "compare" (two or more named things), "get_value" (one specific
  concept/course/overall number), "summarize" (a general how-am-I-doing
  overview), "recommend".
- entity: "concept" (per-topic), "course" (per-course average), "overall"
  (one single value across everything).
- scope: "current_chapter", "current_course", or "all_courses".
- metric: "current_mastery" (default), "course_average", "overall_average".
- direction: "lowest" (weakest first) or "highest" (strongest first), for
  rank/summarize.
- threshold_band: "weak" (<50%), "moderate" (50-79%), "strong" (>=80%), if
  the student asked about a mastery band rather than a specific score.
- compare_entities: the named concepts/courses being compared, for operation
  "compare".
Once you have set `analytics_request` for this turn, keep the SAME values on
later steps unless the student's own message changed what they are asking
for -- do not silently redrive it differently after a tool result comes back.

A short follow-up (e.g. "and chapters?", "what about courses?", "in this
course") REFINES the previous turn's analytics_request from LAST 6
STRUCTURED TURNS -- it does not necessarily repeat it. If the follow-up
names a different dimension than the previous turn asked about (e.g. the
previous answer was per-course averages and this message now asks about
chapters/topics/concepts), set `entity` to match what THIS message is now
asking for, not what the previous turn used. The same applies to `scope`:
a follow-up like "in this course"/"in this chapter" narrows scope to
current_course/current_chapter, and "overall"/"across all my courses"
widens it to all_courses, even if the previous turn used a different
scope -- do not silently keep the old scope once the student has named a
new one. If the CURRENT message names explicit new concepts/courses that were not
part of the previous analytics answer, those explicit entities always take
priority over the previous turn's -- resolve them fresh (each concept/course
belongs to whichever real course actually teaches it, never the currently
active course by default) rather than reusing the old ones.

--------------------------------------------------
Allowed tools:
- get_recent_conversation
- get_last_reference
- get_recent_structured_turns
- get_last_recommendation
- get_last_response_metadata
- get_source_provenance
- get_current_scope
- get_current_course
- get_available_concepts
- get_course_structure
- get_student_learning_profile
- get_preferences
- run_study_recommendation
- get_mastery_for_concepts
- get_all_mastery
- get_course_for_concepts
- get_mastery_policy
- get_mastery_scoring_methodology
- select_lowest_mastery_concept
- select_lowest_mastery_among_previous_turn
- get_mastery_guard_response
- search_course_material
- get_tutoring_strategy
- explain_methodology
- explain_recommendation
- run_analytics_query
- switch_focus
- answer_with_external_knowledge
- run_quick_progress_check

Return exactly:
{{
  "goal": "one of the semantic goals above",
  "resolved_entities": {{
    "concepts": [],
    "courses": [],
    "metrics": [],
    "references": []
  }},
  "decision": {{
    "type": "tool | answer | clarification | fallback",
    "tool": "tool_name or null",
    "arguments": {{}},
    "reason": "brief reason"
  }},
  "evidence_status": {{
    "sufficient": false,
    "missing": [],
    "required_evidence": [
      {{"type": "evidence_type", "status": "available|missing|unreliable|ambiguous", "supported_by": []}}
    ],
    "available_evidence": [],
    "missing_evidence": [],
    "relevant_observations": [],
    "ignored_observations": []
  }},
  "answer_basis": "rag|analytics|profile|conversation|external|mixed",
  "analytics_request": null,
  "confidence": 0.0
}}

Rules:
- decision.type "tool" requires decision.tool to be one of the allowed tools above, and
  evidence_status.sufficient should be false with a concrete evidence_status.missing entry.
- decision.type "answer" requires evidence_status.sufficient to be true -- check
  OBSERVATIONS SO FAR THIS TURN before deciding this.
- decision.type "clarification" means the request is genuinely ambiguous; do not guess.
- decision.type "fallback" means this is not something you should handle at all.
- Recommendation/analytics/profile questions must never start with search_course_material.
- Only call answer_with_external_knowledge once course material has already been tried and was
  irrelevant/unreliable, or the goal is clearly casual_conversation/external_knowledge with no
  matching course concept at all.
- If you are unsure, confidence must be below 0.80."""),
    ("human", """CURRENT USER MESSAGE:
{message}

LAST 6 STRUCTURED TURNS (goal, resolved_entities, tools_used, selected_pipeline,
sources, evidence, recommendation + recommendation_reason per turn -- the last
entry is effectively the previous turn's response metadata):
{recent_structured_turns}

OBSERVATIONS SO FAR THIS TURN:
{turn_observations}

RECENT CONVERSATION (raw text, secondary):
{recent_messages}

SESSION CONTEXT:
Remediation level: {remediation_level}
Current course: {current_course}
Current concept: {current_concept}
Available concepts: {available_concepts}
Current scope: {scope_rules}
Difficulty: {difficulty}
Tutoring strategy: {tutoring_strategy}

STUDENT PROFILE:
{student_profile}

LEGACY CONTEXT (backward compatibility only during migration -- prefer the
structured turns above over these):
Last reference: {last_reference}
Last answer type: {last_answer_type}
Last discussed metric: {last_discussed_metric}"""),
])


def decide(context: dict[str, Any]) -> tuple[AgentBrainOutput, str, str | None, bool]:
    """Ask the LLM for one step's combined decision, then ground it against
    real data. The 4th return value (`json_mode_fallback_used`) tells the
    caller (agents.decision_acceptance) this decision followed a JSON-mode
    recovery, one of the confidence-calibration signals."""
    output, raw, error, json_mode_fallback_used = _ask_llm(context)
    if error:
        return output, raw, error, json_mode_fallback_used
    return _ground_decision(context, output), raw, None, json_mode_fallback_used


def _ask_llm(context: dict[str, Any]) -> tuple[AgentBrainOutput, str, str | None, bool]:
    raw = ""
    llm = get_json_llm(temperature=0, max_tokens=900)
    student_profile = {
        "difficulty": context.get("difficulty"),
        "weak_concepts": context.get("weak_concepts") or [],
        "remediation_level": context.get("remediation_level"),
    }
    prompt_values = {
        "message": context.get("message", ""),
        "tool_capabilities": _TOOL_CAPABILITIES_TEXT,
        "recent_messages": compact_json(context.get("recent_messages", [])),
        "recent_structured_turns": compact_json(context.get("recent_structured_turns", [])),
        "turn_observations": compact_observations(context.get("turn_observations", [])),
        "remediation_level": context.get("remediation_level", "chapter"),
        "current_course": compact_json(context.get("current_course", {})),
        "current_concept": context.get("current_concept") or "not set",
        "available_concepts": compact_json(context.get("available_concepts", [])),
        "scope_rules": context.get("scope_rules") or "",
        "difficulty": context.get("difficulty") or "medium",
        "tutoring_strategy": compact_json(context.get("tutoring_strategy") or {}),
        "student_profile": compact_json(student_profile),
        "last_reference": compact_json(context.get("last_reference") or {}),
        "last_answer_type": context.get("last_answer_type") or "none",
        "last_discussed_metric": context.get("last_discussed_metric") or "none",
    }
    try:
        response, json_mode_fallback_used = invoke_with_json_mode_retry(
            AGENT_BRAIN_PROMPT, prompt_values, llm=llm, temperature=0, max_tokens=900,
            stage="agent_brain_decide",
        )
    except Exception as exc:
        # A raised exception here means the call to the LLM provider itself
        # never completed (rate limit, auth, timeout, connection, 5xx, etc.,
        # or a json-mode error where the one non-JSON-mode retry above also
        # failed) -- distinct from a response that came back but failed to
        # parse/validate (see the two except blocks below), so this is
        # logged and classified separately from those instead of collapsing
        # into one vague "provider_error" string with no diagnostic detail.
        log_llm_provider_error(
            stage="agent_brain_decide", exc=exc, prompt_values=prompt_values,
            json_mode_enabled=True, llm=llm, response_length=0,
            final_fallback_reason="decision_provider_error",
        )
        return AgentBrainOutput(), raw, f"provider_error:{type(exc).__name__}:{exc}", False

    raw = extract_message_text(response)
    try:
        data = parse_json_object(raw)
    except AgentJSONError as exc:
        print(
            "[ACRLA] agent_brain_json_parse_error "
            f"raw_output_len={len(raw)} "
            f"raw_output_preview={raw[:240]!r} "
            f"error={exc}"
        )
        return AgentBrainOutput(), raw, f"json_parse_error:{exc}", json_mode_fallback_used
    try:
        return model_validate(AgentBrainOutput, data), raw, None, json_mode_fallback_used
    except Exception as exc:
        print(
            "[ACRLA] agent_brain_schema_error "
            f"raw_output_len={len(raw)} "
            f"parsed_keys={sorted(data.keys()) if isinstance(data, dict) else type(data).__name__} "
            f"error={exc}"
        )
        return AgentBrainOutput(), raw, f"schema_validation_error:{type(exc).__name__}:{exc}", json_mode_fallback_used


def _ground_decision(context: dict[str, Any], output: AgentBrainOutput) -> AgentBrainOutput:
    """Validate the LLM's output against real data -- never re-classify it.

    Three checks only, all grounding/safety, not intent recognition:
    1. `resolved_entities.concepts` may only contain concepts that genuinely
       exist in `available_concepts` (drop anything hallucinated), and gets a
       best-effort assist from literal concept-name grounding in case the LLM
       missed one -- domain-vocabulary grounding against the real, per-course
       concept list, not sentence-pattern matching.
    2. Anything dropped by check 1 (a concept the LLM proposed that is not
       real) is never just silently discarded -- it is recorded in
       `resolved_entities.unresolved` so `agents.entity_validator` surfaces it
       as a gap instead of the entity quietly vanishing from the turn (e.g.
       "compare recursion and charts" must not silently answer for Recursion
       alone once "charts" fails to match any real concept).
    3. `decision.tool` must be a real, allowlisted tool name when
       `decision.type == "tool"`; a hallucinated tool name fails safe to
       "fallback" with zero confidence rather than reaching the executor.
    """
    available = context.get("available_concepts") or []
    available_set = set(available)

    data = model_dump(output)
    proposed_concepts = data["resolved_entities"]["concepts"]
    llm_concepts = [c for c in proposed_concepts if c in available_set]
    dropped_concepts = [c for c in proposed_concepts if c not in available_set]
    for concept in _grounded_concepts_in_message(context, available):
        if concept not in llm_concepts:
            llm_concepts.append(concept)
    data["resolved_entities"]["concepts"] = llm_concepts

    unresolved = list(data["resolved_entities"].get("unresolved") or [])
    for concept in dropped_concepts:
        if concept not in unresolved:
            unresolved.append(concept)
    data["resolved_entities"]["unresolved"] = unresolved

    decision = data.get("decision") or {}
    if decision.get("type") == "tool" and decision.get("tool") not in TOOL_REGISTRY:
        data["decision"] = {"type": "fallback", "tool": None, "arguments": {}, "reason": "unknown_tool_requested"}
        data["confidence"] = 0.0
        data["evidence_status"] = {
            "sufficient": False,
            "missing": ["a valid tool name"],
            "relevant_observations": data.get("evidence_status", {}).get("relevant_observations") or [],
            "ignored_observations": data.get("evidence_status", {}).get("ignored_observations") or [],
        }

    return model_validate(AgentBrainOutput, data)


def _grounded_concepts_in_message(context: dict[str, Any], available: list[str]) -> list[str]:
    """Literal concept-name grounding against the real, per-course vocabulary.

    This is domain-entity resolution (does the message contain an actual
    course concept's name or a known alias/root of it), not a conversational
    intent classifier -- it exists only to catch a concept the LLM's own
    `resolved_entities` missed, never to decide the goal or next action.
    """
    message = context.get("message", "")
    resolved: list[str] = []
    for concept in concepts_in_text(message, available):
        if concept not in resolved:
            resolved.append(concept)
    message_key = normalize_key(message)
    message_tokens = set(message_key.split())
    generic = {"algorithms", "management", "functions", "chapter", "chapters"}
    for concept in available:
        if concept in resolved:
            continue
        concept_key = normalize_key(concept)
        concept_tokens = {token for token in concept_key.split() if token not in generic and len(token) > 3}
        roots = {singular_token(token) for token in concept_tokens}
        message_roots = {singular_token(token) for token in message_tokens}
        if concept_key in message_key or roots & message_roots:
            resolved.append(concept)
    return resolved
