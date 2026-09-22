# ACRLA Architecture

ACRLA is an adaptive Moodle tutoring assistant. It combines a Moodle local plugin, a FastAPI backend, PostgreSQL persistence, ChromaDB retrieval, and a pluggable LLM provider layer (`services/llm_factory.py`) for generation. See [tutoring_flow.md](tutoring_flow.md) for the adaptive tutor state machine, mastery rules, and Quick Progress Check flow in detail.

## High-Level Flow

1. Moodle renders ACRLA controls on dashboard and course pages.
2. The Moodle plugin sends student, course, mastery, and material data to the FastAPI backend.
3. The backend persists profile, session, mastery, and material metadata.
4. Chat messages go through automatic response routing.
5. ACRLA searches internal course material first.
6. If reliable course context is found, ACRLA answers with internal RAG and returns PDF sources.
7. If no reliable internal context is found, ACRLA uses external fallback support and returns no PDF sources.

## Readable Demo Flow

```text
Moodle mastery button
  -> widget.js reads data-acrla-* attributes
  -> /api/v1/moodle/launch receives student/course/concept/score
  -> frontend iframe starts /api/v1/session/start
  -> backend resolves chapter/course/overall scope
  -> chat_orchestrator handles each message
  -> raw message probes ChromaDB for relevant course chunks
  -> relevance gate chooses internal_rag or external_fallback
  -> response returns selected_pipeline plus sources when internal_rag was used
  -> frontend updates the Auto badge and shows/hides PDF sources
```

This flow is intentionally backend-controlled. The UI can pass launch hints, but
the backend rebuilds and validates scope so stale browser state cannot mix
courses.

## Simple Architecture Diagram

```text
Moodle page
  |
  | local/acrla plugin renders buttons and data-acrla-* attributes
  v
widget.js persistent side panel
  |
  | /api/v1/moodle/sync, /api/v1/moodle/launch, /api/v1/session/start
  v
FastAPI router
  |
  | initializes student, course, launch level, remediation scope
  v
chat_orchestrator.py
  |
  | raw-message retrieval probe
  v
ChromaDB course collection  ---- no reliable context ---->  hybrid_pipeline.py
  | reliable context                                      external_fallback
  v
rag_pipeline.py
  |
  | internal_rag answer + retrieved sources
  v
frontend/index.html
  |
  | Auto badge + PDF source display when internal_rag
  v
Student
```

## Session Flow

1. Moodle opens ACRLA with `student_id`, `course_id`, `course_name`, `level_type`, and optional `concept`/`score`.
2. `/api/v1/moodle/launch` resolves or creates the student/course and prepares a launch redirect.
3. `frontend/index.html` reads the query parameters and calls `/api/v1/session/start`.
4. `/session/start` rebuilds the active scope from backend data:
   - chapter: one clicked concept
   - course: concepts from the clicked course material manifest
   - overall: concepts across available courses
5. Chat messages call `/api/v1/chat`.
6. The orchestrator classifies the message and either handles it deterministically or routes it through RAG/fallback.
7. Quick Progress Check uses `/api/v1/assessment/start` and `/api/v1/assessment/submit`.

## Main File Responsibilities

| File | Responsibility |
|---|---|
| `backend/routers/api.py` | FastAPI endpoints, Moodle launch/session setup, assessment start/submit, RAG diagnostics |
| `backend/services/chat_orchestrator.py` | Intent handling, scope enforcement, automatic routing, source visibility, chat memory |
| `backend/pipelines/rag_pipeline.py` | PDF ingestion, Chroma retrieval, internal RAG prompt, retrieved source labels |
| `backend/pipelines/hybrid_pipeline.py` | External fallback generation when internal material is not reliable |
| `backend/services/memory_manager.py` | Student/session/long-term memory and mastery persistence facade |
| `backend/models/schemas.py` | Stable API request/response contracts |
| `frontend/index.html` | Embedded/standalone chat UI, route badge, source display, assessment modal |
| `moodle_plugin/acrla/lib.php` | Moodle-side context and markup generation |
| `moodle_plugin/acrla/widget.js` | Moodle side panel, grade-click launches, mastery refresh |
| `moodle_plugin/acrla/styles.css` | Moodle-safe ACRLA styling |

## Moodle Plugin Frontend

The plugin under `moodle_plugin/acrla/` injects:

- floating ACRLA launcher
- persistent right-side panel
- dashboard overall/course mastery controls
- course-page inline chapter mastery controls
- material sync button
- iframe pointing to the ACRLA frontend

The Moodle plugin does not connect ACRLA directly to Moodle's database. Moodle sends context to the backend through API endpoints.

## FastAPI Backend

The backend exposes endpoints for:

- session start and chat
- Moodle profile/mastery sync
- Moodle launch context
- Moodle PDF/material sync
- RAG collection diagnostics and rebuild
- Quick Progress Check assessment start/submit
- mastery retrieval for Moodle UI refresh

The router layer initializes launch scope. The chat orchestrator then handles each message within that scope.

## LLM and Retrieval Stack

- `services/llm_factory.py` selects the active LLM provider from
  `LLM_PROVIDER` (`gemini` by default; `groq`, `ollama`, `openai`, and
  `cerebras` are also supported) so generation is not tied to one vendor.
- Ollama `nomic-embed-text` provides embeddings for ChromaDB ingestion/retrieval.
- ChromaDB stores course-specific vector collections.
- PostgreSQL stores students, courses, sessions, mastery, long-term memory, and assessment records.

## Conversation Agent & Routing Layer

The conversation agent (`backend/agents/`) is the **primary** path for
handling a chat turn; `chat_orchestrator.handle_message` is a thin wrapper:
deterministic safety guards, then the agent, and only when the agent cannot
safely handle the turn does it fall back to `legacy_handle_message` (the
original rule-based pipeline, preserved and marked
`# Legacy Conversation Fallback`, retired branch-by-branch only after the
agent proves equivalent for that case).

Two interchangeable agent architectures share the same `AgentResult`
contract, selected via `ACRLA_AGENT_MODE` (default `simple`):

- **`simple`** (`agents/simple_agent.py` + `agents/simple_planner.py`,
  default): one semantic planner call returns a complete turn plan (goal,
  entities, every tool needed, in dependency order), every tool runs
  deterministically, and one final-answer call (or a deterministic reply,
  when a tool's own output is already the answer) closes the turn. A second,
  bounded planner call only fires when something genuinely still needs
  resolving (an unresolved entity, a real tool failure, missing required
  evidence) -- never for a bookkeeping mismatch on top of an already-usable
  result.
- **`iterative`** (`agents/conversation_agent.py` + `agents/agent_brain.py`):
  the original agent brain -> tool -> observe -> re-plan loop (up to 5
  steps/turn), kept as a fallback while `simple` is validated in production.

Neither architecture decides "internal vs external" up front -- that falls
out of which tool actually ran and whether its evidence validated.
`search_course_material` embeds its own evidence verdict
(`reliable`/`coverage`/`reason`) in its result; the coordinator reads that
verdict rather than re-deriving it. Reliable evidence -> `internal_rag` with
real sources. Unreliable/absent evidence -> `external_fallback` with
`sources: []`. `answer_with_external_knowledge` never claims Moodle/PDF
grounding and always returns `sources: []`.

Before any answer is honored, a deterministic gate
(`entity_validator.validate_answer_readiness`) re-checks that every entity
the turn is about is backed by real tool evidence, that the evidence matches
what the goal needs, and that no tool call silently failed -- replanning or
asking for clarification instead of producing a partial or hallucinated
answer. `search_course_material` always uses the chapter/course/overall
scope `chat_orchestrator` already resolved for the turn -- it never accepts
a scope or course-id override from agent-brain/tool arguments, so a
prompt-injected or hallucinated tool call cannot read outside the authorized
scope.

Deterministic tools live in `backend/tools/`, split by domain: `rag_tools.py`
(course material search), `analytics_tools.py` (mastery query
planning/execution), `mastery_tools.py`, `course_tools.py`, `memory_tools.py`
(structured-turn storage/lookup), `profile_tools.py`, `tutoring_tools.py`,
`external_tools.py` (controlled general-knowledge fallback),
`dialogue_tools.py` (provenance/methodology/recommendation/preference/
focus-switch tools), `tutor_state_tools.py` (adaptive tutor state
read/write), `assessment_tools.py` (Quick Progress Check), and
`text_utils.py` (shared normalization helpers).

Recent structured turns (`ConversationTurn`: goal, resolved entities,
observations, selected pipeline, sources, evidence, recommendation + reason)
are the primary context fed to the agent every step -- not raw message text.
This is what lets follow-ups like "why these?" or "what source did you use?"
be answered from real recorded state instead of re-derived phrase matching.

## Privacy Boundary

`backend/services/privacy_context.py` is the single, centralized boundary
between local student/course context and any external LLM call:

- **Student identity minimization** (allow-list, not block-list):
  `build_llm_safe_student_context` only ever returns course name,
  concepts, remediation level/scope, difficulty, and tutoring-strategy
  phrasing. Student name, email, Moodle id, database id, and session id are
  never read from the context dict in the first place -- the safe default
  is "not sent," never "sent unless blocked."
- **Institutional content sensitivity**: every course-document chunk
  carries a `PUBLIC` / `INTERNAL` / `RESTRICTED` sensitivity level (default
  `INTERNAL` for unclassified legacy content).
  `filter_course_chunks_for_external` drops `RESTRICTED` chunks entirely and
  size-caps `PUBLIC`/`INTERNAL` content before it can reach an external LLM
  prompt. This gateway is applied at every retrieval call that can feed an
  external prompt, including dynamic Quick Progress Check question
  generation.
- Every external-content decision is logged as a privacy-safe audit line
  (`log_external_content_decision`) -- chunk counts, sensitivity levels, and
  character counts only, never chunk text, source paths, or student
  identifiers.

This module governs outbound LLM prompt construction only; it is not an
authentication, retention, or transport-encryption control.

## Development & Extension Rules

These invariants keep the routing/agent/mastery architecture consistent as
the codebase grows:

- Do not hardcode Moodle course IDs or names; concepts come from Moodle
  sync/material metadata or persisted mastery.
- Course-level scope must use only the clicked course. Overall-level scope
  is the only scope allowed to aggregate multiple courses.
- Analytics, preference, profile, and provenance questions bypass RAG and
  return `sources: []`. Tutoring questions search course material first;
  fallback never exposes PDF sources.
- The agent layer may plan/tool-call/generate, but deterministic tools must
  not update mastery. Only Quick Progress Check submission
  (`tools/assessment_tools.py`'s `run_quick_progress_check`, and
  `POST /assessment/submit`) may call `MemoryManager.set_mastery`/
  `update_mastery`.
- Weak analytics means `< 50%`; Strong analytics means `>= 80%` -- not a
  generic low-to-high ranking of every row.
- Hard safety checks (mastery-modification guard, name-capture/recall, scope
  enforcement) stay deterministic regex/lookup code, run before the agent --
  never delegated to LLM judgment.
- `classify_intent`, `analyze_user_turn`, and the rule-based routing chain
  must never run on the primary path -- they belong only inside
  `legacy_handle_message`, reached only when the agent fails. Fix the
  agent's own reasoning or the deterministic gates around it
  (`entity_validator.py`) instead of reintroducing phrase/keyword routing.
- Both agent architectures (`simple`, `iterative`) must return the same
  `AgentResult` contract -- extend `AgentResult` and populate new fields on
  both sides.
- A raised LLM-call exception (rate limit, auth, timeout, connection, 5xx)
  must be classified and logged via `agents/llm_errors.py`, never collapsed
  into a generic parse/schema-error message.
- Tutor-state transitions must never be detected via hardcoded
  sentence/keyword patterns -- the semantic planner classifies
  `tutor_signal` by meaning, in the same JSON call as `goal`.
- Chat/practice never updates mastery, including inside the adaptive tutor
  state machine.
- New reusable dialogue behavior (canned explanations, provenance lookups,
  guard responses) belongs in `agents/policies.py` (+ a matching tool in
  `tools/dialogue_tools.py`), not as another exact-phrase handler in
  `chat_orchestrator.py`.

## Automatic Response Routing

ACRLA no longer requires students to select Internal or External mode.

For every tutoring message:

1. The raw current user message is used as the retrieval probe.
2. ChromaDB returns scored course chunks.
3. A relevance gate checks distance-style scores and lexical/course overlap.
4. If reliable context exists, the selected pipeline is `internal_rag`.
5. Otherwise, the selected pipeline is `external_fallback`.

Raw-message probing is important because contextual rewrites can accidentally make unrelated questions look related to a previous topic.

## Remediation Levels

ACRLA supports three launch levels:

- `chapter`: locked to the clicked concept/chapter.
- `course`: restricted to the clicked Moodle course.
- `overall`: allowed to aggregate across enrolled courses.

Course-level scope must use only concepts from the clicked course's synced material manifest. Overall is the only level allowed to combine courses.

## Independent Mastery Architecture

ACRLA tracks independent current mastery values for:

- chapter level: `chapter:{course_id}:{concept}`
- course level: `course:{course_id}`
- overall level: `overall:{student_id}`

Initial Moodle mastery is the baseline. Current ACRLA mastery starts at that baseline and may increase after Quick Progress Check submission.

Chat messages, explanations, and ordinary practice questions do not update mastery.

## Quick Progress Check

The Quick Progress Check is the only MVP flow that updates mastery.

Formula:

```text
calculated_mastery = 0.7 * current_mastery + 0.3 * assessment_score
updated_mastery = max(current_mastery, calculated_mastery, initial_moodle_mastery)
```

Mastery never decreases in the MVP.

## Source Visibility

Sources are shown only when internal RAG is selected and retrieved chunks are used.

Analytics, preference, profile, and external fallback responses return no PDF sources.
