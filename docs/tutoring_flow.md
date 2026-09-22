# ACRLA Tutoring Flow

This document covers the adaptive tutor state machine, mastery/analytics
rules, and the Quick Progress Check assessment flow in detail. For the
high-level request routing and agent architecture, see
[architecture.md](ARCHITECTURE.md).

## Difficulty vs. Mastery

ACRLA separates question difficulty from mastery-based support. These two
axes are independent and must not override each other.

Student-selected difficulty controls question type:

- `easy`: multiple choice
- `medium`: open-ended explanation
- `hard`: code-writing, algorithm design, or deeper reasoning

Mastery controls tutoring strategy (phrasing/support level, not question type):

- `simplified_remediation`: simple language, reminders, hints, step-by-step support
- `guided_practice`: normal explanation, examples, and feedback
- `advanced_challenge`: deeper reasoning, edge cases, stretch prompts

Low mastery does not downgrade a Hard question. High mastery does not
override a student-selected Easy question.

## Adaptive Tutor State Machine

Personalized tutoring (`goal=concept_explanation` or `personalized_tutoring`)
follows a structured six-state pedagogical loop instead of a fresh free-form
reply every turn:

- `EXPLAIN` -- a short (3-4 sentence) concept explanation from RAG.
- `EXAMPLE` -- one concrete worked example.
- `GUIDED_PRACTICE` -- one practice question at the current difficulty.
- `FEEDBACK` -- evaluates the student's answer (transient -- computed the
  same turn as `RETRY_OR_NEXT`, never persisted on its own).
- `RETRY_OR_NEXT` -- the Adaptive Policy's decision (also transient):
  correct and confident escalates difficulty and continues; correct but
  weak keeps the same difficulty; wrong retries with a hint; two or more
  consecutive wrong answers on the same concept drop back to `EXAMPLE` at a
  lower difficulty.
- `PROGRESS_CHECK_READY` -- after enough practice rounds, suggests a Quick
  Progress Check.

Only the four "resting" states (`EXPLAIN`, `EXAMPLE`, `GUIDED_PRACTICE`,
`PROGRESS_CHECK_READY`) are ever persisted between turns
(`backend/services/tutor_state_machine.py`).

State/transition detection never uses hardcoded sentence patterns. The
semantic planner (`agents/simple_planner.py`) classifies a `tutor_signal`
(`continue` | `practice_answer` | `new_topic`) by meaning, riding along in
the same JSON call that already classifies `goal` -- not a second LLM call,
not a keyword match. The deterministic `agents/plan_compiler.py` then maps
`(tutor_state, tutor_signal)` to concrete tools -- `advance_tutor_state`,
`generate_practice_question`, `evaluate_practice_answer`
(`tools/tutor_state_tools.py`) -- the same goal-keyed, message-wording-blind
pattern every other goal already uses.

The Error Analyzer (`services/error_analyzer.py`) judges each practice
answer with one LLM call (`correct`, `confident`, and -- if wrong -- an
`error_type`: `conceptual_misunderstanding`, `logic_error`,
`missing_base_case`, or `algorithm_misuse`) and records per-concept error
patterns so a later `EXPLAIN` can address a student's specific recurring
difficulty.

Tutor state and error patterns are stored per student/course inside the
existing `StudentLongTermMemory.learning_state` JSON
(`services/tutor_state_machine.py`), reusing the same storage mechanism as
course memory instead of adding new database columns -- no migration
required.

Chat/practice turns in this loop never update mastery -- only Quick
Progress Check does (see Mastery Updates below); `advance_tutor_state`,
`generate_practice_question`, `evaluate_practice_answer`, and
`services/error_analyzer.py` never call
`MemoryManager.set_mastery`/`update_mastery`.

## Proactive Remediation Bootstrap

A student opening ACRLA through a chapter, course, or overall remediation
launch does not have to type "explain recursion"/"help me learn"/"where
should I start?" before tutoring begins -- the tutor proactively selects a
concept and starts `EXPLAIN` on the session's first turn.

- `services/remediation_bootstrap.py` decides, purely from structured
  session state -- never message wording -- whether THIS turn should
  bootstrap: no tutor state or Quick Progress Check already active, an
  empty structured-turn history (genuinely the session's first turn -- a
  specific, purposeful first message like "check my progress" must still
  reach its own real classification instead of being silently overridden),
  and a one-shot session-scoped marker
  (`learning_state.courses[course_id]["remediation_bootstrap"]`) not yet
  set for this `session_id`. A brand-new Moodle launch always creates a new
  session, so a genuinely new remediation session still bootstraps
  normally.
- Concept selection is fully deterministic (no LLM call): a **chapter**
  launch uses the concept already known from launch context directly,
  never a weakest-concept search. A **course** launch picks the
  lowest-mastery concept within the current course; an **overall** launch
  picks the lowest-mastery concept across every canonical synced course the
  student is authorized to see -- both reuse `tools/analytics_tools.py`'s
  `execute_analytics_query` (the same manifest-backed, stale-duplicate-safe
  canonical course data `run_analytics_query` already trusts), so the
  winning concept's real `course_db_id`/`course_id`/`course_name` is never
  invented.
- `agents/simple_agent.py`'s `_try_remediation_bootstrap_fast_path` (checked
  right after the Quick Progress Check fast path, before the planner call)
  hands the selected concept to the existing tutor-state fresh-start
  compilation (`agents/plan_compiler.py`'s `_compile_tutor_state`) via a
  hand-built `SemanticPlan` -- the exact same `search_course_material`/
  `advance_tutor_state` tool sequence an ordinary "explain \<concept\>" turn
  already produces, so EXPLAIN phrasing, RAG grounding, and evidence
  validation are all unchanged. Net cost: 0 planner calls, 1
  response-generator call (only if the deterministic short-circuit doesn't
  already apply).
- The response generator adds one short, dynamic phrasing hint for a
  bootstrapped EXPLAIN turn (see `agents/response_generator.py`'s
  `_bootstrap_intro`, fed by `context["tutor_bootstrap"]`): briefly say why
  this concept was picked ("your lowest-mastery concept in this course, at
  55%") before teaching -- except for a chapter launch, which never calls
  its concept "weakest" since the student clicked it directly.
- After bootstrap, the student remains free to type anything -- a new
  topic, an analytics question, an external question, or Quick Progress
  Check -- since the one-shot marker (and the tutor state the bootstrap
  itself just started) means the fast path never fires again for this
  session; every subsequent turn goes through the ordinary planner-driven
  flow untouched.
- Logged as `[ACRLA] proactive_remediation_bootstrap level=... course_id=...
  concept=... mastery=... reason=... planner_skipped=true` for
  debugging/quota tracking.

## Mastery Updates

Mastery changes only after a Quick Progress Check assessment.

Chatting, explanations, and ordinary practice questions do not directly
increase mastery.

ACRLA keeps updated mastery independent at each remediation level:

- Chapter current ACRLA mastery key: `chapter:{course_id}:{concept}`
- Course current ACRLA mastery key: `course:{course_id}`
- Overall current ACRLA mastery key: `overall:{student_id}`

Initial Moodle mastery may be derived across levels:

- Chapter initial mastery comes from the chapter/concept grade.
- Course initial mastery can be the average of chapter Moodle grades.
- Overall initial mastery can be the average of course Moodle grades.

After ACRLA assessments, the levels do not recalculate from each other:

- Chapter assessments update only the clicked chapter's current ACRLA mastery.
- Course assessments update only the clicked course's current ACRLA mastery.
- Overall assessments update only the student's overall current ACRLA mastery.
- Course assessments do not update chapter mastery.
- Overall assessments do not update course or chapter mastery.
- Chapter assessments do not update course or overall mastery.

MVP update formula:

```text
calculated_mastery = 0.7 * current_mastery + 0.3 * assessment_score
updated_mastery = max(current_mastery, calculated_mastery, initial_moodle_mastery)
```

Mastery does not decrease in the MVP. If the assessment score is low,
mastery stays the same.

ACRLA stores: initial Moodle mastery, current ACRLA mastery, previous
mastery, assessment score, calculated mastery, updated mastery, mastery
delta, and assessment scope metadata.

## Mastery Analytics

Analytics answers are based on canonical synced courses:

1. The backend discovers valid synced Moodle courses.
2. It reads each course's material manifest to get the course-local concept list.
3. It deduplicates stale course rows by normalized course name plus normalized concept set.
4. It keeps the manifest-backed course row when duplicates exist.
5. It reads mastery values only for concepts that belong to that canonical course.

This prevents stale records from producing invalid combinations such as
"Recursion" under "Data Science".

Analytics thresholds use the same mastery bands as the Moodle UI:

- Weak: `0-49%`
- Moderate: `50-79%`
- Strong: `80-100%`

A "weak concepts" analytics request returns only concepts below `50%`; it
does not merely sort all concepts from low to high.

The analytics agent prefers the deterministic `run_analytics_query` tool for
broad performance questions. That tool stores the planned operation
(`list`/`rank`/`compare`/`recommend`), the scope (current course or all
courses), and the exact filtered mastery rows used to answer. If the final
LLM response synthesis is empty, fails, or the provider is rate-limited,
ACRLA formats the already-gathered analytics rows deterministically instead
of falling back to a legacy clarification question.

### Analytics Follow-up Scope Refinement

A follow-up like "in this course" (after an overall/all-courses mastery
answer) must narrow the next query to just that course, not repeat the
previous overall figure -- narrowing scope is a distinct refinement from
switching entity (e.g. concepts vs. courses vs. overall), and both must be
re-evaluated on every follow-up:

- The semantic planner (`agents/simple_planner.py`) and the iterative agent
  brain (`agents/agent_brain.py`) both instruct the model to switch `scope`
  (not just `entity`) to match what THIS message asks for: "in this
  course"/"in this chapter" narrows to `current_course`/`current_chapter`;
  "overall"/"across all my courses" widens to `all_courses`.
- `tools/analytics_tools.py`'s `execute_analytics_query` filters `courses`
  down to the current course BEFORE computing an `entity=overall` average
  when `scope=current_course`, so the number itself is genuinely that one
  course's average, not the same across-everything figure relabeled.
- `format_analytics_result` names the course in the reply when the result is
  scoped to a single course, instead of always saying "Overall mastery
  average".

## Assessment System (Quick Progress Check)

Quick Progress Check generates 3 scoped questions with metadata:

```json
{
  "question_id": "...",
  "variant_id": "...",
  "sub_concept": "...",
  "concepts_used": ["..."],
  "course_ids_used": [2],
  "question_type": "single_concept"
}
```

Question variety:

- Used question IDs/texts are stored in `assessment_records`.
- Generated variants are stored in `assessment_question_variants`.
- Recent variants are avoided for the same student/scope.
- Hardcoded CS/Math-style questions are fallback examples only.
- New Moodle course concepts can generate variants from synced PDF chunks
  using the LLM/RAG path.

### Chat-Triggered Quick Progress Check

"Check my progress" (typed in chat, or a UI trigger sending "start quick
progress check") launches the Quick Progress Check through the conversation
agent instead of reporting a stored analytics value. This is distinct from
the modal-based flow above (`POST /assessment/start` / `/assessment/submit`,
driven by the `#progress-btn` button in `frontend/index.html`) -- the same
underlying goal, a second, conversational entry point.

- The semantic planner classifies "check my progress" / "start quick
  progress check" / "take a quiz" / "assess me" / "progress check" as
  `goal=assessment` -- launching a check, never `analytics_query` (which
  only reports an already-stored value).
- `tools/assessment_tools.py`'s `run_quick_progress_check` tool selects 3-5
  concepts practiced this session (from tutor state, the active concept,
  and recent structured turns -- padded from weak/available concepts only
  if fewer than 3 were actually practiced) and asks one question per
  concept at the student's current difficulty, from the same deterministic
  question bank the adaptive tutor state machine uses -- never an LLM call.
- Answer judging is deterministic (a plain letter comparison, no LLM call)
  for easy/MCQ questions; only open-ended medium/hard questions fall back to
  `services/error_analyzer.py`'s LLM judgment.
- Once a check is started, every following message is treated as that
  turn's answer regardless of what goal the planner assigns it --
  `agents/plan_compiler.py`'s `compile_plan` checks for an active, pending
  assessment question before looking at `goal` at all, and
  `agents/simple_agent.py` skips the planner LLM call entirely for these
  turns (`_try_quick_progress_check_fast_path`).
- After the last question, mastery is updated with the same non-decreasing
  MVP formula `POST /assessment/submit` uses, per concept, via
  `MemoryManager.set_mastery`. This is the only chat-tool path allowed to
  write mastery -- ordinary tutoring/practice tools never do.
- The completion summary is a deterministic, score-tiered template (no LLM
  call): a 0% score gets an encouraging, actionable message, a 50-79% score
  gets "Good effort!", and 80-100% gets "Excellent!" with the full
  per-concept `+X%` breakdown (`tools/assessment_tools._score_tier_reply`).
- The Quick Progress Check state and the tutor state machine's state are
  reset once the check completes, so the next tutoring turn starts fresh at
  `EXPLAIN`.
- State lives in the same `StudentLongTermMemory.learning_state` JSON as
  tutor state (`MemoryManager.get_quick_progress_check`/
  `set_quick_progress_check`), session-scoped the same way.
