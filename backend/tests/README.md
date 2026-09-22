# ACRLA backend regression tests

Permanent home for the in-process regression tests written during the RQ1
(adaptivity), RQ2 (privacy), and RQ3 (accuracy) research work. These are **technical
verification** scripts — they prove specific code behaves as claimed. They
are not the RQ2/RQ3 research experiments themselves (see `evaluation/` at
the project root for those).

## Convention

Each file is a **standalone script**, not a `pytest`-collected test module:
hand-rolled `check(name, condition, detail)` assertions accumulate into a
`results` list, and the script ends with a summary line (`N/M passed`) and a
plain `assert` that fails the process (non-zero exit code) if anything
didn't pass. This project has no live LLM-provider credentials wired into
CI, so every test mocks the LLM **constructor** (`get_llm`/`get_json_llm`,
patched on the *calling* module, e.g. `agents.simple_planner.get_json_llm`
— the name each module actually imported, not `services.llm_factory`'s
own) with a real LangChain `RunnableLambda`, so prompt composition
(`prompt_template | llm`) runs exactly as it does in production and the
test inspects the literal outbound prompt content, never a real network
call.

Each file resolves its own project root (`Path(__file__).resolve().parent.parent`)
before importing anything, so it works regardless of the current working
directory.

## Files

| File | What it verifies |
|---|---|
| `test_privacy_context.py` | RQ2 student-identity minimization — `services.privacy_context.build_llm_safe_student_context` and all 5 active (`ACRLA_AGENT_MODE=simple`) LLM call sites never leak student name/email/Moodle id/db id into an outbound prompt. |
| `test_institutional_privacy.py` | RQ2 institutional-content sensitivity policy — `services.privacy_context.filter_course_chunks_for_external`, a real ChromaDB ingest→retrieve round trip, cross-course isolation, and the deterministic RESTRICTED-content fallback reply. |
| `test_compact_value_depth_fix.py` | RQ3 RAG-grounding fix — `agents.agent_json._compact_value` no longer replaces leaf scalars (chunk text/concept/source) with a truncation placeholder purely for their nesting depth, while deep containers are still bounded; includes a full `generate_final_answer` integration check. |
| `test_qpc_question_validity.py` | RQ3 QPC generated-question validity gate — `services.question_validity` and its wiring into `routers.api:_stored_or_generated_variants`: valid/malformed/wrong-answer/ambiguous/wrong-concept/ungrounded generated questions are each accepted or rejected correctly; predefined bank questions are unaffected; a rejected batch falls through to the existing deterministic fallback template and is never cached/served; a validated batch is what gets persisted and remains scoreable; the retrieved course-material text used to prompt the LLM is confirmed to be the same text the validator grounds against. |
| `test_guided_practice_evaluation.py` | RQ3 guided-practice answer evaluation grounding — `services.error_analyzer.judge_answer`'s new `course_evidence` parameter and its wiring into `tools.tutor_state_tools.evaluate_practice_answer_tool`: EASY stays deterministic (no retrieval attempted); MODERATE/HARD correct/wrong/alternative-wording answers are judged grounded in real retrieved evidence (real ChromaDB round trip); insufficient evidence produces hedged feedback instead of a forced verdict; feedback text always matches the decision; existing adaptive-policy transitions, support-request routing, and the mastery invariant (chat/practice never calls `set_mastery`/`update_mastery`) are all confirmed unchanged; a real RESTRICTED-classified chunk is confirmed to never reach the judge's prompt (RQ2 gateway reused, not bypassed). |
| `test_qpc_dynamic_generation_robustness.py` | RQ3 dynamic-QPC token-budget fix — `routers.api._llm_dynamic_variants`'s `max_tokens` (900→3500, diagnosed via a live `finish_reason: MAX_TOKENS` truncation) and the new `_extract_llm_text` helper: a truncated/unparsable response safely returns no variants; list-shaped and plain-string `response.content` are both handled; a complete valid response (either shape) still parses into exactly 5 variants. |
| `test_qpc_dynamic_privacy_gateway.py` | RQ2 institutional-privacy gateway fix for dynamic QPC generation — `routers.api._dynamic_material_context`/`_stored_or_generated_variants`: PUBLIC and INTERNAL evidence reach the captured outbound Gemini prompt (INTERNAL size-capped); RESTRICTED sentinel text never does; no student identifier is ever a parameter of either function or present in the prompt; the full unfiltered local evidence still reaches the RQ3 validity gate and local fallback template (the dual-context flow); an all-RESTRICTED concept makes zero LLM calls and falls back safely; rejected/invalid candidates still cannot affect mastery; the hand-authored bank path is structurally untouched. Real ChromaDB round trip, no live Gemini call. |
| `test_rq1_adaptivity_fixes.py` | RQ1 adaptivity fixes (read-only audit → 3 approved, minimal fixes): (1) `services.chat_orchestrator._build_agent_context`'s mastery-band strategy instructions (`services.intent_classifier.get_strategy_instruction`) now actually reach the final Gemini prompt — verified both in isolation and via a real end-to-end `handle_message` turn with a capturing LLM, confirming no extra LLM call was introduced and no student identity leaks as a side effect; (2) `routers.api._weakest_course_for_student`'s `NameError` (undefined `student`) is fixed — verified via a real `GET /moodle/launch?level_type=overall` call that the weakest *authorized* course and the weakest concept inside it are genuinely selected, an excluded/test course is never chosen, and course-level launch is unaffected; (3) `consecutive_wrong` is now reset at genuine step-back-to-EXAMPLE boundaries (both the `AdaptivePolicy` repeated-wrong path and `agents.plan_compiler`'s two confusion-streak paths) — verified that a single wrong answer immediately after remediation no longer triggers an immediate second step-back, while normal first-wrong behavior is unchanged. Also includes the broader RQ1 regression scenarios not already covered by the guided-practice/QPC suites above. |
| `test_rq1_lowest_mastery_concept_fix.py` | RQ1 final edge-case fix — `tools.mastery_tools.select_lowest_mastery_concept_tool` returned `selected_concept=None` when called with no explicit concept arguments, no `current_concept`, and no `last_reference` (the real "help me study" / nothing-named case) because its shared helper `_concepts_from_arguments` never falls back to `context["available_concepts"]`. Fixed entirely locally (that one tool falls back to the already-authorized `available_concepts` scope itself; `_concepts_from_arguments` and every other caller of it are unchanged). Verifies: empty-arguments selection picks the lowest-mastery concept within the authorized scope; a lower-mastery concept from an unauthorized/out-of-scope course is never selected; explicit `concepts`/`last_reference`/`current_concept` fallbacks behave exactly as before; a single- or zero-concept scope degrades gracefully (no invented concept); no mastery is written; no LLM call is introduced. |
| `test_rq2_legacy_fallback_privacy.py` | RQ2 legacy-fallback privacy fix (found by the final RQ2 read-only audit, `evaluation/rq2_privacy/final_privacy_audit.md` §4). The rule-based `chat_orchestrator.legacy_handle_message` fallback — reachable in `simple` mode when the primary agent returns a non-provider failure on a tutoring turn — built a Gemini prompt via `pipelines.rag_pipeline.generate_rag_response` / `pipelines.hybrid_pipeline.generate_hybrid_response` that carried the student's real Moodle full name + overall mastery percentage (also embedded in `strategy.reason`/`prompt_block`) and retrieved course context **without** `for_external=True`, so `services.privacy_context.filter_course_chunks_for_external` never ran (RESTRICTED not excluded, INTERNAL not size-capped). Minimal fix (reuses the existing gateway, no new privacy logic): both legacy generation functions now retrieve with `for_external=True` (+ audit + `log_external_content_decision`); `legacy_handle_message` no longer puts `username`/`mastery_level` on `student_context` and rebuilds the strategy block from instructions only; `rag_pipeline.SYSTEM_PROMPT` drops the `{student_name}`/`{mastery_level}` fields and the "address by name" rule. Verifies (temp ChromaDB + FakeEmbeddings + capturing `RunnableLambda`, zero live Gemini): real name/email/Moodle id/DB id absent from the Gemini-bound prompt; no student mastery percentage; RESTRICTED course content blocked; INTERNAL content size-minimized; PUBLIC/INTERNAL authorized content still supports a response; fallback still functional; exactly one legacy generation LLM call (no new Gemini call); plus a forced end-to-end `handle_message` → legacy path run and structural guards that the primary agent's identity allow-list and the shared gateway were not touched. |
| `test_tutor_continuation_classification_fix.py` | Routing/orchestration fix (found by a read-only forensic diagnosis of a live Moodle Course Remediation session): a first-time `needs_support` turn during an active `GUIDED_PRACTICE` question compiles to exactly one tool, `advance_tutor_state` (`agents.plan_compiler._compile_tutor_state`'s consecutive-confusion-`<2` branch — by design, no fresh `search_course_material` call, since the concept's material was already retrieved earlier in the same session). Because `advance_tutor_state` belongs to none of `agents.simple_agent`'s three named tool-category sets, `_finalize_answer`'s classification cascade fell through to its generic "nothing tool-related happened" bucket, mislabeling a live, internal, course-grounded tutoring continuation as `external_fallback` / `no_tool_executed_external` (`sources=[]`, general-knowledge-only system-prompt rule) even though real internal material for the active concept exists and was already used this session. Minimal fix: a new, dedicated `elif "advance_tutor_state" in executed_all and context.get("tutor_needs_support"):` branch in `agents/simple_agent.py:_finalize_answer` classifies this exact turn shape as a new, honestly-named `tutor_continuation` pipeline, with its own grounding rule added to `agents/response_generator.py`'s `FINAL_PROMPT` system message. No change to the tutor state machine, to `advance_tutor_state`'s own semantics, or to retrieval behavior. Verifies (real end-to-end `handle_message` turn, capturing `RunnableLambda`, zero live Gemini): the exact bug scenario now classifies `tutor_continuation` (never `external_fallback`/`no_tool_executed_external`), `sources`/`evidence_reliable` stay empty/`None` (no fabricated evidence), exactly one planner call and one response call (no extra LLM call introduced), the final-answer prompt carries both the new grounding rule and the pending question; a control case (a genuine off-syllabus request with truly zero tools executed) still correctly classifies as `external_fallback` / `no_tool_executed_external`, proving the fix didn't blur that case. |
| `test_remediation_bootstrap_message_priority_fix.py` | Routing/orchestration fix (found by a read-only forensic diagnosis of a Moodle Course Remediation launch): `agents.simple_agent._try_remediation_bootstrap_fast_path` fired unconditionally on a session's genuinely first turn whenever a launch concept was resolvable, building a hardcoded `SemanticPlan(goal="concept_explanation", ...)` **without ever reading the actual message text** (`services.remediation_bootstrap.get_bootstrap_target`'s own docstring said so — "purely from structured launch/session state ... never message wording"). This silently discarded and misanswered ANY genuine first message — an explicit external-resource request ("recommend a YouTube video about sorting algorithms"), a tutoring request ("give me an example"), or anything else — as if it had been "explain `<bootstrap's own selected concept>`" instead, with the real semantic planner never running at all (`planner_call_count=0`). Minimal, deliberately content-agnostic fix (no keyword list): `services/remediation_bootstrap.py:get_bootstrap_target` now returns `None` immediately whenever `context["message"]` is non-empty (stripped) — deferring to the real planner for ANY genuine message, regardless of content. Bootstrap now only auto-fires when the message is truly empty/whitespace-only, matching the module's own stated purpose of teaching "before the student has to type anything" (`models/schemas.py:ChatRequest.message: str` already accepts an empty string with no `min_length`, so this distinction is drawn from the existing request/context structure, not invented). Separately, `agents/simple_planner.py`'s `SIMPLE_PLANNER_PROMPT` system message gained one explicit rule: a request naming an external resource type (YouTube/video, website, online tutorial, external link/resource) is `external_knowledge` even when it also names an in-course concept. No change to the tutor state machine, to `advance_tutor_state`'s own semantics, or to `answer_with_external_knowledge`'s own separately-flagged double-generation behavior (left untouched for a later forensic investigation). Verifies (real end-to-end `handle_message` turns, capturing `RunnableLambda`, zero live Gemini): a truly empty first message still takes the bootstrap fast path (planner skipped, hardcoded EXPLAIN turn); a genuine external-resource first message, a genuine tutoring-phrased first message, and an ordinary "explain X" first message all now correctly reach real planner classification instead of being silently overridden; the external-resource turn classifies `external_knowledge` (never the old hardcoded `concept_explanation`) without ever calling `advance_tutor_state`; the active remediation concept/scope (`current_topic`) is preserved unchanged through that turn; a follow-up "explain" turn right after the external-resource turn still starts real tutoring normally; and the planner prompt's new external-resource rule text is present in the static template. |

## Running

From the `backend/` directory (matches how the app itself is normally run):

```bash
cd backend
export PYTHONPATH=.          # PowerShell: $env:PYTHONPATH = "."
python tests/test_privacy_context.py
python tests/test_institutional_privacy.py
python tests/test_compact_value_depth_fix.py
python tests/test_qpc_question_validity.py
python tests/test_guided_practice_evaluation.py
python tests/test_qpc_dynamic_generation_robustness.py
python tests/test_qpc_dynamic_privacy_gateway.py
python tests/test_rq1_adaptivity_fixes.py
python tests/test_rq1_lowest_mastery_concept_fix.py
python tests/test_rq2_legacy_fallback_privacy.py
python tests/test_tutor_continuation_classification_fix.py
python tests/test_remediation_bootstrap_message_priority_fix.py
```

Or from the repository root — each file resolves its own path, so this also
works:

```bash
python backend/tests/test_privacy_context.py
```

Exit code `0` means every check in that file passed; a non-zero exit means
at least one `check(...)` failed (see the printed `FAIL:` lines) or an
exception was raised.

## What these tests are *not*

They do not measure educational accuracy, pedagogical quality, or run the
~20-case RQ3 expert-rated evaluation. They confirm specific, narrow claims
about code behavior (e.g. "this sentinel string never appears in this
prompt"), which is a precondition for trusting any later research
experiment built on this code — not a substitute for that experiment. See
`evaluation/README.md` for the distinction between **technical
verification** and **research experiment** results.
