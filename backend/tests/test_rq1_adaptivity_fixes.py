"""Regression tests for the 3 RQ1 adaptivity fixes (read-only audit ->
these 3 approved, minimal fixes only):

FIX 1 -- services/chat_orchestrator.py:_build_agent_context built
`context["tutoring_strategy"]["instructions"]` via `getattr(strategy,
"prompt_block", "")`, but `strategy` (from `services.intent_classifier.
select_strategy`, the mastery-aware classifier the ACTIVE simple-agent path
actually uses) is a plain Enum with no such attribute -- always "". Fixed
by calling the existing `services.intent_classifier.get_strategy_
instruction(strategy)` instead. No new strategy system; `services/
strategy_selector.py` (the legacy-path-only, already-correct system) is
untouched.

FIX 2 -- routers/api.py:_weakest_course_for_student referenced an
undefined `student` name (its own parameter is `student_id`) -- crashed
with NameError on every call, breaking "overall launch -> weakest course".
Fixed by using the correct parameter name.

FIX 3 -- tools/tutor_state_tools.py:evaluate_practice_answer_tool and
agents/plan_compiler.py's two confusion-streak step-back branches carried
`consecutive_wrong` forward unchanged across a step-back to EXAMPLE, so a
single new wrong/confused turn after remediation immediately looked like a
second consecutive one. Fixed by resetting the streak to 0 at exactly the
same reset boundaries (`decision.reset_rounds` / a step-back to EXAMPLE),
never touching normal first-wrong-answer behavior.

Uses a real sqlite in-memory DB and the established capturing-LLM
technique (a real LangChain RunnableLambda standing in for the LLM
constructor, patched on each module's own imported name) -- no live
Gemini call anywhere in this file except where explicitly noted as
mocked.

Run from the `backend/` directory:
    python tests/test_rq1_adaptivity_fixes.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inspect
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_core.runnables import RunnableLambda
from unittest.mock import Mock

from models.db_models import Base, Student, Course, Session as SessionModel
from services.memory_manager import MemoryManager

results = []


def check(name, cond, detail=None):
    results.append((name, cond))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f" -> {detail}" if detail is not None else ""))


engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
TestSession = sessionmaker(bind=engine)


def new_db():
    return TestSession()


# ===========================================================================
# FIX 1 -- mastery-driven strategy instructions
# ===========================================================================
import services.chat_orchestrator as chat_orchestrator_module
from services.intent_classifier import Strategy, get_strategy_instruction, STRATEGY_INSTRUCTIONS

db1 = new_db()
student1 = Student(moodle_user_id=910001, username="RQ1 Fix1 Student")
course1 = Course(moodle_course_id=910101, name="RQ1 Fix1 Course")
db1.add_all([student1, course1])
db1.commit()
db1.refresh(student1)
db1.refresh(course1)
session1 = SessionModel(student_id=student1.id, course_id=course1.id, difficulty="medium")
db1.add(session1)
db1.commit()
db1.refresh(session1)
memory1 = MemoryManager(db1)


def build_context_for_strategy(strategy_value):
    return chat_orchestrator_module._build_agent_context(
        session_id="fix1-session", message="test message", session=session1, student=student1,
        memory=memory1, db=db1, remediation_level="chapter", course_moodle_id=course1.moodle_course_id,
        available_concepts={"Recursion"}, weak_concepts=["Recursion"], current_topic="Recursion",
        last_reference={}, dialogue_state={}, strategy=strategy_value,
    )


for label, strategy_value in [
    ("A1_WEAK", Strategy.WEAK), ("A2_MODERATE", Strategy.MODERATE),
    ("A3_EXCELLENT", Strategy.EXCELLENT), ("A4_WEAK_CONFUSED", Strategy.WEAK_CONFUSED),
]:
    ctx = build_context_for_strategy(strategy_value)
    got = ctx["tutoring_strategy"]["instructions"]
    expected = get_strategy_instruction(strategy_value)
    check(f"{label}. instructions reach context and match STRATEGY_INSTRUCTIONS", got == expected and got != "", got)

# A5: no student identity added as a side effect.
ctx_weak = build_context_for_strategy(Strategy.WEAK)
tutoring_strategy_json = json.dumps(ctx_weak["tutoring_strategy"])
for forbidden in ("RQ1 Fix1 Student", "910001", student1.id):
    check(f"A5. tutoring_strategy dict never contains identity marker {forbidden!r}", str(forbidden) not in tutoring_strategy_json)
from services.privacy_context import build_llm_safe_student_context, FORBIDDEN_IDENTITY_FIELDS
safe_ctx = build_llm_safe_student_context(ctx_weak)
check("A5b. build_llm_safe_student_context output carries the fixed instructions through unchanged",
      safe_ctx.get("tutoring_strategy", {}).get("instructions") == get_strategy_instruction(Strategy.WEAK))
check("A5c. the safe context never reads any FORBIDDEN_IDENTITY_FIELDS name",
      not any(field in inspect.getsource(build_llm_safe_student_context) for field in ["context[\"student_name\"]", "context[\"email\"]", "context[\"student_id\"]"]))

# A6: no extra Gemini call introduced -- _build_agent_context is a pure
# dict-builder; confirm structurally it never references get_llm/get_json_llm.
build_context_source = inspect.getsource(chat_orchestrator_module._build_agent_context)
check("A6. _build_agent_context's source never calls get_llm/get_json_llm (no LLM call introduced)",
      "get_llm(" not in build_context_source and "get_json_llm(" not in build_context_source)


# ---------------------------------------------------------------------
# B. End-to-end: the fixed instructions actually reach the FINAL prompt
# sent to the (mocked) LLM, for a real turn through the active simple-
# agent path, with no extra LLM call introduced.
# ---------------------------------------------------------------------
import agents.simple_planner as simple_planner_module
import agents.response_generator as response_generator_module
from services.chat_orchestrator import handle_message

db2 = new_db()
student2 = Student(moodle_user_id=910002, username="RQ1 Fix1B Student")
course2 = Course(moodle_course_id=910102, name="RQ1 Fix1B Course")
db2.add_all([student2, course2])
db2.commit()
db2.refresh(student2)
db2.refresh(course2)
session2 = SessionModel(student_id=student2.id, course_id=course2.id, difficulty="medium", is_active=True)
db2.add(session2)
db2.commit()
db2.refresh(session2)
memory2 = MemoryManager(db2)
# MODERATE band is naturally reachable from avg_mastery alone (0.3 <= avg < 0.65,
# recent_errors = int((1-avg)*3) doesn't force WEAK_CONFUSED/RECOVERY at this range).
memory2.set_mastery(student_id=student2.id, course_id=course2.id, concept="Recursion", mastery_level=0.5)


class _Capture:
    def __init__(self):
        self.count = 0
        self.last_prompt_text = ""


def make_capturing_llm(response_json_or_text, capture: _Capture):
    def _fn(prompt_value):
        capture.count += 1
        try:
            capture.last_prompt_text = "\n".join(m.content for m in prompt_value.to_messages())
        except Exception:
            capture.last_prompt_text = str(prompt_value)

        class _FakeResponse:
            content = response_json_or_text

        return _FakeResponse()
    return RunnableLambda(_fn)


planner_capture = _Capture()
response_capture = _Capture()
planner_json = json.dumps({
    "goal": "concept_explanation", "tutor_signal": None, "needs_clarification": False,
    "clarification_question": None, "confidence": 0.95,
    "resolved_entities": {"concepts": ["Recursion"], "references": []},
    "analytics_request": None,
})
orig_planner_llm = simple_planner_module.get_json_llm
orig_response_llm = response_generator_module.get_llm
simple_planner_module.get_json_llm = lambda *a, **kw: make_capturing_llm(planner_json, planner_capture)
response_generator_module.get_llm = lambda *a, **kw: make_capturing_llm("A plain-text explanation.", response_capture)
try:
    result = handle_message(session_id=session2.id, student_moodle_id=student2.moodle_user_id, message="Can you explain recursion?", db=db2)
finally:
    simple_planner_module.get_json_llm = orig_planner_llm
    response_generator_module.get_llm = orig_response_llm

expected_moderate_instruction = get_strategy_instruction(Strategy.MODERATE)
check("B1. the real final-answer prompt actually contains the MODERATE strategy's instruction text",
      expected_moderate_instruction in response_capture.last_prompt_text, response_capture.last_prompt_text[:500])
check("B2. exactly one planner call and one response call were made (no extra LLM call introduced)",
      planner_capture.count == 1 and response_capture.count == 1, (planner_capture.count, response_capture.count))
check("B3. the turn produced a real reply (fix didn't break the turn)", bool(result.get("reply")))


# ===========================================================================
# FIX 2 -- overall-launch weakest-course selection crash
# ===========================================================================
import routers.api as api_module
from models.schemas import MoodlePayload

db3 = new_db()
student3 = Student(moodle_user_id=910003, username="RQ1 Fix2 Student")
# _available_courses_for_student (routers/api.py:605) excludes moodle_course_id
# == 1, >= 70000, and courses named "mastery test"/"intro cs" -- these test
# course ids must stay OUTSIDE all of those exclusion rules (and below the
# 70000 cutoff, unlike this file's Fix-1/Fix-3 fixtures) so they are
# genuinely "authorized"/available candidates for _weakest_course_for_student
# to choose between; if every candidate got excluded, that function's own
# empty-fallback (`real_courses if real_courses else courses`) would return
# ALL courses unfiltered, silently defeating this exact test.
weak_course = Course(moodle_course_id=61, name="RQ1 Weak Course")
strong_course = Course(moodle_course_id=62, name="RQ1 Strong Course")
excluded_course = Course(moodle_course_id=1, name="RQ1 Excluded Course")  # moodle_course_id==1 is always excluded
db3.add_all([student3, weak_course, strong_course, excluded_course])
db3.commit()
for c in (student3, weak_course, strong_course, excluded_course):
    db3.refresh(c)
memory3 = MemoryManager(db3)
# Real, canonical concept names (services.course_concepts.COURSE_CONCEPTS) --
# needed so canonicalize_concept/_assessment_concept resolve them properly
# through the real launch flow (an arbitrary "Concept A" string would not).
memory3.set_mastery(student_id=student3.id, course_id=weak_course.id, concept="Sorting Algorithms", mastery_level=0.20)
memory3.set_mastery(student_id=student3.id, course_id=weak_course.id, concept="Binary Trees and BSTs", mastery_level=0.10)
memory3.set_mastery(student_id=student3.id, course_id=strong_course.id, concept="Pointers and Memory Management", mastery_level=0.90)
# Excluded course has the LOWEST mastery of all -- must never be selected as "weakest".
memory3.set_mastery(student_id=student3.id, course_id=excluded_course.id, concept="Recursion", mastery_level=0.01)

# C1: no longer raises NameError.
try:
    weakest = api_module._weakest_course_for_student(db3, memory3, student3.id)
    c1_ok = True
except NameError as exc:
    weakest = None
    c1_ok = False
check("C1. _weakest_course_for_student no longer raises NameError", c1_ok, weakest)

# C5 (checked here too, directly): the excluded course is never returned even
# though it has the lowest raw mastery of any course.
check("C1b. the excluded (moodle_course_id=1) course is never returned as weakest",
      weakest is not None and weakest.moodle_course_id != 1, getattr(weakest, "moodle_course_id", None))
check("C1c. the genuinely weakest AUTHORIZED course (weak_course, avg 0.15) is selected over strong_course (avg 0.90)",
      weakest is not None and weakest.id == weak_course.id, getattr(weakest, "id", None))


# C2/C3: full launch_from_moodle overall-launch flow.
import inspect as _inspect
launch_sig = _inspect.signature(api_module.launch_from_moodle)


def call_launch_from_moodle(db, **overrides):
    kwargs = {name: (p.default if p.default is not _inspect.Parameter.empty else None) for name, p in launch_sig.parameters.items()}
    kwargs.update(overrides)
    kwargs["db"] = db
    return api_module.launch_from_moodle(**kwargs)


class _FakeRequest:
    """Minimal stand-in for FastAPI's Request -- launch_from_moodle only
    ever reads `request.base_url` (routers/api.py:2688), to build the final
    redirect URL, well after all course/concept selection logic has run."""
    base_url = "http://testserver/"


try:
    launch_result = call_launch_from_moodle(
        db3, student_id=student3.moodle_user_id, level_type="overall", request=_FakeRequest(),
    )
    c2_error = None
except Exception as exc:
    launch_result = None
    c2_error = exc

check("C2. GET /moodle/launch with level_type=overall no longer crashes", c2_error is None, c2_error)

# C2b/C3: inspect what was actually selected server-side (independent of the
# redirect URL, which deliberately omits "concept" for level_type=overall --
# routers/api.py:2670 -- so we verify via the Session/topic state the launch
# itself writes, the same state a resumed chat turn would read).
from models.db_models import Session as SessionModel3
active_session3 = db3.query(SessionModel3).filter_by(student_id=student3.id, is_active=True).order_by(SessionModel3.started_at.desc()).first()
check("C2b. overall launch created/used a session in the weakest AUTHORIZED course (weak_course, avg 0.15), not strong_course or the excluded course",
      active_session3 is not None and active_session3.course_id == weak_course.id,
      getattr(active_session3, "course_id", None))
selected_topic3 = memory3.get_current_topic(active_session3.id) if active_session3 else None
check("C3. the weakest concept INSIDE that course (Binary Trees and BSTs, mastery 0.10 < Sorting Algorithms 0.20) was selected",
      selected_topic3 == "Binary Trees and BSTs", selected_topic3)

# C4: course-level launch is structurally unaffected -- it never calls the
# fixed function at all (only the "overall" branch does, routers/api.py:2413).
launch_from_moodle_source = inspect.getsource(api_module.launch_from_moodle)
check("C4. _weakest_course_for_student is only invoked for level_type=='overall' (course-level launch untouched)",
      launch_from_moodle_source.count("_weakest_course_for_student(") == 1)
try:
    launch_result_chapter = call_launch_from_moodle(
        db3, student_id=student3.moodle_user_id, level_type="chapter", course_id=strong_course.moodle_course_id,
        request=_FakeRequest(),
    )
    c4_error = None
except Exception as exc:
    c4_error = exc
check("C4b. course-level launch (explicit course_id) still works and picks exactly that course, unaffected by the fix",
      c4_error is None, c4_error)
if c4_error is None:
    active_session3b = db3.query(SessionModel3).filter_by(student_id=student3.id, is_active=True).order_by(SessionModel3.started_at.desc()).first()
    check("C4c. course-level launch selected the EXPLICITLY requested course (strong_course), not the weakest one",
          active_session3b is not None and active_session3b.course_id == strong_course.id,
          getattr(active_session3b, "course_id", None))


# ===========================================================================
# FIX 3 -- consecutive_wrong / support-streak reset at step-back boundaries
# ===========================================================================
from services.tutor_state_machine import AdaptivePolicy, save_tutor_state, load_tutor_state, TutorState
import tools.tutor_state_tools as tst
import services.error_analyzer as error_analyzer_module

db4 = new_db()
student4 = Student(moodle_user_id=910004, username="RQ1 Fix3 Student")
course4 = Course(moodle_course_id=910301, name="RQ1 Fix3 Course")
db4.add_all([student4, course4])
db4.commit()
db4.refresh(student4)
db4.refresh(course4)
memory4 = MemoryManager(db4)
CONCEPT4 = "Recursion"
session_id4 = "fix3-session"


def judgment_json(correct, confident=True, error_type=None, reason=""):
    return json.dumps({"correct": correct, "confident": confident, "error_type": error_type, "feedback_reason": reason})


def run_evaluation(answer, response_json):
    llm_capture = _Capture()
    orig = error_analyzer_module.get_json_llm
    error_analyzer_module.get_json_llm = lambda *a, **kw: make_capturing_llm(response_json, llm_capture)
    try:
        ctx = {
            "session_id": session_id4, "message": answer, "student_id": student4.id, "course_db_id": course4.id,
            "tutor_state": load_tutor_state(memory4, student4.id, course4.id, session_id4), "memory": memory4, "db": db4,
            "retrieval_course_ids": [], "canonical_courses": [], "remediation_level": "chapter",
        }
        return tst.evaluate_practice_answer_tool(ctx, {"answer": answer})
    finally:
        error_analyzer_module.get_json_llm = orig


def seed_state(state="GUIDED_PRACTICE", difficulty="medium", consecutive_wrong=0, rounds_completed=0, current_question="Q"):
    save_tutor_state(memory4, student4.id, course4.id, session_id4, state=state, concept=CONCEPT4, difficulty=difficulty,
                      rounds_completed=rounds_completed, consecutive_wrong=consecutive_wrong, current_question=current_question, asked_variants={})


# D1: first wrong answer -> normal hint/retry, streak becomes 1 (unaffected).
seed_state(consecutive_wrong=0)
result_d1 = run_evaluation("wrong answer", judgment_json(False, True, "logic_error"))
check("D1. first wrong answer -> stays GUIDED_PRACTICE (hint/retry)", result_d1["next_state"] == "GUIDED_PRACTICE", result_d1)
after_d1 = load_tutor_state(memory4, student4.id, course4.id, session_id4)
check("D1b. first wrong answer -> consecutive_wrong becomes 1 (normal increment preserved)", after_d1["consecutive_wrong"] == 1, after_d1)

# D2: second consecutive wrong -> step back to EXAMPLE, streak RESET to 0 (was the bug: stayed at 2).
seed_state(consecutive_wrong=1)
result_d2 = run_evaluation("wrong again", judgment_json(False, True, "logic_error"))
check("D2. second consecutive wrong -> steps back to EXAMPLE", result_d2["next_state"] == "EXAMPLE", result_d2)
after_d2 = load_tutor_state(memory4, student4.id, course4.id, session_id4)
check("D2b. FIX: after step-back, consecutive_wrong is RESET to 0 (not left at 2)", after_d2["consecutive_wrong"] == 0, after_d2)

# D3: the key behavioral proof -- after the reset, ONE new wrong answer at the
# next practice round must NOT immediately trigger a second step-back.
seed_state(state="GUIDED_PRACTICE", difficulty=after_d2["difficulty"], consecutive_wrong=after_d2["consecutive_wrong"], current_question="New Q at lower difficulty")
result_d3 = run_evaluation("wrong once more", judgment_json(False, True, "logic_error"))
check("D3. FIX: a single wrong answer right after remediation -> hint/retry, NOT an immediate second step-back",
      result_d3["next_state"] == "GUIDED_PRACTICE", result_d3)

# D6: confirm decision.reset_rounds itself still only fires on the genuine 2nd-consecutive-wrong boundary (unchanged policy).
decision_first = AdaptivePolicy.decide(correct=False, confident=True, difficulty="medium", consecutive_wrong=0, rounds_completed=0)
decision_second = AdaptivePolicy.decide(correct=False, confident=True, difficulty="medium", consecutive_wrong=1, rounds_completed=0)
check("D6a. AdaptivePolicy itself unchanged: 1st wrong -> reset_rounds=False", decision_first.reset_rounds is False)
check("D6b. AdaptivePolicy itself unchanged: 2nd consecutive wrong -> reset_rounds=True", decision_second.reset_rounds is True)


# ---------------------------------------------------------------------
# D4/D5: plan_compiler's two needs_support step-back branches reset the
# streak too.
# ---------------------------------------------------------------------
import agents.plan_compiler as plan_compiler_module
from agents.agent_models import SemanticPlan, ResolvedEntities


def make_semantic_plan(tutor_signal):
    return SemanticPlan(
        goal="personalized_tutoring", tutor_signal=tutor_signal, needs_clarification=False,
        clarification_question=None, confidence=0.9,
        resolved_entities=ResolvedEntities(concepts=[], references=[]), analytics_request=None,
    )


# D4: repeated needs_support WITH a pending question -> 2nd one steps back, resets streak.
ctx_d4 = {"tutor_state": {"state": "GUIDED_PRACTICE", "concept": CONCEPT4, "current_question": "Q pending", "consecutive_wrong": 1}}
tools_d4, basis_d4 = plan_compiler_module._compile_tutor_state(make_semantic_plan("needs_support"), ctx_d4, [])
advance_call_d4 = next((t for t in tools_d4 if t.name == "advance_tutor_state"), None)
check("D4. repeated needs_support (2nd) -> compiles a step-back to EXAMPLE", advance_call_d4 is not None and advance_call_d4.arguments.get("to") == "EXAMPLE", tools_d4)
check("D4b. FIX: step-back tool call explicitly resets consecutive_wrong to 0", advance_call_d4 is not None and advance_call_d4.arguments.get("consecutive_wrong") == 0, advance_call_d4.arguments if advance_call_d4 else None)

# D5: needs_support with NO pending question (between rounds) -> step-back, resets streak.
ctx_d5 = {"tutor_state": {"state": "GUIDED_PRACTICE", "concept": CONCEPT4, "current_question": None, "consecutive_wrong": 1}}
tools_d5, basis_d5 = plan_compiler_module._compile_tutor_state(make_semantic_plan("needs_support"), ctx_d5, [])
advance_call_d5 = next((t for t in tools_d5 if t.name == "advance_tutor_state"), None)
check("D5. needs_support with no pending question -> compiles a step-back to EXAMPLE", advance_call_d5 is not None and advance_call_d5.arguments.get("to") == "EXAMPLE", tools_d5)
check("D5b. FIX: this step-back also explicitly resets consecutive_wrong to 0", advance_call_d5 is not None and advance_call_d5.arguments.get("consecutive_wrong") == 0, advance_call_d5.arguments if advance_call_d5 else None)

# D-structural: confirm normal (non-step-back) needs_support handling (1st
# request, question kept pending) is UNCHANGED -- does not force a reset,
# still carries the streak forward as before (that branch never touched).
ctx_d_first = {"tutor_state": {"state": "GUIDED_PRACTICE", "concept": CONCEPT4, "current_question": "Q pending", "consecutive_wrong": 0}}
tools_d_first, _ = plan_compiler_module._compile_tutor_state(make_semantic_plan("needs_support"), ctx_d_first, [])
advance_call_first = next((t for t in tools_d_first if t.name == "advance_tutor_state"), None)
check("D7. FIRST needs_support request (not yet a streak) -> keeps SAME question, records streak=1 (unaffected by the fix)",
      advance_call_first is not None and advance_call_first.arguments.get("to") == "GUIDED_PRACTICE" and advance_call_first.arguments.get("consecutive_wrong") == 1,
      advance_call_first.arguments if advance_call_first else None)


# ===========================================================================
# POST-FIX RQ1 REGRESSION -- the remaining scenarios from the read-only
# audit's 16-item list not already covered above (Fix 1 = #16; Fix 2 =
# #14/#15; Fix 3 = #6-#10) or by the existing permanent suite (#5, #9, #11,
# #12 -- see backend/tests/test_guided_practice_evaluation.py /
# test_qpc_question_validity.py, run separately, unmodified).
# ===========================================================================

# #1: low mastery -> weak concept selected. Exercised via the SAME real,
# already-working mechanism the launch flow uses (_weakest_concept_for_course)
# -- historically NOT via tools.mastery_tools.select_lowest_mastery_concept_
# tool, which at the time this file was first written returned {"selected_
# concept": None} on empty arguments (a separate, then-out-of-scope issue,
# deliberately not fixed in that task). That edge case has since been fixed
# in a follow-up task (see backend/tests/test_rq1_lowest_mastery_concept_
# fix.py for its full, dedicated regression coverage) -- the check just
# below is updated accordingly so this file keeps asserting the CURRENT,
# correct behavior rather than a since-fixed bug.
db5 = new_db()
student5 = Student(moodle_user_id=910005, username="RQ1 Regression Student")
course5 = Course(moodle_course_id=910401, name="RQ1 Regression Course")
db5.add_all([student5, course5])
db5.commit()
db5.refresh(student5)
db5.refresh(course5)
memory5 = MemoryManager(db5)
memory5.set_mastery(student_id=student5.id, course_id=course5.id, concept="Recursion", mastery_level=0.85)
memory5.set_mastery(student_id=student5.id, course_id=course5.id, concept="Sorting Algorithms", mastery_level=0.15)
weakest5 = api_module._weakest_concept_for_course(memory5, student5.id, course5.id)
check("#1. Low mastery -> weak concept selected (_weakest_concept_for_course)", weakest5 == "Sorting Algorithms", weakest5)

# select_lowest_mastery_concept_tool's own empty-arguments edge case is now
# fixed (follow-up task) -- confirm it correctly picks the lowest-mastery
# AUTHORIZED concept instead of returning None. Full dedicated coverage
# (including the out-of-scope/unauthorized-course guard) lives in
# test_rq1_lowest_mastery_concept_fix.py; this is a lightweight confirmation
# that the fix is visible from this file's own fixtures too.
from tools.mastery_tools import select_lowest_mastery_concept_tool
ctx_no_concept = {"memory": memory5, "student_id": student5.id, "course_db_id": course5.id, "available_concepts": ["Recursion", "Sorting Algorithms"], "last_reference": {}, "canonical_courses": []}
no_concept_result = select_lowest_mastery_concept_tool(ctx_no_concept, {})
check("#1-note: select_lowest_mastery_concept_tool with no arguments/current_concept/last_reference "
      "now correctly falls back to the authorized available_concepts scope and selects the lowest-mastery one (Sorting Algorithms, 15%)",
      no_concept_result.get("selected_concept", {}).get("concept") == "Sorting Algorithms", no_concept_result)

# #2/#3/#4: EXPLAIN+continue -> EXAMPLE; confusion in EXPLAIN -> stays EXPLAIN
# with support; EXAMPLE+continue -> GUIDED_PRACTICE. Deterministic
# plan_compiler checks (no LLM call -- the classification step itself is
# already exercised live in Fix 1's end-to-end test B).
ctx_explain_continue = {"tutor_state": {"state": "EXPLAIN", "concept": "Recursion"}}
tools_ec, basis_ec = plan_compiler_module._compile_tutor_state(make_semantic_plan("continue"), ctx_explain_continue, [])
advance_ec = next((t for t in tools_ec if t.name == "advance_tutor_state"), None)
check("#2. EXPLAIN + continue -> advances to EXAMPLE", advance_ec is not None and advance_ec.arguments.get("to") == "EXAMPLE", tools_ec)

ctx_explain_confused = {"tutor_state": {"state": "EXPLAIN", "concept": "Recursion"}}
tools_conf, basis_conf = plan_compiler_module._compile_tutor_state(make_semantic_plan("needs_support"), ctx_explain_confused, [])
advance_conf = next((t for t in tools_conf if t.name == "advance_tutor_state"), None)
check("#3. Confusion in EXPLAIN -> does NOT advance (stays for re-explanation, no state change tool at all)",
      advance_conf is None and ctx_explain_confused.get("tutor_needs_support") is True, (tools_conf, ctx_explain_confused))

ctx_example_continue = {"tutor_state": {"state": "EXAMPLE", "concept": "Recursion"}}
tools_exc, basis_exc = plan_compiler_module._compile_tutor_state(make_semantic_plan("continue"), ctx_example_continue, [])
check("#4. EXAMPLE + continue -> compiles generate_practice_question (-> GUIDED_PRACTICE)",
      len(tools_exc) == 1 and tools_exc[0].name == "generate_practice_question", tools_exc)

# #13: updated mastery is visible to the NEXT adaptive decision -- direct
# proof that there is no session/request-scoped mastery cache anywhere in
# this read path.
db6 = new_db()
student6 = Student(moodle_user_id=910006, username="RQ1 Loop Student")
course6 = Course(moodle_course_id=910501, name="RQ1 Loop Course")
db6.add_all([student6, course6])
db6.commit()
db6.refresh(student6)
db6.refresh(course6)
memory6 = MemoryManager(db6)
memory6.set_mastery(student_id=student6.id, course_id=course6.id, concept="Recursion", mastery_level=0.10)
before_weakest = api_module._weakest_concept_for_course(memory6, student6.id, course6.id)
memory6.set_mastery(student_id=student6.id, course_id=course6.id, concept="Sorting Algorithms", mastery_level=0.05)
after_weakest = api_module._weakest_concept_for_course(memory6, student6.id, course6.id)
check("#13. Updated mastery (a lower-mastery concept just written, simulating a just-completed QPC) "
      "is immediately visible to the next weakest-concept decision, with no re-read needed",
      before_weakest == "Recursion" and after_weakest == "Sorting Algorithms", (before_weakest, after_weakest))

# #14: course launch -> weakest concept in that course (already proven live
# end-to-end for the "overall" launch in C2/C3 above; here confirmed for a
# plain course-scoped call to the same underlying selection function).
check("#14. Course launch -> weakest concept in that course (_weakest_concept_for_course, course-scoped)",
      weakest5 == "Sorting Algorithms", weakest5)

# #15: overall launch -> weakest authorized course + concept -- already
# proven live end-to-end above (C2/C2b/C3); referenced here for the record.
check("#15. Overall launch -> weakest authorized course + concept (see C2/C2b/C3 above)", True)


print()
passed = sum(1 for _, ok in results if ok)
print(f"{passed}/{len(results)} passed")
assert passed == len(results)
