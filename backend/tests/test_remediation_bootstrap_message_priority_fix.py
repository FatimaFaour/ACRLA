"""Regression tests for the proactive-remediation-bootstrap message-priority
fix (read-only forensic diagnosis approved before this fix).

Bug: `agents.simple_agent._try_remediation_bootstrap_fast_path` fired
unconditionally on the session's genuinely first turn whenever a launch
concept was resolvable (`services.remediation_bootstrap.get_bootstrap_
target`), building a hardcoded `SemanticPlan(goal="concept_explanation",
...)` WITHOUT ever looking at the actual message text -- the module's own
docstring said so explicitly ("purely from structured launch/session state
... never message wording"). This meant a genuine, purposeful first message
-- "recommend a YouTube video about sorting algorithms", "give me an
example of sorting", even "check my progress" -- was silently discarded
and answered as if it had been "explain <bootstrap's own selected
concept>" instead, with `planner_call_count=0` (the real semantic planner
never ran at all this turn).

Fix (deliberately content-agnostic, no keyword list):
`services/remediation_bootstrap.py:get_bootstrap_target` now returns `None`
immediately whenever `context["message"]` is non-empty (stripped) --
deferring to the real planner for ANY genuine message, regardless of its
content or apparent intent. Bootstrap now only auto-fires when the message
is truly empty/whitespace-only (matching the module's own stated purpose:
teaching "before the student has to type anything" -- an empty message is
exactly what "hasn't typed anything" means; `models/schemas.py:ChatRequest.
message: str` has no `min_length`, so this is already a real, reachable
distinction through the existing request/context structure, not an
invented signal).

Separately, `agents/simple_planner.py`'s `SIMPLE_PLANNER_PROMPT` system
message gained one explicit rule: a request naming an external resource
TYPE (YouTube/video, website, online tutorial, external link/resource) is
`external_knowledge` even when it also names an in-course concept -- so the
real planner, once it actually runs, has a rule to apply instead of silent,
unguided judgment.

No change to the tutoring state machine, to `advance_tutor_state`'s own
semantics, or to `answer_with_external_knowledge`'s own (separately
flagged, deliberately untouched) double-generation behavior.

Uses a real sqlite in-memory DB and the established capturing-LLM
technique (a real LangChain RunnableLambda standing in for the LLM
constructor, patched on each module's own imported name) -- no live Gemini
call anywhere in this file.

Run from the `backend/` directory:
    python tests/test_remediation_bootstrap_message_priority_fix.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_core.runnables import RunnableLambda

from models.db_models import Base, Student, Course, Session as SessionModel
from services.memory_manager import MemoryManager

results = []


def check(name, cond, detail=None):
    results.append((name, cond))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f" -> {detail}" if detail is not None else ""))


engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
TestSession = sessionmaker(bind=engine)


class _Capture:
    def __init__(self):
        self.count = 0
        self.last_prompt_text = ""


def make_capturing_llm(response_text, capture: _Capture):
    def _fn(prompt_value):
        capture.count += 1
        try:
            capture.last_prompt_text = "\n".join(m.content for m in prompt_value.to_messages())
        except Exception:
            capture.last_prompt_text = str(prompt_value)

        class _FakeResponse:
            content = response_text

        return _FakeResponse()
    return RunnableLambda(_fn)


import agents.simple_planner as simple_planner_module
import agents.response_generator as response_generator_module
import tools.external_tools as external_tools_module
from services.chat_orchestrator import handle_message


def new_launch_student_course_session(tag, moodle_user_id, moodle_course_id):
    db = TestSession()
    student = Student(moodle_user_id=moodle_user_id, username=f"BootstrapFix {tag}")
    course = Course(moodle_course_id=moodle_course_id, name="Introduction to Computer Science")
    db.add_all([student, course])
    db.commit()
    db.refresh(student)
    db.refresh(course)
    session = SessionModel(student_id=student.id, course_id=course.id, difficulty="easy", is_active=True)
    db.add(session)
    db.commit()
    db.refresh(session)
    memory = MemoryManager(db)
    # A fresh chapter-remediation launch: current_concept resolvable,
    # remediation_level defaults to "chapter", no prior turns, no active
    # tutor_state/QPC -- exactly the preconditions get_bootstrap_target
    # requires (services/remediation_bootstrap.py:44-83).
    memory.set_current_topic(session.id, "Sorting Algorithms")
    return db, student, course, session, memory


def planner_json(goal, concepts=None, tutor_signal=None, confidence=0.9):
    return json.dumps({
        "goal": goal, "tutor_signal": tutor_signal, "needs_clarification": False,
        "clarification_question": None, "confidence": confidence,
        "resolved_entities": {"concepts": concepts or [], "references": []},
        "analytics_request": None,
    })


orig_planner_llm = simple_planner_module.get_json_llm
orig_response_llm = response_generator_module.get_llm
orig_external_llm = external_tools_module.get_llm


# ===========================================================================
# Test 1 -- truly empty message: bootstrap MUST still fire (planner skipped,
# hardcoded EXPLAIN turn on the bootstrap-selected concept) -- preserves
# "automatic proactive EXPLAIN bootstrap when there is no purposeful request".
# ===========================================================================
db1, student1, course1, session1, memory1 = new_launch_student_course_session("T1", 930001, 930101)
planner_capture_1, response_capture_1 = _Capture(), _Capture()
simple_planner_module.get_json_llm = lambda *a, **kw: make_capturing_llm(planner_json("concept_explanation"), planner_capture_1)
response_generator_module.get_llm = lambda *a, **kw: make_capturing_llm("Here's an explanation of sorting.", response_capture_1)
try:
    result1 = handle_message(session_id=session1.id, student_moodle_id=student1.moodle_user_id, message="", db=db1)
finally:
    simple_planner_module.get_json_llm = orig_planner_llm
    response_generator_module.get_llm = orig_response_llm

check("1a. an empty first message still SKIPS the planner (bootstrap fast path taken)",
      planner_capture_1.count == 0, planner_capture_1.count)
check("1b. agent_goal is concept_explanation (the bootstrap plan)",
      result1.get("agent_goal") == "concept_explanation", result1.get("agent_goal"))
check("1c. tools include both search_course_material and advance_tutor_state",
      set(result1.get("agent_tools_used") or []) >= {"search_course_material", "advance_tutor_state"},
      result1.get("agent_tools_used"))
check("1d. a real reply was still generated (response-generator call happened)",
      response_capture_1.count == 1 and bool(result1.get("reply")))


# ===========================================================================
# Test 2 -- a genuine external-resource request as the literal first message:
# must NOT be swallowed by bootstrap -- must reach real planner
# classification (the exact reported bug).
# ===========================================================================
db2, student2, course2, session2, memory2 = new_launch_student_course_session("T2", 930002, 930102)
planner_capture_2, response_capture_2, external_capture_2 = _Capture(), _Capture(), _Capture()
simple_planner_module.get_json_llm = lambda *a, **kw: make_capturing_llm(
    planner_json("external_knowledge", concepts=["Sorting Algorithms"]), planner_capture_2)
response_generator_module.get_llm = lambda *a, **kw: make_capturing_llm("Here's a general answer.", response_capture_2)
external_tools_module.get_llm = lambda *a, **kw: make_capturing_llm("Try searching YouTube for sorting algorithm tutorials.", external_capture_2)
try:
    result2 = handle_message(
        session_id=session2.id, student_moodle_id=student2.moodle_user_id,
        message="Recommend a YouTube video about sorting algorithms", db=db2,
    )
finally:
    simple_planner_module.get_json_llm = orig_planner_llm
    response_generator_module.get_llm = orig_response_llm
    external_tools_module.get_llm = orig_external_llm

check("2a. a genuine external-resource first message does NOT skip the planner",
      planner_capture_2.count >= 1, planner_capture_2.count)
check("2b. agent_goal is external_knowledge, NOT the hardcoded concept_explanation",
      result2.get("agent_goal") == "external_knowledge", result2.get("agent_goal"))
check("2c. advance_tutor_state was never called for this turn (no false tutor-state advance)",
      "advance_tutor_state" not in (result2.get("agent_tools_used") or []), result2.get("agent_tools_used"))
check("2d. the turn produced a real reply", bool(result2.get("reply")))
check("2e. active remediation concept/scope is preserved (current_topic unchanged)",
      memory2.get_current_topic(session2.id) == "Sorting Algorithms", memory2.get_current_topic(session2.id))

# Subsequent-turn check on the SAME session: a normal "explain" request right
# after the external-resource turn must still start real tutoring normally --
# "preserve subsequent remediation flow after that external request".
planner_capture_2b, response_capture_2b = _Capture(), _Capture()
simple_planner_module.get_json_llm = lambda *a, **kw: make_capturing_llm(
    planner_json("concept_explanation", concepts=["Sorting Algorithms"]), planner_capture_2b)
response_generator_module.get_llm = lambda *a, **kw: make_capturing_llm("Here's an explanation of sorting.", response_capture_2b)
try:
    result2b = handle_message(
        session_id=session2.id, student_moodle_id=student2.moodle_user_id,
        message="Explain sorting algorithms", db=db2,
    )
finally:
    simple_planner_module.get_json_llm = orig_planner_llm
    response_generator_module.get_llm = orig_response_llm

check("2f. a follow-up explain request after the external turn still starts real tutoring",
      set(result2b.get("agent_tools_used") or []) >= {"search_course_material", "advance_tutor_state"},
      result2b.get("agent_tools_used"))
check("2g. the follow-up turn produced a real reply", bool(result2b.get("reply")))


# ===========================================================================
# Test 3 -- a genuine tutoring-phrased request as the literal first message
# ("give me an example") must also reach real classification, not the
# hardcoded EXPLAIN bootstrap plan.
# ===========================================================================
db3, student3, course3, session3, memory3 = new_launch_student_course_session("T3", 930003, 930103)
planner_capture_3, response_capture_3 = _Capture(), _Capture()
simple_planner_module.get_json_llm = lambda *a, **kw: make_capturing_llm(
    planner_json("concept_explanation", concepts=["Sorting Algorithms"]), planner_capture_3)
response_generator_module.get_llm = lambda *a, **kw: make_capturing_llm("Here's a worked example.", response_capture_3)
try:
    result3 = handle_message(
        session_id=session3.id, student_moodle_id=student3.moodle_user_id,
        message="Give me an example of sorting", db=db3,
    )
finally:
    simple_planner_module.get_json_llm = orig_planner_llm
    response_generator_module.get_llm = orig_response_llm

check("3a. a genuine tutoring-phrased first message does NOT skip the planner",
      planner_capture_3.count == 1, planner_capture_3.count)
check("3b. the turn produced a real reply", bool(result3.get("reply")))


# ===========================================================================
# Test 4 -- an ordinary internal concept request as the literal first message
# ("explain sorting algorithms") must reach real classification and still
# land on internal course material behavior.
# ===========================================================================
db4, student4, course4, session4, memory4 = new_launch_student_course_session("T4", 930004, 930104)
planner_capture_4, response_capture_4 = _Capture(), _Capture()
simple_planner_module.get_json_llm = lambda *a, **kw: make_capturing_llm(
    planner_json("concept_explanation", concepts=["Sorting Algorithms"]), planner_capture_4)
response_generator_module.get_llm = lambda *a, **kw: make_capturing_llm("Here's an explanation.", response_capture_4)
try:
    result4 = handle_message(
        session_id=session4.id, student_moodle_id=student4.moodle_user_id,
        message="Explain sorting algorithms", db=db4,
    )
finally:
    simple_planner_module.get_json_llm = orig_planner_llm
    response_generator_module.get_llm = orig_response_llm

check("4a. an ordinary explain request does NOT skip the planner",
      planner_capture_4.count == 1, planner_capture_4.count)
check("4b. agent_goal is concept_explanation", result4.get("agent_goal") == "concept_explanation", result4.get("agent_goal"))
check("4c. tools include search_course_material (internal course-material path)",
      "search_course_material" in (result4.get("agent_tools_used") or []), result4.get("agent_tools_used"))


# ===========================================================================
# Test 5 -- structural: the planner prompt's new external-resource rule is
# actually present in the static template (no LLM call needed).
# ===========================================================================
planner_system_template = simple_planner_module.SIMPLE_PLANNER_PROMPT.messages[0].prompt.template
check("5a. planner system prompt names YouTube/video as an external_knowledge cue",
      "YouTube" in planner_system_template and "external_knowledge" in planner_system_template)
check("5b. planner system prompt explicitly says this applies even when a concept is named",
      "in-course concept" in planner_system_template)


print(f"\n{sum(1 for _, c in results if c)}/{len(results)} checks passed")
assert all(c for _, c in results), "One or more bootstrap message-priority checks failed"
