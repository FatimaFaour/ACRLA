"""Regression test for the advance_tutor_state -> external_fallback /
no_tool_executed_external misclassification bug (read-only forensic
diagnosis approved before this fix).

Bug: a first-time `needs_support` ("I don't understand"/"give me a hint")
turn during an active GUIDED_PRACTICE question compiles to exactly one
tool, `advance_tutor_state` (agents.plan_compiler._compile_tutor_state's
consecutive_confusion < 2 branch) -- by design, no fresh
`search_course_material` call, since the concept's course material was
already retrieved earlier in this same tutoring session. Because
`advance_tutor_state` belongs to none of agents.simple_agent's three named
tool-category sets (_DETERMINISTIC_DATA_TOOLS, _DIALOGUE_REPLY_TOOLS,
_CONTEXT_TOOLS), `_finalize_answer`'s classification cascade used to fall
through to its generic "nothing tool-related happened" bucket, mislabeling
a live, internal, course-grounded tutoring continuation as
`external_fallback` / `no_tool_executed_external` -- with `sources=[]` and
a system-prompt rule telling the LLM to answer from general knowledge with
no course-material claim, even though real internal material for the
active concept exists and was already used earlier in the same session.

Fix: `agents/simple_agent.py:_finalize_answer` now has a dedicated
`elif "advance_tutor_state" in executed_all and
context.get("tutor_needs_support"):` branch classifying this exact turn
shape as a new, honestly-named `tutor_continuation` pipeline, with its own
system-prompt grounding rule added to `agents/response_generator.py`'s
FINAL_PROMPT. No change to the tutoring state machine, to
advance_tutor_state's own semantics, or to retrieval behavior -- this is a
classification-only fix.

Test 1 (the bug's exact reproduction) asserts the turn is now classified
`tutor_continuation`, never `external_fallback`/`no_tool_executed_external`.
Test 2 (control) asserts a genuine external/off-syllabus request with truly
zero tools executed still correctly classifies as `external_fallback` /
`no_tool_executed_external` -- proving the fix did not blur that case.

Uses a real sqlite in-memory DB and the established capturing-LLM
technique (a real LangChain RunnableLambda standing in for the LLM
constructor, patched on each module's own imported name) -- no live Gemini
call anywhere in this file.

Run from the `backend/` directory:
    python tests/test_tutor_continuation_classification_fix.py
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
from services.tutor_state_machine import save_tutor_state

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
from services.chat_orchestrator import handle_message


# ===========================================================================
# Test 1 -- the exact bug scenario: first-time needs_support mid-GUIDED_PRACTICE
# ===========================================================================
db1 = TestSession()
student1 = Student(moodle_user_id=920001, username="TutorContinuation Fix Student")
course1 = Course(moodle_course_id=920101, name="Introduction to Computer Science")
db1.add_all([student1, course1])
db1.commit()
db1.refresh(student1)
db1.refresh(course1)
session1 = SessionModel(student_id=student1.id, course_id=course1.id, difficulty="easy", is_active=True)
db1.add(session1)
db1.commit()
db1.refresh(session1)
memory1 = MemoryManager(db1)

PENDING_QUESTION = "Which sorting algorithm has O(n log n) average-case time complexity?"
save_tutor_state(
    memory1, student1.id, course1.id, session1.id,
    state="GUIDED_PRACTICE", concept="Sorting Algorithms", difficulty="easy",
    rounds_completed=1, consecutive_wrong=0, current_question=PENDING_QUESTION,
    asked_variants={},
)

planner_json_1 = json.dumps({
    "goal": "personalized_tutoring", "tutor_signal": "needs_support", "needs_clarification": False,
    "clarification_question": None, "confidence": 0.92,
    "resolved_entities": {"concepts": [], "references": []},
    "analytics_request": None,
})

planner_capture_1 = _Capture()
response_capture_1 = _Capture()
orig_planner_llm = simple_planner_module.get_json_llm
orig_response_llm = response_generator_module.get_llm
simple_planner_module.get_json_llm = lambda *a, **kw: make_capturing_llm(planner_json_1, planner_capture_1)
response_generator_module.get_llm = lambda *a, **kw: make_capturing_llm("Here's a small hint toward that question.", response_capture_1)
try:
    result1 = handle_message(
        session_id=session1.id, student_moodle_id=student1.moodle_user_id,
        message="I'm still confused, can you give me a hint?", db=db1,
    )
finally:
    simple_planner_module.get_json_llm = orig_planner_llm
    response_generator_module.get_llm = orig_response_llm

check("1a. agent_tools_used is exactly ['advance_tutor_state']",
      result1.get("agent_tools_used") == ["advance_tutor_state"], result1.get("agent_tools_used"))
check("1b. selected_pipeline is 'tutor_continuation', NOT 'external_fallback'",
      result1.get("selected_pipeline") == "tutor_continuation", result1.get("selected_pipeline"))
check("1c. selected_pipeline is never the old buggy value",
      result1.get("selected_pipeline") != "external_fallback")
check("1d. evidence_reason is the new honest reason, never 'no_tool_executed_external'",
      result1.get("evidence_reason") == "tutor_state_continuation_no_fresh_retrieval", result1.get("evidence_reason"))
check("1e. evidence_reliable stays None (no fresh retrieval this turn, no fabricated evidence)",
      result1.get("evidence_reliable") is None)
check("1f. sources stays empty (no source re-verified this turn -- no fabrication)",
      result1.get("sources") == [])
check("1g. exactly one planner call and one response call (no extra LLM call introduced)",
      planner_capture_1.count == 1 and response_capture_1.count == 1,
      (planner_capture_1.count, response_capture_1.count))
check("1h. the final-answer prompt actually carries the tutor_continuation grounding rule",
      "tutor_continuation:" in response_capture_1.last_prompt_text)
check("1i. the final-answer prompt still carries the pending question (hint context preserved)",
      PENDING_QUESTION in response_capture_1.last_prompt_text)
check("1j. the turn produced a real reply (fix didn't break the turn)", bool(result1.get("reply")))


# ===========================================================================
# Test 2 -- control: a genuine external/off-syllabus request with zero tools
# executed must still classify as external_fallback / no_tool_executed_external
# ===========================================================================
db2 = TestSession()
student2 = Student(moodle_user_id=920002, username="TutorContinuation Control Student")
course2 = Course(moodle_course_id=920102, name="Introduction to Computer Science")
db2.add_all([student2, course2])
db2.commit()
db2.refresh(student2)
db2.refresh(course2)
session2 = SessionModel(student_id=student2.id, course_id=course2.id, difficulty="easy", is_active=True)
db2.add(session2)
db2.commit()
db2.refresh(session2)
# No tutor_state seeded -- this student has no active tutoring session at all.

planner_json_2 = json.dumps({
    "goal": "casual_conversation", "tutor_signal": None, "needs_clarification": False,
    "clarification_question": None, "confidence": 0.9,
    "resolved_entities": {"concepts": [], "references": []},
    "analytics_request": None,
})

planner_capture_2 = _Capture()
response_capture_2 = _Capture()
simple_planner_module.get_json_llm = lambda *a, **kw: make_capturing_llm(planner_json_2, planner_capture_2)
response_generator_module.get_llm = lambda *a, **kw: make_capturing_llm("The Eiffel Tower is in Paris, France.", response_capture_2)
try:
    result2 = handle_message(
        session_id=session2.id, student_moodle_id=student2.moodle_user_id,
        message="Where is the Eiffel Tower?", db=db2,
    )
finally:
    simple_planner_module.get_json_llm = orig_planner_llm
    response_generator_module.get_llm = orig_response_llm

check("2a. agent_tools_used is empty for a genuine no-tool casual turn",
      result2.get("agent_tools_used") == [], result2.get("agent_tools_used"))
check("2b. selected_pipeline is still 'external_fallback' for a genuine off-syllabus request",
      result2.get("selected_pipeline") == "external_fallback", result2.get("selected_pipeline"))
check("2c. evidence_reason is still 'no_tool_executed_external' for a genuine off-syllabus request",
      result2.get("evidence_reason") == "no_tool_executed_external", result2.get("evidence_reason"))
check("2d. the turn produced a real reply", bool(result2.get("reply")))


print(f"\n{sum(1 for _, c in results if c)}/{len(results)} checks passed")
assert all(c for _, c in results), "One or more tutor_continuation classification checks failed"
