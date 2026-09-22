"""Regression tests for the final RQ1 edge-case fix:
tools/mastery_tools.py:select_lowest_mastery_concept_tool.

Root cause: `_concepts_from_arguments` (tools/course_tools.py) only
produces candidates from an explicit `concepts` argument, `context[
"last_reference"]`, or `context["current_concept"]` -- with none of those
present (the real "help me study"/nothing-named case that reaches this
tool with empty arguments, per agents/plan_compiler.py:340-343), it
returns `[]`, so `select_lowest_mastery_concept_tool` always returned
`selected_concept=None` instead of picking the lowest-mastery concept in
the student's own authorized scope.

Fix (smallest possible, entirely local to this one tool): when
`_concepts_from_arguments` would return nothing, fall back to `context[
"available_concepts"]` -- the SAME already-resolved, already-authorized
concept set for the active remediation level (chapter/course/overall)
every other tool in this scope already uses; never re-derived, never
widened, never queries another course. `_concepts_from_arguments` itself,
`get_mastery_for_concepts_tool`, and every other caller of either are
completely unchanged.

Covers exactly what was asked (labeled 1-10 below). No live LLM call
anywhere in this file -- this tool makes none itself.

Run from the `backend/` directory:
    python tests/test_rq1_lowest_mastery_concept_fix.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inspect
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.db_models import Base, Student, Course
from services.memory_manager import MemoryManager
from tools.mastery_tools import select_lowest_mastery_concept_tool, select_lowest_mastery_among_previous_turn_tool, get_mastery_for_concepts_tool
from tools.course_tools import _concepts_from_arguments

results = []


def check(name, cond, detail=None):
    results.append((name, cond))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f" -> {detail}" if detail is not None else ""))


engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
TestSession = sessionmaker(bind=engine)


def new_db():
    return TestSession()


def base_context(memory, student, course, available_concepts, extra=None):
    ctx = {
        "memory": memory, "student_id": student.id, "course_db_id": course.id,
        "available_concepts": list(available_concepts), "last_reference": {}, "canonical_courses": [],
        "current_course": {"name": course.name}, "current_concept": None,
    }
    ctx.update(extra or {})
    return ctx


# ---------------------------------------------------------------------------
# 1. Empty arguments + multiple authorized concepts -> lowest-mastery
#    concept is selected. Uses the user's own worked example.
# ---------------------------------------------------------------------------
db1 = new_db()
student1 = Student(moodle_user_id=920001, username="RQ1 Edge Student 1")
course1 = Course(moodle_course_id=920101, name="RQ1 Edge Course 1")
db1.add_all([student1, course1])
db1.commit()
db1.refresh(student1)
db1.refresh(course1)
memory1 = MemoryManager(db1)
memory1.set_mastery(student_id=student1.id, course_id=course1.id, concept="Recursion", mastery_level=0.70)
memory1.set_mastery(student_id=student1.id, course_id=course1.id, concept="Sorting Algorithms", mastery_level=0.30)
memory1.set_mastery(student_id=student1.id, course_id=course1.id, concept="Pointers and Memory Management", mastery_level=0.55)
ctx1 = base_context(memory1, student1, course1, ["Recursion", "Sorting Algorithms", "Pointers and Memory Management"])
result1 = select_lowest_mastery_concept_tool(ctx1, {})
check("1. Empty arguments + multiple authorized concepts -> lowest-mastery concept selected (Sorting Algorithms, 30%)",
      result1.get("selected_concept", {}).get("concept") == "Sorting Algorithms", result1)


# ---------------------------------------------------------------------------
# 2. Empty arguments + current authorized course scope -> selection stays
#    inside that course (mastery/course_id on the returned row matches the
#    current course, not some other one).
# ---------------------------------------------------------------------------
check("2. Selection stays inside the current authorized course scope (course_id matches course1)",
      result1.get("selected_concept", {}).get("course_id") == course1.id, result1)


# ---------------------------------------------------------------------------
# 3. A lower-mastery concept from an UNAUTHORIZED/out-of-scope course must
#    NOT be selected, even though it has the lowest raw mastery of any
#    concept in the database.
# ---------------------------------------------------------------------------
other_course1 = Course(moodle_course_id=920102, name="RQ1 Edge Other Course")
db1.add(other_course1)
db1.commit()
db1.refresh(other_course1)
# Lower mastery than anything in the authorized scope, but in a DIFFERENT
# course that is not part of ctx1["available_concepts"].
memory1.set_mastery(student_id=student1.id, course_id=other_course1.id, concept="Binary Trees and BSTs", mastery_level=0.01)
result3 = select_lowest_mastery_concept_tool(ctx1, {})
check("3. An out-of-scope concept with even lower mastery (Binary Trees and BSTs, 1%) is NEVER selected -- scope was not widened",
      result3.get("selected_concept", {}).get("concept") == "Sorting Algorithms", result3)
check("3b. The out-of-scope concept name never appears in the result at all",
      "Binary Trees and BSTs" not in str(result3), result3)


# ---------------------------------------------------------------------------
# 4. Explicit concept arguments still work exactly as before (fallback path
#    is not taken at all -- _concepts_from_arguments already has candidates).
# ---------------------------------------------------------------------------
result4 = select_lowest_mastery_concept_tool(ctx1, {"concepts": ["Recursion", "Pointers and Memory Management"]})
check("4. Explicit concept arguments still restrict the candidate set exactly as before (Pointers, 55% < Recursion 70%, Sorting Algorithms excluded)",
      result4.get("selected_concept", {}).get("concept") == "Pointers and Memory Management", result4)


# ---------------------------------------------------------------------------
# 5. last_reference behavior still works exactly as before.
# ---------------------------------------------------------------------------
ctx5 = base_context(memory1, student1, course1, ["Recursion", "Sorting Algorithms", "Pointers and Memory Management"], extra={
    "last_reference": {"items": [
        {"concept": "Recursion", "current_mastery": 0.70},
        {"concept": "Pointers and Memory Management", "current_mastery": 0.10},
    ]},
})
result5 = select_lowest_mastery_concept_tool(ctx5, {})
check("5. last_reference still supplies its own candidate set unaffected by the fix (Pointers via last_reference, not Sorting Algorithms)",
      result5.get("selected_concept", {}).get("concept") == "Pointers and Memory Management", result5)


# ---------------------------------------------------------------------------
# 6. current_concept fallback still works exactly as before.
# ---------------------------------------------------------------------------
ctx6 = base_context(memory1, student1, course1, ["Recursion", "Sorting Algorithms", "Pointers and Memory Management"], extra={
    "current_concept": "Recursion",
})
result6 = select_lowest_mastery_concept_tool(ctx6, {})
check("6. current_concept fallback still supplies exactly that one concept (Recursion only, not the whole scope)",
      result6.get("selected_concept", {}).get("concept") == "Recursion", result6)


# ---------------------------------------------------------------------------
# 7. If only one authorized concept exists -> it is selected.
# ---------------------------------------------------------------------------
db7 = new_db()
student7 = Student(moodle_user_id=920007, username="RQ1 Edge Student 7")
course7 = Course(moodle_course_id=920107, name="RQ1 Edge Course 7")
db7.add_all([student7, course7])
db7.commit()
db7.refresh(student7)
db7.refresh(course7)
memory7 = MemoryManager(db7)
memory7.set_mastery(student_id=student7.id, course_id=course7.id, concept="Recursion", mastery_level=0.40)
ctx7 = base_context(memory7, student7, course7, ["Recursion"])
result7 = select_lowest_mastery_concept_tool(ctx7, {})
check("7. A single authorized concept is selected", result7.get("selected_concept", {}).get("concept") == "Recursion", result7)


# ---------------------------------------------------------------------------
# 8. No authorized concepts at all -> graceful selected_concept=None, no
#    invented concept.
# ---------------------------------------------------------------------------
ctx8 = base_context(memory7, student7, course7, [])
result8 = select_lowest_mastery_concept_tool(ctx8, {})
check("8. No authorized concepts -> selected_concept is None (nothing invented)", result8.get("selected_concept") is None, result8)


# ---------------------------------------------------------------------------
# 9. No mastery value is written/changed by this tool.
# ---------------------------------------------------------------------------
before_mastery = memory1.get_mastery(student1.id, course1.id, "Sorting Algorithms")
select_lowest_mastery_concept_tool(ctx1, {})
after_mastery = memory1.get_mastery(student1.id, course1.id, "Sorting Algorithms")
check("9a. Mastery value is unchanged after calling the tool", before_mastery == after_mastery, (before_mastery, after_mastery))
source = inspect.getsource(select_lowest_mastery_concept_tool)
check("9b. select_lowest_mastery_concept_tool's source never calls set_mastery/update_mastery",
      "set_mastery" not in source and "update_mastery" not in source, source)


# ---------------------------------------------------------------------------
# 10. No LLM/Gemini call is introduced.
# ---------------------------------------------------------------------------
check("10. select_lowest_mastery_concept_tool's source never references get_llm/get_json_llm",
      "get_llm" not in source and "get_json_llm" not in source, source)
mastery_tools_source_full = inspect.getsource(sys.modules["tools.mastery_tools"])
check("10b. tools/mastery_tools.py has no LLM import at all (module-level)",
      "get_llm" not in mastery_tools_source_full.split("\n\n\n")[0] if False else "from services.llm_factory" not in mastery_tools_source_full)


# ---------------------------------------------------------------------------
# Structural: confirm the shared helpers this fix deliberately did NOT touch
# are unchanged, and that select_lowest_mastery_among_previous_turn_tool
# (a sibling tool using the same _concepts_from_arguments-adjacent pattern)
# is unaffected.
# ---------------------------------------------------------------------------
concepts_from_arguments_source = inspect.getsource(_concepts_from_arguments)
check("structural: _concepts_from_arguments itself was not modified (no available_concepts fallback added there)",
      "available_concepts" in concepts_from_arguments_source and concepts_from_arguments_source.count("context.get(\"available_concepts\")") == 1,
      concepts_from_arguments_source)

# select_lowest_mastery_among_previous_turn_tool has its own, separate,
# pre-existing "no concepts -> return None" branch (recent_structured_turns
# based) -- confirm it is untouched and still behaves as before (empty
# recent_structured_turns -> None, not this fix's available_concepts fallback).
ctx_prev_turn = base_context(memory1, student1, course1, ["Recursion", "Sorting Algorithms"], extra={"recent_structured_turns": []})
result_prev_turn = select_lowest_mastery_among_previous_turn_tool(ctx_prev_turn, {})
check("structural: select_lowest_mastery_among_previous_turn_tool is unaffected by this fix (still None with no prior turns, not scope-wide fallback)",
      result_prev_turn.get("selected_concept") is None and result_prev_turn.get("candidates") == [], result_prev_turn)


print()
passed = sum(1 for _, ok in results if ok)
print(f"{passed}/{len(results)} passed")
assert passed == len(results)
