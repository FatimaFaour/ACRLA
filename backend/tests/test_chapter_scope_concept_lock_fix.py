"""Regression tests for the post-RQ1 production bug fix:
services/chat_orchestrator.py:_concepts_for_level's "chapter" branch.

Root cause: _concepts_for_level and _retrieval_course_ids_for_level both
branched only on `level == "overall"` vs. not -- "chapter" and "course"
took the exact same code path (the current course's FULL concept pool via
_dynamic_concepts_for_course), so a chapter-scoped launch silently behaved
exactly like a course-scoped launch at the concept-eligibility step. A
learner who opened (clicked into) one specific chapter/concept could have
tools.mastery_tools.select_lowest_mastery_concept_tool select a DIFFERENT
concept from elsewhere in the same course, if that other concept happened
to have lower mastery -- violating the chapter-launch guarantee that
remediation stays on the concept the student actually clicked into.

This bug did NOT affect the REST assessment/QPC path (routers/api.py:
_assessment_scope), which already correctly reads course_memory[
"locked_concept"] / launch_context["locked_concept"] (written by the
Moodle-launch handler specifically for level_type=="chapter", routers/
api.py:2580,2594) to lock remediation to a single concept. This fix makes
the chat path (_concepts_for_level) read the SAME existing field the same
way -- no new chapter/concept mapping was invented.

Fix (smallest possible, entirely local to _concepts_for_level's chapter
branch): when level=="chapter", read course_memory["locked_concept"] (or
launch_context["locked_concept"]/["concept"] as fallbacks, matching
_assessment_scope's own read order) and, if it names a concept that is
part of this course's real material, return ONLY that concept. If no
locked concept is recorded, or it names something not in this course's
material, fall back to the full course concept pool (never an empty list,
never silently invented). "course" and "overall" branches are completely
unchanged.

Covers exactly what was asked (labeled A-D below, plus structural checks).
No live LLM/Gemini call anywhere in this file.

Run from the `backend/` directory:
    python tests/test_chapter_scope_concept_lock_fix.py
"""
import sys, inspect
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.db_models import Base, Student, Course
from services.memory_manager import MemoryManager
from services.chat_orchestrator import _concepts_for_level, _retrieval_course_ids_for_level
from tools.mastery_tools import select_lowest_mastery_concept_tool

results = []


def check(name, cond, detail=None):
    results.append((name, cond))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f" -> {detail}" if detail is not None else ""))


def new_fixture(moodle_course_id: int, course_name: str, uid: int):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)
    db = TestSession()
    student = Student(moodle_user_id=uid, username=f"ChapterLockTest-{uid}")
    course = Course(moodle_course_id=moodle_course_id, name=course_name)
    db.add_all([student, course])
    db.commit()
    db.refresh(student)
    db.refresh(course)
    memory = MemoryManager(db)
    return db, student, course, memory


def set_locked_concept(memory, student_id, course_db_id, concept):
    memory.update_course_memory(student_id, course_db_id, {
        "level_type": "chapter",
        "locked_concept": concept,
        "launch_context": {"level_type": "chapter", "concept": concept, "locked_concept": concept},
    })


def set_scope(memory, student_id, course_db_id, level_type):
    memory.update_course_memory(student_id, course_db_id, {
        "level_type": level_type,
        "launch_context": {"level_type": level_type},
    })


# ---------------------------------------------------------------------------
# A. Chapter scope: learner opens Recursion while Binary Trees has lower
#    mastery. EXPECTED: Recursion remains the remediation target; Binary
#    Trees must NOT be selected.
# ---------------------------------------------------------------------------
db_a, student_a, course_a, memory_a = new_fixture(2, "Intro to CS (chapter test A)", 930001)
memory_a.set_mastery(student_a.id, course_a.id, "Recursion", 0.60)
memory_a.set_mastery(student_a.id, course_a.id, "Binary Trees and BSTs", 0.10)  # deliberately the lowest in the course
set_locked_concept(memory_a, student_a.id, course_a.id, "Recursion")

eligible_a = _concepts_for_level("chapter", 2, db_a, memory_a, student_a.id)
check("A1. Chapter scope with locked_concept=Recursion returns ONLY Recursion as eligible",
      eligible_a == ["Recursion"], eligible_a)
check("A2. Binary Trees and BSTs (lower mastery, different chapter) is NOT eligible",
      "Binary Trees and BSTs" not in eligible_a, eligible_a)

selection_a = select_lowest_mastery_concept_tool(
    {"memory": memory_a, "student_id": student_a.id, "course_db_id": course_a.id,
     "available_concepts": eligible_a, "last_reference": {}, "current_concept": None}, {})
check("A3. Weak-concept selection over the chapter-locked eligible set selects Recursion, not Binary Trees",
      selection_a.get("selected_concept", {}).get("concept") == "Recursion", selection_a)


# ---------------------------------------------------------------------------
# B. Chapter scope: learner opens Sorting Algorithms while another course
#    concept has lower mastery. EXPECTED: Sorting Algorithms remains the
#    remediation target.
# ---------------------------------------------------------------------------
db_b, student_b, course_b, memory_b = new_fixture(2, "Intro to CS (chapter test B)", 930002)
memory_b.set_mastery(student_b.id, course_b.id, "Sorting Algorithms", 0.50)
memory_b.set_mastery(student_b.id, course_b.id, "Pointers and Memory Management", 0.05)  # lowest in the course
set_locked_concept(memory_b, student_b.id, course_b.id, "Sorting Algorithms")

eligible_b = _concepts_for_level("chapter", 2, db_b, memory_b, student_b.id)
check("B1. Chapter scope with locked_concept=Sorting Algorithms returns ONLY Sorting Algorithms",
      eligible_b == ["Sorting Algorithms"], eligible_b)

selection_b = select_lowest_mastery_concept_tool(
    {"memory": memory_b, "student_id": student_b.id, "course_db_id": course_b.id,
     "available_concepts": eligible_b, "last_reference": {}, "current_concept": None}, {})
check("B2. Weak-concept selection selects Sorting Algorithms, not Pointers and Memory Management",
      selection_b.get("selected_concept", {}).get("concept") == "Sorting Algorithms", selection_b)


# ---------------------------------------------------------------------------
# C. Course scope: multiple concepts, different mastery values. EXPECTED:
#    the lowest-mastery concept WITHIN that course is selected (unchanged
#    behavior -- course scope must still see the whole course).
# ---------------------------------------------------------------------------
db_c, student_c, course_c, memory_c = new_fixture(2, "Intro to CS (course test C)", 930003)
memory_c.set_mastery(student_c.id, course_c.id, "Recursion", 0.70)
memory_c.set_mastery(student_c.id, course_c.id, "Sorting Algorithms", 0.20)
memory_c.set_mastery(student_c.id, course_c.id, "Pointers and Memory Management", 0.55)
memory_c.set_mastery(student_c.id, course_c.id, "Binary Trees and BSTs", 0.40)
set_scope(memory_c, student_c.id, course_c.id, "course")

eligible_c = _concepts_for_level("course", 2, db_c, memory_c, student_c.id)
check("C1. Course scope returns the FULL course concept pool (not restricted to one concept)",
      set(eligible_c) >= {"Recursion", "Sorting Algorithms", "Pointers and Memory Management", "Binary Trees and BSTs"},
      eligible_c)

selection_c = select_lowest_mastery_concept_tool(
    {"memory": memory_c, "student_id": student_c.id, "course_db_id": course_c.id,
     "available_concepts": eligible_c, "last_reference": {}, "current_concept": None}, {})
check("C2. Course scope selects the true lowest-mastery concept in the course (Sorting Algorithms, 20%)",
      selection_c.get("selected_concept", {}).get("concept") == "Sorting Algorithms", selection_c)


# ---------------------------------------------------------------------------
# D. Overall scope: multiple eligible course concepts across >=2 courses.
#    EXPECTED: existing overall selection behavior remains correct
#    (concepts unioned across all synced courses; unaffected by the
#    chapter-branch fix, since overall's own branch was not touched).
# ---------------------------------------------------------------------------
db_d, student_d, course_d, memory_d = new_fixture(2, "Intro to CS (overall test D)", 930004)
course_d2 = Course(moodle_course_id=3, name="Discrete Math (overall test D)")
db_d.add(course_d2)
db_d.commit()
db_d.refresh(course_d2)
memory_d.set_mastery(student_d.id, course_d.id, "Recursion", 0.65)
memory_d.set_mastery(student_d.id, course_d.id, "Sorting Algorithms", 0.45)
memory_d.set_mastery(student_d.id, course_d.id, "Pointers and Memory Management", 0.50)
memory_d.set_mastery(student_d.id, course_d.id, "Binary Trees and BSTs", 0.55)
memory_d.set_mastery(student_d.id, course_d2.id, "Logic", 0.15)  # lowest, in the SECOND course
memory_d.set_mastery(student_d.id, course_d2.id, "Sets", 0.80)
memory_d.set_mastery(student_d.id, course_d2.id, "Graphs", 0.80)
memory_d.set_mastery(student_d.id, course_d2.id, "Relations Functions", 0.80)
set_scope(memory_d, student_d.id, course_d.id, "overall")

eligible_d = _concepts_for_level("overall", 2, db_d, memory_d, student_d.id)
check("D1. Overall scope unions concepts across multiple synced courses (includes Recursion from course 2 and Logic from course 3)",
      "Recursion" in eligible_d and "Logic" in eligible_d, eligible_d)

selection_d = select_lowest_mastery_concept_tool(
    {"memory": memory_d, "student_id": student_d.id, "course_db_id": course_d.id,
     "available_concepts": eligible_d, "last_reference": {}, "current_concept": None}, {})
check("D2. Overall scope selects the true global lowest-mastery concept across courses (Logic, 15%)",
      selection_d.get("selected_concept", {}).get("concept") == "Logic", selection_d)

retrieval_ids_d = _retrieval_course_ids_for_level("overall", 2, db_d)
check("D3. Overall scope's retrieval-course-id resolution is unaffected by this fix (still includes both synced courses)",
      set(retrieval_ids_d) >= {2, 3}, retrieval_ids_d)


# ---------------------------------------------------------------------------
# E. Graceful fallback: chapter scope with NO locked_concept recorded must
#    NOT return an empty eligible list -- falls back to the full course pool
#    (never silently invents a concept, never returns nothing).
# ---------------------------------------------------------------------------
db_e, student_e, course_e, memory_e = new_fixture(2, "Intro to CS (fallback test E)", 930005)
memory_e.set_mastery(student_e.id, course_e.id, "Recursion", 0.60)
memory_e.set_mastery(student_e.id, course_e.id, "Sorting Algorithms", 0.30)
set_scope(memory_e, student_e.id, course_e.id, "chapter")  # level_type=chapter, but no locked_concept written

eligible_e = _concepts_for_level("chapter", 2, db_e, memory_e, student_e.id)
check("E1. Chapter scope with no locked_concept recorded falls back to the full course pool (not empty)",
      len(eligible_e) > 1 and "Recursion" in eligible_e and "Sorting Algorithms" in eligible_e, eligible_e)


# ---------------------------------------------------------------------------
# F. Graceful fallback: chapter scope with a STALE locked_concept (not part
#    of this course's real material) must NOT return an empty/invented
#    eligible list -- falls back to the full course pool.
# ---------------------------------------------------------------------------
db_f, student_f, course_f, memory_f = new_fixture(2, "Intro to CS (stale-lock test F)", 930006)
memory_f.set_mastery(student_f.id, course_f.id, "Recursion", 0.60)
set_locked_concept(memory_f, student_f.id, course_f.id, "Nonexistent Concept From A Different Course")

eligible_f = _concepts_for_level("chapter", 2, db_f, memory_f, student_f.id)
check("F1. Chapter scope with a stale/invalid locked_concept falls back to the full course pool, not an empty list",
      len(eligible_f) >= 1 and "Nonexistent Concept From A Different Course" not in eligible_f, eligible_f)


# ---------------------------------------------------------------------------
# Structural / no-regression checks
# ---------------------------------------------------------------------------
retrieval_ids_chapter = _retrieval_course_ids_for_level("chapter", 2, db_a)
retrieval_ids_course = _retrieval_course_ids_for_level("course", 2, db_c)
check("G1. Chapter scope's retrieval-course-id resolution is unchanged (still restricted to the single current course)",
      retrieval_ids_chapter == [2], retrieval_ids_chapter)
check("G2. Course scope's retrieval-course-id resolution is unchanged (still restricted to the single current course)",
      retrieval_ids_course == [2], retrieval_ids_course)

source = inspect.getsource(_concepts_for_level)
check("H1. No mastery-mutating call was introduced into _concepts_for_level",
      "set_mastery" not in source and "update_mastery" not in source, None)
check("H2. No LLM/Gemini call was introduced into _concepts_for_level",
      "get_llm" not in source and "get_json_llm" not in source, None)
check("H3. The overall branch's own source is untouched (still iterates every Course row)",
      "for course in db.query(Course).all()" in source, None)

before_mastery = memory_a.get_mastery(student_a.id, course_a.id, "Recursion")
_concepts_for_level("chapter", 2, db_a, memory_a, student_a.id)
after_mastery = memory_a.get_mastery(student_a.id, course_a.id, "Recursion")
check("H4. Calling _concepts_for_level never changes a stored mastery value",
      before_mastery == after_mastery, (before_mastery, after_mastery))


print()
passed = sum(1 for _, ok in results if ok)
print(f"{passed}/{len(results)} passed")
assert passed == len(results)
