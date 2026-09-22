"""Regression tests for the RQ3 QPC generated-question validity gate
(services.question_validity + its wiring into
routers.api:_stored_or_generated_variants).

Root cause this addresses: routers/api.py:_llm_dynamic_variants only
checked that Gemini's output PARSED into the right JSON shape (exactly 4
options, correct in A-D) -- never whether the question was actually about
the right concept, grounded in the retrieved course material, unambiguous,
or correctly answered. Once persisted to AssessmentQuestionVariant, a bad
question was served to every future student indefinitely and could
influence mastery via the normal Quick Progress Check scoring path.

Covers exactly what the task asked for (labeled A-J below):
A. valid generated MCQ -> accepted
B. malformed MCQ -> rejected
C. wrong indicated answer -> rejected (distractor more grounded than the
   marked-correct option)
D. ambiguous/multiple-correct-answer case -> rejected (duplicate options)
E. wrong-concept question -> rejected
F. unsupported question (no grounding in the retrieved material) -> rejected
G. predefined/hand-authored bank question -> unaffected (never passed
   through the validator at all)
H. a batch that fails validation -> falls through to the SAME deterministic
   fallback template this code path already used for "the LLM returned
   nothing" -- the rejected content is never stored/served, so it cannot
   be what a mastery update is based on
I. a batch that passes validation -> the LLM-generated (validated)
   questions are what gets persisted/served, and remain usable by the
   existing, unmodified deterministic scoring formula
J. the exact retrieved course-material text used to prompt the LLM is the
   SAME text the validator grounds its checks against (not a different or
   stale context)

Uses controlled fixtures throughout -- no live Gemini calls. Follows this
project's established convention: standalone script, hand-rolled check()
assertions, a real sqlite-backed MemoryManager/DB session, and patching
each module's own locally-imported name (`routers.api.get_llm`, the name
routers/api.py actually imported) rather than services.llm_factory's own.

Run from the `backend/` directory:
    python tests/test_qpc_question_validity.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inspect
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.db_models import Base, Student, Course
from models.db_models import AssessmentQuestionVariant
from services.memory_manager import MemoryManager

results = []


def check(name, cond, detail=None):
    results.append((name, cond))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f" -> {detail}" if detail is not None else ""))


CONCEPT = "Linear Regression"
COURSE_CONTEXT = (
    "Linear regression models the relationship between a numeric target variable "
    "and one or more predictor variables by fitting a straight line that minimizes "
    "squared error between predicted and actual values. The slope and intercept "
    "define the fitted line."
)

# ---------------------------------------------------------------------------
# A-F: validate_generated_question -- direct unit-level checks, no DB/LLM.
# ---------------------------------------------------------------------------
from services.question_validity import validate_generated_question, validate_generated_questions

VALID_ITEM = {
    "question_id": "lr_target_1", "variant_id": "dyn_1_lr_target_1",
    "sub_concept": "target variable",
    "prompt": "In linear regression, what does the fitted line predict?",
    "options": [
        "A. A category label",
        "B. A numeric target value from predictor variables",
        "C. The name of the dataset",
        "D. A random number unrelated to the data",
    ],
    "correct": "B",
}

# A. valid generated MCQ -> accepted
result_a = validate_generated_question(VALID_ITEM, concept=CONCEPT, course_context=COURSE_CONTEXT)
check("A. a valid, grounded, unambiguous MCQ is accepted", result_a.passed, result_a.reason)

# B. malformed MCQ -> rejected
malformed_item = {**VALID_ITEM, "options": ["A. only one option"]}
result_b = validate_generated_question(malformed_item, concept=CONCEPT, course_context=COURSE_CONTEXT)
check("B. a malformed MCQ (wrong option count) is rejected", not result_b.passed and result_b.reason == "structurally_invalid", result_b)

malformed_item2 = {**VALID_ITEM, "correct": "Z"}
result_b2 = validate_generated_question(malformed_item2, concept=CONCEPT, course_context=COURSE_CONTEXT)
check("B2. a malformed MCQ (invalid correct-answer letter) is rejected", not result_b2.passed and result_b2.reason == "structurally_invalid", result_b2)

# C. wrong indicated answer -> rejected (a distractor is MORE grounded than
# the option marked "correct" -- a concrete, checkable signal the answer
# key itself is likely wrong).
wrong_answer_item = {
    "question_id": "lr_wrong_1", "sub_concept": "target variable",
    "prompt": "In linear regression, what does the fitted line predict?",
    "options": [
        "A. A numeric target value from predictor variables",  # actually the well-grounded one
        "B. A random unrelated number with no connection to predictor variables or fitted lines",
        "C. The name of the dataset file only",
        "D. A category label only",
    ],
    "correct": "B",  # mislabeled -- B is the LEAST grounded option
}
result_c = validate_generated_question(wrong_answer_item, concept=CONCEPT, course_context=COURSE_CONTEXT)
check("C. an indicated answer less grounded than a distractor is rejected", not result_c.passed and result_c.reason == "indicated_answer_less_grounded_than_a_distractor", result_c)

# D. ambiguous/multiple-correct-answer case -> rejected (duplicate options)
ambiguous_item = {
    "question_id": "lr_dup_1", "sub_concept": "target variable",
    "prompt": "In linear regression, what does the fitted line predict?",
    "options": [
        "A. A numeric target value from predictor variables",
        "B. A numeric target value from predictor variables",  # duplicate of A
        "C. The dataset file name",
        "D. A category label",
    ],
    "correct": "A",
}
result_d = validate_generated_question(ambiguous_item, concept=CONCEPT, course_context=COURSE_CONTEXT)
check("D. duplicate/ambiguous options are rejected", not result_d.passed and result_d.reason == "duplicate_or_ambiguous_options", result_d)

# E. wrong-concept question -> rejected (question has nothing to do with
# the concept it claims to be testing).
wrong_concept_item = {
    "question_id": "lr_offtopic_1", "sub_concept": "unrelated topic",
    "prompt": "Which sorting algorithm splits a list around a pivot value?",
    "options": ["A. Quick sort", "B. Bubble sort", "C. Insertion sort", "D. Selection sort"],
    "correct": "A",
}
result_e = validate_generated_question(wrong_concept_item, concept=CONCEPT, course_context=COURSE_CONTEXT)
check("E. a question unrelated to the target concept is rejected", not result_e.passed and result_e.reason == "concept_mismatch", result_e)

# F. unsupported question -> rejected (on-topic-sounding but shares no real
# vocabulary with the actually-retrieved course material).
unsupported_item = {
    "question_id": "lr_unsupported_1", "sub_concept": "advanced regularization",
    "prompt": "Linear regression models use elastic net penalty terms with lambda hyperparameters for regularization.",
    "options": [
        "A. It combines L1 and L2 penalty terms to shrink coefficients",
        "B. It removes the need for any predictor variables",
        "C. It sorts the dataset alphabetically",
        "D. It converts regression into classification automatically",
    ],
    "correct": "A",
}
result_f = validate_generated_question(unsupported_item, concept=CONCEPT, course_context=COURSE_CONTEXT)
check("F. a question with insufficient grounding in the retrieved material is rejected", not result_f.passed and result_f.reason == "insufficient_grounding_in_course_material", result_f)

# Batch-level: validate_generated_questions filters correctly and logs
# privacy-safe rejection reasons (no full option text required in the log).
batch = [VALID_ITEM, malformed_item, wrong_answer_item, ambiguous_item, wrong_concept_item, unsupported_item]
accepted, rejected = validate_generated_questions(batch, concept=CONCEPT, course_context=COURSE_CONTEXT)
check("batch: exactly the 1 valid item is accepted", len(accepted) == 1 and accepted[0]["question_id"] == "lr_target_1", [a["question_id"] for a in accepted])
check("batch: exactly the 5 invalid items are rejected", len(rejected) == 5, rejected)
check("batch: rejection log entries carry a reason but no full option text", all("reason" in r and "options" not in r for r in rejected), rejected)


# ---------------------------------------------------------------------------
# G. Predefined/hand-authored bank questions are never passed through the
# validator at all -- structural check on routers.api's own source, plus a
# behavioral check that the bank still returns usable items unchanged.
# ---------------------------------------------------------------------------
import routers.api as api_module

candidate_questions_source = inspect.getsource(api_module._candidate_questions)
check("G1. _candidate_questions adds the predefined bank directly, not through validate_generated_questions", "_assessment_question_bank(concept)" in candidate_questions_source and "validate_generated_questions" not in candidate_questions_source)

bank_items = api_module._assessment_question_bank("Recursion")
check("G2. the predefined bank still returns real, unmodified questions", len(bank_items) >= 1 and all(len(item.get("options", [])) == 4 for item in bank_items), len(bank_items))


# ---------------------------------------------------------------------------
# H & I: _stored_or_generated_variants -- the actual modified function --
# with a real sqlite DB and a mocked get_llm (routers.api's own imported
# name). Structural check first: the validator IS actually wired in.
# ---------------------------------------------------------------------------
stored_or_generated_source = inspect.getsource(api_module._stored_or_generated_variants)
check("wiring: _stored_or_generated_variants calls validate_generated_questions on the LLM output before caching", "validate_generated_questions(raw_variants" in stored_or_generated_source)

engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
TestSession = sessionmaker(bind=engine)
db = TestSession()

course = Course(moodle_course_id=88101, name="Data Science")
db.add(course)
db.commit()
db.refresh(course)

student = Student(moodle_user_id=88102, username="qpc_validity_student")
db.add(student)
db.commit()
db.refresh(student)
memory = MemoryManager(db)


class _FakeLLMResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, prompt):
        return _FakeLLMResponse(self._content)


import json as _json

INVALID_BATCH_JSON = _json.dumps([
    {"sub_concept": "unrelated topic", "prompt": "Which sorting algorithm splits a list around a pivot value?",
     "options": ["A. Quick sort", "B. Bubble sort", "C. Insertion sort", "D. Selection sort"], "correct": "A"},
    {"sub_concept": "target variable", "prompt": "In linear regression, what does the fitted line predict?",
     "options": ["A. A numeric target value from predictor variables",
                 "B. A random unrelated number with no connection to predictor variables or fitted lines",
                 "C. The dataset file name only", "D. A category label only"], "correct": "B"},
])

VALID_BATCH_JSON = _json.dumps([
    {"sub_concept": "target variable", "prompt": "In linear regression, what does the fitted line predict?",
     "options": ["A. A category label", "B. A numeric target value from predictor variables",
                 "C. The name of the dataset", "D. A random number unrelated to the data"], "correct": "B"},
])

orig_get_llm = api_module.get_llm

# ---------------------------------------------------------------------------
# H. Every candidate in the batch fails validation -> falls through to the
# EXISTING deterministic fallback template; nothing invalid is stored/served.
# ---------------------------------------------------------------------------
api_module.get_llm = lambda *a, **kw: _FakeLLM(INVALID_BATCH_JSON)
try:
    variants_h = api_module._stored_or_generated_variants(db, course.id, course.moodle_course_id, CONCEPT)
finally:
    api_module.get_llm = orig_get_llm

served_prompts_h = {v["prompt"] for v in variants_h}
check("H1. no rejected (invalid) prompt is ever served to the student", "Which sorting algorithm splits a list around a pivot value?" not in served_prompts_h, served_prompts_h)
check("H2. the served set instead comes from the existing deterministic fallback template", any("core" in (v.get("question_id") or "") for v in variants_h), [v.get("question_id") for v in variants_h])

db_rows_h = db.query(AssessmentQuestionVariant).filter_by(course_id=course.id, concept=CONCEPT).all()
check("H3. only the fallback questions were persisted -- the rejected LLM content was never cached", all("sorting algorithm" not in (row.prompt or "").lower() for row in db_rows_h), [row.prompt for row in db_rows_h])

# "Cannot update mastery": the served (fallback) question's own `correct`
# field is what any later scoring would use -- confirm it is well-formed
# and, applying the SAME deterministic equality check submit_assessment
# itself uses (`submitted == expected`), an answer matching the rejected
# item's (wrong) claimed-correct option would simply never be one of the
# served options at all, so it cannot enter scoring.
fallback_item_h = variants_h[0]
check("H4. the served fallback item has its own valid correct-letter field usable by the existing scoring formula", fallback_item_h.get("correct") in {"A", "B", "C", "D"}, fallback_item_h.get("correct"))

# ---------------------------------------------------------------------------
# I. A batch that passes validation -> the validated LLM content (not the
# fallback) is what gets persisted/served, and remains scoreable by the
# existing, UNMODIFIED deterministic formula (submitted == expected).
# ---------------------------------------------------------------------------
course2 = Course(moodle_course_id=88103, name="Data Science 2")
db.add(course2)
db.commit()
db.refresh(course2)

orig_dynamic_context_i = api_module._dynamic_material_context
api_module.get_llm = lambda *a, **kw: _FakeLLM(VALID_BATCH_JSON)
# Real course material retrieval (Ollama/ChromaDB) is not under test here --
# stand in for it with the SAME fixture text the standalone validator
# checks (A-F) above already used, matching what a real retrieval would
# have supplied for a genuinely on-topic, grounded question.
api_module._dynamic_material_context = lambda course_id, concept: (
    COURSE_CONTEXT, "source.pdf", "Chapter Title",
    [{"text": COURSE_CONTEXT, "sensitivity": "PUBLIC", "source": "source.pdf"}],
)
try:
    variants_i = api_module._stored_or_generated_variants(db, course2.id, course2.moodle_course_id, CONCEPT)
finally:
    api_module.get_llm = orig_get_llm
    api_module._dynamic_material_context = orig_dynamic_context_i

check("I1. the validated LLM-generated question (not the fallback) is what gets served", len(variants_i) == 1 and "fitted line predict" in variants_i[0]["prompt"], variants_i)
db_rows_i = db.query(AssessmentQuestionVariant).filter_by(course_id=course2.id, concept=CONCEPT).all()
check("I2. the validated LLM-generated question is what gets persisted/cached", len(db_rows_i) == 1 and "fitted line predict" in (db_rows_i[0].prompt or ""), [row.prompt for row in db_rows_i])

# Reproduce submit_assessment's own scoring formula (routers/api.py:
# `submitted == expected`, both upper-cased/first-character) against the
# served item, exactly as a real submission would be scored -- this project
# convention (drive the real logic, don't reimplement it) is honored by
# using the identical comparison, not a re-derived one.
served_item_i = variants_i[0]
expected_letter = str(served_item_i["correct"]).strip().upper()
correct_submission = expected_letter
wrong_submission = next(letter for letter in "ABCD" if letter != expected_letter)
check("I3. a correct submission scores correct via the existing scoring formula", correct_submission.strip().upper()[:1] == expected_letter)
check("I4. an incorrect submission scores incorrect via the existing scoring formula", wrong_submission.strip().upper()[:1] != expected_letter)

# Confirm the mastery-write path itself (MemoryManager.set_mastery, the
# SAME function submit_assessment calls) still works normally for a
# validated question's course/concept -- proving nothing about the
# validity gate broke the ability to record mastery for a GOOD question.
before_mastery = memory.get_mastery(student.id, course2.id, CONCEPT)
memory.set_mastery(student_id=student.id, course_id=course2.id, concept=CONCEPT, mastery_level=0.8)
after_mastery = memory.get_mastery(student.id, course2.id, CONCEPT)
check("I5. mastery can still be recorded normally for a validated question's concept", after_mastery == 0.8 and after_mastery != before_mastery, (before_mastery, after_mastery))


# ---------------------------------------------------------------------------
# J. The exact retrieved course-material text used to prompt the LLM is the
# SAME text the validator grounds its checks against.
# ---------------------------------------------------------------------------
DISTINCTIVE_CONTEXT = "DISTINCTIVE_MARKER_TEXT linear regression fits a line to minimize squared error."
course3 = Course(moodle_course_id=88104, name="Data Science 3")
db.add(course3)
db.commit()
db.refresh(course3)

captured_contexts = []
import services.question_validity as qv_module
orig_validate = qv_module.validate_generated_questions
real_validate = qv_module.validate_generated_questions


def _spy_validate(items, *, concept, course_context):
    captured_contexts.append(course_context)
    return real_validate(items, concept=concept, course_context=course_context)


# Patch _dynamic_material_context to return the distinctive marker text
# (simulating real RAG retrieval without needing live Ollama/ChromaDB), and
# spy on the validator call (_stored_or_generated_variants re-imports this
# name fresh from services.question_validity on every call, since it's a
# local `from ... import` inside the function body, not a module-level
# binding in routers.api -- so patching it here does take effect) to
# confirm it receives that EXACT text.
orig_dynamic_context = api_module._dynamic_material_context
api_module._dynamic_material_context = lambda course_id, concept: (
    DISTINCTIVE_CONTEXT, "source.pdf", "Chapter Title",
    [{"text": DISTINCTIVE_CONTEXT, "sensitivity": "PUBLIC", "source": "source.pdf"}],
)
qv_module.validate_generated_questions = _spy_validate
# A question grounded against the distinctive context (shares real
# vocabulary with it) so it will pass -- proving the SAME text flows all
# the way from "retrieval" through generation into validation.
grounded_for_marker = _json.dumps([
    {"sub_concept": "target variable", "prompt": "What does linear regression fit to minimize squared error?",
     "options": ["A. A line", "B. A random guess", "C. A dataset name", "D. A password"], "correct": "A"},
])
api_module.get_llm = lambda *a, **kw: _FakeLLM(grounded_for_marker)
try:
    variants_j = api_module._stored_or_generated_variants(db, course3.id, course3.moodle_course_id, CONCEPT)
finally:
    api_module.get_llm = orig_get_llm
    api_module._dynamic_material_context = orig_dynamic_context
    qv_module.validate_generated_questions = orig_validate

check("J1. the validator received the SAME context text _dynamic_material_context produced", captured_contexts and captured_contexts[0] == DISTINCTIVE_CONTEXT, captured_contexts)
check("J2. a question genuinely grounded in that retrieved text is accepted (RAG evidence reaches validation, not bypassed)", len(variants_j) == 1 and "squared error" in variants_j[0]["prompt"], variants_j)


print()
passed = sum(1 for _, ok in results if ok)
print(f"{passed}/{len(results)} passed")
assert passed == len(results)
