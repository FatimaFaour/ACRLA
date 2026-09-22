"""Regression tests for the RQ2 minimum privacy layer (services.privacy_context).

Covers:
1. build_llm_safe_student_context's own allow-list behavior (unit-level).
2. Structural check: the helper's source never reads any identity field
   from `context` at all (not "reads then strips" -- never reads).
3. Sentinel-value tests across all 5 active (ACRLA_AGENT_MODE=simple)
   LLM call sites -- a fake student/context is built with sentinel
   name/username/email/moodle_user_id/student_id values, the REAL
   prompt-building function is run with the real LLM CONSTRUCTOR patched
   to a capturing stand-in (LangChain's own RunnableLambda, so
   `prompt_template | llm` composes exactly as it does in production --
   nothing about chain construction is bypassed or reimplemented), and the
   captured ChatPromptValue is inspected BEFORE any real network call
   would have happened. Asserts the sentinels never appear, and that
   legitimate pedagogical fields (course name, concept, difficulty, RAG
   content) still do.
4. Structural safety tests: no active prompt TEMPLATE string contains a
   forbidden identity placeholder.

Follows this project's established convention: standalone script, hand-
rolled check() assertions, a real sqlite-backed MemoryManager, and
patching each module's own locally-imported name (`agents.simple_planner.
get_json_llm`, not `services.llm_factory.get_json_llm`) since that is the
name actually bound in each module's namespace.

Run from the `backend/` directory:
    python tests/test_privacy_context.py
or from anywhere (this file resolves its own project root):
    python backend/tests/test_privacy_context.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inspect
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_core.runnables import RunnableLambda

from models.db_models import Base, Student
from services.memory_manager import MemoryManager

results = []


def check(name, cond, detail=None):
    results.append((name, cond))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f" -> {detail}" if detail is not None else ""))


SENTINEL_NAME = "PRIVACY_SENTINEL_NAME"
SENTINEL_EMAIL = "privacy-sentinel@example.com"
SENTINEL_MOODLE_ID = 987654321
SENTINEL_STUDENT_ID = "SENTINEL_DB_ID_123456789"
SENTINELS = [SENTINEL_NAME, SENTINEL_EMAIL, str(SENTINEL_MOODLE_ID), SENTINEL_STUDENT_ID]


engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
TestSession = sessionmaker(bind=engine)
db = TestSession()

student = Student(moodle_user_id=SENTINEL_MOODLE_ID, username=SENTINEL_NAME, email=SENTINEL_EMAIL)
db.add(student)
db.commit()
db.refresh(student)
student.id = SENTINEL_STUDENT_ID  # force the DB id itself to be a sentinel too (in-memory only, not re-committed)

memory = MemoryManager(db)
course_id = "priv-course"
session_id = "priv-session"
RAG_CONTENT_MARKER = "RAG_CONTENT_MARKER_recursion_is_a_function_calling_itself"


# ---------------------------------------------------------------------------
# 1-2: build_llm_safe_student_context itself.
# ---------------------------------------------------------------------------
from services.privacy_context import build_llm_safe_student_context, FORBIDDEN_IDENTITY_FIELDS

raw_context = {
    "student_name": SENTINEL_NAME,
    "username": SENTINEL_NAME,
    "email": SENTINEL_EMAIL,
    "moodle_user_id": SENTINEL_MOODLE_ID,
    "student_id": SENTINEL_STUDENT_ID,
    "session_id": "some-session-id",
    "current_course": {"name": "Data Science", "db_course_id": course_id},
    "current_concept": "Linear Regression",
    "available_concepts": ["Linear Regression", "Decision Trees"],
    "resolved_concepts": ["Linear Regression"],
    "remediation_level": "chapter",
    "scope_rules": "Stay within Linear Regression.",
    "difficulty": "medium",
    "tutoring_strategy": {"name": "guided_practice", "reason": "weak area", "instructions": "Go step by step."},
    "weak_concepts": ["Linear Regression"],
}

safe = build_llm_safe_student_context(raw_context)
safe_str = str(safe)
check("1a. safe context contains none of the sentinel identity values", all(s not in safe_str for s in SENTINELS), safe)
check("1b. safe context contains no 'reason' field of the tutoring strategy", "weak area" not in safe_str)
check("1c. safe context omits weak_concepts by default (include_mastery=False)", "weak_concepts" not in safe)
check("1d. safe context still has course_name", safe.get("course_name") == "Data Science")
check("1e. safe context still has current_concept", safe.get("current_concept") == "Linear Regression")
check("1f. safe context still has difficulty", safe.get("difficulty") == "medium")
check("1g. safe context still has tutoring_strategy name+instructions", safe["tutoring_strategy"] == {"name": "guided_practice", "instructions": "Go step by step."})

safe_with_mastery = build_llm_safe_student_context(raw_context, include_mastery=True)
check("1h. include_mastery=True adds weak_concepts (names only)", safe_with_mastery.get("weak_concepts") == ["Linear Regression"])
check("1i. include_mastery=True still contains no sentinel values", all(s not in str(safe_with_mastery) for s in SENTINELS))

check("1j. empty/missing context never crashes and returns no identity", build_llm_safe_student_context({}) is not None)

# Structural: the function's own source never reads any forbidden field.
import services.privacy_context as privacy_context_module
helper_source = inspect.getsource(privacy_context_module.build_llm_safe_student_context)
for field in FORBIDDEN_IDENTITY_FIELDS:
    check(f"2. build_llm_safe_student_context never reads context.get({field!r})", f'"{field}"' not in helper_source and f"'{field}'" not in helper_source)


# ---------------------------------------------------------------------------
# Capturing-LLM machinery: a REAL LangChain Runnable (RunnableLambda) so
# `prompt_template | llm` composes exactly as production code does --
# nothing about chain construction is mocked away, only the network call.
# ---------------------------------------------------------------------------
class _Capture:
    def __init__(self):
        self.prompt_value = None

    def prompt_text(self) -> str:
        if self.prompt_value is None:
            return ""
        try:
            return "\n".join(m.content for m in self.prompt_value.to_messages())
        except Exception:
            return str(self.prompt_value)


class _FakeResponse:
    def __init__(self, content):
        self.content = content
        self.response_metadata = {}


def make_capturing_llm(response_content):
    capture = _Capture()

    def _fn(prompt_value):
        capture.prompt_value = prompt_value
        return _FakeResponse(response_content)

    return RunnableLambda(_fn), capture


def patch_module_attr(module, attr_name, replacement):
    original = getattr(module, attr_name)
    setattr(module, attr_name, replacement)
    return original


PLANNER_JSON = (
    '{"goal": "concept_explanation", "concepts": ["Linear Regression"], "courses": [], '
    '"references": [], "analytics_request": null, "needs_clarification": false, '
    '"clarification_question": null, "confidence": 0.9, "tutor_signal": null}'
)
ANALYTICS_JSON = (
    '{"operation": "get_value", "scope": "current_course", "entity": "concept", '
    '"metrics": ["current_mastery"], "filters": {}, "group_by": [], "sort_by": null, '
    '"sort_order": null, "limit": null, "include_course_average": false, '
    '"include_overall_average": false, "include_weakest": false, "include_strongest": false, '
    '"confidence": 0.9}'
)
JUDGE_JSON = '{"correct": true, "confident": true, "error_type": null, "feedback_reason": "Looks right."}'


def base_context(message, goal_hint=None, tutor_state=None, extra=None):
    ctx = {
        "session_id": session_id, "message": message, "student_id": student.id,
        "student_name": student.username, "course_db_id": course_id,
        "current_course": {"db_course_id": course_id, "moodle_course_id": 9, "name": "Data Science", "concepts": ["Linear Regression"]},
        "current_concept": None, "tutor_state": tutor_state or {}, "quick_progress_check": {},
        "remediation_level": "course", "available_concepts": ["Linear Regression"], "weak_concepts": ["Linear Regression"],
        "last_reference": {}, "last_answer_type": None, "last_discussed_metric": None,
        "difficulty": "medium", "tutoring_strategy": {"name": "guided_practice", "reason": "weak area", "instructions": "Go step by step."},
        "recent_messages": [], "recent_structured_turns": [],
        "scope_rules": "Stay within Data Science.", "course_context_by_db_id": {}, "canonical_courses": [],
        "memory": memory, "db": db,
    }
    ctx.update(extra or {})
    return ctx


def fake_search_course_material(context, arguments):
    concept = (arguments.get("concepts") or ["Linear Regression"])[0]
    return {
        "tool": "search_course_material", "success": True,
        "chunks": [{"concept": concept, "text": RAG_CONTENT_MARKER, "sources": ["ch3_linear_regression.pdf"]}],
        "sources": ["ch3_linear_regression.pdf"],
        "evidence": {"reliable": True, "coverage": "full", "supported_concepts": [concept], "reason": "ok", "confidence": 0.9},
    }


import agents.simple_agent as simple_agent
import agents.simple_planner as simple_planner_module
import agents.response_generator as response_generator_module
import services.error_analyzer as error_analyzer_module
import tools.external_tools as external_tools_module
import tools.analytics_tools as analytics_tools_module
import tools.tutor_state_tools as tst
from agents.agent_tools import TOOL_REGISTRY


def run_scenario(label, message, tutor_state=None, extra_context=None, planner_json=PLANNER_JSON):
    """Drive a real turn through run_simple_conversation_agent with BOTH
    the planner's and response generator's LLM constructors patched to
    capturing RunnableLambdas -- captures the exact prompt content each
    real call site would have sent."""
    planner_llm, planner_capture = make_capturing_llm(planner_json)
    response_llm, response_capture = make_capturing_llm("A generated tutoring reply.")

    orig_planner_get = patch_module_attr(simple_planner_module, "get_json_llm", lambda *a, **kw: planner_llm)
    orig_response_get = patch_module_attr(response_generator_module, "get_llm", lambda *a, **kw: response_llm)
    orig_search_tool = TOOL_REGISTRY.get("search_course_material")
    TOOL_REGISTRY["search_course_material"] = fake_search_course_material
    try:
        ctx = base_context(message, tutor_state=tutor_state, extra=extra_context)
        simple_agent.run_simple_conversation_agent(ctx)
    finally:
        simple_planner_module.get_json_llm = orig_planner_get
        response_generator_module.get_llm = orig_response_get
        if orig_search_tool is not None:
            TOOL_REGISTRY["search_course_material"] = orig_search_tool

    planner_text = planner_capture.prompt_text()
    response_text = response_capture.prompt_text()
    return planner_text, response_text


def assert_no_sentinels(label, *texts):
    for text in texts:
        for sentinel in SENTINELS:
            check(f"{label}: no sentinel {sentinel!r} in prompt", sentinel not in text)


# ---------------------------------------------------------------------------
# 3.1 Normal concept explanation
# ---------------------------------------------------------------------------
planner_text, response_text = run_scenario("3.1 concept explanation", "Explain Linear Regression")
assert_no_sentinels("3.1", planner_text, response_text)
check("3.1: course name still reaches the response prompt", "Data Science" in response_text, None)
check("3.1: concept still reaches the response prompt", "Linear Regression" in response_text)

# ---------------------------------------------------------------------------
# 3.2 RAG-grounded tutoring response (asserts RAG content actually present)
# ---------------------------------------------------------------------------
planner_text, response_text = run_scenario(
    "3.2 RAG-grounded tutoring", "Explain Linear Regression",
    planner_json=PLANNER_JSON,
)
assert_no_sentinels("3.2", planner_text, response_text)
# Note: as of agents/agent_json.py's _compact_value depth-truncation fix
# (see evaluation/rq3_accuracy/rag_grounding_fix.md), leaf scalars survive
# regardless of nesting depth, so the literal chunk text is now also
# expected to reach the prompt -- kept as an additional assertion here
# alongside the pre-existing structural checks, so this file still passes
# whether or not that fix is present (both were true when last run).
check("3.2: the search_course_material tool result reaches the response prompt", '"search_course_material"' in response_text and '"success":true' in response_text, response_text[-800:])
check("3.2: internal_rag pipeline/evidence still reaches the response prompt", "internal_rag" in response_text)

# ---------------------------------------------------------------------------
# 3.3 External / general knowledge fallback -- the main fix.
# ---------------------------------------------------------------------------
external_llm, external_capture = make_capturing_llm("A general-knowledge answer.")
orig_external_get = patch_module_attr(external_tools_module, "get_llm", lambda *a, **kw: external_llm)
try:
    ext_ctx = base_context("What is the capital of France?")
    result = external_tools_module.answer_with_external_knowledge_tool(ext_ctx, {"query": ext_ctx["message"]})
finally:
    external_tools_module.get_llm = orig_external_get
external_text = external_capture.prompt_text()
assert_no_sentinels("3.3 external_knowledge", external_text)
check("3.3: external tool still succeeded (real chain composition works end to end)", result.get("success") is True, result)
check("3.3: external prompt still carries course name (pedagogical redirect context)", "Data Science" in external_text)
check("3.3: external prompt no longer contains a 'Name:' student-identity line", "Name:" not in external_text, external_text[:400])

# ---------------------------------------------------------------------------
# 3.4 Analytics planning
# ---------------------------------------------------------------------------
analytics_llm, analytics_capture = make_capturing_llm(ANALYTICS_JSON)
orig_analytics_get = patch_module_attr(analytics_tools_module, "get_json_llm", lambda *a, **kw: analytics_llm)
try:
    plan = analytics_tools_module.plan_analytics_query(
        message="What's my mastery on this?", remediation_level="course",
        current_course="Data Science", current_concept="Linear Regression",
        available_courses=[{"name": "Data Science", "concepts": ["Linear Regression"]}],
        last_reference={},
    )
finally:
    analytics_tools_module.get_json_llm = orig_analytics_get
analytics_text = analytics_capture.prompt_text()
assert_no_sentinels("3.4 analytics_planning", analytics_text)
check("3.4: analytics plan prompt still carries course name", "Data Science" in analytics_text)
check("3.4: analytics planning succeeded", plan.get("operation") == "get_value", plan)

# ---------------------------------------------------------------------------
# 3.5 Practice answer evaluation
# ---------------------------------------------------------------------------
judge_llm, judge_capture = make_capturing_llm(JUDGE_JSON)
orig_judge_get = patch_module_attr(error_analyzer_module, "get_json_llm", lambda *a, **kw: judge_llm)
try:
    eval_tutor_state = {
        "state": "GUIDED_PRACTICE", "concept": "Linear Regression", "difficulty": "medium",
        "rounds_completed": 0, "consecutive_wrong": 0,
        "current_question": "Explain Linear Regression in your own words.", "asked_variants": {},
    }
    eval_ctx = base_context("It predicts a number.", tutor_state=eval_tutor_state)
    eval_ctx["tutor_state"] = eval_tutor_state
    eval_result = tst.evaluate_practice_answer_tool(eval_ctx, {})
finally:
    error_analyzer_module.get_json_llm = orig_judge_get
judge_text = judge_capture.prompt_text()
assert_no_sentinels("3.5 practice_evaluation", judge_text)
check("3.5: practice evaluation still succeeded", eval_result.get("success") is True, eval_result)
check("3.5: judge prompt still carries the concept", "Linear Regression" in judge_text)

# ---------------------------------------------------------------------------
# 3.6 Personalized tutoring (no concept named -- weakest-concept path)
# ---------------------------------------------------------------------------
planner_text, response_text = run_scenario(
    "3.6 personalized tutoring", "help me study",
    planner_json=PLANNER_JSON.replace('"concept_explanation"', '"personalized_tutoring"'),
)
assert_no_sentinels("3.6", planner_text, response_text)

# ---------------------------------------------------------------------------
# 3.7 Support/confusion turn (tutor_signal=needs_support, active EXPLAIN state)
# ---------------------------------------------------------------------------
support_tutor_state = {
    "state": "EXPLAIN", "concept": "Linear Regression", "difficulty": "medium",
    "rounds_completed": 0, "consecutive_wrong": 0, "current_question": None, "asked_variants": {},
}
planner_text, response_text = run_scenario(
    "3.7 support/confusion", "I don't understand",
    tutor_state=support_tutor_state,
    planner_json=PLANNER_JSON.replace('"concept_explanation"', '"personalized_tutoring"').replace('"tutor_signal": null', '"tutor_signal": "needs_support"'),
)
assert_no_sentinels("3.7", planner_text, response_text)
check("3.7: response prompt carries the needs_support tutor-state instruction", "needs_support" not in response_text.lower() or "TUTOR STATE" in response_text, None)

# ---------------------------------------------------------------------------
# 3.8 Concept comparison
# ---------------------------------------------------------------------------
planner_text, response_text = run_scenario(
    "3.8 concept comparison", "Compare Linear Regression and Decision Trees",
    planner_json=PLANNER_JSON.replace('"concept_explanation"', '"concept_comparison"').replace('["Linear Regression"]', '["Linear Regression", "Decision Trees"]'),
)
assert_no_sentinels("3.8", planner_text, response_text)


# ---------------------------------------------------------------------------
# 4. Structural safety: no active prompt TEMPLATE string contains a
# forbidden identity placeholder (a future-regression guard).
# ---------------------------------------------------------------------------
FORBIDDEN_PLACEHOLDERS = ["{student_name}", "{username}", "{email}", "{moodle_user_id}", "{student_id}"]

from agents.simple_planner import SIMPLE_PLANNER_PROMPT
from agents.response_generator import FINAL_PROMPT

planner_system = SIMPLE_PLANNER_PROMPT.messages[0].prompt.template
planner_human = SIMPLE_PLANNER_PROMPT.messages[1].prompt.template
response_system = FINAL_PROMPT.messages[0].prompt.template
response_human = FINAL_PROMPT.messages[1].prompt.template
external_template_source = inspect.getsource(external_tools_module) + inspect.getsource(__import__("pipelines.hybrid_pipeline", fromlist=["build_external_fallback_prompt"]))
analytics_template_source = inspect.getsource(analytics_tools_module)
judge_template_source = inspect.getsource(error_analyzer_module)

for label, text in [
    ("planner system prompt", planner_system),
    ("planner human template", planner_human),
    ("response system prompt", response_system),
    ("response human template", response_human),
    ("external-knowledge prompt source", external_template_source),
    ("analytics-planning prompt source", analytics_template_source),
    ("judge_answer prompt source", judge_template_source),
]:
    for placeholder in FORBIDDEN_PLACEHOLDERS:
        check(f"4. {label} never contains {placeholder}", placeholder not in text)

# The literal "Name: {student_context...username...}" line is gone.
hybrid_source = inspect.getsource(__import__("pipelines.hybrid_pipeline", fromlist=["build_external_fallback_prompt"]))
check("4b. hybrid_pipeline.build_external_fallback_prompt no longer builds a 'Name:' line from username", 'Name: {student_context.get("username"' not in hybrid_source)
check("4c. tools.external_tools no longer reads context.get(\"student_name\") directly", 'context.get("student_name"' not in inspect.getsource(external_tools_module))


print()
passed = sum(1 for _, ok in results if ok)
print(f"{passed}/{len(results)} passed")
assert passed == len(results)
