"""Regression tests for the RQ1.B root-cause-analysis architectural fixes
(see evaluation/rq1_architecture/hybrid_routing/rq1b_root_cause_analysis.md
and rq1b_fix_implementation_report.md).

Covers, with NO external model call anywhere in this file (every planner
decision and every tool result is a fixed/monkey-patched fixture -- see
`agents.simple_planner.plan` and `agents.agent_tools.TOOL_REGISTRY` being
replaced below, restored in `finally` blocks):

Fix 1 (agents/simple_agent.py:_ground_plan entity-completeness logic):
  1. Reliable internal RAG path (sanity: unaffected by the fix)
  2. A single-concept content request whose concept fails resolution is no
     longer pre-empted before retrieval -- it reaches evidence validation
  3. ... and external fallback correctly follows when that evidence is
     unreliable
  4. Direct external path unchanged
  5. Analytics path unchanged
  6. Deterministic tutoring (tutor-state) path unchanged
  7. concept_comparison with a SILENTLY OMITTED second concept (no
     `unresolved` marker at all) is now correctly forced to clarification --
     the exact gap the fix closes -- while a comparison with two genuinely
     resolved concepts, and a comparison with an explicitly-named-and-failed
     concept (the already-working case), are both unaffected.

Fix 2 (services/course_concepts.py:resolve_concept_identity):
  8. Raw canonical label resolves a clean-form planner proposal; an
     unrelated concept does NOT match; a cross-course concept does NOT
     leak; existing exact matches keep working.

Fix 3 (services/chat_orchestrator.py pre-agent guard observability):
  9. The guard still intercepts a mastery-modification message exactly as
     before (no safety/behavior change)
 10. ... and now reports selected_pipeline="deterministic_reply" instead of
     None.

Run from the `backend/` directory:
    python tests/test_rq1b_root_cause_fixes.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

results = []


def check(name, cond, detail=None):
    results.append((name, cond))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f" -> {detail}" if detail is not None else ""))


# ===========================================================================
# Fix 2 unit tests: services.course_concepts.resolve_concept_identity
# ===========================================================================
from services.course_concepts import resolve_concept_identity

COURSE_A_CONCEPTS = ["Recursion", "Sorting Algorithms", "Pointers and Memory Management", "Binary Trees and BSTs"]
COURSE_B_CONCEPTS = ["Chap: Linear Regression"]

check("8a. raw canonical label resolves a clean-form proposal",
      resolve_concept_identity("Linear Regression", COURSE_B_CONCEPTS) == "Chap: Linear Regression")
check("8b. exact match on the raw label itself still works",
      resolve_concept_identity("Chap: Linear Regression", COURSE_B_CONCEPTS) == "Chap: Linear Regression")
check("8c. existing exact match (no prefix at all) still works",
      resolve_concept_identity("Recursion", COURSE_A_CONCEPTS) == "Recursion")
check("8d. an unrelated concept sharing a word does NOT match (no fuzzy/token-overlap)",
      resolve_concept_identity("Logistic Regression", COURSE_B_CONCEPTS) is None)
check("8e. a genuinely absent concept does NOT match",
      resolve_concept_identity("Hash Tables", COURSE_A_CONCEPTS) is None)
check("8f. cross-course: Course A's concept does NOT leak into Course B's vocabulary",
      resolve_concept_identity("Recursion", COURSE_B_CONCEPTS) is None)
check("8g. cross-course: Course B's raw label does NOT leak into Course A's vocabulary",
      resolve_concept_identity("Chap: Linear Regression", COURSE_A_CONCEPTS) is None)
check("8h. None/empty input resolves to None",
      resolve_concept_identity(None, COURSE_A_CONCEPTS) is None and resolve_concept_identity("", COURSE_A_CONCEPTS) is None)


# ===========================================================================
# Fix 1 unit tests: agents.simple_agent._compile_and_ground (grounding logic)
# ===========================================================================
import agents.simple_agent as simple_agent
from agents.agent_models import SemanticPlan, ResolvedEntities

def ctx(message, available, tutor_state=None):
    return {
        "message": message,
        "available_concepts": available,
        "tutor_state": tutor_state or {},
        "quick_progress_check": {},
        "is_followup": False,
        "current_concept": None,
        "agent_selected_concept": None,
        "goal": None,
    }


# 1. Reliable internal RAG path -- single real concept, unaffected by the fix.
plan1 = simple_agent._compile_and_ground(
    ctx("Can you explain what recursion is?", COURSE_A_CONCEPTS),
    SemanticPlan(goal="concept_explanation", resolved_entities=ResolvedEntities(concepts=["Recursion"]), confidence=0.9),
)
check("1a. reliable-RAG-eligible plan: not forced to clarification", plan1.needs_clarification is False)
check("1b. reliable-RAG-eligible plan: search_course_material is compiled", any(t.name == "search_course_material" for t in plan1.tools))
check("1c. reliable-RAG-eligible plan: resolved concept is exactly 'Recursion'", plan1.resolved_entities.concepts == ["Recursion"])

# 2. concept_explanation naming a genuinely absent concept: FIX -- no longer
#    pre-empted before retrieval (pre-fix this wiped tools + forced clarification).
plan2 = simple_agent._compile_and_ground(
    ctx("Since we're working through this course, can you also explain how hash tables work?", COURSE_A_CONCEPTS),
    SemanticPlan(goal="concept_explanation", resolved_entities=ResolvedEntities(concepts=["Hash Tables"]), confidence=0.9),
)
check("2a. FIX: zero-resolved concept_explanation is NOT forced to clarification pre-execution", plan2.needs_clarification is False)
check("2b. FIX: search_course_material is still compiled (retrieval allowed to attempt)", any(t.name == "search_course_material" for t in plan2.tools))
check("2c. the absent concept is still surfaced as unresolved (informational, non-blocking)", plan2.resolved_entities.unresolved == ["Hash Tables"])
check("2d. the absent concept is correctly dropped from resolved concepts", plan2.resolved_entities.concepts == [])

# 3+7. concept_comparison, second concept SILENTLY OMITTED by the planner
#      (never named, so `unresolved` stays empty) -- FIX: must still be forced
#      to clarification (this was the CL-3/CL-4 gap: pre-fix, this silently
#      passed through to retrieval instead).
plan3 = simple_agent._compile_and_ground(
    ctx("Can you compare Recursion and Blorptrees?", COURSE_A_CONCEPTS),
    SemanticPlan(goal="concept_comparison", resolved_entities=ResolvedEntities(concepts=["Recursion"]), confidence=0.9),
)
check("3a. FIX: concept_comparison with only 1 real concept (2nd silently omitted) IS forced to clarification", plan3.needs_clarification is True)
check("3b. FIX: tools are wiped in this case", plan3.tools == [])
check("3c. (characterization) unresolved is empty here -- confirms the fix does not depend on it", plan3.resolved_entities.unresolved == [])

# 4. concept_comparison with two genuinely resolved concepts -- must remain
#    answerable, unaffected by the fix.
plan4 = simple_agent._compile_and_ground(
    ctx("Compare recursion and sorting algorithms.", COURSE_A_CONCEPTS),
    SemanticPlan(goal="concept_comparison", resolved_entities=ResolvedEntities(concepts=["Recursion", "Sorting Algorithms"]), confidence=0.9),
)
check("4a. two real concepts: concept_comparison is NOT forced to clarification", plan4.needs_clarification is False)
check("4b. two real concepts: search_course_material is compiled", any(t.name == "search_course_material" for t in plan4.tools))

# 5. concept_comparison where the planner explicitly named-and-failed the
#    second concept (unresolved IS populated) -- the already-working case,
#    must remain forced to clarification.
plan5 = simple_agent._compile_and_ground(
    ctx("Can you compare recursion and quantum entanglement?", COURSE_A_CONCEPTS),
    SemanticPlan(goal="concept_comparison", resolved_entities=ResolvedEntities(concepts=["Recursion", "quantum entanglement"]), confidence=0.9),
)
check("5a. explicitly-named-and-failed 2nd concept: still forced to clarification (unaffected)", plan5.needs_clarification is True)
check("5b. explicitly-named-and-failed 2nd concept: still surfaced in unresolved", plan5.resolved_entities.unresolved == ["quantum entanglement"])

# 6. Fix 2 wired into grounding: a course whose only real concept is stored
#    under a raw prefixed label; planner naturally proposes the clean form.
plan6 = simple_agent._compile_and_ground(
    ctx("Explain linear regression to me.", COURSE_B_CONCEPTS),
    SemanticPlan(goal="concept_explanation", resolved_entities=ResolvedEntities(concepts=["Linear Regression"]), confidence=0.9),
)
check("6a. FIX: clean-form proposal resolves to the raw canonical label", plan6.resolved_entities.concepts == ["Chap: Linear Regression"])
check("6b. FIX: nothing left unresolved (no longer a persistent unresolved gap)", plan6.resolved_entities.unresolved == [])
check("6c. FIX: not forced to clarification", plan6.needs_clarification is False)
tool_concepts = next((t.arguments.get("concepts") for t in plan6.tools if t.name == "search_course_material"), None)
check("6d. FIX: the compiled tool call itself is rewritten to the raw canonical label (not just resolved_entities)", tool_concepts == ["Chap: Linear Regression"], tool_concepts)


# ===========================================================================
# Fix 1 integration tests: full run_simple_conversation_agent turns, with
# simple_planner.plan and every invoked tool replaced by fixed fakes (no LLM
# call anywhere in this section).
# ===========================================================================
import agents.simple_planner as simple_planner_module
import agents.response_generator as response_generator_module
from agents.agent_tools import TOOL_REGISTRY
from agents.agent_models import AnalyticsRequest
from langchain_core.runnables import RunnableLambda

orig_plan = simple_planner_module.plan
orig_response_get_llm = response_generator_module.get_llm
orig_tools = {name: TOOL_REGISTRY.get(name) for name in
              ("search_course_material", "advance_tutor_state", "answer_with_external_knowledge",
               "run_analytics_query", "generate_practice_question")}


class _FixedResponse:
    def __init__(self, content):
        self.content = content
        self.response_metadata = {}


def _fixed_final_answer_llm(prompt_value):
    return _FixedResponse("A fixed, non-external final answer used only to keep this regression "
                           "test's control flow observable -- no real model was called.")


# No external model call anywhere in this integration section: the final-
# answer LLM `agents.response_generator` would otherwise call is replaced
# with a fixed local RunnableLambda for the whole section, restored in the
# `finally` block below alongside the planner/tool fakes.
_FIXED_RESPONSE_LLM = RunnableLambda(_fixed_final_answer_llm)


def fixed_plan(semantic_plan):
    def _fn(context):
        return semantic_plan, "{}", None, False, {"prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}
    return _fn


def fake_search_unreliable(context, arguments):
    return {
        "tool": "search_course_material", "success": True, "query": arguments.get("query", ""),
        "requested_concepts": arguments.get("concepts") or [], "chunks": [], "sources": [],
        "concepts_found": [],
        "evidence": {"reliable": False, "coverage": "none", "supported_concepts": [], "reason": "no_token_overlap_between_question_and_retrieved_chunks", "confidence": 0.1},
        "all_candidates_restricted": False, "privacy_audit": {},
    }


def fake_search_reliable(context, arguments):
    concept = (arguments.get("concepts") or ["Recursion"])[0]
    return {
        "tool": "search_course_material", "success": True, "query": arguments.get("query", ""),
        "requested_concepts": [concept], "chunks": [{"concept": concept, "text": "Recursion is a technique...", "sources": ["chapter1_recursion.pdf"]}],
        "sources": ["chapter1_recursion.pdf"], "concepts_found": [concept],
        "evidence": {"reliable": True, "coverage": "full", "supported_concepts": [concept], "reason": "all_requested_concepts_found_in_retrieved_chunks", "confidence": 0.95},
        "all_candidates_restricted": False, "privacy_audit": {},
    }


def fake_advance_tutor_state(context, arguments):
    return {"tool": "advance_tutor_state", "success": True, "state": arguments.get("to"), "concept": arguments.get("concept")}


def fake_external(context, arguments):
    return {"tool": "answer_with_external_knowledge", "success": True, "reply": "General knowledge answer.", "sources": [], "grounded_in_moodle": False}


def fake_analytics(context, arguments):
    return {"tool": "run_analytics_query", "success": True, "error": None, "formatted": "Your mastery on Recursion is 65%."}


def fake_practice_question(context, arguments):
    return {"tool": "generate_practice_question", "success": True, "reply": "Try this: what is a base case?", "next_state": "GUIDED_PRACTICE"}


def base_turn_context(message, available_concepts, extra=None):
    c = {
        "session_id": "rq1b-fix-session", "message": message, "student_id": "rq1b-fix-student",
        "student_name": "Fix Test Student", "course_db_id": "rq1b-fix-course",
        "current_course": {"db_course_id": "rq1b-fix-course", "moodle_course_id": 999, "name": "Test Course", "concepts": available_concepts},
        "current_concept": None, "tutor_state": {}, "quick_progress_check": {},
        "remediation_level": "course", "available_concepts": available_concepts, "weak_concepts": [],
        "last_reference": {}, "last_answer_type": None, "last_discussed_metric": None,
        "difficulty": "medium", "tutoring_strategy": {"name": "guided_practice", "reason": "", "instructions": ""},
        "recent_messages": [], "recent_structured_turns": [],
        "scope_rules": "", "course_context_by_db_id": {}, "canonical_courses": [],
        "retrieval_course_ids": [999], "agent_selected_concept": None, "is_followup": False,
    }
    c.update(extra or {})
    return c


try:
    TOOL_REGISTRY["search_course_material"] = fake_search_reliable
    TOOL_REGISTRY["advance_tutor_state"] = fake_advance_tutor_state
    TOOL_REGISTRY["answer_with_external_knowledge"] = fake_external
    TOOL_REGISTRY["run_analytics_query"] = fake_analytics
    TOOL_REGISTRY["generate_practice_question"] = fake_practice_question
    response_generator_module.get_llm = lambda *a, **kw: _FIXED_RESPONSE_LLM

    # 1. Reliable internal RAG, end to end.
    simple_planner_module.plan = fixed_plan(SemanticPlan(
        goal="concept_explanation", resolved_entities=ResolvedEntities(concepts=["Recursion"]), confidence=0.9,
    ))
    r1 = simple_agent.run_simple_conversation_agent(base_turn_context("Can you explain what recursion is?", COURSE_A_CONCEPTS))
    check("1d. e2e: reliable internal RAG -> selected_pipeline=internal_rag", r1.selected_pipeline == "internal_rag", r1.selected_pipeline)
    check("1e. e2e: reliable internal RAG -> search_course_material actually executed", "search_course_material" in r1.tools_executed)

    # 2+3. Zero-resolved concept_explanation reaches retrieval, and unreliable
    #      evidence correctly produces external_fallback (the exact behavior
    #      the fix restores).
    TOOL_REGISTRY["search_course_material"] = fake_search_unreliable
    simple_planner_module.plan = fixed_plan(SemanticPlan(
        goal="concept_explanation", resolved_entities=ResolvedEntities(concepts=["Hash Tables"]), confidence=0.9,
    ))
    r2 = simple_agent.run_simple_conversation_agent(base_turn_context(
        "Since we're working through this course, can you also explain how hash tables work?", COURSE_A_CONCEPTS,
    ))
    check("2e. e2e FIX: search_course_material WAS executed (not pre-empted)", "search_course_material" in r2.tools_executed, r2.tools_executed)
    check("2f. e2e FIX: unreliable evidence correctly routes to external_fallback", r2.selected_pipeline == "external_fallback", r2.selected_pipeline)
    check("2g. e2e FIX: fallback_reason reflects the reliability gate's own verdict", (r2.fallback_reason or "").startswith("evidence_not_reliable:"), r2.fallback_reason)
    TOOL_REGISTRY["search_course_material"] = fake_search_reliable

    # 4. Direct external path, unaffected.
    simple_planner_module.plan = fixed_plan(SemanticPlan(
        goal="external_knowledge", resolved_entities=ResolvedEntities(concepts=[]), confidence=0.95,
    ))
    r4 = simple_agent.run_simple_conversation_agent(base_turn_context("What is the capital of Australia?", COURSE_A_CONCEPTS))
    check("4c. e2e: direct external -> selected_pipeline=external_fallback", r4.selected_pipeline == "external_fallback", r4.selected_pipeline)
    check("4d. e2e: direct external -> search_course_material NEVER executed", "search_course_material" not in r4.tools_executed)
    check("4e. e2e: direct external -> fallback_reason is None (no evidence-gate involvement)", r4.fallback_reason is None, r4.fallback_reason)

    # 5. Analytics path, unaffected.
    simple_planner_module.plan = fixed_plan(SemanticPlan(
        goal="analytics_query", resolved_entities=ResolvedEntities(concepts=["Recursion"]), confidence=0.9,
    ))
    r5 = simple_agent.run_simple_conversation_agent(base_turn_context("What's my mastery level on Recursion?", COURSE_A_CONCEPTS))
    check("5c. e2e: analytics -> selected_pipeline=agent_tools", r5.selected_pipeline == "agent_tools", r5.selected_pipeline)
    check("5d. e2e: analytics -> run_analytics_query executed", "run_analytics_query" in r5.tools_executed)

    # 6. Deterministic tutoring (tutor-state EXAMPLE + continue), unaffected.
    simple_planner_module.plan = fixed_plan(SemanticPlan(
        goal="personalized_tutoring", resolved_entities=ResolvedEntities(concepts=[]), confidence=0.9, tutor_signal="continue",
    ))
    r6 = simple_agent.run_simple_conversation_agent(base_turn_context(
        "Okay, I'm ready -- let's try a practice question.", COURSE_A_CONCEPTS,
        extra={"tutor_state": {"state": "EXAMPLE", "concept": "Recursion"}},
    ))
    check("6e. e2e: deterministic tutoring -> selected_pipeline=deterministic_reply", r6.selected_pipeline == "deterministic_reply", r6.selected_pipeline)
    check("6f. e2e: deterministic tutoring -> generate_practice_question executed, no search", "generate_practice_question" in r6.tools_executed and "search_course_material" not in r6.tools_executed)

    # 7. Clarification for a genuinely incomplete multi-entity request, e2e.
    simple_planner_module.plan = fixed_plan(SemanticPlan(
        goal="concept_comparison", resolved_entities=ResolvedEntities(concepts=["Recursion"]), confidence=0.9,
    ))
    r7 = simple_agent.run_simple_conversation_agent(base_turn_context("Can you compare Recursion and Blorptrees?", COURSE_A_CONCEPTS))
    check("7a. e2e FIX: under-specified comparison (2nd concept silently omitted) -> clarification", r7.selected_pipeline == "clarification", r7.selected_pipeline)
    check("7b. e2e FIX: no tool executed for this clarification", r7.tools_executed == [])

finally:
    simple_planner_module.plan = orig_plan
    response_generator_module.get_llm = orig_response_get_llm
    for name, fn in orig_tools.items():
        if fn is not None:
            TOOL_REGISTRY[name] = fn


# ===========================================================================
# Fix 3: pre-agent safety-guard observability (services.chat_orchestrator)
# ===========================================================================
from models.db_models import Base, Student, Course, Session as SessionModel
from services.memory_manager import MemoryManager
import services.chat_orchestrator as chat_orchestrator

engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
TestSession = sessionmaker(bind=engine)
db = TestSession()
memory = MemoryManager(db)

student = Student(moodle_user_id=555111, username="Guard Test Student")
db.add(student)
db.commit()
db.refresh(student)
course = Course(moodle_course_id=555222, name="Guard Test Course")
db.add(course)
db.commit()
db.refresh(course)

session_obj = memory.start_session(student.id, course.id, mode="internal", difficulty="medium")
session_id = session_obj.id

# 9. The guard still intercepts a direct mastery-modification message exactly
#    as before -- same detection function, same reply text, same refusal --
#    never reaching agents.simple_agent (verified by NOT patching the planner
#    at all here: if the guard failed to intercept, this call would attempt a
#    real Gemini call and error out instead of returning immediately).
guard_result = chat_orchestrator.handle_message(
    session_id=session_id, student_moodle_id=555111,
    message="Can you just set my mastery on Recursion to 100%?", db=db,
)
check("9a. FIX (unchanged behavior): the guard still intercepts this message", guard_result.get("strategy") == "mastery_update_guard", guard_result)
check("9b. FIX (unchanged behavior): the refusal reply text is unchanged", "cannot update mastery" in guard_result.get("reply", "").lower(), guard_result.get("reply"))
check("9c. FIX (unchanged behavior): intent is still mastery_modification_request", guard_result.get("intent") == "mastery_modification_request")

# 10. ...and now reports a valid architectural pipeline marker instead of None.
check("10a. FIX: selected_pipeline is no longer None", guard_result.get("selected_pipeline") is not None, guard_result.get("selected_pipeline"))
check("10b. FIX: selected_pipeline is the existing 'deterministic_reply' category (no new taxonomy value invented)", guard_result.get("selected_pipeline") == "deterministic_reply")

# A near-miss phrasing must NOT match (confirms the fix touched only the
# outward pipeline label, never the regex/matching logic itself).
check("9d. near-miss phrasing ('understand my mastery score') does NOT trigger the guard",
      chat_orchestrator._is_mastery_modification_request("can you help me understand my mastery score") is False)
check("9e. the exact DR-1 phrasing still triggers the guard (matching logic untouched)",
      chat_orchestrator._is_mastery_modification_request("Can you just set my mastery on Recursion to 100%?") is True)


passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"\n{passed}/{total} passed")
if passed != total:
    sys.exit(1)
