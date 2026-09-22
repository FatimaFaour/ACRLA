"""Regression tests for the agents/agent_json.py::_compact_value depth-
truncation bug (RQ3 audit finding): a leaf scalar (chunk text, concept
name, source filename) nested 5+ levels deep inside a tool result was
replaced by a "<str:truncated-depth>" placeholder purely because of ITS
POSITION in the structure, not its own size -- silently stripping the
actual retrieved RAG text out of agents/response_generator.py's prompt to
Gemini for every internal_rag turn, while leaving the metadata that CLAIMS
grounding (evidence_reliable, coverage, source filenames -- built as
separate, shallower dicts) fully intact.

Fix (agents/agent_json.py::_compact_value): the depth>4 cutoff now applies
ONLY to containers (dict/list/tuple/set) -- leaf scalars (str/int/float/
bool/None) are returned as-is (subject only to the existing max_string
length cap) regardless of depth.

Covers exactly what the task asked for:
1. retrieved chunk text survives compact_json
2. concept metadata survives
3. source metadata survives
4. deeply nested CONTAINERS are still safely truncated (protection preserved)
5. generate_final_answer receives real retrieved evidence for an
   internal_rag turn (full integration, real function, capturing LLM)

Run from the `backend/` directory:
    python tests/test_compact_value_depth_fix.py
or from anywhere (this file resolves its own project root):
    python backend/tests/test_compact_value_depth_fix.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.runnables import RunnableLambda

results = []


def check(name, cond, detail=None):
    results.append((name, cond))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f" -> {detail}" if detail is not None else ""))


from agents.agent_json import compact_json, _compact_value
from agents.response_generator import _dedupe_tool_results

CHUNK_TEXT = "Linear regression fits a line y = mx + b to minimize squared error between predicted and actual values."
CONCEPT_NAME = "Linear Regression"
SOURCE_NAME = "ch3_linear_regression.pdf"

# The exact shape agents.response_generator builds from a real
# search_course_material tool observation.
REALISTIC_TOOL_RESULTS = [{
    "tool": "search_course_material", "arguments": {"concepts": [CONCEPT_NAME]},
    "result": {
        "tool": "search_course_material", "success": True, "query": "Explain linear regression",
        "requested_concepts": [CONCEPT_NAME],
        "chunks": [{"concept": CONCEPT_NAME, "text": CHUNK_TEXT, "sources": [SOURCE_NAME]}],
        "sources": [SOURCE_NAME], "concepts_found": [CONCEPT_NAME],
        "evidence": {"reliable": True, "coverage": "full", "supported_concepts": [CONCEPT_NAME], "reason": "ok", "confidence": 0.9},
    },
    "rejected": False,
}]

out = compact_json(_dedupe_tool_results(REALISTIC_TOOL_RESULTS), limit=5000)

# 1. Retrieved chunk text survives.
check("1. retrieved chunk text survives compact_json (not '<str:truncated-depth>')", CHUNK_TEXT in out, out)
check("1b. no literal truncated-depth placeholder leaked in place of the chunk text", '"text":"<str:truncated-depth>"' not in out)

# 2. Concept metadata survives (the chunk's own "concept" field, depth 5).
check("2. chunk-level concept metadata survives ('concept': 'Linear Regression' inside the chunk dict)", f'"concept":"{CONCEPT_NAME}"' in out, out)

# 3. Source metadata survives (the filename string itself, wherever it appears).
check("3. source filename string survives somewhere in the output", SOURCE_NAME in out, out)

# 4. Deeply nested CONTAINERS are still safely truncated -- protection preserved.
deeply_nested_list = {"a": {"b": {"c": {"d": {"e": {"f": ["still a container, should be capped"]}}}}}}
compacted_container = _compact_value(deeply_nested_list)
check("4a. a genuinely deep dict is still truncated at some level", "truncated-depth" in str(compacted_container), compacted_container)

deep_dict_with_leaf = {"a": {"b": {"c": {"d": {"e": "a plain leaf string sitting at depth 5"}}}}}
compacted_leaf = _compact_value(deep_dict_with_leaf)
# The dict CONTAINER at depth 5 (value of "d") is still capped, but note:
# this specific fixture nests the leaf as a dict VALUE at depth 5, so the
# leaf itself is reached via _compact_value(leaf_string, depth=5) --
# confirm it survives as the literal string, not a placeholder.
leaf_survived = "a plain leaf string sitting at depth 5" in str(compacted_leaf)
check("4b. a leaf string value nested at depth 5 survives (not replaced merely for its position)", leaf_survived, compacted_leaf)

# Regression: an actually oversized STRING is still length-capped (max_string),
# regardless of depth -- the fix only changed the DEPTH gate, not the
# pre-existing per-string size cap.
huge_string = "X" * 2000
capped = _compact_value(huge_string, depth=10, max_string=500)
check("4c. an oversized string is still length-capped at depth>4 (existing max_string protection unchanged)", len(capped) < 600 and capped.endswith("...(truncated)"), len(capped))

# A very wide list (many items) at shallow depth is still item-capped.
wide_list = list(range(50))
capped_list = _compact_value(wide_list, depth=0, max_list=8)
check("4d. an oversized list is still item-capped (existing max_list protection unchanged)", len(capped_list) <= 9 and any(isinstance(i, dict) and "_truncated_items" in i for i in capped_list), capped_list)

# A container that itself sits at depth > 4 is still truncated to the
# placeholder (containers are NOT exempted by this fix, only scalars are).
container_at_depth5 = _compact_value({"nested": "dict"}, depth=5)
check("4e. a CONTAINER (dict) landing at depth>4 is still replaced with the truncated-depth placeholder", container_at_depth5 == "<dict:truncated-depth>", container_at_depth5)
list_at_depth5 = _compact_value([1, 2, 3], depth=5)
check("4f. a CONTAINER (list) landing at depth>4 is still replaced with the truncated-depth placeholder", list_at_depth5 == "<list:truncated-depth>", list_at_depth5)

# Scalars at depth>4 explicitly verified NOT to be replaced (the exact
# fix requirement), for each scalar type.
check("5a. a str at depth 10 is returned as-is", _compact_value("plain string", depth=10) == "plain string")
check("5b. an int at depth 10 is returned as-is", _compact_value(42, depth=10) == 42)
check("5c. a float at depth 10 is returned as-is", _compact_value(0.9, depth=10) == 0.9)
check("5d. a bool at depth 10 is returned as-is", _compact_value(True, depth=10) is True)
check("5e. None at depth 10 is returned as-is", _compact_value(None, depth=10) is None)


# ---------------------------------------------------------------------------
# 6. Integration: generate_final_answer receives real retrieved evidence for
# an internal_rag turn -- captured via a real LangChain RunnableLambda so
# nothing about prompt construction is mocked away, only the network call.
# ---------------------------------------------------------------------------
import agents.response_generator as response_generator_module
from agents.agent_models import AgentBrainOutput, ResolvedEntities, EvidenceStatus


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


capture = _Capture()


def _fn(prompt_value):
    capture.prompt_value = prompt_value
    return _FakeResponse("Linear regression models the relationship between a target and predictors.")


capturing_llm = RunnableLambda(_fn)
orig_get_llm = response_generator_module.get_llm
response_generator_module.get_llm = lambda *a, **kw: capturing_llm

try:
    context = {
        "message": "Explain linear regression", "recent_structured_turns": [],
        "remediation_level": "course", "current_course": {"name": "Data Science"},
        "available_concepts": [CONCEPT_NAME], "scope_rules": "", "difficulty": "medium",
        "tutoring_strategy": {"name": "guided_practice", "reason": "", "instructions": ""},
        "tutor_state": {},
    }
    decision = AgentBrainOutput(
        goal="concept_explanation", resolved_entities=ResolvedEntities(concepts=[CONCEPT_NAME]),
        evidence_status=EvidenceStatus(sufficient=True, missing=[], relevant_observations=["search_course_material"], ignored_observations=[]),
        answer_basis="rag", confidence=0.9,
    )
    evidence = {"reliable": True, "coverage": "full", "reason": "ok", "supported_concepts": [CONCEPT_NAME]}
    reply = response_generator_module.generate_final_answer(
        context, decision, REALISTIC_TOOL_RESULTS, "internal_rag", evidence, sources=[SOURCE_NAME],
    )
finally:
    response_generator_module.get_llm = orig_get_llm

sent_prompt_text = capture.prompt_text()
check("6a. generate_final_answer's actual outbound prompt contains the real retrieved chunk text", CHUNK_TEXT in sent_prompt_text, sent_prompt_text[-500:])
check("6b. generate_final_answer's outbound prompt still contains the concept name", CONCEPT_NAME in sent_prompt_text)
check("6c. generate_final_answer's outbound prompt still contains the source filename", SOURCE_NAME in sent_prompt_text)
check("6d. generate_final_answer still returned a reply (turn not broken by the fix)", bool(reply), reply)


print()
passed = sum(1 for _, ok in results if ok)
print(f"{passed}/{len(results)} passed")
assert passed == len(results)
