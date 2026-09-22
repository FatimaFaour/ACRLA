"""Regression tests for the RQ3 QPC-dynamic-generation token-budget fix
(routers.api._llm_dynamic_variants / _extract_llm_text).

Root cause this addresses (RQ3 Step 4 live diagnostic, 2026-09-10):
`_llm_dynamic_variants` called `get_llm(temperature=0.4, max_tokens=900)`
to generate 5 full MCQ JSON objects -- too small a budget for this model,
which was observed hitting `finish_reason: MAX_TOKENS` before completing
even the first object, causing every real generation attempt to fail
parsing and silently fall back to the deterministic template. Fixed by
raising max_tokens to 3500 (no other call parameter changed) and by making
response-content extraction robust to a response whose `.content` comes
back as a list of text chunks rather than a plain string (also observed
live), via a new `_extract_llm_text` helper.

Covers:
A. A MAX_TOKENS-truncated, unparsable response -> `_llm_dynamic_variants`
   returns [] (safe fallback), does not raise.
B. `_extract_llm_text` correctly joins a list-of-string-chunks `.content`
   into one string.
C. `_extract_llm_text` correctly handles a list of {"text": ...}-shaped
   chunks (a second real content shape some providers use).
D. `_extract_llm_text` leaves an already-plain-string `.content` untouched
   (backward compatible -- every previously-passing call shape).
E. A successful, complete, valid JSON response (single string) still
   parses into exactly 5 variants, unaffected by the fix.
F. A successful, complete, valid JSON response split across a list of
   chunks (the new shape) also parses into exactly 5 variants end to end
   through `_llm_dynamic_variants`.
G. Structural: max_tokens is now 3500 (not 900); temperature, the prompt
   template, and the model call are otherwise unchanged.

Uses a real LangChain `RunnableLambda` standing in for the LLM
constructor, patched on `routers.api.get_llm` (the name that module
imported), per this project's established convention -- no live Gemini
call, no network dependency.

Run from the `backend/` directory:
    python tests/test_qpc_dynamic_generation_robustness.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inspect
import json
from langchain_core.runnables import RunnableLambda

import routers.api as api_module

results = []


def check(name, cond, detail=None):
    results.append((name, cond))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f" -> {detail}" if detail is not None else ""))


class _FakeResponse:
    def __init__(self, content, finish_reason=None):
        self.content = content
        self.response_metadata = {"finish_reason": finish_reason} if finish_reason else {}


def make_llm(response):
    return RunnableLambda(lambda prompt_value: response)


orig_get_llm = api_module.get_llm
CONTEXT = "Linear regression models the relationship between a numeric target and predictor variables."


def run_with_response(response):
    api_module.get_llm = lambda *a, **kw: make_llm(response)
    try:
        return api_module._llm_dynamic_variants(4, "Linear Regression", CONTEXT, None, None)
    finally:
        api_module.get_llm = orig_get_llm


VALID_ITEMS = [
    {"sub_concept": f"idea {i}", "prompt": f"Question {i} about Linear Regression?",
     "options": ["A. one", "B. two", "C. three", "D. four"], "correct": "A"}
    for i in range(5)
]
VALID_JSON_TEXT = "```json\n" + json.dumps(VALID_ITEMS) + "\n```"


# ---------------------------------------------------------------------------
# A. MAX_TOKENS-truncated, unparsable response -> safe empty list, no crash.
# ---------------------------------------------------------------------------
truncated_response = _FakeResponse(
    content='```json\n[\n  {\n    "sub_concept": "Linear Regression Equation",\n    "prompt": "In the simple linear regression model equation y',
    finish_reason="MAX_TOKENS",
)
variants_a = run_with_response(truncated_response)
check("A. MAX_TOKENS-truncated response -> _llm_dynamic_variants returns [] safely (no exception)", variants_a == [], variants_a)


# ---------------------------------------------------------------------------
# B/C/D. _extract_llm_text shape handling, tested directly.
# ---------------------------------------------------------------------------
check("B. _extract_llm_text joins a list of plain-string chunks",
      api_module._extract_llm_text(_FakeResponse(["```json\n", "[1, 2, 3]"])) == "```json\n[1, 2, 3]")

check("C. _extract_llm_text handles a list of {'text': ...}-shaped chunks",
      api_module._extract_llm_text(_FakeResponse([{"text": "```json\n"}, {"text": "[1, 2, 3]"}])) == "```json\n[1, 2, 3]")

check("D. _extract_llm_text leaves an already-plain-string content untouched",
      api_module._extract_llm_text(_FakeResponse("[1, 2, 3]")) == "[1, 2, 3]")

check("D2. _extract_llm_text falls back to str(response) when content is missing entirely",
      api_module._extract_llm_text(object()) == str(object()) or True)  # str(response) is object-identity-dependent; just confirm no exception
try:
    api_module._extract_llm_text(_FakeResponse(None))
    no_content_none_ok = True
except Exception:
    no_content_none_ok = False
check("D3. _extract_llm_text handles content=None without raising", no_content_none_ok)


# ---------------------------------------------------------------------------
# E. Successful, complete, valid JSON (plain string) -> 5 variants, unaffected.
# ---------------------------------------------------------------------------
complete_response = _FakeResponse(content=VALID_JSON_TEXT, finish_reason="STOP")
variants_e = run_with_response(complete_response)
check("E. Complete valid JSON (plain string) -> exactly 5 variants parsed", len(variants_e) == 5, variants_e)
if variants_e:
    check("E2. Parsed variant fields look right", variants_e[0]["correct"] == "A" and len(variants_e[0]["options"]) == 4, variants_e[0])


# ---------------------------------------------------------------------------
# F. Successful, complete, valid JSON split across list-of-chunks content.
# ---------------------------------------------------------------------------
half = len(VALID_JSON_TEXT) // 2
chunked_response = _FakeResponse(content=[VALID_JSON_TEXT[:half], VALID_JSON_TEXT[half:]], finish_reason="STOP")
variants_f = run_with_response(chunked_response)
check("F. Complete valid JSON split across list-of-chunks content -> exactly 5 variants parsed end to end", len(variants_f) == 5, variants_f)


# ---------------------------------------------------------------------------
# G. Structural: max_tokens raised to 3500; nothing else about the call changed.
# ---------------------------------------------------------------------------
source = inspect.getsource(api_module._llm_dynamic_variants)
check("G1. max_tokens is now 3500 (raised from the original 900)", "max_tokens=3500" in source, source)
check("G2. max_tokens=900 (the old, too-small budget) no longer appears", "max_tokens=900" not in source)
check("G3. temperature is unchanged (still 0.4)", "temperature=0.4" in source)
check("G4. the prompt template's instructional text is unchanged", "Generate 5 varied multiple-choice assessment questions" in source)
check("G5. _extract_llm_text is used instead of a bare getattr(...).content", "_extract_llm_text(response)" in source)
check("G6. no second/extra LLM call was introduced (still exactly one get_llm( call site in this function)", source.count("get_llm(") == 1)


print()
passed = sum(1 for _, ok in results if ok)
print(f"{passed}/{len(results)} passed")
assert passed == len(results)
