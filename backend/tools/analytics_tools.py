"""Analytics query planning + execution tools.

This module is purely computational: it operates on an already-resolved list
of canonical `{db_course_id, moodle_course_id, name, concepts, ...}` course
dicts (see `chat_orchestrator._canonical_synced_courses`) and stored mastery
data. It has no filesystem/Moodle-manifest or vectorstore dependency of its
own, so it can be called both from the legacy analytics pipeline and as a
single agent-facing tool (`run_analytics_query_tool`).

Nothing here writes mastery; it only ranks/filters/reports it.
"""

from __future__ import annotations

from typing import Any

from langchain.prompts import ChatPromptTemplate

from agents.agent_json import compact_json, extract_message_text, parse_json_object
from agents.debug_log import vprint
from agents.llm_errors import invoke_with_json_mode_retry, log_llm_provider_error
from services.memory_manager import MemoryManager
from services.llm_factory import get_json_llm
from tools.text_utils import normalize_key


ANALYTICS_PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You convert one ACRLA analytics question into a JSON query plan.
Return strict JSON only. Do not answer the student. Do not invent data.

Supported operations: list, summarize, compare, rank, get_value, recommend.
Supported scopes: current_chapter, current_course, all_courses.
Supported entities: concept, course, overall.
Supported metrics: current_mastery, initial_moodle_mastery, course_average, overall_average.

Rules:
- Include only what the student requested or clearly implied.
- A general "how am I doing", "what is my mastery/progress", or any request for
  ONE single number across everything (not per-chapter) means entity=overall,
  metrics=["overall_average"], operation="get_value" -- not a per-chapter list.
- "all chapter scores", "every topic", "chapter by chapter", or "all concept mastery values" means list concept current_mastery grouped by course.
- "course averages only", or "the average for each course", means entity=course, metrics=["course_average"], operation="list", and no weakest/strongest.
- Requests for Weak-band concepts/topics mean include_weakest=true and rank ascending by current_mastery.
- Requests for low/lowest/weakest concepts, or "what needs the most work", without asking for the Weak band mean rank ascending by current_mastery.
- Requests for Strong-band concepts/topics mean include_strongest=true and rank descending by current_mastery.
- Requests for high/highest/strongest/best-performing concepts without asking for the Strong band mean rank descending by current_mastery -- do not default to ascending order for this case.
- A request to rank/order all topics (not just weakest or strongest) still needs operation="rank" with sort_order matching the requested direction.
- Naming two or more concepts/courses to compare means operation="compare" with filters.concepts set to exactly those names.
- "below 50%" means filters={{"current_mastery_lt": 0.5}}.
- If the request is not an analytics query, set confidence below 0.75.

Return exactly:
{{
  "domain": "mastery_analytics",
  "operation": "list",
  "scope": "all_courses",
  "entity": "concept",
  "metrics": ["current_mastery"],
  "filters": {{}},
  "group_by": ["course"],
  "sort_by": null,
  "sort_order": null,
  "limit": null,
  "include_course_average": false,
  "include_overall_average": false,
  "include_weakest": false,
  "include_strongest": false,
  "confidence": 0.0
}}"""),
    ("human", """CONTEXT:
Remediation level: {remediation_level}
Current course: {current_course}
Current concept: {current_concept}
Available courses: {available_courses}
Last reference: {last_reference}

USER ANALYTICS QUESTION:
{message}"""),
])


def plan_analytics_query(
    message: str,
    remediation_level: str,
    current_course: str,
    current_concept: str | None,
    available_courses: list[dict],
    last_reference: dict | None,
) -> dict:
    """Ask the LLM for an analytics query plan, never for the final answer."""
    llm = get_json_llm(temperature=0, max_tokens=360)
    prompt_values = {
        "message": message,
        "remediation_level": remediation_level,
        "current_course": current_course,
        "current_concept": current_concept or "not set",
        "available_courses": compact_json(available_courses, limit=1800),
        "last_reference": compact_json(last_reference or {}, limit=900),
    }
    try:
        response, _json_mode_fallback_used = invoke_with_json_mode_retry(
            ANALYTICS_PLAN_PROMPT, prompt_values, llm=llm, temperature=0, max_tokens=360,
            stage="analytics_planner",
        )
    except Exception as exc:
        # The call to the LLM provider itself never completed here (or the
        # one non-JSON-mode retry for a json-mode error also failed) -- kept
        # separate from the JSON-parse except block in
        # _parse_analytics_plan_json so a provider outage is never
        # misreported as "the model returned bad JSON".
        log_llm_provider_error(
            stage="analytics_planner", exc=exc, prompt_values=prompt_values,
            json_mode_enabled=True, llm=llm, response_length=0,
            final_fallback_reason="analytics_plan_provider_error",
        )
        plan = _empty_analytics_plan(confidence=0.0)
    else:
        try:
            plan = _parse_analytics_plan_json(extract_message_text(response))
        except Exception as exc:
            print(f"[ACRLA] analytics_plan_failed error={exc}")
            plan = _empty_analytics_plan(confidence=0.0)
    plan = _normalize_analytics_plan(plan)
    print(
        "[ACRLA] analytics_plan "
        f"analytics_plan={plan} "
        f"analytics_plan_confidence={plan.get('confidence', 0.0):.2f} "
        f"analytics_operation={plan.get('operation')} "
        f"analytics_scope={plan.get('scope')} "
        f"analytics_entity={plan.get('entity')} "
        f"analytics_metrics={plan.get('metrics')} "
        f"analytics_filters={plan.get('filters')}"
    )
    return plan


def _parse_analytics_plan_json(raw_response) -> dict:
    raw = str(raw_response or "").strip()
    try:
        return parse_json_object(raw)
    except Exception as exc:
        print(f"[ACRLA] analytics_plan_parse_failed error={exc} raw_output_len={len(raw)} raw_output_preview={raw[:240]!r}")
        raise


def _empty_analytics_plan(confidence: float = 0.0) -> dict:
    return {
        "domain": "mastery_analytics",
        "operation": "list",
        "scope": "current_course",
        "entity": "concept",
        "metrics": ["current_mastery"],
        "filters": {},
        "group_by": [],
        "sort_by": None,
        "sort_order": None,
        "limit": None,
        "include_course_average": False,
        "include_overall_average": False,
        "include_weakest": False,
        "include_strongest": False,
        "confidence": confidence,
    }


def _normalize_analytics_plan(plan: dict) -> dict:
    allowed_operations = {"list", "summarize", "compare", "rank", "get_value", "recommend"}
    allowed_scopes = {"current_chapter", "current_course", "all_courses"}
    allowed_entities = {"concept", "course", "overall"}
    allowed_metrics = {"current_mastery", "initial_moodle_mastery", "course_average", "overall_average"}
    normalized = _empty_analytics_plan()
    if isinstance(plan, dict):
        normalized.update(plan)
    normalized["operation"] = normalized.get("operation") if normalized.get("operation") in allowed_operations else "list"
    normalized["scope"] = normalized.get("scope") if normalized.get("scope") in allowed_scopes else "current_course"
    normalized["entity"] = normalized.get("entity") if normalized.get("entity") in allowed_entities else "concept"
    metrics = [metric for metric in (normalized.get("metrics") or []) if metric in allowed_metrics]
    normalized["metrics"] = metrics or ["current_mastery"]
    normalized["filters"] = normalized.get("filters") if isinstance(normalized.get("filters"), dict) else {}
    normalized["group_by"] = normalized.get("group_by") if isinstance(normalized.get("group_by"), list) else []
    normalized["include_course_average"] = _truthy_plan_flag(normalized.get("include_course_average"))
    normalized["include_overall_average"] = _truthy_plan_flag(normalized.get("include_overall_average"))
    normalized["include_weakest"] = _truthy_plan_flag(normalized.get("include_weakest"))
    normalized["include_strongest"] = _truthy_plan_flag(normalized.get("include_strongest"))
    if normalized.get("include_course_average"):
        normalized["entity"] = "course"
        if "course_average" not in normalized["metrics"]:
            normalized["metrics"] = ["course_average"]
    if normalized.get("include_overall_average"):
        normalized["entity"] = "overall"
        if "overall_average" not in normalized["metrics"]:
            normalized["metrics"] = ["overall_average"]
    try:
        if normalized.get("limit") is not None:
            normalized["limit"] = int(normalized.get("limit"))
    except (TypeError, ValueError):
        normalized["limit"] = None
    try:
        normalized["confidence"] = max(0.0, min(1.0, float(normalized.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        normalized["confidence"] = 0.0
    if normalized.get("include_weakest"):
        normalized["operation"] = "rank"
        normalized["sort_by"] = "current_mastery"
        normalized["sort_order"] = "asc"
    if normalized.get("include_strongest"):
        normalized["operation"] = "rank"
        normalized["sort_by"] = "current_mastery"
        normalized["sort_order"] = "desc"
    return normalized


def _truthy_plan_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def execute_analytics_query(
    plan: dict,
    student_id: str,
    memory: MemoryManager,
    courses: list[dict],
    current_course_id: str,
) -> dict:
    """Execute a planned analytics query using stored mastery data only.

    `courses` must already be the canonical, deduplicated course/concept list
    (chat_orchestrator resolves Moodle sync state and passes it in) -- this
    function does not touch the filesystem or vectorstore itself.
    """
    if plan.get("scope") == "current_course":
        courses = [course for course in courses if str(course["db_course_id"]) == str(current_course_id)] or courses[:1]

    concept_rows = []
    for course in courses:
        for concept in course["concepts"]:
            concept_rows.append({
                "course_db_id": course["db_course_id"],
                "course_id": course["moodle_course_id"],
                "course_name": course["name"],
                "concept": concept,
                "current_mastery": _merged_course_mastery(student_id, memory, course, concept),
            })

    rows = _filter_analytics_rows(concept_rows, plan)
    rows = _apply_mastery_band_flags(rows, plan)
    if plan.get("entity") == "course" or "course_average" in plan.get("metrics", []):
        items = _course_average_items(rows)
    elif plan.get("entity") == "overall" or "overall_average" in plan.get("metrics", []):
        overall_item = {"overall_average": _average([row["current_mastery"] for row in rows])}
        # Scoped to one course: label the result with that course's name so
        # the answer reads as "this course's average", never as an
        # across-everything figure -- see format_analytics_result.
        if plan.get("scope") == "current_course" and len(courses) == 1:
            overall_item["course_name"] = courses[0]["name"]
        items = [overall_item]
    else:
        items = rows

    if plan.get("operation") in {"rank", "recommend"} or plan.get("sort_by"):
        reverse = str(plan.get("sort_order") or "").lower() == "desc"
        sort_key = plan.get("sort_by") or "current_mastery"
        items = sorted(items, key=lambda item: item.get(sort_key, item.get("current_mastery", 0.0)) or 0.0, reverse=reverse)
    limit = plan.get("limit")
    if isinstance(limit, int) and limit > 0:
        items = items[:limit]
    validation_error = _validate_analytics_result(plan, items, courses)
    vprint(f"[ACRLA] final_analytics_items={items}")
    if validation_error:
        print(f"[ACRLA] analytics_validation_error={validation_error}")
    return {"items": items, "courses": courses, "error": validation_error}


def _validate_analytics_result(plan: dict, items: list[dict], courses: list[dict]) -> str | None:
    """Hard guard against duplicate courses, wrong compare outputs, and a
    result shape that does not match the requested operation/entity -- e.g.
    an "overall mastery" request must come back as one overall value, not a
    per-chapter list; a "course averages" request must come back as courses,
    not concepts; a ranking must actually be sorted in the requested
    direction and respect any limit.
    """
    if plan.get("entity") == "overall" or "overall_average" in plan.get("metrics", []):
        if len(items) != 1 or "overall_average" not in (items[0] if items else {}):
            return f"overall mastery request did not return exactly one overall_average value: {items}"
    elif plan.get("entity") == "course" or "course_average" in plan.get("metrics", []):
        wrong_shape = [item for item in items if "course_average" not in item or "concept" in item]
        if wrong_shape:
            return f"course-average request returned concept-shaped rows instead of course averages: {wrong_shape}"
    elif plan.get("operation") not in {"compare"}:
        wrong_shape = [item for item in items if "concept" not in item and "current_mastery" not in item]
        if wrong_shape:
            return f"concept-level request returned rows without a concept: {wrong_shape}"

    if plan.get("operation") in {"rank", "recommend"} or plan.get("sort_by"):
        sort_key = plan.get("sort_by") or "current_mastery"
        values = [item.get(sort_key) for item in items if item.get(sort_key) is not None]
        if len(values) > 1:
            descending = str(plan.get("sort_order") or "").lower() == "desc"
            expected = sorted(values, reverse=descending)
            if values != expected:
                return f"ranking did not respect requested direction (sort_order={plan.get('sort_order')}): {values}"
        limit = plan.get("limit")
        if isinstance(limit, int) and limit > 0 and len(items) > limit:
            return f"ranking returned more rows than the requested limit={limit}: {len(items)} rows"

    duplicate_keys = {}
    for course in courses:
        key = (normalize_key(course.get("name")), _concept_set_signature(course.get("concepts") or []))
        duplicate_keys.setdefault(key, []).append(course)
    duplicates = [values for values in duplicate_keys.values() if len(values) > 1]
    if duplicates:
        return f"duplicate canonical courses remained: {duplicates}"

    concept_course_keys = [
        (normalize_key(item.get("course_name")), normalize_key(item.get("concept")))
        for item in items
        if item.get("concept")
    ]
    duplicate_concept_rows = [
        key for key in sorted(set(concept_course_keys))
        if key[0] and key[1] and concept_course_keys.count(key) > 1
    ]
    if duplicate_concept_rows:
        return f"duplicate analytics concept rows remained: {duplicate_concept_rows}"

    if plan.get("include_weakest"):
        not_weak = [
            item for item in items
            if item.get("current_mastery") is not None and float(item.get("current_mastery") or 0.0) >= 0.5
        ]
        if not_weak:
            return f"weak-only analytics returned non-weak rows: {not_weak}"

    if plan.get("include_strongest"):
        not_strong = [
            item for item in items
            if item.get("current_mastery") is not None and float(item.get("current_mastery") or 0.0) < 0.8
        ]
        if not_strong:
            return f"strong-only analytics returned non-strong rows: {not_strong}"

    requested = (plan.get("filters") or {}).get("concepts") or []
    if plan.get("operation") == "compare" and requested:
        requested_keys = [normalize_key(concept) for concept in requested]
        item_keys = [normalize_key(item.get("concept")) for item in items]
        missing = [requested[index] for index, key in enumerate(requested_keys) if key not in item_keys]
        extras = [item.get("concept") for item in items if normalize_key(item.get("concept")) not in requested_keys]
        duplicate_items = [
            key for key in sorted(set(item_keys))
            if key and item_keys.count(key) > 1
        ]
        if missing or extras or duplicate_items:
            return (
                "compare result mismatch "
                f"requested={requested} missing={missing} extras={extras} "
                f"duplicate_items={duplicate_items}"
            )
    return None


def _concept_set_signature(concepts: list[str]) -> str:
    return ",".join(sorted(normalize_key(concept) for concept in concepts or []))


def _merged_course_mastery(student_id: str, memory: MemoryManager, course: dict, concept: str) -> float:
    values = []
    for db_course_id in course.get("merged_db_course_ids") or [course["db_course_id"]]:
        value = memory.get_mastery(student_id, db_course_id, concept)
        if value is not None:
            values.append(value)
    return max(values) if values else 0.0


def _filter_analytics_rows(rows: list[dict], plan: dict) -> list[dict]:
    filters = plan.get("filters") or {}
    result = list(rows)
    concept_filter = filters.get("concept") or filters.get("concepts")
    if concept_filter:
        wanted = {normalize_key(item) for item in (concept_filter if isinstance(concept_filter, list) else [concept_filter])}
        result = [row for row in result if normalize_key(row.get("concept")) in wanted]
    lt = filters.get("current_mastery_lt") or filters.get("mastery_lt") or filters.get("below")
    if lt is not None:
        try:
            threshold = float(lt)
            threshold = threshold / 100 if threshold > 1 else threshold
            result = [row for row in result if (row.get("current_mastery") or 0.0) < threshold]
        except (TypeError, ValueError):
            pass
    return result


def _apply_mastery_band_flags(rows: list[dict], plan: dict) -> list[dict]:
    """Apply semantic mastery-band requests from the analytics plan.

    The planner's `include_weakest`/`include_strongest` flags are already a
    semantic decision, so the executor can use the product's canonical bands
    directly without inspecting the student's wording. Weak means the actual
    UI/policy Weak band (<50%), not merely "sort low to high"; Strong means
    the Strong band (>=80%).
    """
    result = list(rows)
    if plan.get("include_weakest"):
        result = [row for row in result if float(row.get("current_mastery") or 0.0) < 0.5]
    if plan.get("include_strongest"):
        result = [row for row in result if float(row.get("current_mastery") or 0.0) >= 0.8]
    return result


def _course_average_items(rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["course_id"], []).append(row)
    return [
        {
            "course_id": course_id,
            "course_name": values[0]["course_name"],
            "course_average": _average([item["current_mastery"] for item in values]),
        }
        for course_id, values in grouped.items()
    ]


def _average(values: list[float]) -> float:
    clean = [float(value or 0.0) for value in values]
    return sum(clean) / len(clean) if clean else 0.0


def format_analytics_result(plan: dict, result: dict) -> str:
    """Format analytics output according to the executed plan."""
    if result.get("error"):
        return f"I could not safely answer that analytics request: {result['error']}"
    items = result.get("items", [])
    if not items:
        return "I do not have matching mastery data for that analytics request."
    if plan.get("entity") == "course" or "course_average" in plan.get("metrics", []):
        lines = [f"{item['course_name']}: {_format_percent(item['course_average'])}" for item in items]
        return "Course mastery averages:\n" + "\n".join(lines)
    if plan.get("entity") == "overall" or "overall_average" in plan.get("metrics", []):
        course_name = items[0].get("course_name")
        if course_name:
            return f"Mastery average for {course_name}: {_format_percent(items[0].get('overall_average', 0.0))}"
        return f"Overall mastery average: {_format_percent(items[0].get('overall_average', 0.0))}"
    if plan.get("operation") == "compare":
        lines = [
            f"{item['concept']} - {item['course_name']} - {_format_percent(item['current_mastery'])}"
            for item in items
        ]
        if len(items) == 2:
            first, second = items
            diff_points = abs((second["current_mastery"] - first["current_mastery"]) * 100)
            higher = second if second["current_mastery"] >= first["current_mastery"] else first
            lines.append(f"Difference: {higher['concept']} is {_format_points(diff_points)} percentage points higher.")
        return "\n".join(lines)
    if plan.get("operation") in {"rank", "recommend"}:
        label = "Lowest chapter mastery scores:" if str(plan.get("sort_order")).lower() != "desc" else "Highest chapter mastery scores:"
        return label + "\n" + "\n".join(
            f"{index}. {item['concept']} - {item['course_name']} - {_format_percent(item['current_mastery'])}"
            for index, item in enumerate(items, start=1)
        )
    if "course" in (plan.get("group_by") or []) or plan.get("scope") == "all_courses":
        grouped: dict[int, list[dict]] = {}
        for item in items:
            grouped.setdefault(item["course_id"], []).append(item)
        sections = ["All chapter mastery scores:"]
        for course_id, values in grouped.items():
            sections.append("")
            sections.append(values[0]["course_name"])
            sections.extend(
                f"{index}. {item['concept']} - {_format_percent(item['current_mastery'])}"
                for index, item in enumerate(values, start=1)
            )
        return "\n".join(sections)
    return "Chapter mastery scores:\n" + "\n".join(
        f"{index}. {item['concept']} - {item['course_name']} - {_format_percent(item['current_mastery'])}"
        for index, item in enumerate(items, start=1)
    )


def _format_percent(value: float) -> str:
    percent = float(value or 0.0) * 100
    if percent.is_integer():
        return f"{percent:.0f}%"
    one_decimal = round(percent, 1)
    if abs(percent - one_decimal) < 0.01:
        return f"{one_decimal:.1f}%"
    return f"{percent:.2f}%"


def _format_points(value: float) -> str:
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"


def _analytics_request_to_plan(request: dict[str, Any]) -> dict[str, Any]:
    """Deterministically convert the agent brain's structured
    `AnalyticsRequest` (agents.agent_models.AnalyticsRequest, created once by
    the main agent from the user's own goal) into the internal plan shape
    `execute_analytics_query` expects -- no LLM call, no re-inference. Routed
    through `_normalize_analytics_plan` so the include_weakest/
    include_strongest/include_course_average/include_overall_average
    consistency rules stay in exactly one place.
    """
    entity = request.get("entity") or "concept"
    metric = request.get("metric") or "current_mastery"
    threshold_band = request.get("threshold_band")
    direction = request.get("direction")
    compare_entities = [c for c in (request.get("compare_entities") or []) if isinstance(c, str) and c.strip()]
    raw = {
        "operation": "compare" if compare_entities else (request.get("operation") or "list"),
        "scope": request.get("scope") or "current_course",
        "entity": entity,
        "metrics": [metric],
        "filters": {"concepts": compare_entities} if compare_entities else {},
        "group_by": [request["group_by"]] if request.get("group_by") else [],
        "sort_by": "current_mastery" if direction else None,
        "sort_order": {"lowest": "asc", "highest": "desc"}.get(direction),
        "limit": request.get("limit"),
        "include_course_average": entity == "course",
        "include_overall_average": entity == "overall",
        "include_weakest": threshold_band == "weak",
        "include_strongest": threshold_band == "strong",
        "confidence": request.get("confidence") or 1.0,
    }
    return _normalize_analytics_plan(raw)


def run_analytics_query_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Agent-facing tool: plan + execute + format one analytics question.

    Backs "what is my weakest concept?" style questions when the agent (not
    the legacy analytics_only pipeline) is handling the turn. `canonical_courses`
    must be preloaded onto the context by chat_orchestrator (it already
    resolves Moodle sync state once per turn for the legacy path).

    A structured `analytics_request` (in `arguments` or persisted on
    `context` by agents.conversation_agent) is always preferred over
    `plan_analytics_query`'s own LLM-based inference -- it was already
    classified once, semantically, by the main agent brain from the user's
    actual goal, so re-deriving it from raw text again here would risk a
    different (and potentially less-informed) interpretation each call. The
    LLM planner only runs at all when no structured request exists yet (e.g.
    called from the legacy pipeline), which also means "if the analytics LLM
    planner fails" cannot happen when a structured request is available.
    """
    memory: MemoryManager = context["memory"]
    courses = context.get("canonical_courses") or []
    structured_request = arguments.get("analytics_request") or context.get("analytics_request")
    if structured_request:
        plan = _analytics_request_to_plan(structured_request)
        plan_source = "structured_analytics_request"
        vprint(f"[ACRLA] analytics_plan_source=structured_analytics_request analytics_request={structured_request} analytics_plan={plan}")
    else:
        plan = plan_analytics_query(
            message=arguments.get("query") or context.get("message", ""),
            remediation_level=context.get("remediation_level", "chapter"),
            current_course=(context.get("current_course") or {}).get("name") or "current Moodle course",
            current_concept=context.get("current_concept"),
            available_courses=courses,
            last_reference=context.get("last_reference"),
        )
        plan_source = "message_inferred"
    result = execute_analytics_query(
        plan=plan,
        student_id=context["student_id"],
        memory=memory,
        courses=courses,
        current_course_id=context.get("course_db_id"),
    )
    return {
        "plan": plan,
        "plan_source": plan_source,
        "items": result.get("items", []),
        "error": result.get("error"),
        "formatted": format_analytics_result(plan, result),
    }
