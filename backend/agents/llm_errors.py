"""Structured classification + logging for raw LLM-provider call failures.

This is deliberately separate from JSON-parse-error and pydantic-schema-error
handling (see agents.agent_json.AgentJSONError / agents.agent_brain's own
except blocks): those happen *after* a response was already received from the
provider. Everything here is about the call itself never completing --
rate limits, auth failures, timeouts, connection errors, provider 5xxs,
context-too-large, json-mode-unsupported, and model-unavailable errors -- so
a provider outage is never silently reported as "the model returned bad
JSON" or "the final answer was empty" when the real cause is that the call
never returned anything at all.

Never logs: API keys, full prompts, student profiles, or retrieved chunks --
only sizes/counts and the provider's own error code/message/status.
"""

from __future__ import annotations

from typing import Any

_RATE_LIMIT_NAMES = {"RateLimitError", "ResourceExhausted", "TooManyRequests"}
_AUTH_NAMES = {"AuthenticationError", "PermissionDeniedError", "Unauthenticated", "Unauthorized", "PermissionDenied"}
_TIMEOUT_NAMES = {"APITimeoutError", "TimeoutError", "ReadTimeout", "ConnectTimeout", "TimeoutException", "DeadlineExceeded"}
_CONNECTION_NAMES = {"APIConnectionError", "ConnectionError", "ConnectError", "NetworkError"}
_SERVER_ERROR_NAMES = {"InternalServerError", "ServiceUnavailableError", "ServiceUnavailable"}
_NOT_FOUND_NAMES = {"NotFoundError", "NotFound"}
_BAD_REQUEST_NAMES = {"BadRequestError", "UnprocessableEntityError", "InvalidRequestError", "InvalidArgument", "BadRequest"}
# ChatGoogleGenerativeAIError: langchain_google_genai's own wrapper for a
# response it could not use (unsupported message/role shape, blocked/empty
# content) -- semantically "the provider gave back something we could not
# use", the same bucket APIResponseValidationError already covers.
_RESPONSE_VALIDATION_NAMES = {"APIResponseValidationError", "ChatGoogleGenerativeAIError"}

_CONTEXT_LENGTH_MARKERS = (
    "context_length_exceeded", "maximum context length", "context window",
    "too many tokens", "reduce the length", "prompt is too long",
)
_JSON_MODE_MARKERS = (
    "response_format", "json_object", "json mode", "json_schema", "structured output",
    "failed to validate json",
)
# Exact provider error codes known to mean "JSON mode itself rejected this
# completion" -- checked before the fuzzy message markers above since a code
# is a precise, structural signal (e.g. Groq's json_validate_failed), not
# something to fuzzy-match on.
_JSON_MODE_ERROR_CODES = {"json_validate_failed"}
_MODEL_UNAVAILABLE_MARKERS = ("model_not_found", "does not exist", "decommissioned", "has been deprecated", "unknown model")

_PROVIDER_ERROR_MESSAGE_LIMIT = 300


def current_llm_identity() -> tuple[str, str]:
    """(provider, model) currently configured -- mirrors services.llm_factory.get_llm's own branching."""
    from config import get_settings

    settings = get_settings()
    provider = settings.llm_provider
    if provider == "gemini":
        model = settings.gemini_model
    elif provider == "groq":
        model = settings.groq_model
    elif provider == "cerebras":
        model = settings.cerebras_model
    elif provider == "openai":
        model = settings.openai_model
    else:
        model = settings.ollama_model
    return provider, model


def _status_code_of(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None) if response is not None else None
    if isinstance(code, int):
        return code
    # google.api_core.exceptions.* (raised by langchain_google_genai) expose
    # the numeric status as `.code`, not `.status_code`/`.response.status_code`.
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def _body_error_fields(exc: Exception) -> tuple[str | None, str | None]:
    """Best-effort (provider_error_code, provider_error_message) from an
    OpenAI/Groq-style error body -- never anything beyond what the provider
    itself reported in its own error response."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            code = error.get("code") or error.get("type")
            message = error.get("message")
            if code or message:
                return (str(code) if code else None, str(message) if message else None)
    message = getattr(exc, "message", None)
    if message:
        return (None, str(message))
    text = str(exc).strip()
    return (None, text or None)


def _retry_after_of(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for key in ("retry-after", "Retry-After"):
        try:
            value = headers.get(key)
        except AttributeError:
            value = None
        if value:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def classify_llm_exception(exc: Exception) -> dict[str, Any]:
    """Turn a raised LLM-client exception into a structured classification.

    Matches by exception class *name* rather than importing groq/openai
    exception types directly, so this works the same regardless of which
    provider is configured (or whether that provider's package is even
    installed) and regardless of whether langchain re-wraps the underlying
    SDK exception.
    """
    exception_type = type(exc).__name__
    status_code = _status_code_of(exc)
    provider_error_code, provider_error_message = _body_error_fields(exc)
    retry_after = _retry_after_of(exc)
    haystack = f"{provider_error_code or ''} {provider_error_message or ''}".lower()

    if exception_type in _RATE_LIMIT_NAMES or status_code == 429:
        category = "llm_rate_limit"
    elif exception_type in _AUTH_NAMES or status_code in (401, 403):
        category = "llm_auth_error"
    elif exception_type in _TIMEOUT_NAMES:
        category = "llm_timeout"
    elif exception_type in _CONNECTION_NAMES:
        category = "llm_connection_error"
    elif exception_type in _NOT_FOUND_NAMES or status_code == 404 or any(marker in haystack for marker in _MODEL_UNAVAILABLE_MARKERS):
        category = "llm_model_unavailable"
    elif any(marker in haystack for marker in _CONTEXT_LENGTH_MARKERS):
        category = "llm_context_too_large"
    elif (
        provider_error_code in _JSON_MODE_ERROR_CODES
        or (any(marker in haystack for marker in _JSON_MODE_MARKERS) and (exception_type in _BAD_REQUEST_NAMES or status_code == 400))
    ):
        category = "llm_json_mode_error"
    elif exception_type in _SERVER_ERROR_NAMES or (status_code is not None and 500 <= status_code < 600):
        category = "llm_provider_5xx"
    elif exception_type in _RESPONSE_VALIDATION_NAMES:
        category = "llm_invalid_response"
    else:
        category = "unknown_provider_error"

    message = provider_error_message
    if message and len(message) > _PROVIDER_ERROR_MESSAGE_LIMIT:
        message = message[:_PROVIDER_ERROR_MESSAGE_LIMIT] + "...(truncated)"

    return {
        "error_category": category,
        "exception_type": exception_type,
        "status_code": status_code,
        "provider_error_code": provider_error_code,
        "provider_error_message": message,
        "retry_after": retry_after,
    }


def invoke_with_json_mode_retry(
    prompt_template: Any,
    prompt_values: dict[str, Any],
    *,
    llm: Any,
    temperature: float,
    max_tokens: int,
    stage: str,
) -> tuple[Any, bool]:
    """Invoke `prompt_template | llm` (`llm` already built via get_json_llm);
    if that fails with a classified `llm_json_mode_error` (e.g. Groq's
    `json_validate_failed`), retry EXACTLY ONCE with a plain (non-JSON-mode)
    completion from the same provider/model/temperature/max_tokens. The
    existing lenient `agents.agent_json.parse_json_object` (markdown-fence
    stripping, balanced-brace extraction, one repair-parse attempt) plus
    pydantic schema validation downstream already tolerate a plain
    completion that is not strictly JSON-mode-constrained, so this is a safe
    recovery -- it does not change what "valid" means, only how the
    completion was requested.

    Any other error category is re-raised immediately (no retry for rate
    limits, auth, timeouts, connection errors, or context-too-large -- those
    would not be fixed by dropping JSON mode). A second failure on the retry
    is also re-raised as-is, so the caller's own except block still runs its
    existing safe-fallback behavior unchanged.

    Returns (response, json_mode_fallback_used).
    """
    try:
        chain = prompt_template | llm
        return chain.invoke(prompt_values), False
    except Exception as exc:
        classification = classify_llm_exception(exc)
        if classification["error_category"] != "llm_json_mode_error":
            raise
        print(
            "[ACRLA] llm_json_mode_fallback "
            f"llm_stage={stage} "
            f"provider_error_code={classification['provider_error_code']} "
            f"json_mode_fallback_used=True"
        )
        from services.llm_factory import get_llm

        fallback_llm = get_llm(temperature=temperature, max_tokens=max_tokens)
        chain = prompt_template | fallback_llm
        response = chain.invoke(prompt_values)  # a second failure propagates to the caller as-is
        return response, True


def extract_token_usage(response: Any) -> tuple[int, int, int]:
    """(prompt_tokens, completion_tokens, total_tokens) actually reported by
    the provider for one chat-model response, or (0, 0, 0) if the response
    carries no usage metadata at all -- callers fall back to
    `estimate_prompt_size`'s character-based estimate in that case.

    Tries langchain-core's standardized `usage_metadata` first (input_tokens/
    output_tokens/total_tokens), then the raw OpenAI/Groq-style
    `response_metadata["token_usage"]` (prompt_tokens/completion_tokens/
    total_tokens) that ChatOpenAI/ChatGroq attach -- never re-invokes the
    model or inspects prompt text itself, only already-returned counts.
    """
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        prompt = int(usage.get("input_tokens") or 0)
        completion = int(usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        if prompt or completion or total:
            return prompt, completion, total
    metadata = getattr(response, "response_metadata", None)
    token_usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
    if isinstance(token_usage, dict) and token_usage:
        prompt = int(token_usage.get("prompt_tokens") or 0)
        completion = int(token_usage.get("completion_tokens") or 0)
        total = int(token_usage.get("total_tokens") or (prompt + completion))
        if prompt or completion or total:
            return prompt, completion, total
    return 0, 0, 0


def estimate_prompt_size(prompt_values: dict[str, Any] | None) -> tuple[int, int]:
    """(character_count, rough_token_estimate) of everything that would be
    rendered into the prompt -- never the text itself, just its size."""
    total_chars = sum(len(str(value)) for value in (prompt_values or {}).values())
    return total_chars, total_chars // 4


# ---------------------------------------------------------------------------
# Final-answer token budget + truncation detection.
#
# Shared by every LLM call that phrases a user-visible answer (agents.
# response_generator, tools.external_tools, and the legacy fallback answer
# generators in pipelines.hybrid_pipeline / services.chat_orchestrator) --
# NOT agents.simple_planner, which keeps its own separate, already-tuned
# _PLANNER_MAX_TOKENS (2500) unchanged; a JSON plan and a natural-language
# answer are different generation shapes with different safe ceilings, and
# this module intentionally does not conflate them.
#
# Root cause this addresses: a live tutoring answer was observed stopping
# mid-sentence ("Recursion is a method where a function solves a problem by
# calling") -- traced to response_generator's final-answer call completing
# at max_tokens=450, well below what Gemini's "thinking"-capable models need.
# The same hidden-thinking-token mechanism confirmed for the planner (a
# request with prompt_tokens=1243 hit MAX_TOKENS at ceiling=900 despite a
# fully-formed, well under 900-word visible completion) applies here too:
# hidden reasoning tokens count against max_tokens but never appear in the
# visible answer, so a ceiling sized only for the visible answer's own
# length runs out before the visible text does.
FINAL_ANSWER_MAX_TOKENS = 1500
# One-time bounded retry ceiling if FINAL_ANSWER_MAX_TOKENS still truncates
# (see is_truncated_response) -- matches the planner's own Gemini-thinking-
# token-safe ceiling, reused here for the same reason. Never retried a
# second time (see callers).
FINAL_ANSWER_RETRY_MAX_TOKENS = 2500

# finish_reason values (case-insensitive) that mean the provider stopped
# because it hit the output-token ceiling, not because it was done --
# Gemini reports "MAX_TOKENS", OpenAI/Groq-style providers report "length".
TRUNCATION_FINISH_REASONS = {"MAX_TOKENS", "LENGTH"}


def is_truncated_response(response: Any) -> bool:
    """True if `response`'s own finish_reason means the provider stopped
    because it hit the output-token ceiling, not because it produced a
    complete answer -- the same signal agents.simple_planner already checks
    for its own JSON completions, generalized here for any chat-model
    response object (anything exposing `.response_metadata`)."""
    response_metadata = getattr(response, "response_metadata", None) or {}
    finish_reason = str(response_metadata.get("finish_reason") or "")
    return finish_reason.upper() in TRUNCATION_FINISH_REASONS


# error_category values that ACRLA's provider-level retry policy actually
# retries (see services.llm_factory._patch_gemini_retry_policy for Gemini,
# the only provider this was patched for) -- used only to make
# log_llm_provider_error's retry_attempted field accurate. It does not
# itself trigger or suppress any retry; the retry decision already happened
# at the provider-client layer by the time this function runs. Every other
# category (rate limit/quota, auth, permission, model-unavailable, invalid
# response, context-too-large, json-mode errors) is a permanent/client-side
# failure that fails on the first attempt by design -- retrying it again
# here would be the "second retry layer" this fix deliberately avoids.
_RETRYABLE_ERROR_CATEGORIES = {"llm_timeout", "llm_provider_5xx"}

# error_category values that mean an LLM call itself never completed due to
# a provider-level condition -- never a bug in ACRLA's own prompt/response
# handling, and never something a second attempt at the SAME call (or a
# DIFFERENT LLM-based path, like the legacy semantic-intent classifier)
# could fix. When one of these happens on the final-answer call after a
# semantic plan already succeeded, the caller must terminate the LLM path
# for this turn rather than silently falling through to legacy LLM-based
# routing, which would just re-hit the same failing provider with a
# different call and waste quota/latency for no benefit (the confirmed live
# bug this constant exists to fix: response_generation failed with
# llm_rate_limit, then the legacy orchestrator immediately made another
# Gemini call -- semantic_intent_analysis -- and hit the same 429).
# Used by agents.response_generator to populate token_usage["provider_error_category"]
# and by agents.simple_agent to decide the "final_answer_provider_error"
# fallback_reason vs. the generic "empty_final_answer" one.
PROVIDER_LEVEL_ERROR_CATEGORIES = {
    "llm_rate_limit", "llm_auth_error", "llm_model_unavailable",
    "llm_provider_5xx", "llm_timeout", "llm_connection_error",
}


def log_llm_provider_error(
    *,
    stage: str,
    exc: Exception,
    prompt_values: dict[str, Any] | None = None,
    json_mode_enabled: bool = False,
    llm: Any = None,
    response_length: int = 0,
    retry_attempted: bool | None = None,
    final_fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Classify + print one structured `[ACRLA] llm_provider_error` line.

    `llm` (the constructed chat-model instance, if the caller kept a
    reference before piping it into a chain) is used only to read its own
    `request_timeout` config -- never re-invoked here. Returns the
    classification dict in case the caller wants `error_category` for its
    own return value.
    """
    provider, model = current_llm_identity()
    classification = classify_llm_exception(exc)
    prompt_character_count, prompt_token_estimate = estimate_prompt_size(prompt_values)
    request_timeout = getattr(llm, "request_timeout", None) if llm is not None else None
    if retry_attempted is None:
        # Derived from the classified error_category, not the LLM client's
        # own `max_retries` config field -- for Gemini specifically, that
        # field is dead (langchain_google_genai==1.0.10 hardcodes its retry
        # bound internally and never reads it; see
        # services.llm_factory._patch_gemini_retry_policy), so it was always
        # a misleading proxy for whether a retry actually happened. This is
        # about the LLM *client's own* internal retry (not the json-mode-
        # fallback retry in `invoke_with_json_mode_retry`, which callers
        # report separately via their own json_mode_fallback_used logging).
        retry_attempted = classification["error_category"] in _RETRYABLE_ERROR_CATEGORIES
    print(
        "[ACRLA] llm_provider_error "
        f"llm_stage={stage} "
        f"llm_provider={provider} "
        f"llm_model={model} "
        f"error_category={classification['error_category']} "
        f"exception_type={classification['exception_type']} "
        f"status_code={classification['status_code']} "
        f"provider_error_code={classification['provider_error_code']} "
        f"provider_error_message={classification['provider_error_message']!r} "
        f"retry_after={classification['retry_after']} "
        f"request_timeout={request_timeout} "
        f"prompt_character_count={prompt_character_count} "
        f"prompt_token_estimate={prompt_token_estimate} "
        f"response_length={response_length} "
        f"json_mode_enabled={json_mode_enabled} "
        f"retry_attempted={retry_attempted} "
        f"final_fallback_reason={final_fallback_reason or 'none'}"
    )
    return classification
