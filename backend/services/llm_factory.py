from config import get_settings


_gemini_retry_patched = False


def _patch_gemini_retry_policy() -> None:
    """Replace langchain_google_genai's hardcoded provider-level retry policy
    with one that never retries a permanent failure.

    langchain_google_genai==1.0.10 (chat_models._create_retry_decorator)
    hardcodes its own `max_retries = 2` as a LOCAL variable -- it never reads
    `ChatGoogleGenerativeAI.max_retries` (the constructor/pydantic field), so
    passing max_retries=0/1 to the constructor has NO effect on this at all
    (confirmed by reading the installed package: the field is dead for this
    purpose). Worse, its retry predicate matches
    `google.api_core.exceptions.GoogleAPIError`, which is the base class of
    EVERY Google API exception -- so a 429 daily-quota exhaustion
    (ResourceExhausted), a 401/403 auth/permission failure (Unauthenticated/
    PermissionDenied), and a 404 model-not-found (NotFound) all get silently
    retried up to 2 times with exponential backoff (1-60s) before ACRLA's own
    error classification (agents.llm_errors.classify_llm_exception) ever sees
    them. That is pure wasted latency for a daily quota that will not reset
    within any retry window, and equally pointless for a permanent auth/model
    error.

    This is the ONLY retry layer for Gemini calls -- ACRLA itself never
    retries a provider call (see agents.llm_errors.invoke_with_json_mode_retry's
    docstring: "no retry for rate limits, auth, timeouts, connection errors").
    Replacing the policy here (once, idempotently) rather than wrapping calls
    in a second retry keeps exactly one retry policy in effect.

    New policy: only `google.api_core.exceptions.ServerError` (the base of
    every 5xx -- ServiceUnavailable, InternalServerError, BadGateway -- and
    of DeadlineExceeded/GatewayTimeout, i.e. timeouts) is retried, bounded at
    2 attempts total with the same exponential backoff as before -- a
    genuinely transient failure behaves exactly as it did before this patch.
    Every 4xx (ResourceExhausted/Unauthenticated/PermissionDenied/NotFound/
    InvalidArgument/FailedPrecondition -- ClientError, a sibling of
    ServerError, never matched) now fails on the first attempt.
    """
    global _gemini_retry_patched
    if _gemini_retry_patched:
        return
    import logging

    import google.api_core.exceptions as google_exceptions
    import langchain_google_genai.chat_models as gemini_chat_models
    from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    logger = logging.getLogger(gemini_chat_models.__name__)

    def _acrla_gemini_retry_decorator():
        return retry(
            reraise=True,
            stop=stop_after_attempt(2),  # same bound as the replaced upstream policy
            wait=wait_exponential(multiplier=2, min=1, max=60),  # same backoff as the replaced upstream policy
            retry=retry_if_exception_type(google_exceptions.ServerError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

    gemini_chat_models._create_retry_decorator = _acrla_gemini_retry_decorator
    _gemini_retry_patched = True


def get_llm(temperature: float = 0.3, max_tokens: int = 250):
    # get_settings() is deliberately called fresh here (not cached at module
    # scope) -- a module-level `settings = get_settings()` was previously
    # captured once at first import and never re-read, so an edited .env
    # (e.g. switching LLM_PROVIDER) had no effect on an already-running
    # process. See config.get_settings's own docstring for the same reasoning.
    settings = get_settings()
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        _patch_gemini_retry_policy()
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            temperature=temperature,
            max_tokens=max_tokens,
            # Unlike ChatGoogleGenerativeAI's max_retries (dead -- see
            # _patch_gemini_retry_policy), langchain_groq DOES wire this
            # straight into the underlying groq.Groq client's own
            # constructor (confirmed by reading langchain_groq.chat_models.
            # ChatGroq.validate_environment), and that client's default
            # _should_retry already correctly excludes 400/401/403/404 --
            # only 408/409/429/5xx are ever candidates. max_retries=0 closes
            # the one remaining gap ACRLA cares about (429/quota exhaustion
            # must fail immediately, not spend a bounded retry budget on a
            # limit that will not lift within any short backoff window) --
            # no custom retry-policy patch needed here, just this one field.
            max_retries=0,
        )
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if settings.llm_provider == "cerebras":
        # Cerebras Cloud exposes an OpenAI-compatible Chat Completions API,
        # so no dedicated langchain-cerebras package is required (none is in
        # requirements.txt) -- ChatOpenAI pointed at Cerebras's base URL
        # works the same way it does for any other OpenAI-compatible host.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.cerebras_model,
            openai_api_key=settings.cerebras_api_key,
            openai_api_base="https://api.cerebras.ai/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_predict=max_tokens,
    )


def get_json_llm(temperature: float = 0.0, max_tokens: int = 500):
    """Return an LLM configured for JSON object output when supported."""
    settings = get_settings()
    llm = get_llm(temperature=temperature, max_tokens=max_tokens)
    if settings.llm_provider == "gemini" and hasattr(llm, "bind"):
        # Gemini's structured-output switch is its own generation_config
        # field (response_mime_type), not the OpenAI-style response_format
        # used by groq/openai/cerebras -- same "ask the provider to emit
        # valid JSON" intent, provider-specific parameter name.
        try:
            return llm.bind(generation_config={"response_mime_type": "application/json"})
        except Exception:
            return llm
    if settings.llm_provider in {"groq", "openai", "cerebras"} and hasattr(llm, "bind"):
        try:
            return llm.bind(response_format={"type": "json_object"})
        except Exception:
            return llm
    return llm


def get_embeddings():
    settings = get_settings()
    from langchain_community.embeddings import OllamaEmbeddings
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )
