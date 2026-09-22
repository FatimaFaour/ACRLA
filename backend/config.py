from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM provider: "gemini" (default) | "groq" | "ollama" (local) | "openai" | "cerebras"
    llm_provider: str = "gemini"

    # Gemini (Google AI Studio / Gemini Developer API, via langchain-google-genai)
    gemini_api_key: str = "not-needed"
    gemini_model: str = "gemini-2.5-flash"

    # Groq settings (free, fast responses). openai/gpt-oss-120b confirmed
    # live against this project's Groq account via client.models.list() --
    # a capable enough model for both semantic-planner JSON output and
    # final-answer synthesis, the two roles services.llm_factory.get_llm/
    # get_json_llm serve for every provider.
    groq_api_key: str = "not-needed"
    groq_model: str = "openai/gpt-oss-120b"

    # Cerebras (OpenAI-compatible endpoint, see services.llm_factory). Model
    # must be one this account actually has access to -- confirmed live via
    # client.models.list() against the configured key, not assumed from
    # Cerebras's general public catalog (which this account's key does not
    # match: it only has gpt-oss-120b / gemma-4-31b / zai-glm-4.7).
    cerebras_api_key: str = "not-needed"
    cerebras_model: str = "gpt-oss-120b"

    # Ollama settings (local fallback)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "tinyllama"
    ollama_embedding_model: str = "nomic-embed-text"

    # OpenAI (optional)
    openai_api_key: str = "not-needed"
    openai_model: str = "gpt-4o"

    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/acrla"
    chroma_path: str = "./chroma_db"
    secret_key: str = "acrla-secret-key"
    moodle_url: str = "http://localhost:8080"
    allowed_origins: str = "*"
    max_retrieval_chunks: int = 2

    # Conversation agent architecture: "simple" (one planner call + one final
    # answer call, see agents.simple_agent) or "iterative" (the original
    # agent_brain.decide() -> tool -> observe loop, agents.conversation_agent,
    # kept as a fallback behind this flag). Default simple.
    acrla_agent_mode: str = "simple"

    # Gate for large/bulk diagnostic log dumps (full canonical course
    # records, complete manifests, full analytics item arrays, repeated
    # scope-concept dumps, full tool arguments/results) -- see
    # agents.debug_log. Concise INFO logs (goal, selected pipeline, tools
    # executed, call/token counts, fallback reason, provider error, latency)
    # always print regardless of this flag. Default false (quiet).
    acrla_debug_verbose: bool = False

    # Reserved for a future single-LLM-call fast path for simple concept
    # explanations. Field defined (and read at startup, see main.py) but
    # NOT YET WIRED to any routing behavior -- the specific mechanism
    # requested (detect a "simple concept explanation" by checking whether
    # the message contains a known concept name and no analytics/comparison
    # keywords) is keyword/phrase-based routing, which agents.simple_planner
    # was deliberately built to avoid after a real bug (a subjective
    # statement like "I feel weak" or "recursion is confusing" would match
    # "contains a known concept, no analytics keywords" and skip semantic
    # classification entirely, silently reverting that fix). Left False by
    # default and inert until a semantic (not keyword) implementation is
    # agreed -- see the task report for the recommended alternative.
    acrla_single_llm_call_mode: bool = False

    # Quota-saving mode: skip an LLM call wherever a deterministic result is
    # just as correct. Two effects when true (each logged as
    # "[ACRLA] quota_saver_skipped_llm reason=<reason>" at the moment a call
    # is skipped):
    #   - analytics_already_formatted: no new behavior here -- run_analytics_query's
    #     own "formatted" text already short-circuits the final-answer call
    #     unconditionally (agents.simple_agent._deterministic_reply /
    #     agents.response_generator.generate_final_answer's own "agent_tools"
    #     check); this flag only adds the log line, since that short-circuit
    #     is a strict improvement and must not become opt-in.
    #   - greeting_template: a short, common casual_conversation message
    #     ("hi", "thanks", "bye", ...) gets a canned reply instead of a
    #     final-answer call -- narrow on purpose (exact short phrases only,
    #     see agents.simple_agent._QUOTA_SAVER_GREETING_REPLIES) so a
    #     genuinely subjective message ("I feel weak") still gets a real,
    #     context-aware LLM-phrased reply. Default false (unchanged
    #     behavior); this only affects response PHRASING, never goal
    #     classification/routing.
    acrla_quota_saver: bool = False

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    class Config:
        env_file = ".env"
        # An unrecognized .env key (a provider field not yet added here, a
        # leftover/experimental var) must never crash Settings() construction
        # -- pydantic-settings defaults to forbidding unknown keys, which is
        # exactly what made LLM_PROVIDER=cerebras + CEREBRAS_API_KEY/
        # CEREBRAS_MODEL raise a ValidationError before this field existed.
        extra = "ignore"


def get_settings() -> Settings:
    """Construct Settings fresh from the environment/.env every call.

    Deliberately not cached: an lru_cache here would only ever be populated
    once per process anyway (a real fix requires a fresh process, not a
    cleared cache -- see the callers below), but leaving it cached made it
    easy to *believe* a stale value was the cache's fault instead of an old
    process still running. Re-reading pydantic-settings on every call is
    cheap; correctness here matters more than the few microseconds saved.
    """
    return Settings()
