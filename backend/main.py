from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import get_settings
from routers.api import router
from models.db_models import Base, engine

settings = get_settings()
print(
    "[ACRLA] startup_llm_config "
    f"llm_provider={settings.llm_provider} "
    f"gemini_model={settings.gemini_model} "
    f"groq_model={settings.groq_model} "
    f"cerebras_model={settings.cerebras_model} "
    f"openai_model={settings.openai_model} "
    f"ollama_model={settings.ollama_model} "
    f"acrla_single_llm_call_mode={settings.acrla_single_llm_call_mode}"
)

# Diagnostic-only: never logs the full key, only enough to confirm the app is
# reading the intended value (prefix) and that it wasn't truncated/padded by
# stray whitespace or a copy-paste error (length).
if settings.llm_provider == "gemini":
    _key = settings.gemini_api_key or ""
    print(
        "[ACRLA] gemini_key_check "
        f"key_prefix={_key[:8]!r} "
        f"key_length={len(_key)} "
        f"has_leading_or_trailing_whitespace={_key != _key.strip()}"
    )
if settings.llm_provider == "cerebras":
    _key = settings.cerebras_api_key or ""
    print(
        "[ACRLA] cerebras_key_check "
        f"key_prefix={_key[:8]!r} "
        f"key_length={len(_key)} "
        f"has_leading_or_trailing_whitespace={_key != _key.strip()}"
    )
if settings.llm_provider == "groq":
    _key = settings.groq_api_key or ""
    print(
        "[ACRLA] groq_key_check "
        f"key_prefix={_key[:8]!r} "
        f"key_length={len(_key)} "
        f"has_leading_or_trailing_whitespace={_key != _key.strip()}"
    )


def _log_prompt_token_estimates() -> None:
    """Log estimated token counts for the two per-turn LLM prompts (semantic
    planner, final-answer) at startup -- char/4 estimate, the same heuristic
    `agents.llm_errors.estimate_prompt_size` already uses at runtime, applied
    here to just the STATIC template text (no per-turn data filled in yet),
    so a prompt-size regression is visible immediately at boot instead of
    only discoverable by re-running a manual measurement script. Wrapped in
    try/except: this is a diagnostic nice-to-have, never something that
    should be able to prevent the app from starting.
    """
    try:
        from agents.simple_planner import SIMPLE_PLANNER_PROMPT
        from agents.response_generator import FINAL_PROMPT

        def _tok(text: str) -> int:
            return len(text) // 4

        planner_system = SIMPLE_PLANNER_PROMPT.messages[0].prompt.template
        planner_human = SIMPLE_PLANNER_PROMPT.messages[1].prompt.template
        response_system = FINAL_PROMPT.messages[0].prompt.template
        response_human = FINAL_PROMPT.messages[1].prompt.template
        print(
            "[ACRLA] startup_prompt_token_estimate "
            f"planner_system_tokens={_tok(planner_system)} "
            f"planner_human_template_tokens={_tok(planner_human)} "
            f"planner_static_total_tokens={_tok(planner_system) + _tok(planner_human)} "
            f"response_system_tokens={_tok(response_system)} "
            f"response_human_template_tokens={_tok(response_human)} "
            f"response_static_total_tokens={_tok(response_system) + _tok(response_human)} "
            "note=static_template_only_actual_per_turn_size_also_includes_message_and_tool_data"
        )
    except Exception as exc:
        print(f"[ACRLA] startup_prompt_token_estimate_failed error={exc}")


_log_prompt_token_estimates()

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="ACRLA — Adaptive Conversational Remediation and Learning Assistant",
    description="AI-powered adaptive chatbot backend for Moodle using RAG + GPT API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
def healthcheck():
    return {"status": "healthy"}

app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")