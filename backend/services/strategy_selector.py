"""Mastery-band tutoring strategy, legacy-path only.

NOTE: there are two independent, differently-shaped strategy systems in this
codebase -- do not conflate them:
  - THIS module (`select_tutoring_strategy` -> `TutoringStrategy` dataclass,
    with a `.prompt_block` property) is used only by
    `chat_orchestrator._select_strategy_for_turn`, itself only called from
    the legacy rule-based fallback path.
  - `services.intent_classifier.select_strategy` -> `Strategy` enum (+
    `get_strategy_instruction`) is the mastery-aware classifier the ACTIVE
    simple-agent path actually uses to build `context["tutoring_strategy"]`
    (see `chat_orchestrator.py`'s agent-context builder). It has no
    `.prompt_block` attribute -- an earlier bug called `getattr(strategy,
    "prompt_block", "")` on a `Strategy` enum member and silently dropped
    all mastery-band teaching instructions from the prompt.
Both are kept -- extend the one your call site actually consumes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TutoringStrategy:
    name: str
    reason: str
    instructions: list[str]

    @property
    def prompt_block(self) -> str:
        bullet_list = "\n".join(f"- {item}" for item in self.instructions)
        return (
            "BACKEND-CONTROLLED ADAPTIVE STRATEGY:\n"
            f"Strategy: {self.name}\n"
            f"Reason: {self.reason}\n\n"
            "You must follow this strategy:\n"
            f"{bullet_list}"
        )


STRATEGY_INSTRUCTIONS = {
    "simplified_remediation": [
        "Use simple language.",
        "Keep explanations short.",
        "Break ideas into clear step-by-step guidance.",
        "Keep support simple, but do not change the backend-selected question difficulty.",
        "Give hints after wrong answers.",
        "Avoid advanced terminology unless you explain it immediately.",
    ],
    "guided_practice": [
        "Use medium-length explanations.",
        "Ask practice questions.",
        "Give specific feedback.",
        "Connect concepts to examples.",
    ],
    "advanced_challenge": [
        "Use deeper reasoning.",
        "Ask harder questions.",
        "Include edge cases.",
        "Encourage comparison between concepts.",
        "Use less hand-holding.",
    ],
}


def select_tutoring_strategy(
    mastery_score: float,
    difficulty: str,
    concept: str | None = None,
) -> TutoringStrategy:
    score = _normalize_mastery_score(mastery_score)
    if score < 50:
        name = "simplified_remediation"
    elif score < 75:
        name = "guided_practice"
    else:
        name = "advanced_challenge"

    target = concept or "overall course"
    reason = f"mastery for {target} is {score:.0f}%."
    instructions = list(STRATEGY_INSTRUCTIONS[name])

    if difficulty == "easy" and name != "advanced_challenge":
        instructions.append("Use easy questions because the student selected easy difficulty.")
    elif difficulty == "hard" and name == "simplified_remediation":
        instructions.append("Use hard questions because the student selected hard difficulty, but add scaffolding and hints.")
    elif difficulty == "hard" and name == "advanced_challenge":
        instructions.append("Use hard code-writing or algorithm-design questions when testing.")

    return TutoringStrategy(name=name, reason=reason, instructions=instructions)


def _normalize_mastery_score(value: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score <= 1.0:
        score *= 100
    return max(0.0, min(100.0, score))
