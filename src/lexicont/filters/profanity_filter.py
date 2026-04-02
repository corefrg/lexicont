from pydantic import BaseModel, Field
from typing import Optional
from glin_profanity import Filter
from lexicont.config_loader import load_config, load_rules
from lexicont.logger import get_logger

logger = get_logger("profanity_filter")
cfg = load_config()["profanity_filter"]
rules_data = load_rules()

filter_ = Filter(
    {
        "languages": cfg["languages"],
        "detect_leetspeak": cfg["detect_leetspeak"],
        "normalize_unicode": cfg["normalize_unicode"],
    }
)


class ProfanityFilterResult(BaseModel):
    stage: str = "profanity_filter"
    decision: str = Field(..., description="block or pass")
    confidence: float = Field(..., description="0.0-1.0")
    reason: Optional[str] = Field(None, description="what was found")
    original_text: str
    normalized_text: str


def run(text):
    try:
        result = filter_.check_profanity(text)
        contains = result.get("contains_profanity", False)
        profane_words = result.get("profane_words", [])
        normalized = result.get("processed_text", text)
        score = 0.0
        reason = None

        if contains:
            score = 1.0
            reason = f"profanity: {profane_words}"

        if score < cfg["block_threshold"]:
            for category, phrases in rules_data.get("categories", {}).items():
                for phrase in phrases:
                    if phrase in normalized.lower():
                        score = 1.0
                        reason = f"{category}: '{phrase}'"
                        break

        decision = "block" if score >= cfg["block_threshold"] else "pass"
        logger.info(f"profanity_filter: {decision} confidence={score:.2f}")
        return ProfanityFilterResult(
            decision=decision,
            confidence=round(score, 4),
            reason=reason,
            original_text=text,
            normalized_text=normalized,
        )
    except Exception as e:
        logger.error(f"profanity_filter error: {e}")
        return ProfanityFilterResult(
            decision="block",
            confidence=1.0,
            reason="fail-safe error",
            original_text=text,
            normalized_text=text,
        )
