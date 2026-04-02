import re
from rapidfuzz import fuzz
from pydantic import BaseModel, Field
from typing import Optional
from lexicont.config_loader import load_config, load_rules
from lexicont.logger import get_logger

logger = get_logger("fuzzy_trigger")
cfg = load_config()["fuzzy_trigger"]
rules_data = load_rules()


class FuzzyTriggerResult(BaseModel):
    stage: str = "fuzzy_trigger"
    decision: str = Field(..., description="block or pass")
    confidence: float = Field(..., description="0.0-1.0")
    reason: Optional[str] = Field(None, description="exact or fuzzy")
    original_text: str
    normalized_text: str


def normalize_text(text):
    return re.sub(r"[^a-zа-яё0-9\s]", "", text.lower()).strip()


norm_to_info = {}
for category, phrases in rules_data.get("categories", {}).items():
    for phrase in phrases:
        p_norm = normalize_text(phrase)
        if p_norm:
            norm_to_info[p_norm] = (category, phrase)


def run(text):
    try:
        normalized = normalize_text(text)
        reason = None
        score = 0.0

        for p_norm, (category, orig) in norm_to_info.items():
            if p_norm in normalized:
                reason = f"{category}: '{orig}' (exact)"
                score = 1.0
                break
        if score == 0.0:
            for p_norm, (category, orig) in norm_to_info.items():
                ratio = fuzz.partial_ratio(normalized, p_norm)
                if ratio >= cfg["fuzzy_min"]:
                    reason = f"{category}: '{orig}' (fuzzy {ratio}%)"
                    score = round(ratio / 100.0, 4)
                    break

        decision = "block" if score >= cfg["block_threshold"] else "pass"
        logger.info(f"fuzzy_trigger: {decision} confidence={score:.2f}")
        return FuzzyTriggerResult(
            decision=decision,
            confidence=round(score, 4),
            reason=reason,
            original_text=text,
            normalized_text=normalized,
        )
    except Exception as e:
        logger.error(f"fuzzy_trigger error: {e}")
        return FuzzyTriggerResult(
            decision="block",
            confidence=1.0,
            reason="fail-safe error",
            original_text=text,
            normalized_text=text,
        )
