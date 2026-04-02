from detoxify import Detoxify
from pydantic import BaseModel, Field
from typing import Optional, Dict
from lexicont.config_loader import load_config
from lexicont.logger import get_logger

logger = get_logger("toxicity_ml")
cfg = load_config()["toxicity_ml"]
model = Detoxify(cfg["model"])
thresholds = cfg["thresholds"]


class ToxicityMlResult(BaseModel):
    stage: str = "toxicity_ml"
    decision: str = Field(..., description="block or pass")
    confidence: float = Field(..., description="0.0-1.0")
    top_category: Optional[str] = Field(
        None, description="category with the highest score"
    )
    category_scores: Dict[str, float] = Field(..., description="all scores")
    reason: Optional[str] = Field(None, description="reason")
    original_text: str
    normalized_text: str


def run(text):
    try:
        preds = model.predict(text)
        preds = {k: round(float(v), 4) for k, v in preds.items()}
        top_category = max(preds, key=preds.get)
        top_score = preds[top_category]

        decision = "pass"
        reason = f"{top_category} ({top_score})"
        for cat, thr in thresholds.items():
            if preds.get(cat, 0) >= thr:
                decision = "block"
                reason = f"{cat} ({preds[cat]})"
                break

        logger.info(f"toxicity_ml: {decision} top={top_category}={top_score:.2f}")
        return ToxicityMlResult(
            decision=decision,
            confidence=round(top_score, 4),
            top_category=top_category,
            category_scores=preds,
            reason=reason,
            original_text=text,
            normalized_text=text.lower(),
        )
    except Exception as e:
        logger.error(f"toxicity_ml error: {e}")
        return ToxicityMlResult(
            decision="block",
            confidence=1.0,
            top_category=None,
            category_scores={},
            reason="fail-safe error",
            original_text=text,
            normalized_text=text.lower(),
        )
