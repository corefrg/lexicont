from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lexicont.logger import get_logger
from lexicont.pipeline import run as pipeline_run

logger = get_logger("api")

app = FastAPI(
    title="lexicont",
    description="Policy-driven agent for real-time text moderation",
    version="0.1.0",
)


class ModerateRequest(BaseModel):
    text: str = Field(
        ..., description="Text to moderate", min_length=1, max_length=10_000
    )


class StageDetail(BaseModel):
    stage: str = Field(
        ...,
        description="profanity_filter | fuzzy_trigger | toxicity_ml | llm_entry_judge | llm_judge",
    )
    decision: str = Field(..., description="block | review | pass")
    confidence: float = Field(..., description="0.0-1.0")
    reason: Optional[str] = Field(
        None,
        description="Reason for this stage decision. None if no violation found.",
    )


class ModerateResponse(BaseModel):
    decision: str = Field(
        ...,
        description=(
            "Final pipeline decision. "
            "block - block the text, can come from any stage. "
            "review - manual review, only from llm_entry_judge or llm_judge. "
            "pass - text is clean."
        ),
    )
    confidence: float = Field(
        ...,
        description="Confidence 0.0-1.0. Taken from the stage with the highest score.",
    )
    explanation: str = Field(
        ...,
        description=(
            "Brief reason from the stage that made the final decision. "
            "Empty string if text is clean and no stage found violations."
        ),
    )
    trace: str = Field(
        ...,
        description=(
            "Chain of stages that ran, separated by -. "
            "Short if early stopping triggered. "
            "fail-safe if system crashed internally."
        ),
    )
    stages: List[StageDetail] = Field(
        default_factory=list,
        description="Details of each stage that ran. Useful for debugging.",
    )


@app.get("/health", tags=["service"])
def health():
    return {"status": "ok"}


@app.post("/moderate", response_model=ModerateResponse, tags=["moderation"])
def moderate(request: ModerateRequest) -> ModerateResponse:
    logger.info(f"API /moderate: text_len={len(request.text)}")

    try:
        result = pipeline_run(request.text)
    except Exception as e:
        logger.error(f"API unexpected error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal error: {type(e).__name__}"
        )

    stages = [
        StageDetail(
            stage=s.get("stage", ""),
            decision=s.get("decision", ""),
            confidence=round(float(s.get("confidence", 0.0)), 4),
            reason=s.get("reason") or None,
        )
        for s in result.stages
    ]

    return ModerateResponse(
        decision=result.final_decision,
        confidence=round(result.max_confidence, 4),
        explanation=result.explanation or "",
        trace=result.trace,
        stages=stages,
    )
