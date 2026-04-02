from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from lexicont.agent import ModerationAgent, format_pipeline_trace
from lexicont.logger import get_logger

logger = get_logger("pipeline")


class ModerationContext(BaseModel):
    text: str
    stages: List[Dict[str, Any]] = Field(default_factory=list)
    final_decision: str = Field(..., description="block | review | pass")
    max_confidence: float = Field(0.0, description="0.0-1.0")
    trace: str = Field("")
    explanation: Optional[str] = Field(
        None, description="brief reason for the decision"
    )


_agents = {}


def _get_agent(config_path=None):
    if config_path not in _agents:
        _agents[config_path] = ModerationAgent(config_path=config_path)
    return _agents[config_path]


def _extract_explanation(stages, final_decision):
    if not stages:
        return None
    if final_decision in ("block", "review"):
        winning = next((s for s in stages if s.get("decision") == final_decision), None)
        if winning:
            return winning.get("reason") or None
    return stages[-1].get("reason") or None


def run(text, verbose=False, config_path=None):
    try:
        agent_ctx = _get_agent(config_path).run(text)

        if verbose:
            print(format_pipeline_trace(agent_ctx))

        explanation = _extract_explanation(agent_ctx.stages, agent_ctx.final_decision)

        return ModerationContext(
            text=agent_ctx.text,
            stages=agent_ctx.stages,
            final_decision=agent_ctx.final_decision,
            max_confidence=agent_ctx.max_confidence,
            trace=agent_ctx.trace,
            explanation=explanation,
        )

    except Exception as e:
        logger.error(f"pipeline fail-safe: {e}")
        return ModerationContext(
            text=text,
            final_decision="block",
            max_confidence=1.0,
            trace="fail-safe",
            explanation=f"system error: {type(e).__name__}",
        )
