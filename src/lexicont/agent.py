from __future__ import annotations
import json
from typing import Any, Dict, List, Literal, Set
from pydantic import BaseModel, ConfigDict, Field
from lexicont.config_loader import load_config
from lexicont.filters.fuzzy_trigger import run as fuzzy_run
from lexicont.filters.llm_judge import run as llm_run
from lexicont.filters.llm_entry_judge import run as triage_run
from lexicont.filters.profanity_filter import run as profanity_run
from lexicont.filters.toxicity_ml import run as toxicity_run
from lexicont.logger import get_logger

logger = get_logger("agent")

Action = Literal[
    "profanity_filter",
    "fuzzy_trigger",
    "toxicity_ml",
    "llm_entry_judge",
    "llm_judge",
    "finalize",
    "stop",
]


class AgentContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    text: str
    effective_text: str = Field("", description="normalized text after entry_judge")
    stages: List[Dict[str, Any]] = Field(default_factory=list)
    final_decision: str = Field("pass", description="block | review | pass")
    max_confidence: float = Field(0.0, ge=0.0, le=1.0)
    trace: str = Field("")
    current_step: int = Field(0, ge=0)
    executed_stages: Set[str] = Field(default_factory=set)
    triage_send_to_stage4: bool = Field(True)

    def model_post_init(self, __context):
        if not self.effective_text:
            self.effective_text = self.text


class Policy:
    def __init__(self, cfg):
        general = cfg["general"]
        self.EARLY_STOP = general["early_stop_confidence"]
        self.STAGE4_TRIGGER = general["stage4_trigger_confidence"]
        self.ENABLE_STAGE1 = general.get("enable_stage1", True)
        self.ENABLE_STAGE2 = general.get("enable_stage2", True)
        self.ENABLE_STAGE3 = general.get("enable_stage3", True)
        self.ENABLE_llm_entry_judge = general.get("enable_llm_entry_judge", True)
        self.ENABLE_LLM_JUDGE = general.get("enable_llm_judge", True)

    def decide(self, ctx):
        executed = ctx.executed_stages
        if self.ENABLE_STAGE1 and "profanity_filter" not in executed:
            return "profanity_filter"
        if self._is_high_confidence_block(ctx):
            return "stop"
        if self.ENABLE_STAGE2 and "fuzzy_trigger" not in executed:
            return "fuzzy_trigger"
        if self._is_high_confidence_block(ctx):
            return "stop"
        if self.ENABLE_STAGE3 and "toxicity_ml" not in executed:
            return "toxicity_ml"
        if self._is_high_confidence_block(ctx):
            return "stop"
        if self.ENABLE_llm_entry_judge and "llm_entry_judge" not in executed:
            return "llm_entry_judge"
        if (
            "llm_entry_judge" in executed
            and ctx.final_decision == "block"
            and not ctx.triage_send_to_stage4
        ):
            return "stop"
        if self.ENABLE_LLM_JUDGE and "llm_judge" not in executed:
            triage_done = "llm_entry_judge" in executed
            if triage_done:
                if ctx.triage_send_to_stage4:
                    return "llm_judge"
            else:
                if ctx.max_confidence < self.STAGE4_TRIGGER:
                    return "llm_judge"
        return "finalize"

    def _is_high_confidence_block(self, ctx):
        return ctx.final_decision == "block" and ctx.max_confidence >= self.EARLY_STOP


class ExecutionLayer:
    def run(self, action, ctx):
        if action == "profanity_filter":
            result = profanity_run(ctx.text)
        elif action == "fuzzy_trigger":
            result = fuzzy_run(ctx.text)
        elif action == "toxicity_ml":
            result = toxicity_run(ctx.text)
        elif action == "llm_entry_judge":
            s = ctx.stages
            result = triage_run(
                ctx.text,
                s[0] if len(s) > 0 else None,
                s[1] if len(s) > 1 else None,
                s[2] if len(s) > 2 else None,
            )
            ctx.triage_send_to_stage4 = result.send_to_stage4
            if result.clean_text and result.clean_text != ctx.text:
                logger.info(
                    f"llm_entry_judge: text normalized for stage4: "
                    f"{ctx.text!r} - {result.clean_text!r}"
                )
                ctx.effective_text = result.clean_text
            logger.info(
                f"llm_entry_judge agent decision: send_to_stage4={result.send_to_stage4}"
            )
        elif action == "llm_judge":
            s = ctx.stages
            text_changed = ctx.effective_text != ctx.text
            if text_changed:
                logger.info(
                    f"llm_judge: RAG on normalized: {ctx.effective_text!r} "
                    f"(original: {ctx.text!r})"
                )
            else:
                logger.info(f"llm_judge: RAG on original text: {ctx.effective_text!r}")
            result = llm_run(
                ctx.effective_text,
                s[0] if len(s) > 0 else None,
                s[1] if len(s) > 1 else None,
                s[2] if len(s) > 2 else None,
                original_text=ctx.text if text_changed else None,
            )
        else:
            raise ValueError(f"ExecutionLayer: unknown action '{action}'")
        self._apply(ctx, result)

    @staticmethod
    def _apply(ctx, result):
        dump = result.model_dump()
        ctx.stages.append(dump)
        ctx.executed_stages.add(result.stage)
        confidence = dump.get("confidence", 0.0)
        if confidence > ctx.max_confidence:
            ctx.max_confidence = confidence
        decision = dump.get("decision", "pass")
        if decision == "block":
            ctx.final_decision = "block"
        elif decision == "review" and ctx.final_decision != "block":
            ctx.final_decision = "review"


_STAGE_LABELS = {
    "profanity_filter": "Stage 1 = Profanity Filter",
    "fuzzy_trigger": "Stage 2 = Fuzzy Trigger",
    "toxicity_ml": "Stage 3 = Toxicity ML",
    "llm_entry_judge": "Stage 3.5 = LLM Entry Judge",
    "llm_judge": "Stage 4 = LLM Judge",
}


def _format_stage_dump(dump):
    stage = dump.get("stage", "unknown")
    label = _STAGE_LABELS.get(stage, stage)
    decision = dump.get("decision", "?")
    confidence = dump.get("confidence", 0.0)
    reason = dump.get("reason") or ""
    lines = [
        f"   {label}",
        f"  decision : {decision.upper()}",
        f"  confidence : {confidence:.4f}",
    ]
    if reason:
        lines.append(f"  reason : {reason}")
    if stage == "toxicity_ml":
        top_cat = dump.get("top_category") or ""
        scores = dump.get("category_scores") or {}
        notable = {k: round(v, 3) for k, v in scores.items() if v >= 0.15}
        if top_cat:
            lines.append(f"  top_cat : {top_cat}")
        if notable:
            lines.append(f"  scores : {json.dumps(notable, ensure_ascii=False)}")
    elif stage == "llm_entry_judge":
        flags = dump.get("flags") or ""
        clean = dump.get("clean_text") or ""
        s4 = dump.get("send_to_stage4", True)
        if flags:
            lines.append(f"  flags : {flags}")
        if clean:
            lines.append(f"  clean_text : {clean!r}")
        lines.append(f"  - stage4 : {'YES' if s4 else 'NO (entry_judge blocked)'}")
        meta = dump.get("meta") or {}
        if meta.get("digest_sent"):
            lines.append(f"  digest_in : {meta['digest_sent']}")
    elif stage == "llm_judge":
        meta = dump.get("meta") or {}
        scores = meta.get("scores") or {}
        flags = meta.get("flags") or ""
        check = meta.get("check") or ""
        rag_used = meta.get("rag_used", False)
        rag_text = meta.get("rag_search_text") or ""
        rag_original = meta.get("rag_original_text") or ""
        rag_context = meta.get("rag_context") or ""
        if rag_text:
            lines.append(f"  rag_text : {rag_text!r} - text for RAG search")
        if rag_original:
            lines.append(
                f"  original : {rag_original!r} - original (also passed to LLM)"
            )
        if check:
            lines.append(f"  _check : {check}")
        if flags:
            lines.append(f"  flags : {flags}")
        if scores:
            notable = {k: round(v, 3) for k, v in scores.items() if v >= 0.15}
            if notable:
                lines.append(f"  scores : {json.dumps(notable, ensure_ascii=False)}")
        lines.append(f"  rag_used : {rag_used}")
        if rag_used and rag_context and rag_context != "N/A":
            lines.append(f"  rag_match : {rag_context[:200]}")
    elif stage == "profanity_filter":
        norm = dump.get("normalized_text") or ""
        orig = dump.get("original_text") or ""
        if norm and norm != orig:
            lines.append(f" | normalized : {norm!r}")
    elif stage == "fuzzy_trigger":
        norm = dump.get("normalized_text") or ""
        lines.append(f"  normalized : {norm!r}")
    lines.append("                    ")
    return "\n".join(lines)


def format_pipeline_trace(ctx):
    sep = "=" * 40
    lines = [
        "",
        sep,
        " LEXICONT PIPELINE TRACE",
        f" text: {ctx.text!r}",
        sep,
    ]
    for dump in ctx.stages:
        lines.append(_format_stage_dump(dump))
    lines += [
        "",
        "  RESULT",
        f"  final_decision : {ctx.final_decision.upper()}",
        f"   max_confidence : {ctx.max_confidence:.4f}",
        f"   effective_text : {ctx.effective_text!r}",
        f"  trace : {ctx.trace}",
        "          ",
        sep,
        "",
    ]
    return "\n".join(lines)


class ModerationAgent:
    MAX_STEPS = 6

    def __init__(self, config_path=None):
        cfg = load_config(config_path)
        self.policy = Policy(cfg)
        self.execution = ExecutionLayer()

    def run(self, text):
        ctx = AgentContext(text=text)
        logger.info(f"agent start: {text!r}")
        while ctx.current_step < self.MAX_STEPS:
            action = self.policy.decide(ctx)
            logger.debug(f"step={ctx.current_step} policy-{action}")
            if action == "stop":
                logger.info(
                    f"early stop at step {ctx.current_step} "
                    f"(confidence={ctx.max_confidence:.2f})"
                )
                break
            if action == "finalize":
                self._finalize(ctx)
                break
            try:
                self.execution.run(action, ctx)
            except Exception as exc:
                logger.error(
                    f"fail-safe triggered at step={ctx.current_step} "
                    f"action={action}: {exc}"
                )
                ctx.final_decision = "block"
                ctx.max_confidence = 1.0
                break
            ctx.current_step += 1
        ctx.trace = " - ".join(s["stage"] for s in ctx.stages)
        logger.info(
            f"agent done: {ctx.final_decision} "
            f"confidence={ctx.max_confidence:.2f} "
            f"steps={ctx.current_step} trace=[{ctx.trace}]"
        )
        return ctx

    @staticmethod
    def _finalize(ctx):
        decisions = [s.get("decision") for s in ctx.stages]
        if "block" in decisions:
            ctx.final_decision = "block"
        elif "review" in decisions:
            ctx.final_decision = "review"
        else:
            ctx.final_decision = "pass"
