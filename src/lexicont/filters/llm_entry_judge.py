from __future__ import annotations
import json
import re
import time
from typing import Any, Dict, Optional
import requests
from pydantic import BaseModel, Field
from lexicont.config_loader import load_config
from lexicont.logger import get_logger

logger = get_logger("llm_entry_judge")

# intentionally left as module-level for patching in notebooks
# they are re-read from config inside run() on every call
BACKEND = None
MODEL = None
API_URL = None
TIMEOUT = None
MAX_TOKENS = None
RETRIES = None
THINKING_ENABLED = None
BLOCK_THRESHOLD = None
SYSTEM_PROMPT = None

_CONNECT_TIMEOUT = 3.0


def _resolve_module_vars():
    global BACKEND, MODEL, API_URL, TIMEOUT, MAX_TOKENS, RETRIES
    global THINKING_ENABLED, BLOCK_THRESHOLD, SYSTEM_PROMPT, _CONNECT_TIMEOUT

    cfg = load_config()
    triage_cfg = cfg.get("llm_entry_judge", {})

    BACKEND = triage_cfg.get(
        "backend", cfg.get("llm_judge", {}).get("backend", "llamacpp")
    )
    backend_cfg = triage_cfg.get(BACKEND, cfg.get("llm_judge", {}).get(BACKEND, {}))
    MODEL = backend_cfg.get("model", "qwen3:4b")
    API_URL = backend_cfg.get("url", "http://localhost:11434/v1/chat/completions")
    TIMEOUT = triage_cfg.get("timeout", 120)
    MAX_TOKENS = triage_cfg.get("max_tokens", 1024)
    RETRIES = triage_cfg.get("retries", 2)
    THINKING_ENABLED = triage_cfg.get("thinking_enabled", False)
    BLOCK_THRESHOLD = triage_cfg.get("block_threshold", 0.80)
    SYSTEM_PROMPT = triage_cfg.get("system_prompt", "")
    _CONNECT_TIMEOUT = 3.0


class LlmTriageResult(BaseModel):
    stage: str = "llm_entry_judge"
    decision: str = Field(..., description="block | review | pass")
    confidence: float = Field(..., ge=0.0, le=1.0)
    flags: str = Field("", description="found categories")
    clean_text: str = Field("", description="normalized text for RAG/stage4")
    reason: Optional[str] = None
    original_text: str
    send_to_stage4: bool = Field(
        True, description="agent decision: whether to pass to stage4"
    )
    meta: Dict[str, Any] = Field(default_factory=dict)


def _build_digest(
    stage1,
    stage2,
    stage3,
):
    parts = []

    if stage1 and stage1.get("decision") == "block":
        reason = stage1.get("reason") or ""
        if reason:
            parts.append(f"profanity: {reason}")

    if stage2 and stage2.get("decision") == "block":
        reason = stage2.get("reason") or ""
        if reason:
            parts.append(f"pattern: {reason}")

    if stage3 and stage3.get("decision") == "block":
        top_cat = stage3.get("top_category") or ""
        if top_cat:
            parts.append(f"ml: {top_cat}")

    return " | ".join(parts)


def _parse_json(text):
    if not text or not text.strip():
        logger.debug("_parse_json: empty text")
        return None

    logger.debug(f"_parse_json: received text preview: {text[:300]}...")

    if "</think>" in text:
        candidate = text.split("</think>", 1)[-1].strip()
        logger.debug("_parse_json: found </think>, extracting JSON after it")
        try:
            return json.loads(candidate)
        except Exception:
            pass
        md = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.DOTALL).strip()
        try:
            return json.loads(md)
        except Exception:
            pass

    last_brace = text.rfind("{")
    if last_brace != -1:
        candidate = text[last_brace:]
        logger.debug(f"_parse_json: found last {{ at position {last_brace}")
        try:
            return json.loads(candidate)
        except Exception:
            pass

        fragment = re.sub(r',\s*"[^"]*$', "", candidate)
        fragment = re.sub(r':\s*"[^"]*$', ': ""', fragment)
        if not fragment.rstrip().endswith("}"):
            fragment = fragment + "}"
        try:
            return json.loads(fragment)
        except Exception:
            pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        logger.debug("_parse_json: found JSON pattern via regex")
        try:
            return json.loads(m.group())
        except Exception:
            pass

    m2 = re.search(r"(\{.*)", text, re.DOTALL)
    if m2:
        fragment = m2.group(1)
        fragment = re.sub(r',\s*"[^"]*$', "", fragment)
        fragment = re.sub(r':\s*"[^"]*$', ': ""', fragment)
        try:
            return json.loads(fragment + "}")
        except Exception:
            pass

    logger.warning(f"_parse_json: all strategies failed. Text: {text[:300]}...")
    return None


def _fallback_normalize_text(text):
    clean = text
    # TODO personalize for your language!
    leet_map = {
        "0": "о",
        "1": "и",
        "3": "з",
        "4": "ч",
        "5": "с",
        "6": "ш",
        "7": "т",
        "8": "в",
        "9": "р",
    }
    for digit, letter in leet_map.items():
        clean = clean.replace(digit, letter)

    clean = re.sub(r"([а-яёa-z])\1{2,}", r"\1", clean, flags=re.IGNORECASE)
    clean = clean.replace("ё", "е")
    clean = re.sub(r"[^а-яёa-z0-9\s.,!?;:()]", " ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()
    clean = clean.lower()

    return clean


def _check_server_reachable():
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(API_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=_CONNECT_TIMEOUT):
            return True
    except OSError as e:
        logger.error(
            f"llm_entry_judge: server unreachable {host}:{port} - {e}\n"
            f"  backend={BACKEND} url={API_URL}\n"
            f"  Check: is llamacpp/ollama running? Is the port correct?"
        )
        return False


def call_llm(messages):
    try:
        if BACKEND == "ollama":
            payload = {
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "think": THINKING_ENABLED,
                "format": "json",
                "options": {
                    "temperature": 0.0 if not THINKING_ENABLED else 0.2,
                    "top_p": 0.95,
                    "top_k": 20,
                    "num_predict": MAX_TOKENS,
                },
            }
        else:
            payload = {
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "thinking_forced_open": THINKING_ENABLED,
            }

        logger.debug(
            f"llm_entry_judge - {BACKEND} | model={MODEL} | url={API_URL} | "
            f"max_tokens={MAX_TOKENS} | thinking={THINKING_ENABLED}"
        )

        t0 = time.perf_counter()
        r = requests.post(
            API_URL,
            json=payload,
            timeout=(_CONNECT_TIMEOUT, float(TIMEOUT)),
        )
        r.raise_for_status()
        resp = r.json()

        if BACKEND == "ollama":
            msg = resp.get("message", {})
            content = msg.get("content", "")
            thinking = msg.get("thinking", "")
        else:
            msg = resp["choices"][0]["message"]
            content = msg.get("content", "")
            thinking = msg.get("reasoning_content", "")

        elapsed = round(time.perf_counter() - t0, 2)
        logger.debug(
            f"llm_entry_judge call took {elapsed}s thinking_enabled={THINKING_ENABLED}"
        )
        if content:
            logger.debug(
                f"llm_entry_judge RAW content (first 500 chars): {content[:500]}..."
            )
        else:
            logger.warning(
                f"llm_entry_judge: content is empty! thinking={thinking[:200]!r} | "
                f"full_msg_keys={list(msg.keys())}"
            )
        return thinking, content, elapsed

    except requests.exceptions.ConnectTimeout:
        logger.error(
            f"llm_entry_judge: CONNECT TIMEOUT after {_CONNECT_TIMEOUT}s - server not running?\n"
            f"  url={API_URL} backend={BACKEND}\n"
            f"  Run: llamacpp-server -m <model> --port 11434"
        )
        return "", "", 0.0
    except requests.exceptions.ReadTimeout:
        logger.error(
            f"llm_entry_judge: READ TIMEOUT after {TIMEOUT}s - generation too slow\n"
            f"  model={MODEL} max_tokens={MAX_TOKENS}\n"
            f"  Options: increase timeout in config | reduce max_tokens | "
            f"use a lighter model"
        )
        return "", "", 0.0
    except requests.exceptions.ConnectionError as e:
        logger.error(
            f"llm_entry_judge: CONNECTION ERROR - server unreachable\n"
            f"  url={API_URL} backend={BACKEND}\n"
            f"  Details: {e}"
        )
        return "", "", 0.0
    except Exception as e:
        logger.error(f"llm_entry_judge call error: {type(e).__name__}: {e}")
        return "", "", 0.0


def run(
    text,
    stage1=None,
    stage2=None,
    stage3=None,
):
    _resolve_module_vars()

    try:
        if not _check_server_reachable():
            raise ConnectionError(
                f"LLM server unreachable: {API_URL} (backend={BACKEND})"
            )

        digest = _build_digest(stage1, stage2, stage3)

        hints_prefix = f"HINTS: {digest}\n" if digest else ""
        user_content = f'{hints_prefix}Text: """{text}"""'
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        parsed = None
        elapsed_total = 0.0
        thinking_log = ""
        content = ""
        total_attempts = max(1, RETRIES)
        for attempt in range(total_attempts):
            logger.debug(f"llm_entry_judge: attempt {attempt + 1}/{total_attempts}")
            thinking_log, content, elapsed = call_llm(messages)
            elapsed_total += elapsed
            if not content:
                logger.warning(
                    f"llm_entry_judge: empty response attempt={attempt + 1}/{total_attempts} "
                    f"- retry (cold start / timeout)"
                )
                if attempt + 1 < total_attempts:
                    time.sleep(2)
                continue
            parsed = _parse_json(content)
            if parsed:
                break
            logger.warning(
                f"llm_entry_judge: invalid JSON attempt={attempt + 1}/{total_attempts} - retry"
            )
        if not parsed:
            if not content:
                raise TimeoutError(
                    f"LLM did not respond after {total_attempts} attempts of {float(TIMEOUT)}s "
                    f"(model={MODEL})"
                )
            raise ValueError(
                f"json parse failed after {total_attempts} attempts. "
                f"Raw response: {content[:200]!r}"
            )

        decision = parsed.get("decision", "review")
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        flags = parsed.get("flags", "")
        clean_text = parsed.get("clean_text", "") or text
        reason = parsed.get("reason", "")

        if decision == "block" and confidence >= BLOCK_THRESHOLD:
            send_to_stage4 = False
        else:
            send_to_stage4 = True

        logger.info(
            f"llm_entry_judge: {decision} - send_to_stage4={send_to_stage4} "
            f"confidence={confidence:.2f}"
        )
        return LlmTriageResult(
            decision=decision,
            confidence=round(confidence, 4),
            flags=flags,
            clean_text=clean_text,
            reason=reason or None,
            original_text=text,
            send_to_stage4=send_to_stage4,
            meta={
                "digest_sent": digest,
                "elapsed_s": elapsed_total,
                "model": MODEL,
                "backend": BACKEND,
                "thinking": thinking_log,
                "normalization_method": "llm",
            },
        )
    except Exception as e:
        logger.error(f"llm_entry_judge error: {e}")

        clean_text = _fallback_normalize_text(text)

        logger.info(
            f"llm_entry_judge: AUTO-NORMALIZATION via regex (NOT LLM) | "
            f"original: {text!r} - normalized: {clean_text!r}"
        )

        return LlmTriageResult(
            decision="review",
            confidence=0.5,
            flags="",
            clean_text=clean_text,
            reason="fail-safe error (regex normalization)",
            original_text=text,
            send_to_stage4=True,
            meta={
                "error": str(e),
                "normalization_method": "fallback_regex",
                "fallback_applied": True,
            },
        )
