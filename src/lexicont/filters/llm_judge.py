import re
import time
import json
import requests
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from lexicont.config_loader import load_config, load_rules, get_patterns_path
from lexicont.logger import get_logger
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer

logger = get_logger("llm_judge")

# read rules once at import - rules don't change per-call
rules_data = load_rules()
llm_rules = rules_data.get("llm", {})
THREAT_MARKERS = llm_rules.get("threat_markers", [])
BLACKLIST = llm_rules.get("blacklist", [])
PROFANITY = llm_rules.get("profanity", [])

# intentionally left as module-level for patching in notebooks
BACKEND = None
MODEL = None
API_URL = None
RAG_ENABLED = None
COLLECTION_NAME = None
EMBEDDING_MODEL_NAME = None
TOP_K = None
SCORE_THRESHOLD = None
STORE_BACKEND = None
SERVER_HOST = None
SERVER_PORT = None
SERVER_API_KEY = None
AUTO_CREATE = None
AUTO_LOAD = None
PATTERNS_REL_PATH = None

_rag_client = None
_rag_embedder = None


def _get_cfg():
    return load_config()["llm_judge"]


def _get_rag_cfg(cfg):
    return cfg.get("rag", {"enabled": False})


def _resolve_module_vars():
    """read config and update module-level vars. Called at start of each run()."""
    global BACKEND, MODEL, API_URL, RAG_ENABLED, COLLECTION_NAME
    global EMBEDDING_MODEL_NAME, TOP_K, SCORE_THRESHOLD, STORE_BACKEND
    global SERVER_HOST, SERVER_PORT, SERVER_API_KEY, AUTO_CREATE, AUTO_LOAD
    global PATTERNS_REL_PATH

    cfg = _get_cfg()
    rag_cfg = _get_rag_cfg(cfg)

    BACKEND = cfg["backend"]
    MODEL = cfg[BACKEND]["model"]
    API_URL = cfg[BACKEND]["url"]
    RAG_ENABLED = rag_cfg.get("enabled", False)
    COLLECTION_NAME = rag_cfg.get("collection_name", "rag_patterns")
    EMBEDDING_MODEL_NAME = rag_cfg.get(
        "embedding_model", "intfloat/multilingual-e5-small"
    )
    TOP_K = rag_cfg.get("top_k", 1)
    SCORE_THRESHOLD = rag_cfg.get("score_threshold", 0.65)
    STORE_BACKEND = rag_cfg.get("store_backend", "inmemory")
    SERVER_HOST = rag_cfg.get("server_host", "localhost")
    SERVER_PORT = rag_cfg.get("server_port", 6333)
    SERVER_API_KEY = rag_cfg.get("server_api_key", None) or None
    AUTO_CREATE = rag_cfg.get("auto_create", False)
    AUTO_LOAD = rag_cfg.get("auto_load", False)
    PATTERNS_REL_PATH = rag_cfg.get("patterns_path", "rag/patterns.jsonl")


def _load_patterns_into_collection():
    path = get_patterns_path()

    logger.info("=" * 70)
    logger.info(f"RAG: get_patterns_path() → {path}")
    logger.info(f"RAG: file exists? {path.exists()}")
    logger.info(f"RAG: absolute path: {path.absolute()}")

    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        logger.info(f"RAG: file opened successfully, {len(lines)} lines total")

        patterns = [
            json.loads(line.strip())
            for line in lines
            if line.strip() and not line.startswith("#")
        ]

        logger.info(f"RAG: successfully parsed {len(patterns)} valid JSON patterns")

        if not patterns:
            logger.warning("RAG: patterns list is empty after parsing!")
            return

    except Exception as e:
        logger.error(f"RAG: FAILED to read/parse {path} → {e}")
        raise

    if patterns:
        first = patterns[0]
        logger.info(f"RAG: FIRST pattern example: {first.get('text', '')[:80]!r}...")

    logger.info(f"RAG: indexing {len(patterns)} patterns...")

    texts = [p["text"] for p in patterns]
    embeddings = _rag_embedder.encode(texts, normalize_embeddings=True)

    points = [
        PointStruct(
            id=i,
            vector=emb.tolist(),
            payload={
                "text": p["text"],
                "label": p["label"],
                "category": p.get("category", ""),
            },
        )
        for i, (emb, p) in enumerate(zip(embeddings, patterns))
    ]

    _rag_client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info(f"RAG: upserted {len(points)} patterns into collection")


def _init_rag():
    global _rag_client, _rag_embedder

    logger.info("RAG: _init_rag() started")

    if _rag_embedder is not None:
        logger.info("RAG: embedder already loaded, skipping init")
        return

    logger.info(f"RAG: loading embedder {EMBEDDING_MODEL_NAME} (CPU)")
    _rag_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    backend = STORE_BACKEND
    if backend == "qdrant_server":
        try:
            if "://" in SERVER_HOST:
                _rag_client = QdrantClient(
                    url=SERVER_HOST, api_key=SERVER_API_KEY or None
                )
            else:
                _rag_client = QdrantClient(
                    host=SERVER_HOST,
                    port=SERVER_PORT,
                    api_key=SERVER_API_KEY,
                )
            _rag_client.get_collections()
            logger.info(
                f"RAG: connected to Qdrant server at {SERVER_HOST}:{SERVER_PORT}"
            )

            if not _rag_client.collection_exists(COLLECTION_NAME):
                if AUTO_CREATE:
                    dim = _rag_embedder.get_sentence_embedding_dimension()
                    _rag_client.create_collection(
                        collection_name=COLLECTION_NAME,
                        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                    )
                    logger.info(f"RAG: created collection '{COLLECTION_NAME}'")
                    if AUTO_LOAD:
                        _load_patterns_into_collection()
                else:
                    raise RuntimeError(
                        f"Collection '{COLLECTION_NAME}' does not exist and auto_create is false"
                    )
            else:
                if AUTO_LOAD and _rag_client.count(COLLECTION_NAME).count == 0:
                    _load_patterns_into_collection()

            logger.info(
                f"RAG ready: {_rag_client.count(COLLECTION_NAME).count} patterns"
            )
            return

        except Exception as e:
            logger.warning(
                f"RAG: failed to connect to Qdrant server: {e}. Falling back to inmemory."
            )

    # inmemory fallback
    _rag_client = QdrantClient(":memory:")
    if not _rag_client.collection_exists(COLLECTION_NAME):
        dim = _rag_embedder.get_sentence_embedding_dimension()
        _rag_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        _load_patterns_into_collection()
    else:
        if _rag_client.count(COLLECTION_NAME).count == 0:
            _load_patterns_into_collection()

    logger.info(
        f"RAG ready (inmemory): {_rag_client.count(COLLECTION_NAME).count} patterns"
    )


def get_rag_context(text):
    if not RAG_ENABLED:
        logger.debug("RAG: disabled, skipping search.")
        return ""

    if _rag_embedder is None:
        _init_rag()

    try:
        query_vec = _rag_embedder.encode([text], normalize_embeddings=True)[0].tolist()
        start_time = time.perf_counter()
        results = _rag_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=TOP_K,
            score_threshold=SCORE_THRESHOLD,
        ).points

        search_duration = time.perf_counter() - start_time
        logger.debug(f"RAG: search took {search_duration:.4f}s")

        if not results:
            logger.debug(f"RAG: no matches found for query text (len={len(text)}).")
            return ""

        scores_found = [hit.score for hit in results]
        max_score = max(scores_found) if scores_found else 0.0
        logger.debug(
            f"RAG: Found {len(results)} matches. Max score: {max_score:.4f}. Scores: {scores_found}"
        )
        for i, hit in enumerate(results):
            label = hit.payload.get("label", "N/A")
            matched_text_snippet = hit.payload.get("text", "")[:100]
            logger.debug(
                f"RAG: Match {i + 1}/{len(results)} - Score: {hit.score:.4f}, Label: '{label}', Text snippet: '{matched_text_snippet}...'"
            )

        lines = []
        for hit in results:
            label = hit.payload.get("label", "")
            matched_text_snippet = hit.payload.get("text", "")[:100]
            line = f"- Label: {label}; Matched Pattern: '{matched_text_snippet}...'"
            lines.append(line)

        rag_context_str = "RAG_CONTEXT:\n" + "\n".join(lines)
        logger.debug(f"RAG: Final context sent to LLM: {rag_context_str}")
        return rag_context_str

    except Exception as e:
        logger.warning(f"RAG search failed: {e}")
        return ""


class LlmJudgeResult(BaseModel):
    stage: str = "llm_judge"
    decision: str
    confidence: float
    reason: Optional[str]
    original_text: str
    normalized_text: str
    meta: Dict[str, Any] = Field(default_factory=dict)


def clamp(val, default=0.0):
    try:
        return max(0.0, min(1.0, float(val)))
    except Exception:
        return default


def parse_json(text):
    if not text:
        return None

    if "</think>" in text:
        candidate = text.split("</think>", 1)[-1].strip()
        logger.debug("parse_json: found </think>, extracting JSON after it")
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

    return None


def apply_threat_boost(scores, text_lower):
    if any(m in text_lower for m in THREAT_MARKERS):
        scores["violence"] = min(1.0, scores.get("violence", 0) + 0.15)
        scores["manipulation"] = min(1.0, scores.get("manipulation", 0) + 0.10)
    return scores


def make_decision(top_score, top_cat, conf, cfg):
    if conf < cfg["min_confidence"]:
        return "review", f"low confidence {conf:.2f}"
    if top_score >= cfg["block_threshold"]:
        return "block", f"{top_cat}={top_score:.2f}"
    if top_score >= cfg["review_threshold"]:
        return "review", f"{top_cat}={top_score:.2f}"
    return "pass", "no violation"


def call_llm(messages, cfg):
    try:
        if BACKEND == "ollama":
            payload = {
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "think": cfg["thinking_enabled"],
                "format": "json",
                "options": {
                    "temperature": 0.0 if not cfg["thinking_enabled"] else 0.2,
                    "top_p": 0.95,
                    "top_k": 20,
                    "num_predict": cfg["max_tokens"],
                },
            }
        else:
            payload = {
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "max_tokens": cfg["max_tokens"],
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 20,
                "thinking_forced_open": cfg["thinking_enabled"],
            }

        t0 = time.perf_counter()
        r = requests.post(API_URL, json=payload, timeout=cfg["timeout"])
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
            f"llm_judge call took {elapsed}s thinking_enabled={cfg['thinking_enabled']}"
        )
        return thinking, content, elapsed
    except Exception as e:
        logger.error(f"llm_judge call error: {e}")
        return "", "", 0.0


def run(
    text,
    stage1=None,
    stage2=None,
    stage3=None,
    original_text=None,
):
    _resolve_module_vars()
    cfg = _get_cfg()

    try:
        rag_info = get_rag_context(text)

        if RAG_ENABLED and rag_info:
            system_content = cfg["rag_system_prompt"]
            logger.debug("RAG: Using rag_system_prompt.")
        else:
            if RAG_ENABLED:
                logger.debug(
                    "RAG: Enabled but no context found, using standard system_prompt."
                )
            else:
                logger.debug("RAG: Not enabled, using standard system_prompt.")
            system_content = cfg["system_prompt"]

        if original_text and original_text != text:
            text_block = (
                f'Original text (before normalization): """{original_text}"""\n'
                f'Normalized text (use for analysis): """{text}"""'
            )
            logger.debug(
                f"llm_judge: passing both texts - original={original_text!r}, "
                f"normalized={text!r}"
            )
        else:
            text_block = f'Text: """{text}"""'

        user_content = f"{rag_info}\n{text_block}" if rag_info else text_block
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        logger.debug(
            f"Final LLM prompt - System: {system_content}, User: {user_content}"
        )
        parsed = None
        thinking = ""
        for attempt in range(cfg["retries"]):
            thinking, content, _ = call_llm(messages, cfg)
            logger.error(f"RAW LLM RESPONSE:\n{content!r}")
            logger.error(f"Response length: {len(content) if content else 0}")
            parsed = parse_json(content)
            if parsed:
                break
        if not parsed:
            raise ValueError("json parse failed")
        scores = {c: clamp(parsed.get(c, 0.0)) for c in cfg["categories"]}
        confidence = clamp(parsed.get("confidence", 0.0))
        scores = apply_threat_boost(scores, text.lower())
        top_category = max(scores, key=scores.get)
        top_score = scores[top_category]
        decision, reason = make_decision(top_score, top_category, confidence, cfg)
        logger.info(
            f"llm_judge: {decision} {top_category}={top_score:.2f} conf={confidence:.2f}"
        )
        return LlmJudgeResult(
            decision=decision,
            confidence=round(confidence, 4),
            reason=reason,
            original_text=text,
            normalized_text=text.lower(),
            meta={
                "scores": scores,
                "flags": parsed.get("flags"),
                "check": parsed.get("_check"),
                "thinking": thinking,
                "stage1": stage1,
                "stage2": stage2,
                "stage3": stage3,
                "rag_used": bool(rag_info),
                "rag_context": rag_info if rag_info else "N/A",
                "rag_search_text": text,
                "rag_original_text": original_text
                if original_text and original_text != text
                else "",
            },
        )
    except Exception as e:
        logger.error(f"llm_judge error: {e}")
        return LlmJudgeResult(
            decision="block",
            confidence=1.0,
            reason="fail-safe error",
            original_text=text,
            normalized_text=text.lower(),
        )
