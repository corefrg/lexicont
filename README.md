# Lexicont: lightweight policy driven agent for text moderation    

![Lexicont Logo](assets/logo.png)  

Lexicont is a moderation system built as a **policy-driven agent** that combines fast rule-based filters, machine learning, and LLM reasoning. It processes the majority of inputs in milliseconds and only invokes the LLM when confidence is low, making it suitable for production environments where low latency is required.

## Overview
- Fast layers - profanity detection, fuzzy matching, toxicity ML - run quickly.
- Early stop - high-confidence blocks skip further processing.
- LLM layers - intermediate triage and final judgment with Qwen via llamacpp or Ollama.
- RAG support - Qdrant vector database for retrieving similar examples.
- Policy driven agent - explicit control loop ensures deterministic behaviour.

## Architecture

The moderation pipeline follows a fixed order of stages:

1. profanity_filter - dictionary + leetspeak detection
2. fuzzy_trigger - partial ratio matching
3. toxicity_ml - multilingual toxicity classifier
4. llm_entry_judge - LLM for text normalisation and decision on stage 4
5. llm_judge - final LLM with RAG

Control logic:
- After stage 1 or stage 2, if confidence >= 0.85 and decision is block, the pipeline stops.
- Stage 4 is invoked only if either the triage layer (stage 3.5) explicitly allows it, or triage is disabled and max confidence < 0.80.

## Quick start

### Local (Poetry)

```bash
git clone https://github.com/corefrg/lexicont.git
cd lexicont
poetry install
poetry run lexicont check "offensive text"
```

### Docker Compose
```bash
docker-compose up -d
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"text":"buy fake documents"}'
```

## Usage

### Command line

```bash
# legacy syntax
poetry run lexicont "text to moderate"
poetry run lexicont "text" --log-level DEBUG --verbose

# explicit subcommand
poetry run lexicont check "text to moderate"
poetry run lexicont check "text" --log-level DEBUG --verbose

# interactive mode
poetry run lexicont
```

### HTTP API
Start the server:
```bash
poetry run uvicorn lexicont.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- POST /moderate - returns decision, confidence, and stage details
- GET /health - status check

### From Python

```python
from lexicont.pipeline import run
result = run("text")
print(result.final_decision)  # block, review, or pass
print(result.max_confidence)
```

## Configuration

### Config files

| File                     | Purpose                                      | Env var              |
|--------------------------|----------------------------------------------|----------------------|
| moderation_config.yaml   | Main settings: thresholds, stage toggles, LLM, RAG | LEXICONT_CONFIG      |
| moderation_rules.v1.yaml | Phrase lists for profanity and fuzzy stages   | LEXICONT_RULES       |
| patterns.jsonl           | Semantic patterns for the RAG layer           | LEXICONT_PATTERNS    |

Each file has a built-in default bundled with the package. To override, either set the env var or place the file in the working directory.

### Copy defaults and edit

```bash
poetry run lexicont init --dir my_configs

set LEXICONT_CONFIG=my_configs\moderation_config.yaml
set LEXICONT_RULES=my_configs\moderation_rules.v1.yaml
set LEXICONT_PATTERNS=my_configs\patterns.jsonl

# Linux / macOS
export LEXICONT_CONFIG=my_configs/moderation_config.yaml
export LEXICONT_RULES=my_configs/moderation_rules.v1.yaml
export LEXICONT_PATTERNS=my_configs/patterns.jsonl
```

### Key options in moderation_config.yaml
- general.early_stop_confidence - threshold for early termination (default 0.85)
- general.stage4_trigger_confidence - threshold to invoke LLM when triage disabled (default 0.80)
- general.enable_stage1/2/3 - toggle individual stages on or off
- llm_judge.backend - llamacpp or ollama
- llm_judge.rag.store_backend - inmemory or qdrant_server

For Docker, use host.docker.internal to reach LLM services running on the host.

### moderation_rules.v1.yaml structure
```yaml
categories:
  profanity:
    - phrase one
  illegal:
    - buy fake license
  my_category:
    - custom phrase
```

### patterns.jsonl structure
One JSON object per line:
```json
{"text": "buy a license through traffic police", "label": "offer to buy forged documents", "category": "illegal"}
```

## Development
```bash
poetry install
pre-commit install
ruff format src tests
ruff check --fix src tests
```

To add a new filter, implement a function in filters/ and register it in agent.py.

## License
MIT

