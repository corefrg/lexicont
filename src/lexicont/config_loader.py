from __future__ import annotations

import importlib.resources
import os
from pathlib import Path

import yaml

from lexicont.logger import get_logger

logger = get_logger("config_loader")


def _load_yaml(path):
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"Cannot load YAML from {path}: {e}") from e


def _bundled(relative):
    ref = importlib.resources.files("lexicont") / relative
    with importlib.resources.as_file(ref) as p:
        return p


def _resolve_yaml(explicit, env_var, local_name, bundled_relative, label):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"[{label}] path not found: {p}")
        logger.debug(f"[{label}] explicit: {p}")
        return _load_yaml(p)

    env_val = os.environ.get(env_var)
    if env_val:
        p = Path(os.path.expanduser(env_val))
        if p.exists():
            logger.debug(f"[{label}] env {env_var}: {p}")
            return _load_yaml(p)
        logger.warning(
            f"[{label}] {env_var}={env_val} not found, falling back to default"
        )

    local = Path.cwd() / local_name
    if local.exists():
        logger.debug(f"[{label}] local cwd: {local}")
        return _load_yaml(local)

    bundled = _bundled(bundled_relative)
    logger.debug(f"[{label}] bundled default: {bundled}")
    return _load_yaml(bundled)


def _resolve_path(explicit, env_var, local_name, bundled_relative, label):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"[{label}] path not found: {p}")
        logger.debug(f"[{label}] explicit: {p}")
        return p

    env_val = os.environ.get(env_var)
    if env_val:
        p = Path(os.path.expanduser(env_val))
        if p.exists():
            logger.debug(f"[{label}] env {env_var}: {p}")
            return p
        logger.warning(
            f"[{label}] {env_var}={env_val} not found, falling back to default"
        )

    local = Path.cwd() / local_name
    if local.exists():
        logger.debug(f"[{label}] local cwd: {local}")
        return local

    bundled = _bundled(bundled_relative)
    logger.debug(f"[{label}] bundled default: {bundled}")
    return bundled


def load_config(config_path=None):
    return _resolve_yaml(
        explicit=config_path,
        env_var="LEXICONT_CONFIG",
        local_name="moderation_config.yaml",
        bundled_relative="config/moderation_config.yaml",
        label="config",
    )


def load_rules(rules_path=None):
    return _resolve_yaml(
        explicit=rules_path,
        env_var="LEXICONT_RULES",
        local_name="moderation_rules.v1.yaml",
        bundled_relative="config/moderation_rules.v1.yaml",
        label="rules",
    )


def get_patterns_path(patterns_path=None):
    return _resolve_path(
        explicit=patterns_path,
        env_var="LEXICONT_PATTERNS",
        local_name="patterns.jsonl",
        bundled_relative="rag/patterns.jsonl",
        label="patterns",
    )
