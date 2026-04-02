from __future__ import annotations

import argparse
import importlib.resources
import logging
import shutil
from pathlib import Path

from lexicont.logger import get_logger

logger = get_logger()

_BUNDLED_FILES = [
    ("config/moderation_config.yaml", "moderation_config.yaml"),
    ("config/moderation_rules.v1.yaml", "moderation_rules.v1.yaml"),
    ("rag/patterns.jsonl", "patterns.jsonl"),
]


def _cmd_init(target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    for bundled_rel, out_name in _BUNDLED_FILES:
        ref = importlib.resources.files("lexicont") / bundled_rel
        with importlib.resources.as_file(ref) as src:
            dst = target_dir / out_name
            if dst.exists():
                print(f"  [skip]  {dst}  (already exists)")
            else:
                shutil.copy2(src, dst)
                print(f"  [copy]  {dst}")
    print(
        f"\nConfigs written to {target_dir}\n"
        f"Set env vars to use them:\n"
        f"  export LEXICONT_CONFIG={target_dir}/moderation_config.yaml\n"
        f"  export LEXICONT_RULES={target_dir}/moderation_rules.v1.yaml\n"
        f"  export LEXICONT_PATTERNS={target_dir}/patterns.jsonl\n"
    )


def _cmd_check(text, config_path, verbose):
    from lexicont.pipeline import run

    result = run(text, verbose=verbose, config_path=config_path)
    print(
        f"DECISION: {result.final_decision.upper()} | "
        f"confidence={result.max_confidence:.2f}"
    )
    if result.explanation:
        print(f"REASON:   {result.explanation}")


def _add_log_level(p):
    p.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING"],
    )


def main():
    parser = argparse.ArgumentParser(
        prog="lexicont",
        description="Policy-driven text moderation",
    )
    _add_log_level(parser)

    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="copy default configs to a directory")
    p_init.add_argument("--dir", default=".", metavar="PATH")
    _add_log_level(p_init)

    p_check = sub.add_parser("check", help="moderate a text string")
    p_check.add_argument("text")
    p_check.add_argument("--config", default=None, metavar="PATH")
    p_check.add_argument("--verbose", action="store_true")
    _add_log_level(p_check)

    p_legacy = sub.add_parser("_legacy", help=argparse.SUPPRESS)
    p_legacy.add_argument("text")
    p_legacy.add_argument("--config", default=None, metavar="PATH")
    p_legacy.add_argument("--verbose", action="store_true")
    _add_log_level(p_legacy)

    args, remaining = parser.parse_known_args()

    if args.command is None and remaining:
        legacy_args = p_legacy.parse_args(remaining)
        args.command = "_legacy"
        args.text = legacy_args.text
        args.config = legacy_args.config
        args.verbose = legacy_args.verbose
        if legacy_args.log_level:
            args.log_level = legacy_args.log_level

    log_level = getattr(args, "log_level", None) or "INFO"
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))

    if args.command == "init":
        _cmd_init(Path(args.dir))
        return

    if args.command in ("check", "_legacy"):
        _cmd_check(args.text, args.config, args.verbose)
        return

    # interactive mode
    print("lexicont - enter text or 'exit'")
    while True:
        try:
            t = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        if t.lower() in ("exit", "quit"):
            break
        if t.strip():
            _cmd_check(t, config_path=None, verbose=False)


if __name__ == "__main__":
    main()
