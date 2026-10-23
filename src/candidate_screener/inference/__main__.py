"""Inference artifacts (plan 2026-10-23-inference-api).

    uv run python -m candidate_screener.inference --cache-parts   # concept df + TF-IDF space
    uv run python -m candidate_screener.inference --freeze        # L1 -> output/rerank/l1-chat3-model.json
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-parts", action="store_true", help="save the fitted concept df and TF-IDF space")
    ap.add_argument("--freeze", action="store_true", help="fit L1 on chat-3 once and write its record")
    args = ap.parse_args(argv)
    if not (args.cache_parts or args.freeze):
        ap.error("nothing to do")
    if args.cache_parts:
        from candidate_screener.inference import parts
        parts.cache_parts()
    if args.freeze:
        from candidate_screener.inference import model
        model.freeze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
