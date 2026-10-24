"""Inference artifacts, the API's check and a demo (plan 2026-10-23-inference-api).

    uv run python -m candidate_screener.inference --cache-parts   # concept df + TF-IDF space
    uv run python -m candidate_screener.inference --freeze        # L1 -> output/rerank/l1-chat3-model.json
    uv run python -m candidate_screener.inference --check         # the API vs the committed records (~10 min)
    uv run python -m candidate_screener.inference --demo JD_ID    # one final-set JD's top 10 (no CV text)
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-parts", action="store_true", help="save the fitted concept df and TF-IDF space")
    ap.add_argument("--freeze", action="store_true", help="fit L1 on chat-3 once and write its record")
    ap.add_argument("--check", action="store_true", help="drive the API against the committed records; write nothing")
    ap.add_argument("--demo", metavar="JD_ID", help="print a final-set JD's top 10 and one explanation")
    ap.add_argument("--rerank", action="store_true", help="with --demo: D64 (C2 on the top 50)")
    args = ap.parse_args(argv)
    if not (args.cache_parts or args.freeze or args.check or args.demo):
        ap.error("nothing to do")
    if args.rerank and not args.demo:
        ap.error("--rerank is only valid with --demo")
    if args.cache_parts:
        from candidate_screener.inference import parts
        parts.cache_parts()
    if args.freeze:
        from candidate_screener.inference import model
        model.freeze()
    if args.check:
        from candidate_screener.inference import verify
        same = verify.check()
        for name, ok in same.items():
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        return 0 if all(same.values()) else 1
    if args.demo:
        from candidate_screener.inference import verify
        verify.demo(args.demo, rerank=args.rerank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
