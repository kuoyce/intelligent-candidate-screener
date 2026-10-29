"""Synthetic CUI study tooling. Live execution is always explicit; nothing trains/downloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fixtures import load_fixtures, validate_fixtures
from .protocol import freeze, verify_freeze, write_new


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--fixtures", type=Path)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--fixtures", type=Path)
    frozen.add_argument("--protocol", type=Path, required=True)
    frozen.add_argument("--model-manifest", type=Path, required=True)
    frozen.add_argument("--qa-manifest", type=Path, required=True)
    frozen.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--fixtures", type=Path)
    execute.add_argument("--split", choices=("dev", "eval"), default="dev")
    execute.add_argument("--provider", choices=("stub", "live"), default="stub")
    execute.add_argument("--mode", choices=("paired",), default="paired")
    execute.add_argument("--manifest", type=Path, help="Frozen study manifest, required for evaluation")
    execute.add_argument("--protocol", type=Path)
    execute.add_argument("--model-manifest", type=Path)
    execute.add_argument("--qa-manifest", type=Path)
    execute.add_argument("--approved-live-run", action="store_true",
                         help="Record that owner-approved configuration and runtime budget were obtained")
    execute.add_argument("--run-id", required=True)
    execute.add_argument("--output", type=Path, required=True)
    recovery = commands.add_parser("recover")
    recovery.add_argument("--provider", choices=("stub",), default="stub")
    recovery.add_argument("--output", type=Path, required=True)
    review = commands.add_parser("export-review")
    review.add_argument("--run", type=Path, required=True)
    review.add_argument("--output-dir", type=Path, required=True)
    review.add_argument("--seed", type=int, default=0)
    report = commands.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--reviews", type=Path)
    report.add_argument("--mapping", type=Path)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            result = validate_fixtures(load_fixtures(args.fixtures))
        elif args.command == "freeze":
            result = freeze(load_fixtures(args.fixtures), args.protocol,
                            model_manifest=args.model_manifest, qa_manifest=args.qa_manifest)
            write_new(args.output, result)
            result = {"study_id": result["study_id"], "output": str(args.output)}
        elif args.command == "run":
            result = execute_run(args)
        elif args.command == "recover":
            from .recovery import run_recovery
            result = run_recovery()
            write_new(args.output, result)
            if not result["passed"]:
                parser.exit(1, "Recovery assertions failed; inspect output artifact\n")
        elif args.command == "export-review":
            from .review import create_review_packet
            bundle = create_review_packet(read(args.run), seed=args.seed)
            # Refuse partial overwrite, including identity mappings.
            args.output_dir.mkdir(parents=True, exist_ok=False)
            write_new(args.output_dir / "packet.json", bundle["packet"])
            write_new(args.output_dir / "private-mapping.json", bundle["mapping"])
            template = {"run_id": bundle["packet"]["run_id"], "ratings": [
                {"item_id": item["item_id"], "reviewer_id": reviewer,
                 "total_claims": None, "supported_claims": None, "complete_answer": None,
                 "abstained": None, "correct_abstention": None, "reference_correct": None,
                 "unnecessary_clarification": None, "critical_errors": [],
                 "directness": None, "clarity": None, "usefulness": None}
                for reviewer in ("REVIEWER_1", "REVIEWER_2") for item in bundle["packet"]["items"]]}
            write_new(args.output_dir / "ratings-template.json", template)
            result = {"items": len(bundle["packet"]["items"]),
                      "instruction": "Keep private-mapping.json from reviewers; replace placeholder IDs and complete ratings independently"}
        else:
            from .metrics import summarize_run
            run = read(args.run)
            reviews = None
            if bool(args.reviews) != bool(args.mapping):
                raise ValueError("Supply both --reviews and --mapping")
            if args.reviews:
                reviews = {**read(args.reviews), "mapping": read(args.mapping)}
            report = summarize_run(run, reviews=reviews)
            write_new(args.output, report)
            result = {"run_id": run["run_id"], "human_review": reviews is not None,
                      "output": str(args.output)}
        print(json.dumps(result, indent=2, sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"CUI evaluation: {exc}\n")


def execute_run(args):
    from .runner import run
    from .metrics import validate_run
    if args.output.exists():
        raise FileExistsError("Run output already exists; keep original records")
    fixtures = load_fixtures(args.fixtures)
    composer, qa, manifest = None, None, None
    if args.split == "eval":
        if args.provider != "live" or not args.manifest or not args.protocol:
            raise ValueError("Evaluation requires live provider, frozen manifest and protocol")
        manifest = read(args.manifest)
        verify_freeze(manifest, fixtures, args.protocol)
        for key in ("model_manifest", "qa_manifest"):
            explicit = getattr(args, key)
            path = Path(manifest[key]["path"])
            if explicit and explicit.resolve() != path:
                raise ValueError("Explicit model selection differs from frozen manifest")
            setattr(args, key, path)
    if args.provider == "live":
        if not args.approved_live_run or not args.model_manifest or not args.qa_manifest:
            raise ValueError("Live execution needs approved budget and explicit generator/QA manifests")
        from .protocol import configured_generator
        from candidate_screener.llm.extractive_qa import QaClient
        composer = configured_generator(args.model_manifest)
        qa = QaClient(args.qa_manifest, timeout=30.)
    try:
        result = run(fixtures, split=args.split, provider=args.provider, run_id=args.run_id,
                     composer=composer, qa=qa, manifest=manifest, protocol_path=args.protocol)
        validate_run(result)
        write_new(args.output, result)
    finally:
        if qa is not None:
            qa.close()
    return {"run_id": result["run_id"], "kind": result["kind"], "turn_records": len(result["records"]),
            "errors": sum(r["error"] is not None for r in result["records"]), "output": str(args.output)}


if __name__ == "__main__":
    main()
