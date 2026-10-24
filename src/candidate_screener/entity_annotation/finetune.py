"""Offline frozen-backbone GLiNER pilot training with separately verified heads."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

from candidate_screener.config import PROJECT_ROOT
from candidate_screener.extraction.medium_assets import MODEL_DIR, digest, verify_assets
from candidate_screener.extraction.medium_model import load_local_model, runtime_versions

VERSION = "entity-frozen-backbone-v1"
CONFIG = {"seed": 0, "epochs": 3, "batch_size": 1, "gradient_accumulation": 4,
          "learning_rate": 1e-4, "weight_decay": .01, "max_gradient_norm": 1.,
          "threads": 4, "device": "cpu", "precision": "float32",
          "loss_alpha": .8, "loss_gamma": 0., "label_smoothing": 0.,
          "loss_reduction": "sum", "negatives": 1., "masking": "label"}
PROMPTS = ["technology, software tool or programming language", "professional skill or competency",
           "job title or occupation", "academic degree or qualification"]


def apply_tokenizer(model, policy: dict) -> None:
    """Restore a frozen text-only tokenizer and markerV0 geometry without weights."""
    from .character_tokenization import BoundaryTokenizer
    tokenizer = BoundaryTokenizer(policy)
    width = policy["max_width_units"]
    if type(width) is not int or width < 1:
        raise ValueError("Tokenizer requires a positive frozen span width")
    if model.config.span_mode != "markerV0":
        raise ValueError("Tokenizer width adaptation supports verified markerV0 head only")
    before = {name: tuple(value.shape) for name, value in model.model.state_dict().items()}
    head = model.model.span_rep_layer.span_rep_layer
    if not hasattr(head, "max_width"):
        raise ValueError("Verified markerV0 head geometry unavailable")
    model.config.max_width = width
    model.data_processor.config.max_width = width
    head.max_width = width
    model.data_processor.words_splitter = tokenizer
    after = {name: tuple(value.shape) for name, value in model.model.state_dict().items()}
    if before != after:
        raise ValueError("Tokenizer geometry changed learned parameter shapes")
    model.tokenizer_policy = policy


def freeze_backbone(model) -> dict:
    trainable, frozen = [], []
    for name, parameter in model.model.named_parameters():
        parameter.requires_grad_(not name.startswith("token_rep_layer."))
        (trainable if parameter.requires_grad else frozen).append(name)
    if not trainable or not frozen:
        raise ValueError("Expected nonempty frozen backbone and trainable representation parameters")
    return {"trainable": trainable, "frozen": frozen}


def _code_digests():
    directory = Path(__file__).parent
    return {name: digest(directory / name) for name in ("finetune.py", "finetune_data.py", "character_tokenization.py")}


def _snapshot(path: Path) -> tuple[list[dict], dict]:
    gold = path / "gold.jsonl" if path.is_dir() else path
    manifest_path = gold.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if digest(gold) != manifest["jsonl_sha256"]:
        raise ValueError("Training snapshot digest mismatch")
    rows = [json.loads(line) for line in gold.read_text().splitlines() if line.strip()]
    if len(rows) != 10 or any(row["split"] != "train" or row.get("reviewed_all") is not True for row in rows):
        raise ValueError("Approved snapshot requires exactly ten completely reviewed train documents")
    if len({row["doc_id"] for row in rows}) != 10:
        raise ValueError("Duplicate training document")
    if any(set(row["categories"]) != {"KNOWLEDGE", "SKILL", "TITLE", "DEGREE"} for row in rows):
        raise ValueError("Training snapshot ontology mismatch")
    for row in rows:
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["source_sha256"]:
            raise ValueError("Training source digest mismatch")
    return rows, {"jsonl_sha256": digest(gold), "manifest_sha256": digest(manifest_path),
                  "source_revisions": manifest.get("source_revisions", [])}


def train_records(model, records: list[dict], *, pilot=False) -> dict:
    """Native collator/forward loss; every accumulation group normalizes its size."""
    import torch
    from gliner.data_processing.collator import DataCollator
    if not records:
        raise ValueError("No aligned training windows")
    torch.set_num_threads(CONFIG["threads"])
    torch.manual_seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])
    names = freeze_backbone(model)
    collator = DataCollator(model.config, data_processor=model.data_processor,
                            prepare_labels=True, entity_types=[PROMPTS])
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    model.train()
    # Frozen weights also keep their stochastic encoder dropout disabled.
    model.model.token_rep_layer.eval()
    started = time.monotonic()
    losses, steps = [], 0
    randomizer = random.Random(CONFIG["seed"])
    for _ in range(1 if pilot else CONFIG["epochs"]):
        order = list(range(len(records)))
        randomizer.shuffle(order)
        for offset in range(0, len(order), CONFIG["gradient_accumulation"]):
            group = order[offset:offset + CONFIG["gradient_accumulation"]]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                batch = collator([records[index]])
                output = model(**batch, alpha=CONFIG["loss_alpha"], gamma=CONFIG["loss_gamma"],
                               label_smoothing=CONFIG["label_smoothing"], reduction=CONFIG["loss_reduction"],
                               negatives=CONFIG["negatives"], masking=CONFIG["masking"])
                loss = output.loss
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                (loss / len(group)).backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(parameters, CONFIG["max_gradient_norm"], error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            if pilot:
                break
        if pilot:
            break
    model.eval()
    elapsed = time.monotonic() - started
    return {"version": VERSION, "parameters": names, "windows": len(records), "optimizer_steps": steps,
            "microbatches": len(losses), "elapsed_seconds": elapsed,
            "training_loss_mean": sum(losses) / len(losses),
            "estimated_full_seconds": elapsed / steps * math.ceil(len(records) / CONFIG["gradient_accumulation"]) * CONFIG["epochs"]}


def save_checkpoint(model, destination: Path, *, snapshot: dict, result: dict) -> dict:
    import torch
    destination = destination.resolve()
    if not destination.is_relative_to((PROJECT_ROOT / "data").resolve()):
        raise ValueError("Fine-tuned weights must remain under ignored data/")
    if destination.exists():
        raise FileExistsError("Fine-tuned revision already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".finetune-", dir=destination.parent))
    state = {name: value.detach().cpu() for name, value in model.model.state_dict().items()
             if not name.startswith("token_rep_layer.")}
    try:
        torch.save(state, staging / "representation.pt")
        manifest = {"version": VERSION, "base": model.medium_manifest,
                    "snapshot": snapshot, "runtime": runtime_versions(), "config": CONFIG,
                    "code_sha256": _code_digests(), "result": result,
                    "artifact_sha256": {"representation.pt": digest(staging / "representation.pt")}}
        if hasattr(model, "tokenizer_policy"):
            manifest["tokenizer_policy"] = model.tokenizer_policy
            manifest["tokenizer_sha256"] = model.tokenizer_policy["tokenizer_sha256"]
            if model.tokenizer_policy["snapshot_sha256"] != snapshot["jsonl_sha256"]:
                raise ValueError("Tokenizer and training snapshot mismatch")
        payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        manifest["model_id"] = hashlib.sha256(payload.encode()).hexdigest()[:24]
        (staging / "finetune-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.rename(staging, destination)
        return manifest
    finally:
        if staging.exists():
            import shutil
            shutil.rmtree(staging)


def load_finetuned_model(path: Path, base_path: Path = MODEL_DIR, *, snapshot_sha256=None):
    """Verify the unchanged base and separately identified learned representation."""
    import torch
    path = Path(path)
    manifest = json.loads((path / "finetune-manifest.json").read_text())
    payload = {key: value for key, value in manifest.items() if key != "model_id"}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    if manifest.get("model_id") != expected or manifest.get("version") != VERSION:
        raise ValueError("Fine-tuned manifest identity mismatch")
    if snapshot_sha256 is not None and manifest["snapshot"]["jsonl_sha256"] != snapshot_sha256:
        raise ValueError("Fine-tuned snapshot mismatch")
    base = verify_assets(base_path)
    for key in ("model_id", "revision", "artifact_sha256", "tokenizer", "encoder"):
        if base[key] != manifest["base"][key]:
            raise ValueError("Fine-tuned base provenance mismatch")
    if digest(path / "representation.pt") != manifest["artifact_sha256"]["representation.pt"]:
        raise ValueError("Fine-tuned representation digest mismatch")
    model = load_local_model(base_path)
    if "tokenizer_policy" in manifest:
        if (manifest["tokenizer_sha256"] != manifest["tokenizer_policy"]["tokenizer_sha256"] or
                manifest["snapshot"]["jsonl_sha256"] != manifest["tokenizer_policy"]["snapshot_sha256"]):
            raise ValueError("Fine-tuned tokenizer provenance mismatch")
        apply_tokenizer(model, manifest["tokenizer_policy"])
    state = torch.load(path / "representation.pt", map_location="cpu", weights_only=True)
    full = model.model.state_dict()
    expected_names = {name for name in full if not name.startswith("token_rep_layer.")}
    if set(state) != expected_names:
        raise ValueError("Fine-tuned representation parameter set mismatch")
    full.update(state)
    model.model.load_state_dict(full, strict=True)
    model.eval()
    model.finetune_manifest = manifest
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--tokenizer", type=Path, help="Frozen policy; otherwise derive from the training snapshot")
    args = parser.parse_args()
    if (args.train or args.synthetic_smoke) and args.destination is None:
        parser.error("Saving requires a new --destination")
    if not args.synthetic_smoke and args.snapshot is None:
        parser.error("--pilot/--train require --snapshot")
    import torch
    torch.set_num_threads(CONFIG["threads"])
    torch.manual_seed(CONFIG["seed"])
    model = load_local_model()
    policy = json.loads(args.tokenizer.read_text()) if args.tokenizer else None
    if args.synthetic_smoke:
        if policy is not None:
            apply_tokenizer(model, policy)
            units = list(model.data_processor.words_splitter("x" + " " * (policy["max_width_units"] - 1)))
            if len(units) != policy["max_width_units"]:
                raise ValueError("Synthetic fixture does not exercise the frozen span width")
            sample = {"tokenized_text": [unit[0] for unit in units],
                      "ner": [[0, len(units) - 1, PROMPTS[0]]]}
        else:
            sample = {"tokenized_text": ["Python", "developer"], "ner": [[0, 0, PROMPTS[0]], [1, 1, PROMPTS[2]]]}
        before = {name: parameter.detach().clone() for name, parameter in model.model.named_parameters()
                  if not name.startswith("token_rep_layer.")}
        frozen_before = {name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
                         for name, value in model.model.state_dict().items() if name.startswith("token_rep_layer.")}
        result = train_records(model, [sample], pilot=True)
        if not any(not torch.equal(before[name], value) for name, value in model.model.named_parameters() if name in before):
            raise ValueError("Synthetic training did not change representation weights")
        if any(frozen_before[name] != hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
               for name, value in model.model.state_dict().items() if name in frozen_before):
            raise ValueError("Synthetic training changed frozen backbone")
        identity = {"jsonl_sha256": policy["snapshot_sha256"] if policy else "synthetic-only",
                    "manifest_sha256": "synthetic-only"}
        save_checkpoint(model, args.destination, snapshot=identity, result=result)
        restored = load_finetuned_model(args.destination, snapshot_sha256=identity["jsonl_sha256"])
        if any(not torch.equal(value, restored.model.state_dict()[name]) for name, value in model.model.state_dict().items()):
            raise ValueError("Synthetic save/reload differs")
        print(json.dumps({"status": "synthetic-smoke-passed", "result": result, "reload_verified": True}))
        return
    from .finetune_data import prepare_training
    rows, identity = _snapshot(args.snapshot)
    from .character_tokenization import derive_tokenizer
    policy = policy or derive_tokenizer(rows, model.data_processor.words_splitter)
    if policy["snapshot_sha256"] != identity["jsonl_sha256"]:
        raise ValueError("Tokenizer policy belongs to another snapshot")
    apply_tokenizer(model, policy)
    prepared = prepare_training(rows, model, tokenizer_manifest=policy)
    if prepared["diagnostics"]["blocked"]:
        print(json.dumps({"status": "blocked", "diagnostics": prepared["diagnostics"]}))
        raise SystemExit(2)
    result = train_records(model, prepared["records"], pilot=args.pilot)
    result["snapshot"] = identity
    if args.train:
        manifest = save_checkpoint(model, args.destination, snapshot=identity, result=result)
        reloaded = load_finetuned_model(args.destination, snapshot_sha256=identity["jsonl_sha256"])
        if any(not torch.equal(value, reloaded.model.state_dict()[name]) for name, value in model.model.state_dict().items()):
            raise ValueError("Saved checkpoint reload differs")
        result["model_id"] = manifest["model_id"]
        result["reload_verified"] = True
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
