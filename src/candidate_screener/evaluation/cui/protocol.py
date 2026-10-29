"""Content-addressed study identities and immutable local artifact writes."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import subprocess


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    """Never replace a run, review mapping or frozen manifest silently."""
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def code_identity():
    root = Path(__file__).resolve().parents[2]
    hashes = {str(p.relative_to(root)): file_digest(p)
              for p in sorted(root.rglob("*")) if p.suffix in {".py", ".txt", ".json"}}
    # The content digest includes uncommitted package changes. Git identity is descriptive only.
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return {"package_sha256": digest(hashes), "files": hashes,
            "git_head": git.stdout.strip() if git.returncode == 0 else None}


def runtime_identity(composer=None):
    from candidate_screener.llm.conversation_prose import SYSTEM, PROMPT_VERSION
    config = getattr(composer, "config", None)
    # Explicit allowlist: never persist API credentials or the entire environment.
    fields = ("model", "revision", "runtime", "quantization", "max_context_tokens",
              "max_output_tokens", "timeout_seconds", "temperature", "json_mode",
              "enable_thinking", "expected_sha256")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
            "generator": {key: getattr(config, key, None) for key in fields} if config else None}


def freeze(fixtures, protocol_path, *, model_manifest=None, qa_manifest=None):
    from .fixtures import validate_fixtures
    validate_fixtures(fixtures)
    review = fixtures.get("review", {})
    if (fixtures.get("review_status") != "reviewed" or not review.get("author")
            or not isinstance(review.get("reviewers"), list)
            or len(set(review["reviewers"])) != 2
            or any(not isinstance(x, str) or not x.strip() for x in review["reviewers"])):
        raise ValueError("Freeze requires reviewed fixtures, an author and two named reviewers")
    if model_manifest is None or qa_manifest is None:
        raise ValueError("Freeze requires explicit generator and QA manifests")
    composer = configured_generator(model_manifest)
    result = {"schema_version": 1, "fixtures_sha256": digest(fixtures),
              "protocol_sha256": file_digest(protocol_path), "code": code_identity(),
              "runtime": runtime_identity(composer), "review": review,
              "bootstrap": {"seed": 0, "replicates": 10000},
              "cache_policy": "shared frozen QA outputs within standalone pairs; independent dialogue histories",
              "scope": "service-level synthetic QA; no portal UI or asynchronous GLiNER benchmark"}
    for key, path in (("model_manifest", model_manifest), ("qa_manifest", qa_manifest)):
        if path is not None:
            result[key] = {"path": str(Path(path).resolve()), "sha256": file_digest(path)}
    result["study_id"] = digest(result)
    return result


def verify_freeze(manifest, fixtures, protocol_path):
    identity = {key: value for key, value in manifest.items() if key != "study_id"}
    if manifest.get("study_id") != digest(identity):
        raise ValueError("Study manifest identity mismatch")
    if manifest.get("fixtures_sha256") != digest(fixtures):
        raise ValueError("Frozen fixtures changed")
    if manifest.get("protocol_sha256") != file_digest(protocol_path):
        raise ValueError("Frozen protocol changed")
    if manifest["code"]["package_sha256"] != code_identity()["package_sha256"]:
        raise ValueError("Frozen package changed; create a new study identity")
    current = runtime_identity()
    if {k: v for k, v in manifest["runtime"].items() if k != "generator"} != {k: v for k, v in current.items() if k != "generator"}:
        raise ValueError("Frozen runtime or prompt changed")
    for key in ("model_manifest", "qa_manifest"):
        item = manifest.get(key)
        if not item or file_digest(item["path"]) != item["sha256"]:
            raise ValueError("Live evaluation requires unchanged model and QA manifests")


def configured_generator(path):
    from candidate_screener.llm.conversation_prose import configured_prose_composer
    composer = configured_prose_composer({"SCREENER_QWEN_MANIFEST": str(Path(path).resolve())})
    if composer is None or composer.config.model != "Qwen/Qwen3.5-4B":
        raise ValueError("Pinned Qwen3.5 configuration unavailable; no substitute provider")
    return composer


def verify_selected_runtime(manifest, composer, qa):
    if runtime_identity(composer) != manifest["runtime"]:
        raise ValueError("Selected generation configuration differs from frozen runtime")
    selected = Path(getattr(qa, "manifest_path", "")).resolve()
    if selected != Path(manifest["qa_manifest"]["path"]).resolve():
        raise ValueError("Selected QA manifest differs from frozen runtime")
    if getattr(qa, "timeout", None) != 30.:
        raise ValueError("QA requires the frozen 30-second limit")
