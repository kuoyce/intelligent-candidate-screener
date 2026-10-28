"""G3 equivalence probe for the pinned Qwen3.5-4B GGUF service (plan 2026-10-28-gguf-cross-platform, 03).

Read-only. It starts nothing: it talks to an already running facade (``gguf_server``, 8001) for
``/v1/models`` and to its llama.cpp backend (8002) for ``/apply-template``, ``/tokenize`` and a
greedy ``/completion`` with ``n_probs``, since the facade accepts only the app's own requests.
Only the six invented prompts below are ever sent; there is no option to send other text.

    python -m candidate_screener.llm.gguf_probe --manifest M --out OUT.json
    python -m candidate_screener.llm.gguf_probe --manifest M --out OUT.json --prompts 1 --max-tokens 8
    python -m candidate_screener.llm.gguf_probe --compare REF.json OTHER.json --rule identical|near-tie

``--prompts 1`` limits only the completions (E5 under emulation): the template and tokenizer
records are always taken for all six prompts, since E2 needs them on every platform.

Comparison (03, "The bar"; the bar is not changed here):

* E1: the facade's ``/v1/models`` model, revision and GGUF SHA-256, and the manifest's, are equal
  to the pin (``gguf_assets``) in both records.
* E2: same committed prompt set, and for all six prompts the ``/apply-template`` text and the
  ``/tokenize`` ids are identical.
* ``--rule identical`` (E3): generated token ids identical for all six prompts.
* ``--rule near-tie`` (the E4 rule, also used for E5/E6): per prompt present in both, the ids are
  identical over the tokens both runs generated, or they first differ at a step where the
  REFERENCE run's top-2 log-probability gap is <= 0.05 nats. Nothing after the first difference
  is compared.

Standard library and ``requests`` only, so it runs inside the ``llm4b`` image.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import monotonic
from urllib.parse import urlsplit

import requests

from candidate_screener.llm import gguf_assets

PROBE_VERSION = 1
NEAR_TIE_NATS = 0.05
TEMPLATE_KWARGS = {"enable_thinking": False}     # as gguf_server.TEMPLATE_KWARGS
TOKENIZE_FLAGS = {"add_special": True, "parse_special": True, "with_pieces": False}  # as gguf_server
MAX_RESPONSE_BYTES = 8_000_000
DEFAULT_TIMEOUT = 600.0
PROC_ROOT = Path("/proc")

_SOURCES = ("You answer questions about invented job applicants. Use only the invented sources "
            "below.\n"
            "Invented job: software developer. Requirement: a completed Bachelor degree.\n"
            "Candidate 1 (invented source CV-A): Completed Bachelor degree in Computer Science.\n"
            "Candidate 2 (invented source CV-B): Master degree in Data Science, in progress.")

#: Six invented prompts (03): four in the D67 walkthrough's shape, two short plain prompts.
PROMPTS = (
    {"name": "degree-question",
     "messages": [{"role": "system", "content": _SOURCES},
                  {"role": "user", "content": "What degree does Candidate 1 have?"}]},
    {"name": "ongoing-master",
     "messages": [{"role": "system", "content": _SOURCES},
                  {"role": "user", "content": "Has Candidate 2 completed a Master degree?"}]},
    {"name": "compare-two",
     "messages": [{"role": "system", "content": _SOURCES},
                  {"role": "user", "content": "Compare Candidate 1 and Candidate 2 on the degree requirement."}]},
    {"name": "why-1-better",
     "messages": [{"role": "system", "content": _SOURCES},
                  {"role": "user", "content": "Why is Candidate 1 better than Candidate 2?"}]},
    {"name": "plain-greeting",
     "messages": [{"role": "user", "content": "Say hello in one short, invented sentence."}]},
    {"name": "plain-colours",
     "messages": [{"role": "user", "content": "Name three colours of the rainbow."}]},
)


def prompts_digest():
    return hashlib.sha256(json.dumps(PROMPTS, sort_keys=True).encode()).hexdigest()


def check_loopback(url):
    """The origin `http://127.0.0.1:PORT`, or ValueError: the probe talks to this machine only."""
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError:
        port = None
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or port is None
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError(f"Refusing a non-loopback URL (http://127.0.0.1:PORT only): {url!r}")
    return f"http://127.0.0.1:{port}"


def completion_request(prompt, max_tokens):
    """Greedy, seeded, uncached; pre-sampling top-2 probabilities and the generated token ids."""
    return {"prompt": prompt, "n_predict": max_tokens, "temperature": 0, "top_k": 1, "seed": 0,
            "n_probs": 2, "post_sampling_probs": False, "return_tokens": True,
            "cache_prompt": False, "stream": False}


def _log(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValueError("Invalid probability")
    return math.log(value) if value > 0 else None


def parse_probabilities(response):
    """Per-step {id, logprob, top: [{id, logprob}, ...]} and the format the backend used.

    llama.cpp's `completion_probabilities` is `{id, logprob, top_logprobs}` by default, `{id, prob,
    top_probs}` with `post_sampling_probs`, and `{content, probs: [{tok_str, prob}]}` in older
    releases. Probabilities are converted with the natural log (a zero becomes None, unknown).
    """
    rows = response.get("completion_probabilities")
    if not isinstance(rows, list):
        return [], None
    steps, kinds = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid completion_probabilities")
        if "top_logprobs" in row:
            kinds.add("logprob")
            top = [{"id": item.get("id"), "logprob": float(item["logprob"])} for item in row["top_logprobs"]]
            chosen = row.get("logprob")
            chosen = float(chosen) if chosen is not None else None
        elif "top_probs" in row or "probs" in row:
            kinds.add("prob converted with ln")
            items = row.get("top_probs", row.get("probs"))
            top = [{"id": item.get("id"), "logprob": _log(item["prob"])} for item in items]
            chosen = _log(row["prob"]) if "prob" in row else (top[0]["logprob"] if top else None)
        else:
            raise ValueError("Unknown completion_probabilities format")
        steps.append({"id": row.get("id"), "logprob": chosen, "top": top})
    if len(kinds) > 1:
        raise ValueError("Mixed probability formats")
    return steps, (kinds.pop() if kinds else None)


def _request(session, method, url, payload, timeout):
    started = monotonic()
    with session.request(method, url, json=payload, timeout=(min(10.0, timeout), timeout),
                         allow_redirects=False, stream=True) as response:
        if response.status_code != 200:
            raise RuntimeError(f"{method} {url} answered HTTP {response.status_code}")
        raw = bytearray()
        for chunk in response.iter_content(65536):
            raw.extend(chunk)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise RuntimeError(f"{method} {url}: response too large")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"{method} {url}: not a JSON object")
    return value, round(monotonic() - started, 3)


def _session():
    session = requests.Session()
    session.trust_env = False          # no proxy, no .netrc: loopback only
    return session


def host_info():
    info = {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "processor": platform.processor(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "cpu_model": None}
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            key, _, value = line.partition(":")
            if key.strip() in {"model name", "Model", "Hardware"} and value.strip():
                info["cpu_model"] = value.strip()
                break
    except OSError:
        pass
    if info["cpu_model"] is None and info["system"] == "Darwin":
        try:
            info["cpu_model"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                               text=True, timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def backend_process(proc_root=None):
    """The running llama-server's CPU back end library and thread flag, read from /proc (Linux).

    The backend runs with `--log-disable`, so its start-up line ("loaded CPU backend from …") is
    not available; the mapped `libggml-cpu-*.so` is the same fact. Elsewhere this is unknown.
    """
    root = Path(proc_root or PROC_ROOT)
    result = {"source": str(root), "processes": 0, "cpu_backends": None, "threads": None,
              "threads_batch": None}
    if not root.is_dir():
        result["source"] = f"unavailable ({root} not present)"
        return result
    backends = set()
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = [part.decode(errors="replace") for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
            if not argv or Path(argv[0]).name != "llama-server":
                continue
            maps = (entry / "maps").read_text(errors="replace")
        except OSError:
            continue
        result["processes"] += 1
        for flag, key in (("--threads", "threads"), ("--threads-batch", "threads_batch")):
            if flag in argv[:-1]:
                result[key] = argv[argv.index(flag) + 1]
        for line in maps.splitlines():
            name = Path(line.split()[-1]).name if line.split() else ""
            if name.startswith("libggml-cpu") and ".so" in name:
                backends.add(name)
    result["cpu_backends"] = sorted(backends) if result["processes"] else None
    return result


def _manifest_fields(path):
    value = json.loads(Path(path).read_text())
    keys = ("model", "revision", "sha256", "gguf", "quantization", "llama_release", "runtime_platform",
            "runtime_archive", "runtime_archive_sha256", "start_check")
    return {key: value.get(key) for key in keys}


def run_probe(manifest, *, facade="http://127.0.0.1:8001", backend="http://127.0.0.1:8002", max_tokens=64,
              prompts="all", timeout=DEFAULT_TIMEOUT, session=None, proc_root=None):
    facade, backend = check_loopback(facade), check_loopback(backend)
    if type(max_tokens) is not int or not 1 <= max_tokens <= 256:
        raise ValueError("--max-tokens must be within 1..256")
    if prompts not in {"all", "1"}:
        raise ValueError("--prompts must be 'all' or '1'")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("--timeout must be positive")
    record = {"probe_version": PROBE_VERSION, "prompts_digest": prompts_digest(),
              "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "settings": {"facade": facade, "backend": backend, "max_tokens": max_tokens, "prompts": prompts,
                           "timeout_s": timeout, "template_kwargs": TEMPLATE_KWARGS,
                           "near_tie_nats": NEAR_TIE_NATS},
              "pin": {"model": gguf_assets.MODEL, "revision": gguf_assets.REVISION, "sha256": gguf_assets.SHA256},
              "manifest": _manifest_fields(manifest), "host": host_info(),
              "backend_process": backend_process(proc_root), "errors": []}
    own = session is None
    session = _session() if own else session
    session.trust_env = False
    try:
        response, elapsed = _request(session, "GET", facade + "/v1/models", None, timeout)
        record["facade_models"] = {"request": "GET /v1/models", "response": response, "elapsed_s": elapsed}
        props, elapsed = _request(session, "GET", backend + "/props", None, timeout)
        settings = props.get("default_generation_settings") or {}
        model_path = props.get("model_path")
        record["backend_props"] = {
            "build_info": props.get("build_info"), "total_slots": props.get("total_slots"),
            "model_file": Path(model_path).name if isinstance(model_path, str) else None,
            "n_ctx": settings.get("n_ctx") if isinstance(settings, dict) else None, "elapsed_s": elapsed}
        rows = []
        for index, prompt in enumerate(PROMPTS, start=1):
            row = {"index": index, "name": prompt["name"], "messages": prompt["messages"],
                   "apply_template": None, "tokenize": None, "completion": None}
            rows.append(row)
            try:
                body = {"messages": prompt["messages"], "chat_template_kwargs": TEMPLATE_KWARGS}
                result, elapsed = _request(session, "POST", backend + "/apply-template", body, timeout)
                text = result.get("prompt")
                if not isinstance(text, str):
                    raise RuntimeError("/apply-template returned no prompt text")
                row["apply_template"] = {"request": body, "prompt": text, "elapsed_s": elapsed}
                body = {"content": text, **TOKENIZE_FLAGS}
                result, elapsed = _request(session, "POST", backend + "/tokenize", body, timeout)
                tokens = result.get("tokens")
                if not isinstance(tokens, list) or any(type(t) is not int for t in tokens):
                    raise RuntimeError("/tokenize returned no token ids")
                row["tokenize"] = {"request": body, "tokens": tokens, "elapsed_s": elapsed}
                if prompts == "1" and index != 1:
                    continue
                body = completion_request(text, max_tokens)
                result, elapsed = _request(session, "POST", backend + "/completion", body, timeout)
                steps, kind = parse_probabilities(result)
                generated = result.get("tokens")
                source = "tokens"
                if not isinstance(generated, list) or any(type(t) is not int for t in generated):
                    generated, source = [step["id"] for step in steps], "completion_probabilities ids"
                    if any(type(t) is not int for t in generated):
                        raise RuntimeError("/completion returned no generated token ids")
                row["completion"] = {"request": body, "tokens": generated, "tokens_source": source,
                                     "content": result.get("content"), "stop_type": result.get("stop_type"),
                                     "steps": steps, "probability_format": kind,
                                     "timings": result.get("timings"), "elapsed_s": elapsed}
            except (requests.RequestException, RuntimeError, ValueError, KeyError, TypeError) as error:
                record["errors"].append(f"prompt {index}: {type(error).__name__}: {error}")
        record["prompts"] = rows
    except (requests.RequestException, RuntimeError, ValueError) as error:
        record["errors"].append(f"{type(error).__name__}: {error}")
        record.setdefault("prompts", [])
    finally:
        if own:
            session.close()
    record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return record


# ---------------------------------------------------------------------------------- comparison

def _identity(record):
    """The (model, revision, sha256) the facade advertises and the manifest records."""
    rows = ((record.get("facade_models") or {}).get("response") or {}).get("data") or []
    facade = [(r.get("id"), r.get("revision"), r.get("sha256")) for r in rows if isinstance(r, dict)]
    manifest = record.get("manifest") or {}
    return facade, (manifest.get("model"), manifest.get("revision"), manifest.get("sha256"))


def _check_identity(ref, other):
    pin = (gguf_assets.MODEL, gguf_assets.REVISION, gguf_assets.SHA256)
    lines, passed = [f"pin: {pin[0]} {pin[1]} {pin[2]}"], True
    seen = []
    for label, record in (("reference", ref), ("other", other)):
        facade, manifest = _identity(record)
        seen.append((facade, manifest))
        if facade != [pin]:
            passed = False
            lines.append(f"{label}: facade /v1/models {facade} is not exactly the pin")
        if manifest != pin:
            passed = False
            lines.append(f"{label}: manifest identity {manifest} is not the pin")
    if seen[0] != seen[1]:
        passed = False
        lines.append("the two records' identities differ")
    return passed, lines


def _rows(record):
    return {row.get("index"): row for row in record.get("prompts") or [] if isinstance(row, dict)}


def _check_template(ref, other):
    lines, passed = [], True
    if ref.get("prompts_digest") != prompts_digest() or other.get("prompts_digest") != prompts_digest():
        passed = False
        lines.append("prompt set differs from the committed prompts (probe version mismatch)")
    a, b = _rows(ref), _rows(other)
    for index in range(1, len(PROMPTS) + 1):
        ra, rb = a.get(index) or {}, b.get(index) or {}
        ta, tb = ra.get("apply_template"), rb.get("apply_template")
        ka, kb = ra.get("tokenize"), rb.get("tokenize")
        if not (ta and tb and ka and kb):
            passed = False
            lines.append(f"prompt {index}: template or tokens missing")
            continue
        same_text = ta.get("prompt") == tb.get("prompt")
        same_ids = ka.get("tokens") == kb.get("tokens")
        if not (same_text and same_ids):
            passed = False
        lines.append(f"prompt {index}: template {'identical' if same_text else 'DIFFERS'}, "
                     f"{len(ka.get('tokens') or [])} token ids {'identical' if same_ids else 'DIFFER'}")
    return passed, lines


def _first_difference(ref, other):
    """(step, compared) where step is the first differing index or None; an early end counts."""
    a, b = ref["tokens"], other["tokens"]
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, n
    if len(a) != len(b):
        shorter = ref if len(a) < len(b) else other
        if shorter.get("stop_type") != "limit":   # it ended (eos/word) where the other went on
            return n, n
    return None, n


def _gap(completion, step):
    steps = completion.get("steps") or []
    if step >= len(steps):
        return None
    values = [item.get("logprob") for item in steps[step].get("top") or []]
    if len(values) < 2 or any(v is None for v in values[:2]):
        return None
    first, second = sorted(values, reverse=True)[:2]
    return first - second


def _check_generation(ref, other, rule):
    lines, passed = [], True
    a, b = _rows(ref), _rows(other)
    compared = []
    for index in range(1, len(PROMPTS) + 1):
        ca, cb = (a.get(index) or {}).get("completion"), (b.get(index) or {}).get("completion")
        if not (ca and cb):
            if rule == "identical":
                passed = False
                lines.append(f"prompt {index}: completion missing in "
                             f"{'both' if not (ca or cb) else 'reference' if not ca else 'other'}")
            else:
                lines.append(f"prompt {index}: not in both runs, not compared")
            continue
        compared.append(index)
        step, n = _first_difference(ca, cb)
        if rule == "identical":
            budgets = (ca.get("request") or {}).get("n_predict"), (cb.get("request") or {}).get("n_predict")
            if budgets[0] != budgets[1]:
                passed = False
                lines.append(f"prompt {index}: max_tokens differ {budgets}")
            if step is None and len(ca["tokens"]) == len(cb["tokens"]):
                lines.append(f"prompt {index}: {n} tokens identical")
            else:
                passed = False
                lines.append(f"prompt {index}: DIFFERS at step {step if step is not None else n}")
            continue
        if step is None:
            lines.append(f"prompt {index}: {n} tokens identical (compared over the tokens both generated)")
            continue
        gap = _gap(ca, step)
        if gap is not None and gap <= NEAR_TIE_NATS:
            lines.append(f"prompt {index}: first differs at step {step} of {n} tokens; reference top-2 gap "
                         f"{gap:.6f} nats <= {NEAR_TIE_NATS} (near tie); later steps not compared")
        else:
            passed = False
            shown = "unknown" if gap is None else f"{gap:.6f} nats"
            lines.append(f"prompt {index}: first differs at step {step} of {n} tokens; reference top-2 gap "
                         f"{shown} > {NEAR_TIE_NATS} or unknown: FAIL")
    if not compared:
        passed = False
        lines.append("no prompt has a completion in both runs")
    return passed, lines


def compare(ref, other, rule):
    """[(check, passed, lines)] for E1, E2 and E3 (`identical`) or E4 (`near-tie`)."""
    if rule not in {"identical", "near-tie"}:
        raise ValueError("rule must be 'identical' or 'near-tie'")
    return [("E1", *_check_identity(ref, other)), ("E2", *_check_template(ref, other)),
            ("E3" if rule == "identical" else "E4", *_check_generation(ref, other, rule))]


# ---------------------------------------------------------------------------------- command line

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", help="The pinned GGUF manifest the running service uses")
    parser.add_argument("--out", help="Where to write the probe record (JSON)")
    parser.add_argument("--facade", default="http://127.0.0.1:8001")
    parser.add_argument("--backend", default="http://127.0.0.1:8002")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompts", choices=["all", "1"], default="all",
                        help="Completions for all six prompts, or prompt 1 only (template and tokens: always all)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="Per-request read timeout in seconds (generous for emulation)")
    parser.add_argument("--compare", nargs=2, metavar=("REF.json", "OTHER.json"),
                        help="Compare two records; the first is the reference")
    parser.add_argument("--rule", choices=["identical", "near-tie"])
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.compare:
        if args.rule is None or args.manifest or args.out:
            parser.error("--compare takes two records and --rule, nothing else")
        ref, other = (json.loads(Path(path).read_text()) for path in args.compare)
        results = compare(ref, other, args.rule)
        print(f"reference {args.compare[0]} ({(ref.get('manifest') or {}).get('runtime_platform')}) vs "
              f"{args.compare[1]} ({(other.get('manifest') or {}).get('runtime_platform')}), rule {args.rule}")
        for label, record in (("reference", ref), ("other", other)):
            if record.get("errors"):
                print(f"  note: {label} record has probe errors: {record['errors']}")
        for check, passed, lines in results:
            print(f"{check} {'PASS' if passed else 'FAIL'}")
            for line in lines:
                print(f"  {line}")
        return 0 if all(passed for _, passed, _ in results) else 1
    if not (args.manifest and args.out) or args.rule:
        parser.error("a probe run needs --manifest and --out (and no --rule)")
    record = run_probe(args.manifest, facade=args.facade, backend=args.backend, max_tokens=args.max_tokens,
                       prompts=args.prompts, timeout=args.timeout)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    partial.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    partial.replace(out)
    print(json.dumps({"out": str(out), "errors": record["errors"]}))
    return 1 if record["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
