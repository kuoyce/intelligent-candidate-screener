"""G3 equivalence probe (plan 2026-10-28-gguf-cross-platform, 03): synthetic, no network.

A fake requests session stands in for the facade (8001) and the llama.cpp backend (8002);
records for the comparison rules are built inline.
"""
from copy import deepcopy
import json
import math
import re

import pytest

from candidate_screener.llm import gguf_assets
from candidate_screener.llm import gguf_probe as probe


# ---------------------------------------------------------------------------- fakes

class FakeResponse:
    def __init__(self, status, value):
        self.status_code = status
        self.raw = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, size):
        for start in range(0, len(self.raw), size):
            yield self.raw[start:start + size]


class FakeSession:
    """Answers the five endpoints the probe uses; records every request."""

    def __init__(self, probability_format="logprob"):
        self.calls = []
        self.trust_env = True
        self.format = probability_format

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass

    def request(self, method, url, json=None, timeout=None, allow_redirects=True, stream=False):
        self.calls.append({"method": method, "url": url, "json": deepcopy(json), "timeout": timeout,
                           "allow_redirects": allow_redirects, "trust_env": self.trust_env})
        path = url.split("127.0.0.1", 1)[1].split("/", 1)[1]
        if path == "v1/models":
            return FakeResponse(200, {"object": "list", "data": [
                {"id": gguf_assets.MODEL, "revision": gguf_assets.REVISION, "sha256": gguf_assets.SHA256}]})
        if path == "props":
            return FakeResponse(200, {"build_info": "b11896-9125a2ebe", "total_slots": 1,
                                      "model_path": "/secret/dir/Qwen3.5-4B-Q4_K_M.gguf",
                                      "default_generation_settings": {"n_ctx": 4096}})
        if path == "apply-template":
            return FakeResponse(200, {"prompt": "<|im_start|>user\n" + json["messages"][-1]["content"]
                                      + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"})
        if path == "tokenize":
            return FakeResponse(200, {"tokens": [len(json["content"]), 7, 8]})
        if path == "completion":
            n = json["n_predict"]
            ids = list(range(100, 100 + n))
            if self.format == "logprob":
                steps = [{"id": i, "token": "t", "bytes": [116], "logprob": -0.1,
                          "top_logprobs": [{"id": i, "token": "t", "bytes": [116], "logprob": -0.1},
                                           {"id": i + 1, "token": "u", "bytes": [117], "logprob": -2.4}]}
                         for i in ids]
            else:
                steps = [{"id": i, "token": "t", "bytes": [116], "prob": 0.9,
                          "top_probs": [{"id": i, "token": "t", "bytes": [116], "prob": 0.9},
                                        {"id": i + 1, "token": "u", "bytes": [117], "prob": 0.1}]}
                         for i in ids]
            return FakeResponse(200, {"content": "t" * n, "tokens": ids, "stop_type": "limit",
                                      "completion_probabilities": steps,
                                      "timings": {"prompt_n": 3, "predicted_n": n}})
        return FakeResponse(404, {"error": "unknown"})


@pytest.fixture
def manifest(tmp_path):
    value = {"model": gguf_assets.MODEL, "revision": gguf_assets.REVISION, "sha256": gguf_assets.SHA256,
             "gguf": "Qwen3.5-4B-Q4_K_M.gguf", "llama_server": "/x/llama-server", "llama_release": "b11896",
             "runtime_platform": "linux-x64", "runtime_archive": "llama-b11896-bin-ubuntu-x64.tar.gz",
             "runtime_archive_sha256": "b" * 64, "quantization": "Q4_K_M"}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def fake_proc(tmp_path):
    root = tmp_path / "proc"
    server = root / "4242"
    server.mkdir(parents=True)
    (server / "cmdline").write_bytes(b"\0".join([b"/opt/rt/llama-server", b"-m", b"x.gguf",
                                                  b"--threads", b"4", b"--threads-batch", b"4"]) + b"\0")
    (server / "maps").write_text("7f00-7f01 r-xp 0 0:0 1 /opt/rt/libggml-cpu-alderlake.so\n"
                                 "7f02-7f03 r-xp 0 0:0 1 /opt/rt/libggml-base.so\n")
    other = root / "77"
    other.mkdir()
    (other / "cmdline").write_bytes(b"bash\0")
    (other / "maps").write_text("")
    return root


def run(manifest, session, tmp_path, **kwargs):
    options = dict(facade="http://127.0.0.1:8001", backend="http://127.0.0.1:8002", max_tokens=64,
                   prompts="all", timeout=600.0, session=session, proc_root=tmp_path / "noproc")
    options.update(kwargs)
    return probe.run_probe(manifest, **options)


# ---------------------------------------------------------------------------- request shapes

@pytest.mark.parametrize("url", ["http://10.0.0.5:8002", "http://localhost.evil.com:8002",
                                 "https://127.0.0.1:8002", "http://user:pw@127.0.0.1:8002",
                                 "http://127.0.0.1:8002/v1", "http://127.0.0.1:8002?x=1",
                                 "http://example.com:8001", "ftp://127.0.0.1:8001"])
def test_non_loopback_urls_are_refused(url):
    with pytest.raises(ValueError, match="loopback"):
        probe.check_loopback(url)


def test_refused_url_sends_nothing(manifest, tmp_path):
    session = FakeSession()
    with pytest.raises(ValueError, match="loopback"):
        run(manifest, session, tmp_path, backend="http://192.168.1.2:8002")
    assert session.calls == []


def test_loopback_url_is_accepted():
    assert probe.check_loopback("http://127.0.0.1:8002/") == "http://127.0.0.1:8002"


def test_every_request_is_loopback_bounded_and_without_proxy(manifest, tmp_path):
    session = FakeSession()
    run(manifest, session, tmp_path, timeout=123.0)
    assert session.calls
    for call in session.calls:
        assert call["url"].startswith(("http://127.0.0.1:8001/", "http://127.0.0.1:8002/"))
        assert call["trust_env"] is False
        assert call["allow_redirects"] is False
        connect, read = call["timeout"]
        assert 0 < connect <= read == 123.0


def test_default_session_ignores_environment_proxies(monkeypatch, manifest, tmp_path):
    made = []

    def factory():
        made.append(FakeSession())
        return made[-1]

    monkeypatch.setattr(probe.requests, "Session", factory)
    run(manifest, None, tmp_path)
    assert made and all(call["trust_env"] is False for session in made for call in session.calls)


def test_facade_serves_identity_and_backend_serves_template_tokens_and_completion(manifest, tmp_path):
    session = FakeSession()
    run(manifest, session, tmp_path)
    by_path = {}
    for call in session.calls:
        by_path.setdefault(call["url"], []).append(call)
    assert [c["method"] for c in by_path["http://127.0.0.1:8001/v1/models"]] == ["GET"]
    assert len(by_path["http://127.0.0.1:8002/apply-template"]) == 6
    assert len(by_path["http://127.0.0.1:8002/tokenize"]) == 6
    assert len(by_path["http://127.0.0.1:8002/completion"]) == 6


def test_template_request_disables_thinking_and_matches_committed_prompts(manifest, tmp_path):
    session = FakeSession()
    run(manifest, session, tmp_path)
    bodies = [c["json"] for c in session.calls if c["url"].endswith("/apply-template")]
    assert [body["messages"] for body in bodies] == [p["messages"] for p in probe.PROMPTS]
    assert all(body["chat_template_kwargs"] == {"enable_thinking": False} for body in bodies)


def test_tokenize_and_completion_use_the_templated_text(manifest, tmp_path):
    session = FakeSession()
    record = run(manifest, session, tmp_path)
    for row in record["prompts"]:
        text = row["apply_template"]["prompt"]
        assert row["tokenize"]["request"] == {"content": text, "add_special": True, "parse_special": True,
                                              "with_pieces": False}
        assert row["completion"]["request"]["prompt"] == text


def test_completion_is_greedy_seeded_uncached_with_two_probabilities(manifest, tmp_path):
    session = FakeSession()
    run(manifest, session, tmp_path, max_tokens=64)
    body = next(c["json"] for c in session.calls if c["url"].endswith("/completion"))
    assert body["temperature"] == 0 and body["top_k"] == 1 and body["seed"] == 0
    assert body["n_predict"] == 64 and body["n_probs"] == 2
    assert body["cache_prompt"] is False and body["stream"] is False
    assert body["post_sampling_probs"] is False and body["return_tokens"] is True


def test_prompt_subset_limits_completions_but_not_template_or_tokens(manifest, tmp_path):
    session = FakeSession()
    record = run(manifest, session, tmp_path, prompts="1", max_tokens=8)
    assert sum(c["url"].endswith("/completion") for c in session.calls) == 1
    assert sum(c["url"].endswith("/tokenize") for c in session.calls) == 6
    assert record["prompts"][0]["completion"]["tokens"] == list(range(100, 108))
    assert all(row["completion"] is None for row in record["prompts"][1:])
    assert record["settings"]["prompts"] == "1" and record["settings"]["max_tokens"] == 8


def test_record_holds_identity_runtime_host_and_timing(manifest, tmp_path, fake_proc):
    record = run(manifest, FakeSession(), tmp_path, proc_root=fake_proc)
    assert record["facade_models"]["response"]["data"][0]["sha256"] == gguf_assets.SHA256
    assert record["manifest"]["runtime_platform"] == "linux-x64"
    assert record["manifest"]["runtime_archive_sha256"] == "b" * 64
    assert record["manifest"]["revision"] == gguf_assets.REVISION
    assert record["backend_props"]["build_info"] == "b11896-9125a2ebe"
    assert record["backend_props"]["model_file"] == "Qwen3.5-4B-Q4_K_M.gguf"
    assert "/secret/dir" not in json.dumps(record)
    assert {"machine", "processor", "system", "cpu_count"} <= set(record["host"])
    assert record["backend_process"]["cpu_backends"] == ["libggml-cpu-alderlake.so"]
    assert record["backend_process"]["threads"] == "4"
    assert record["prompts_digest"] == probe.prompts_digest()
    for row in record["prompts"]:
        for part in ("apply_template", "tokenize", "completion"):
            assert row[part]["elapsed_s"] >= 0


def test_main_writes_the_record(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(probe.requests, "Session", FakeSession)
    monkeypatch.setattr(probe, "PROC_ROOT", tmp_path / "noproc")
    out = tmp_path / "out" / "probe.json"
    assert probe.main(["--manifest", str(manifest), "--out", str(out), "--max-tokens", "4"]) == 0
    value = json.loads(out.read_text())
    assert value["settings"]["max_tokens"] == 4 and len(value["prompts"]) == 6


# ---------------------------------------------------------------------------- probabilities

def test_logprob_format_is_recorded_as_is():
    steps, kind = probe.parse_probabilities({"completion_probabilities": [
        {"id": 5, "logprob": -0.2, "top_logprobs": [{"id": 5, "logprob": -0.2}, {"id": 9, "logprob": -1.7}]}]})
    assert kind == "logprob"
    assert steps == [{"id": 5, "logprob": -0.2, "top": [{"id": 5, "logprob": -0.2}, {"id": 9, "logprob": -1.7}]}]


def test_prob_format_is_converted_with_natural_log_and_marked():
    steps, kind = probe.parse_probabilities({"completion_probabilities": [
        {"id": 5, "prob": 0.5, "top_probs": [{"id": 5, "prob": 0.5}, {"id": 9, "prob": 0.25}]}]})
    assert kind == "prob converted with ln"
    assert steps[0]["top"][1]["logprob"] == pytest.approx(math.log(0.25))
    assert steps[0]["logprob"] == pytest.approx(math.log(0.5))


def test_legacy_probs_format_without_ids_is_converted():
    steps, kind = probe.parse_probabilities({"completion_probabilities": [
        {"content": "a", "probs": [{"tok_str": "a", "prob": 0.75}, {"tok_str": "b", "prob": 0.2}]}]})
    assert kind == "prob converted with ln"
    assert steps[0]["id"] is None and steps[0]["top"][0]["logprob"] == pytest.approx(math.log(0.75))


def test_zero_probability_is_recorded_as_unknown_not_infinite():
    steps, _ = probe.parse_probabilities({"completion_probabilities": [
        {"id": 1, "prob": 1.0, "top_probs": [{"id": 1, "prob": 1.0}, {"id": 2, "prob": 0.0}]}]})
    assert steps[0]["top"][1]["logprob"] is None
    json.dumps(steps, allow_nan=False)


def test_probe_records_prob_format_when_backend_sends_probs(manifest, tmp_path):
    record = run(manifest, FakeSession("prob"), tmp_path, max_tokens=2)
    completion = record["prompts"][0]["completion"]
    assert completion["probability_format"] == "prob converted with ln"
    assert completion["steps"][0]["top"][1]["logprob"] == pytest.approx(math.log(0.1))


# ---------------------------------------------------------------------------- comparison rules

def make_record(tokens=None, gaps=None, max_tokens=64, completions=6, stop_type="limit", prompts="all"):
    """A probe record: prompt k generates `tokens[k]`; `gaps[k][i]` is step i's top-2 gap (nats)."""
    rows = []
    for k, prompt in enumerate(probe.PROMPTS):
        ids = list(tokens[k]) if tokens else [10 + k, 20 + k, 30 + k]
        gap = gaps[k] if gaps else [1.0] * len(ids)
        steps = [{"id": t, "logprob": -0.01,
                  "top": [{"id": t, "logprob": -0.01}, {"id": t + 1, "logprob": -0.01 - g}]}
                 for t, g in zip(ids, gap)]
        completion = ({"request": {"n_predict": max_tokens}, "tokens": ids, "stop_type": stop_type,
                       "steps": steps, "probability_format": "logprob"} if k < completions else None)
        rows.append({"index": k + 1, "name": prompt["name"], "messages": prompt["messages"],
                     "apply_template": {"prompt": f"<t>{k}</t>"}, "tokenize": {"tokens": [k, k + 1]},
                     "completion": completion})
    return {"probe_version": probe.PROBE_VERSION, "prompts_digest": probe.prompts_digest(),
            "settings": {"max_tokens": max_tokens, "prompts": prompts},
            "manifest": {"model": gguf_assets.MODEL, "revision": gguf_assets.REVISION,
                         "sha256": gguf_assets.SHA256, "runtime_platform": "linux-x64"},
            "facade_models": {"response": {"data": [{"id": gguf_assets.MODEL, "revision": gguf_assets.REVISION,
                                                     "sha256": gguf_assets.SHA256}]}},
            "prompts": rows}


def verdicts(ref, other, rule):
    return {check: passed for check, passed, _ in probe.compare(ref, other, rule)}


def test_identical_records_pass_both_rules():
    assert verdicts(make_record(), make_record(), "identical") == {"E1": True, "E2": True, "E3": True}
    assert verdicts(make_record(), make_record(), "near-tie") == {"E1": True, "E2": True, "E4": True}


def test_identity_mismatch_fails_e1():
    other = make_record()
    other["facade_models"]["response"]["data"][0]["sha256"] = "0" * 64
    assert verdicts(make_record(), other, "identical")["E1"] is False


def test_identity_equal_between_runs_but_not_the_pin_fails_e1():
    ref, other = make_record(), make_record()
    for record in (ref, other):
        record["facade_models"]["response"]["data"][0]["revision"] = "f" * 40
        record["manifest"]["revision"] = "f" * 40
    assert verdicts(ref, other, "identical")["E1"] is False


def test_manifest_identity_mismatch_fails_e1():
    other = make_record()
    other["manifest"]["model"] = "Qwen/other"
    assert verdicts(make_record(), other, "near-tie")["E1"] is False


def test_template_text_mismatch_fails_e2():
    other = make_record()
    other["prompts"][3]["apply_template"]["prompt"] += " "
    assert verdicts(make_record(), other, "near-tie")["E2"] is False


def test_token_id_mismatch_fails_e2():
    other = make_record()
    other["prompts"][5]["tokenize"]["tokens"][0] += 1
    assert verdicts(make_record(), other, "identical")["E2"] is False


def test_missing_template_for_a_prompt_fails_e2():
    other = make_record()
    other["prompts"][2]["tokenize"] = None
    assert verdicts(make_record(), other, "identical")["E2"] is False


def test_different_prompt_set_fails_e2():
    other = make_record()
    other["prompts_digest"] = "0" * 64
    assert verdicts(make_record(), other, "identical")["E2"] is False


def test_identical_rule_fails_on_any_generated_difference():
    other = make_record(tokens=[[10, 20, 30], [11, 21, 31], [12, 22, 32], [13, 23, 99], [14, 24, 34], [15, 25, 35]])
    assert verdicts(make_record(), other, "identical")["E3"] is False


def test_identical_rule_needs_all_six_completions():
    assert verdicts(make_record(), make_record(completions=1), "identical")["E3"] is False


def test_near_tie_passes_when_first_difference_is_at_reference_near_tie():
    gaps = [[1.0, 1.0, 1.0]] * 6
    gaps = [list(g) for g in gaps]
    gaps[0][1] = 0.05                              # the bar is inclusive
    ref = make_record(gaps=gaps)
    other_tokens = [[10 + k, 20 + k, 30 + k] for k in range(6)]
    other_tokens[0] = [10, 777, 888]              # differs at step 1, and later steps too
    other = make_record(tokens=other_tokens)
    assert verdicts(ref, other, "near-tie")["E4"] is True


def test_near_tie_fails_when_first_difference_is_at_a_clear_gap():
    gaps = [[1.0, 1.0, 1.0] for _ in range(6)]
    gaps[0][1] = 0.0501
    ref = make_record(gaps=gaps)
    other_tokens = [[10 + k, 20 + k, 30 + k] for k in range(6)]
    other_tokens[0] = [10, 777, 30]
    assert verdicts(ref, make_record(tokens=other_tokens), "near-tie")["E4"] is False


def test_near_tie_uses_the_reference_gap_not_the_other_run():
    near = [[1.0, 0.01, 1.0] for _ in range(6)]
    clear = [[1.0, 1.0, 1.0] for _ in range(6)]
    tokens = [[10 + k, 20 + k, 30 + k] for k in range(6)]
    tokens[2] = [12, 99, 32]
    ref_clear = make_record(gaps=clear)
    other_near = make_record(tokens=tokens, gaps=near)
    assert verdicts(ref_clear, other_near, "near-tie")["E4"] is False
    assert verdicts(make_record(gaps=near), make_record(tokens=tokens, gaps=clear), "near-tie")["E4"] is True


def test_near_tie_ignores_everything_after_the_first_difference():
    gaps = [[1.0, 0.0, 9.0] for _ in range(6)]
    tokens = [[10 + k, 555, 666] for k in range(6)]  # every prompt differs from step 1 on
    assert verdicts(make_record(gaps=gaps), make_record(tokens=tokens), "near-tie")["E4"] is True


def test_near_tie_with_unknown_reference_gap_fails():
    ref = make_record()
    ref["prompts"][0]["completion"]["steps"][1]["top"] = ref["prompts"][0]["completion"]["steps"][1]["top"][:1]
    tokens = [[10 + k, 20 + k, 30 + k] for k in range(6)]
    tokens[0] = [10, 99, 30]
    assert verdicts(ref, make_record(tokens=tokens), "near-tie")["E4"] is False


def test_early_end_of_sequence_is_a_difference():
    other_tokens = [[10 + k, 20 + k, 30 + k] for k in range(6)]
    other_tokens[4] = [14, 24]
    other = make_record(tokens=other_tokens, stop_type="eos")
    assert verdicts(make_record(), other, "identical")["E3"] is False


def test_subset_run_compares_only_shared_prompts_and_tokens():
    """E5: prompt 1 with 8 tokens against the reference's 6 prompts with 64."""
    ref = make_record(tokens=[list(range(100 + 10 * k, 164 + 10 * k)) for k in range(6)],
                      gaps=[[1.0] * 64 for _ in range(6)])
    other = make_record(tokens=[list(range(100, 108))] + [[0]] * 5, max_tokens=8, completions=1,
                        prompts="1")
    result = probe.compare(ref, other, "near-tie")
    assert {check: passed for check, passed, _ in result} == {"E1": True, "E2": True, "E4": True}
    lines = " ".join(line for check, _, details in result if check == "E4" for line in details)
    assert "prompt 1" in lines and "8 tokens" in lines


def test_subset_run_with_no_shared_completion_fails():
    other = make_record(completions=0, prompts="1")
    assert verdicts(make_record(), other, "near-tie")["E4"] is False


def test_near_tie_fails_when_a_full_run_lost_completions():
    # E4 judges each prompt: a full Mac run whose completions errored on 5 of 6 prompts must not
    # pass on the one that worked (G5 review).
    other = make_record(completions=1)
    assert verdicts(make_record(), other, "near-tie")["E4"] is False


def test_near_tie_fails_when_the_subset_run_lost_prompt_1():
    other = make_record(completions=0, prompts="1")
    ref = make_record()
    result = {check: (passed, lines) for check, passed, lines in probe.compare(ref, other, "near-tie")}
    assert result["E4"][0] is False and any("prompt 1" in line for line in result["E4"][1])


def test_compare_cli_exit_codes(tmp_path, capsys):
    ref, same = tmp_path / "ref.json", tmp_path / "same.json"
    ref.write_text(json.dumps(make_record()))
    same.write_text(json.dumps(make_record()))
    bad = tmp_path / "bad.json"
    broken = make_record()
    broken["prompts"][0]["tokenize"]["tokens"] = [9]
    bad.write_text(json.dumps(broken))
    assert probe.main(["--compare", str(ref), str(same), "--rule", "identical"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"E1 .*PASS", out) and re.search(r"E3 .*PASS", out)
    assert probe.main(["--compare", str(ref), str(bad), "--rule", "near-tie"]) == 1
    assert re.search(r"E2 .*FAIL", capsys.readouterr().out)


# ---------------------------------------------------------------------------- invented text only

def test_six_prompts_four_walkthrough_shaped_and_two_plain():
    assert len(probe.PROMPTS) == 6
    assert len({p["name"] for p in probe.PROMPTS}) == 6
    walkthrough = probe.PROMPTS[:4]
    assert all("Candidate 1" in json.dumps(p) for p in walkthrough)
    assert any("Candidate 2" in json.dumps(p) for p in walkthrough)
    assert all(len(json.dumps(p["messages"])) < 400 for p in probe.PROMPTS[4:])


def test_prompts_hold_only_invented_text():
    """No CV-like content: no contact details, links, dates, employers or real names."""
    text = json.dumps(probe.PROMPTS)
    assert len(text) < 6000
    assert "invented" in text.lower()
    assert not re.search(r"[\w.+-]+@[\w-]+\.\w+", text)                 # e-mail
    assert not re.search(r"\+?\d[\d\s().-]{7,}\d", text)                 # phone-like numbers
    assert not re.search(r"https?://|www\.", text)                         # links
    assert not re.search(r"\b(19|20)\d\d\b", text)                         # years
    names = re.findall(r"\b(?:Candidate|CV)[ -]?(\w+)", text)
    assert set(names) <= {"1", "2", "A", "B"}
    for prompt in probe.PROMPTS:
        for message in prompt["messages"]:
            assert set(message) == {"role", "content"}
            assert message["role"] in {"system", "user", "assistant"}


def test_probe_has_no_way_to_send_other_text():
    parser = probe.build_parser()
    options = {action.dest for action in parser._actions}
    assert not options & {"prompt", "prompt_file", "messages", "text", "input"}
