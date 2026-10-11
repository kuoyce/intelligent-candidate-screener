"""D42: no path in the repository may dispatch prompts or spawn a Claude subprocess.

Kept apart from `test_llm_recheck.py`, whose fixture reopens the gate in-process so the
frozen module's own regressions still run with the subprocess mocked.
"""
import subprocess

import pytest

from candidate_screener.annotation import llm_recheck as llm
from candidate_screener.annotation import prompt_calibration as calibration


@pytest.fixture
def no_spawn(monkeypatch):
    """Fail loudly if anything reaches `subprocess.run` — the guard must fire first."""
    def refuse(*args, **kwargs):
        raise AssertionError("a subprocess was started despite D42")
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(llm.subprocess, "run", refuse)
    monkeypatch.setattr(calibration.subprocess, "run", refuse)


def test_the_gate_is_closed_by_default():
    assert llm.DISPATCH_CLOSED is True


@pytest.mark.parametrize("call", [
    lambda: llm.dispatch(run=99),
    lambda: llm.dispatch_all_a1(run=99),
    lambda: llm.dispatch_a2("generic", run=99),
    lambda: llm.judge(run=99),
    lambda: llm.judge_one({"pair_id": "p", "run": 99, "prompt": "x"}, cwd=None),
    lambda: calibration.calibrate_one("pair", "system"),
])
def test_every_dispatch_and_spawn_path_refuses(call, no_spawn):
    with pytest.raises(llm.DispatchClosed, match="D42"):
        call()


def test_the_cli_refuses_dispatch_and_judge(monkeypatch, no_spawn):
    for flag in ("--dispatch", "--judge"):
        monkeypatch.setattr("sys.argv", ["llm_recheck", flag, "--run", "99"])
        with pytest.raises(llm.DispatchClosed):
            llm.main()
    monkeypatch.setattr("sys.argv", ["prompt_calibration"])
    with pytest.raises(llm.DispatchClosed):
        calibration.main()
