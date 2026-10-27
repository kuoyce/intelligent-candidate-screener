"""`SCREENER_MODELS_ROOT` moves the three pretrained model folders and nothing else.

Plan 2026-10-27-docker (01-design-spec.md §7, §10): the Docker image keeps the pretrained
weights under `/opt/models`, outside the data folder; natively the variable is unset and
every path stays where it was. `config` reads the variable at import, so each layout is read
in a fresh interpreter, which loads the asset modules by file path as the image build runs
them. The download pins run the scripts' command lines against a recording stand-in for
`huggingface_hub`. Synthetic: no network, no `data/`.
"""
from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import candidate_screener
from candidate_screener.config import PROJECT_ROOT

PACKAGE = Path(candidate_screener.__file__).resolve().parent
SCRIPTS = {"qa_assets": "llm/qa_assets.py", "medium_assets": "extraction/medium_assets.py",
           "qwen_assets": "llm/qwen_assets.py"}
#: Where the model folders sit under the models root.
MODEL_FOLDERS = {"qa_assets.ROOT": "chatbot/qa/extractive",
                 "medium_assets.MODEL_DIR": "extraction/models/gliner-medium-v2.5",
                 "qwen_assets.MODEL_DIR": "chatbot/qwen",
                 "qwen_assets.DEFAULT_MANIFEST": "chatbot/qwen/manifest.json"}
#: Every path constant as it is with the variable unset, relative to PROJECT_ROOT.
LAYOUT = {"config.PROJECT_ROOT": ".", "config.DATA": "data", "config.RAW": "data/raw",
          "config.INTERIM": "data/interim", "config.PROCESSED": "data/processed",
          "config.VOCAB": "data/vocab", "config.DOCS": "docs", "config.DOCS_DATA": "docs/data",
          "config.NOTEBOOKS": "notebooks", "config.OUTPUT": "output",
          "config.MANIFEST": "docs/data/acquisition-manifest.json",
          "config.PROFILE_METRICS": "docs/data/profile-metrics.json",
          "config.BASELINE_METRICS": "output/baselines/baseline-metrics.json",
          "config.RETRIEVAL_METRICS": "output/baselines/retrieval-metrics.json",
          "config.BASELINE_CACHE": "data/processed/baselines",
          "config.MODELS": "data/processed",
          **{key: f"data/processed/{folder}" for key, folder in MODEL_FOLDERS.items()}}

PROBE = r"""
import importlib.util, json, sys
from pathlib import Path
before = set(sys.modules)
import candidate_screener.config as config
paths = {f"config.{k}": str(v) for k, v in vars(config).items() if isinstance(v, Path)}
for name, relative in json.loads(sys.argv[1]).items():
    spec = importlib.util.spec_from_file_location(f"probe_{name}", Path(config.__file__).parent / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths.update({f"{name}.{k}": str(v) for k, v in vars(module).items()
                  if isinstance(v, Path) and v is not getattr(config, k, None)})  # not config's own
imported = sorted(m for m in set(sys.modules) - before
                  if m.split(".")[0] not in sys.stdlib_module_names and not m.startswith("probe_"))
print(json.dumps({"paths": paths, "imported": imported}))
"""


def probe(cwd, models_root=None):
    """`config`'s and the asset modules' path constants, read in a fresh interpreter."""
    env = {key: value for key, value in os.environ.items() if key != "SCREENER_MODELS_ROOT"}
    if models_root is not None:
        env["SCREENER_MODELS_ROOT"] = str(models_root)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PACKAGE.parent), env.get("PYTHONPATH")]))
    result = subprocess.run([sys.executable, "-c", PROBE, json.dumps(SCRIPTS)], cwd=cwd, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def assert_only_model_folders_moved(default, moved, root):
    changed = {key for key in default.keys() | moved.keys() if default.get(key) != moved.get(key)}
    assert changed == {"config.MODELS", *MODEL_FOLDERS}
    assert moved["config.MODELS"] == str(root)
    for key, folder in MODEL_FOLDERS.items():
        assert moved[key] == str(root / folder)
    for key in ("config.DATA", "config.PROCESSED", "config.VOCAB"):
        assert moved[key] == default[key]


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_without_a_models_root_every_path_is_unchanged(tmp_path, value):
    """Mutation: default `MODELS` to anything but `PROCESSED`, or drop it."""
    seen = probe(tmp_path, value)
    paths = seen["paths"]
    assert {key: paths.get(key) for key in LAYOUT} == {
        key: str(PROJECT_ROOT / relative) for key, relative in LAYOUT.items()}
    # qwen_assets spelled its folder out before `config.MODELS` existed.
    spelled_out = (PACKAGE / "llm/qwen_assets.py").resolve().parents[3] / "data/processed/chatbot/qwen"
    assert paths["qwen_assets.MODEL_DIR"] == str(spelled_out)
    # The image build runs these scripts with only the standard library and `config`.
    assert set(seen["imported"]) <= {"candidate_screener", "candidate_screener.config"}, seen["imported"]


def test_an_absolute_models_root_moves_only_the_model_folders(tmp_path):
    root = tmp_path / "models"
    assert_only_model_folders_moved(probe(tmp_path)["paths"], probe(tmp_path, root)["paths"], root)


def test_a_relative_models_root_is_resolved_against_the_project_root(tmp_path):
    """The interpreter starts in `tmp_path`: a root read against the working directory fails."""
    moved = probe(tmp_path, "build/models")["paths"]
    assert_only_model_folders_moved(probe(tmp_path)["paths"], moved, PROJECT_ROOT / "build/models")


PINNED = "0123456789abcdef0123456789abcdef01234567"
LATEST = "f" * 40
BACKBONE = "microsoft/deberta-v3-base"
MALFORMED = ["main", "v2.5", PINNED[:-1], PINNED + "8", PINNED.upper(), "g" * 40, ""]


@pytest.fixture
def hub(monkeypatch):
    """Records every Hugging Face call and writes synthetic files instead of downloading."""
    import huggingface_hub
    calls = []

    class Api:
        def model_info(self, repo_id, revision=None, **kwargs):
            calls.append(("model_info", repo_id, revision))
            names = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]
            return SimpleNamespace(sha=revision or LATEST, card_data={"license": "cc-by-4.0"},
                                   siblings=[SimpleNamespace(rfilename=name) for name in names])

    def snapshot_download(repo_id, *, revision, local_dir, allow_patterns, **kwargs):
        calls.append(("snapshot_download", repo_id, revision))
        for name in allow_patterns:
            path = Path(local_dir) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model_name": BACKBONE}) if name == "gliner_config.json"
                            else "synthetic")

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return calls


def run_script(monkeypatch, script, *argv):
    """The asset command line, run by file path as the image build runs it."""
    monkeypatch.setattr(sys, "argv", [script, *argv])
    runpy.run_path(str(PACKAGE / script), run_name="__main__")


def test_qa_revision_reaches_the_download(tmp_path, monkeypatch, capsys, hub):
    from candidate_screener.llm.qa_assets import MODEL_ID
    run_script(monkeypatch, "llm/qa_assets.py", "--destination", str(tmp_path / "pinned"),
               "--revision", PINNED)
    assert ("snapshot_download", MODEL_ID, PINNED) in hub
    assert all(revision == PINNED for _, _, revision in hub), "the latest commit was resolved"
    assert json.loads(capsys.readouterr().out)["revision"] == PINNED
    assert json.loads((tmp_path / "pinned/manifest.json").read_text())["revision"] == PINNED

    hub.clear()  # omitted: the latest commit is resolved, as before
    run_script(monkeypatch, "llm/qa_assets.py", "--destination", str(tmp_path / "latest"))
    assert hub[-1] == ("snapshot_download", MODEL_ID, LATEST)

    hub.clear()  # assets already present at another commit are never passed off as the pin
    with pytest.raises(ValueError, match="revision"):
        run_script(monkeypatch, "llm/qa_assets.py", "--destination", str(tmp_path / "latest"),
                   "--revision", PINNED)
    assert hub == []


def test_medium_encoder_revision_reaches_the_download(tmp_path, monkeypatch, capsys, hub):
    from candidate_screener.extraction.medium_assets import MODEL_ID, MODEL_REVISION
    run_script(monkeypatch, "extraction/medium_assets.py", "--destination", str(tmp_path / "pinned"),
               "--encoder-revision", PINNED)
    assert hub == [("snapshot_download", MODEL_ID, MODEL_REVISION),
                   ("snapshot_download", BACKBONE, PINNED)], "the latest encoder commit was resolved"
    assert json.loads(capsys.readouterr().out)["encoder"] == {"model_id": BACKBONE, "revision": PINNED}
    manifest = json.loads((tmp_path / "pinned/model-manifest.json").read_text())
    assert manifest["encoder"] == {"model_id": BACKBONE, "revision": PINNED}

    hub.clear()  # omitted: the latest encoder commit is resolved, as before
    run_script(monkeypatch, "extraction/medium_assets.py", "--destination", str(tmp_path / "latest"))
    assert hub[-2:] == [("model_info", BACKBONE, None), ("snapshot_download", BACKBONE, LATEST)]


@pytest.mark.parametrize("value", MALFORMED)
def test_malformed_revisions_are_refused_before_any_download(tmp_path, monkeypatch, hub, value):
    for script, flag in (("llm/qa_assets.py", "--revision"),
                         ("extraction/medium_assets.py", "--encoder-revision")):
        with pytest.raises(ValueError, match="40-character"):
            run_script(monkeypatch, script, "--destination", str(tmp_path / "out"), flag, value)
    assert hub == [] and not (tmp_path / "out").exists()
