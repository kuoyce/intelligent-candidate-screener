"""Contract checks for the Docker files (plan 2026-10-27-docker, 01-design-spec.md §2, §5-§7).

Plain text: no Docker and no YAML library. Each rule is a function of a file's text that returns
the problems it finds. One test applies it to the committed file; the negative cases apply it to
small broken texts to show that it fails when its rule is broken (AGENTS testing rule 2).

Dockerfile syntax understood: comments, `\\` continuations, ENV and ARG (`k=v`, and the legacy
`ENV k v`), COPY/ADD flags and the JSON form; no heredocs. Compose: block (`- item`) and flow
(`[a, b]`) sequences, `<<: *anchor` merges.
"""
from __future__ import annotations

import ast
import json
import re
import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------- Dockerfile text

def instructions(dockerfile: str) -> list[tuple[str, str]]:
    """(KEYWORD, arguments) per instruction; comment lines dropped, continuations joined."""
    result, pending = [], ""
    for raw in dockerfile.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        keyword, _, arguments = (pending + line).partition(" ")
        result.append((keyword.upper(), arguments.strip()))
        pending = ""
    if pending.strip():
        keyword, _, arguments = pending.strip().partition(" ")
        result.append((keyword.upper(), arguments.strip()))
    return result


def stages(dockerfile: str) -> dict[str, list[tuple[str, str]]]:
    """Each stage's instructions, from its FROM to the next FROM or the end, in file order."""
    result: dict[str, list[tuple[str, str]]] = {}
    current = None
    for keyword, arguments in instructions(dockerfile):
        if keyword == "FROM":
            words = [word for word in arguments.split() if not word.startswith("--")]
            name = words[2] if len(words) == 3 and words[1].upper() == "AS" else words[0]
            current = result[name.lower()] = []
        elif current is not None:
            current.append((keyword, arguments))
    return result


def stage_env(stage: list[tuple[str, str]]) -> dict[str, str]:
    """The ENV values a stage sets itself, the last setting winning."""
    env: dict[str, str] = {}
    for keyword, arguments in stage:
        if keyword == "ENV":
            words = shlex.split(arguments)
            if words and "=" not in words[0]:
                env[words[0]] = " ".join(words[1:])
            else:
                env.update(word.split("=", 1) for word in words)
    return env


def copy_parts(arguments: str) -> tuple[list[str], list[str]]:
    """(flags, sources) of a COPY or ADD; sources without a leading `./`."""
    flags, rest = [], arguments.strip()
    while rest.startswith("--"):
        flag, _, rest = rest.partition(" ")
        flags.append(flag)
        rest = rest.lstrip()
    words = json.loads(rest) if rest.startswith("[") else shlex.split(rest)
    return flags, [word.removeprefix("./") for word in words[:-1]]


def run_mounts(arguments: str) -> list[dict[str, str]]:
    """The `--mount` options of a RUN, as key/value maps."""
    return [dict(part.split("=", 1) if "=" in part else (part, "") for part in spec.split(","))
            for spec in re.findall(r"--mount=(\S+)", arguments)]


def build_args(dockerfile: str) -> dict[str, str]:
    """Every ARG's default, wherever it is declared."""
    result = {}
    for keyword, arguments in instructions(dockerfile):
        if keyword == "ARG":
            for word in shlex.split(arguments):
                name, _, default = word.partition("=")
                result[name] = default
    return result


# ------------------------------------------------------------------------------ Dockerfile rules

#: The image people run must not download anything at run time (AGENTS: no downloads at startup).
APP_OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "UV_OFFLINE": "1"}
#: Self-contained (standard library plus `config`), so no other change invalidates the downloads.
MODELS_STAGE_FILES = frozenset({
    "src/candidate_screener/__init__.py",
    "src/candidate_screener/config.py",
    "src/candidate_screener/llm/qa_assets.py",
    "src/candidate_screener/llm/qwen_assets.py",
    "src/candidate_screener/extraction/medium_assets.py",
})
FRONTEND = "src/candidate_screener/app"
#: S1: `COPY --exclude` patterns are relative to the source path (`src/`).
FRONTEND_EXCLUDE = "candidate_screener/app"


def app_stage_problems(dockerfile: str) -> list[str]:
    """The `app` stage runs offline, finds the models, runs as a non-root user and is the
    default (last) stage."""
    all_stages = stages(dockerfile)
    if "app" not in all_stages:
        return ["no stage named app"]
    app = all_stages["app"]
    env = stage_env(app)
    problems = [f"the app stage does not set {name}={value}"
                for name, value in APP_OFFLINE.items() if env.get(name) != value]
    if not env.get("SCREENER_MODELS_ROOT"):
        problems.append("the app stage does not set SCREENER_MODELS_ROOT")
    users = [arguments for keyword, arguments in app if keyword == "USER"]
    if not users:
        problems.append("the app stage sets no USER, so it runs as root")
    elif users[-1].split(":")[0].strip() in {"root", "0"}:
        problems.append(f"the app stage runs as root (USER {users[-1]})")
    if list(all_stages)[-1] != "app":
        problems.append("app is not the last stage, so a plain `docker build` builds another one")
    return problems


def models_stage_problems(dockerfile: str) -> list[str]:
    """The `models` stage takes nothing from the build context but the five asset files."""
    all_stages = stages(dockerfile)
    if "models" not in all_stages:
        return ["no stage named models"]
    problems = []
    for keyword, arguments in all_stages["models"]:
        if keyword == "ADD":
            problems.append(f"the models stage uses ADD {arguments}")
        elif keyword == "COPY":
            problems += [f"the models stage copies {source}, not one of the five asset files"
                         for source in copy_parts(arguments)[1] if source not in MODELS_STAGE_FILES]
        elif keyword == "RUN" and any(mount.get("type", "bind") == "bind"
                                      for mount in run_mounts(arguments)):
            problems.append("the models stage bind-mounts files into a RUN")
    return problems


def frontend_order_problems(dockerfile: str) -> list[str]:
    """In `app`, the backend copy of src/ leaves the frontend out; the frontend comes after it,
    with no RUN after it, so a frontend change rebuilds only the last layers."""
    app = stages(dockerfile).get("app")
    if app is None:
        return ["no stage named app"]
    problems, backend, frontend = [], None, None
    for index, (keyword, arguments) in enumerate(app):
        if keyword not in {"COPY", "ADD"}:
            continue
        flags, sources = copy_parts(arguments)
        sources = {source.rstrip("/") for source in sources}
        if "src" in sources:
            backend = index
            excluded = {flag.split("=", 1)[1].rstrip("/") for flag in flags
                        if flag.startswith("--exclude=")}
            if FRONTEND_EXCLUDE not in excluded:
                problems.append(f"the backend copy of src/ does not exclude {FRONTEND_EXCLUDE} "
                                "(patterns are relative to the source path)")
        if FRONTEND in sources:
            frontend = index
    if backend is None:
        problems.append("the app stage has no backend copy of src/")
    if frontend is None:
        problems.append(f"the app stage has no frontend copy of {FRONTEND}/")
    else:
        if backend is not None and frontend < backend:
            problems.append("the frontend copy comes before the backend copy")
        if any(keyword == "RUN" for keyword, _ in app[frontend + 1:]):
            problems.append("a RUN follows the frontend copy, so a frontend change re-runs it")
    return problems


def c2_pin(cross_source: str) -> dict:
    """The C2 pin of `rerank/cross.py` (its `QWEN` assignment)."""
    for node in ast.parse(cross_source).body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "QWEN"
                                                for target in node.targets):
            return ast.literal_eval(node.value)
    return {}


def c2_pin_problems(dockerfile: str, cross_source: str) -> list[str]:
    """The Dockerfile downloads C2 at the model and revision the code loads."""
    pin = c2_pin(cross_source)
    if not pin:
        return ["rerank/cross.py has no QWEN pin"]
    args = build_args(dockerfile)
    return [f"the Dockerfile's {arg} is {args.get(arg)!r}; rerank/cross.py pins {pin.get(key)!r}"
            for arg, key in (("C2_MODEL", "model"), ("C2_REVISION", "revision"))
            if args.get(arg) != pin.get(key)]


# ------------------------------------------------------------------------ ignore and attributes

DOCKERIGNORE_LINES = ("data/*", "!data/README.md", "!data/vocab/", ".git", ".venv", "**/.venv",
                      "logs", ".env", ".streamlit/secrets.toml")
#: Their SHA-256 digests enter cache fingerprints, so Git never converts their line endings.
BYTE_EXACT = ("docs/data/**", "output/**", "data/vocab/**")


def _rules(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


def dockerignore_problems(text: str) -> list[str]:
    """data/ (but README.md and vocab/), environments, logs and secrets stay out of the context;
    the last matching line wins, so nothing later may let them back in."""
    lines = _rules(text)
    problems = [f".dockerignore lacks {line}" for line in DOCKERIGNORE_LINES if line not in lines]
    last = {line: index for index, line in enumerate(lines)}
    for exception in ("!data/README.md", "!data/vocab/"):
        if exception in last and "data/*" in last and last[exception] < last["data/*"]:
            problems.append(f"{exception} comes before the last data/*, which excludes it again")
    for pattern in DOCKERIGNORE_LINES:
        if pattern.startswith("!") or pattern not in last:
            continue
        for later in lines[last[pattern] + 1:]:
            target = later[1:].removesuffix("/**").rstrip("/") if later.startswith("!") else None
            if target in {pattern.rstrip("/"), "*", "**", "**/*"}:
                problems.append(f"{later} lets {pattern} back into the build context")
    return problems


def gitattributes_problems(text: str) -> list[str]:
    """docs/data/, output/ and data/vocab/ are marked -text, and no later line re-enables
    conversion there (the last matching line wins)."""
    rules = [line.split() for line in _rules(text)]
    problems = []
    for pattern in BYTE_EXACT:
        marked = [index for index, (rule, *attributes) in enumerate(rules)
                  if rule == pattern and {"-text", "binary"} & set(attributes)]
        if not marked:
            problems.append(f".gitattributes does not mark {pattern} as -text")
            continue
        folder = pattern.removesuffix("/**")
        for rule, *attributes in rules[marked[-1] + 1:]:
            converts = any(value in {"text", "!text"} or value.startswith("text=")
                           for value in attributes)
            prefix = re.split(r"[*?\[]", rule, maxsplit=1)[0].lstrip("/")
            reaches = ("/" not in rule.rstrip("/") or folder.startswith(prefix)
                       or prefix.startswith(folder + "/"))
            if converts and reaches:
                problems.append(f"`{rule} {' '.join(attributes)}` re-enables conversion "
                                f"under {folder}/")
    return problems


# ------------------------------------------------------------------------------- compose text

def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _strip_comment(value: str) -> str:
    return re.sub(r"(^|\s)#.*$", "", value).strip()


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def sequences(text: str, key: str) -> list[list[str]]:
    """Every `key:` sequence in a YAML text as unquoted items, block or flow style. A mapping
    item (compose's long syntax) comes back as the text of its first line."""
    lines = text.splitlines()
    pattern = re.compile(r"^(\s*)(?:-\s+)?" + re.escape(key) + r":(?:\s+(.*))?$")
    found = []
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        value = _strip_comment(match.group(2) or "")
        if value.startswith("["):
            following = iter(lines[index + 1:])
            while "]" not in value:
                value += " " + _strip_comment(next(following, "]"))
            inner = value[1:value.index("]")]
            found.append([_unquote(item) for item in inner.split(",") if item.strip()])
        elif not value:
            items, item_indent = [], None
            for later in lines[index + 1:]:
                stripped = later.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                is_item = stripped == "-" or stripped.startswith("- ")
                if item_indent is None:
                    if not is_item or _indent(later) < len(match.group(1)):
                        break
                    item_indent = _indent(later)
                if is_item and _indent(later) == item_indent:
                    items.append(_unquote(_strip_comment(stripped[1:])))
                elif _indent(later) <= item_indent:
                    break
            if item_indent is not None:
                found.append(items)
    return found


def _block(lines: list[str], start: int) -> list[str]:
    """The lines nested under lines[start]."""
    block = []
    for line in lines[start + 1:]:
        if line.strip() and not line.lstrip().startswith("#") and _indent(line) <= _indent(lines[start]):
            break
        block.append(line)
    return block


def _service_block(compose: str, service: str) -> str | None:
    lines = compose.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^services:\s*(#.*)?$", line):
            services = _block(lines, index)
            child = min((_indent(entry) for entry in services
                         if entry.strip() and not entry.lstrip().startswith("#")), default=0)
            for position, entry in enumerate(services):
                if _indent(entry) == child and re.match(rf"^\s*{re.escape(service)}:\s*(#.*)?$", entry):
                    return "\n".join(_block(services, position))
    return None


def service_volumes(compose: str, service: str) -> list[str]:
    """A service's volumes: its own `volumes:`, else those of the anchors it merges."""
    block = _service_block(compose, service)
    if block is None:
        return []
    if re.search(r"^\s*volumes:", block, re.MULTILINE):
        return [item for items in sequences(block, "volumes") for item in items]
    lines = compose.splitlines()
    volumes = []
    for anchor in re.findall(r"<<:\s*\*([\w-]+)", block):
        for index, line in enumerate(lines):
            if re.search(rf"&{re.escape(anchor)}\b", line):
                anchored = "\n".join(_block(lines, index))
                volumes += [item for items in sequences(anchored, "volumes") for item in items]
    return volumes


def split_mount(item: str) -> tuple[str, str, str]:
    """(host, container, mode) of a short-syntax volume; a `:` inside `${...}` does not split."""
    parts, current, depth, index = [], "", 0, 0
    while index < len(item):
        if item.startswith("${", index):
            depth, current, index = depth + 1, current + "${", index + 2
            continue
        char = item[index]
        if char == "}" and depth:
            depth -= 1
        if char == ":" and not depth:
            parts.append(current)
            current = ""
        else:
            current += char
        index += 1
    host, container, mode = (parts + [current, "", ""])[:3]
    return host.rstrip("/") or host, container.rstrip("/"), mode


def published_port_problems(compose: str) -> list[str]:
    """Every published port is bound to host loopback (local token sign-in is not SSO)."""
    problems = [f"port {item!r} is not published on 127.0.0.1 only (short syntax "
                "127.0.0.1:HOST:CONTAINER)"
                for items in sequences(compose, "ports") for item in items
                if not item.startswith("127.0.0.1:")]
    if re.search(r"^\s*network_mode:\s*[\"']?host[\"']?\s*(#.*)?$", compose, re.MULTILINE):
        problems.append("network_mode: host publishes every port on all interfaces")
    return problems


DATA_FOLDER = ("./data/datapack", "/app/data")


def data_folder_problems(compose: str, services=("app", "tools")) -> list[str]:
    """`./data/datapack` is the only folder mounted at /app/data, read-write, in `services`."""
    problems = []
    for item in (entry for items in sequences(compose, "volumes") for entry in items):
        host, container, mode = split_mount(item)
        if container == "/app/data" and host != DATA_FOLDER[0]:
            problems.append(f"{item!r} mounts {host} at /app/data, not ./data/datapack")
        elif container == "/app/data" and "ro" in mode.split(","):
            problems.append(f"{item!r} mounts the data folder read-only")
    for service in services:
        if not any(split_mount(item)[:2] == DATA_FOLDER for item in service_volumes(compose, service)):
            problems.append(f"the {service} service does not mount ./data/datapack at /app/data")
    return problems


# ---------------------------------------------------------------------- the committed files

def test_ports_are_published_on_host_loopback_only():
    compose = read("compose.yaml")
    assert published_port_problems(compose) == []
    assert published_port_problems(read("compose.dev.yaml")) == []
    ports = [item for items in sequences(compose, "ports") for item in items]
    assert "127.0.0.1:${HIRING_DESK_PORT:-8503}:8503" in ports


def test_app_stage_runs_offline_as_a_non_root_user():
    assert app_stage_problems(read("Dockerfile")) == []


def test_models_stage_copies_only_the_five_asset_files():
    assert models_stage_problems(read("Dockerfile")) == []


def test_frontend_is_copied_after_the_backend():
    assert frontend_order_problems(read("Dockerfile")) == []


def test_dockerignore_keeps_data_environments_logs_and_secrets_out():
    assert dockerignore_problems(read(".dockerignore")) == []


def test_gitattributes_keeps_digested_files_byte_exact():
    assert gitattributes_problems(read(".gitattributes")) == []


def test_c2_download_matches_the_code_pin():
    assert c2_pin_problems(read("Dockerfile"), read("src/candidate_screener/rerank/cross.py")) == []


def test_compose_mounts_the_data_folder_read_write():
    assert data_folder_problems(read("compose.yaml")) == []
    assert data_folder_problems(read("compose.dev.yaml"), services=()) == []


# ------------------------------------------------------------- negative cases: broken texts

GOOD_DOCKERFILE = """\
# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
FROM base AS models
ENV SCREENER_MODELS_ROOT=/opt/models
ARG C2_MODEL=Qwen/Qwen3-Reranker-0.6B
ARG C2_REVISION=e61197ed45024b0ed8a2d74b80b4d909f1255473
COPY src/candidate_screener/__init__.py src/candidate_screener/config.py /app/src/candidate_screener/
COPY src/candidate_screener/llm/qa_assets.py /app/src/candidate_screener/llm/
RUN --mount=type=cache,target=/cache --mount=type=secret,id=hf_token,required=false \\
    python /app/src/candidate_screener/llm/qa_assets.py
FROM models AS app
COPY --exclude=candidate_screener/app src/ /app/src/
RUN python -m compileall /app/src
COPY src/candidate_screener/app/ /app/src/candidate_screener/app/
ENV SCREENER_MODELS_ROOT=/opt/models \\
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1
USER 1000:1000
"""
GOOD_CROSS = 'QWEN ={"model": "Qwen/Qwen3-Reranker-0.6B", "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473"}\n'
GOOD_DOCKERIGNORE = "\n".join(DOCKERIGNORE_LINES) + "\n"
GOOD_GITATTRIBUTES = """\
* text=auto
docs/data/** -text
output/** -text
data/vocab/** binary
*.toml eol=lf
"""
GOOD_COMPOSE = """\
x-run: &run
  image: example
  volumes:
    - ./data/datapack:/app/data
    - ./logs:/app/logs
services:
  app:
    <<: *run
    ports:
      - "127.0.0.1:${HIRING_DESK_PORT:-8503}:8503"
  tools:
    <<: *run
    volumes:
      - ./data/datapack:/app/data
      - "${SCREENER_DATA_SOURCE:-./data}:/source:ro"
"""


def broken(text: str, *replacements: tuple[str, str]) -> str:
    """`text` with each (old, new) applied; old must occur exactly once."""
    for old, new in replacements:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text


def test_every_check_passes_on_the_good_texts():
    assert app_stage_problems(GOOD_DOCKERFILE) == []
    assert models_stage_problems(GOOD_DOCKERFILE) == []
    assert frontend_order_problems(GOOD_DOCKERFILE) == []
    assert c2_pin_problems(GOOD_DOCKERFILE, GOOD_CROSS) == []
    assert dockerignore_problems(GOOD_DOCKERIGNORE) == []
    assert gitattributes_problems(GOOD_GITATTRIBUTES) == []
    assert published_port_problems(GOOD_COMPOSE) == []
    assert data_folder_problems(GOOD_COMPOSE) == []


APP_OFFLINE_ENV = "ENV SCREENER_MODELS_ROOT=/opt/models \\\n    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1\n"
FRONTEND_COPY = "COPY src/candidate_screener/app/ /app/src/candidate_screener/app/\n"
BACKEND_COPY = "COPY --exclude=candidate_screener/app src/ /app/src/\n"
QA_COPY = "COPY src/candidate_screener/llm/qa_assets.py /app/src/candidate_screener/llm/\n"
MODELS_RUN = "RUN --mount=type=cache,target=/cache"

BROKEN_DOCKERFILES = {
    # app stage
    "app-without-HF_HUB_OFFLINE": (app_stage_problems, broken(GOOD_DOCKERFILE, ("HF_HUB_OFFLINE=1 ", ""))),
    "app-with-HF_HUB_OFFLINE=0": (app_stage_problems, broken(GOOD_DOCKERFILE, ("HF_HUB_OFFLINE=1", "HF_HUB_OFFLINE=0"))),
    "app-without-TRANSFORMERS_OFFLINE": (app_stage_problems, broken(GOOD_DOCKERFILE, ("TRANSFORMERS_OFFLINE=1 ", ""))),
    "app-without-UV_OFFLINE": (app_stage_problems, broken(GOOD_DOCKERFILE, (" UV_OFFLINE=1", ""))),
    "app-without-SCREENER_MODELS_ROOT": (app_stage_problems, broken(
        GOOD_DOCKERFILE, ("ENV SCREENER_MODELS_ROOT=/opt/models \\\n    HF_HUB", "ENV HF_HUB"))),
    "offline-set-in-models-not-app": (app_stage_problems, broken(
        GOOD_DOCKERFILE, (APP_OFFLINE_ENV, ""), ("ENV SCREENER_MODELS_ROOT=/opt/models\nARG", APP_OFFLINE_ENV + "ARG"))),
    "offline-turned-off-later": (app_stage_problems, GOOD_DOCKERFILE + "ENV HF_HUB_OFFLINE=0\n"),
    "app-runs-as-root": (app_stage_problems, broken(GOOD_DOCKERFILE, ("USER 1000:1000", "USER root"))),
    "app-runs-as-uid-0": (app_stage_problems, broken(GOOD_DOCKERFILE, ("USER 1000:1000", "USER 0:0"))),
    "app-without-USER": (app_stage_problems, broken(GOOD_DOCKERFILE, ("USER 1000:1000\n", ""))),
    "app-not-the-last-stage": (app_stage_problems, GOOD_DOCKERFILE + "FROM base AS test\n"),
    # models stage
    "models-copies-all-of-src": (models_stage_problems, broken(GOOD_DOCKERFILE, (QA_COPY, QA_COPY + "COPY src/ /app/src/\n"))),
    "models-copies-a-frontend-file": (models_stage_problems, broken(
        GOOD_DOCKERFILE, (QA_COPY, QA_COPY + "COPY src/candidate_screener/app/portal_ui.py /app/\n"))),
    "models-copies-a-sixth-module": (models_stage_problems, broken(
        GOOD_DOCKERFILE, ("llm/qa_assets.py /app/", "llm/qa_assets.py src/candidate_screener/llm/qwen.py /app/"))),
    "models-copies-in-json-form": (models_stage_problems, broken(
        GOOD_DOCKERFILE, (QA_COPY, QA_COPY + 'COPY ["src/candidate_screener/app", "/app/x"]\n'))),
    "models-uses-ADD": (models_stage_problems, broken(GOOD_DOCKERFILE, (QA_COPY, "ADD" + QA_COPY[4:]))),
    "models-binds-the-context": (models_stage_problems, broken(
        GOOD_DOCKERFILE, (MODELS_RUN, "RUN --mount=type=bind,source=src,target=/src --mount=type=cache,target=/cache"))),
    "models-binds-by-default": (models_stage_problems, broken(
        GOOD_DOCKERFILE, (MODELS_RUN, "RUN --mount=source=src,target=/src --mount=type=cache,target=/cache"))),
    # frontend layer order
    "frontend-before-backend": (frontend_order_problems, broken(
        GOOD_DOCKERFILE, (BACKEND_COPY, FRONTEND_COPY), ("RUN python -m compileall /app/src\n" + FRONTEND_COPY,
                                                         "RUN python -m compileall /app/src\n" + BACKEND_COPY))),
    "backend-copy-includes-the-frontend": (frontend_order_problems, broken(
        GOOD_DOCKERFILE, ("--exclude=candidate_screener/app ", ""))),
    "exclude-written-from-the-context-root": (frontend_order_problems, broken(
        GOOD_DOCKERFILE, ("--exclude=candidate_screener/app ", "--exclude=src/candidate_screener/app/** "))),
    "RUN-after-the-frontend-copy": (frontend_order_problems, broken(
        GOOD_DOCKERFILE, (FRONTEND_COPY, FRONTEND_COPY + "RUN python -m compileall /app/src\n"))),
    "no-frontend-copy": (frontend_order_problems, broken(GOOD_DOCKERFILE, (FRONTEND_COPY, ""))),
    # C2 pin
    "c2-revision-differs": (lambda text: c2_pin_problems(text, GOOD_CROSS), broken(
        GOOD_DOCKERFILE, ("C2_REVISION=e61197ed", "C2_REVISION=0000000d"))),
    "c2-model-differs": (lambda text: c2_pin_problems(text, GOOD_CROSS), broken(
        GOOD_DOCKERFILE, ("Reranker-0.6B", "Reranker-4B"))),
    "c2-revision-missing": (lambda text: c2_pin_problems(text, GOOD_CROSS), broken(
        GOOD_DOCKERFILE, ("ARG C2_REVISION=e61197ed45024b0ed8a2d74b80b4d909f1255473\n", ""))),
    "c2-code-pin-moved-on": (lambda text: c2_pin_problems(GOOD_DOCKERFILE, text), broken(
        GOOD_CROSS, ("e61197ed", "1111111d"))),
}


@pytest.mark.parametrize("case", sorted(BROKEN_DOCKERFILES))
def test_dockerfile_checks_fail_on_broken_texts(case):
    check, text = BROKEN_DOCKERFILES[case]
    assert check(text), case


BROKEN_IGNORE_AND_ATTRIBUTES = {
    **{f"dockerignore-lacks-{line}": (dockerignore_problems, broken(GOOD_DOCKERIGNORE, (line + "\n", "")))
       for line in DOCKERIGNORE_LINES if line != ".venv"},
    "dockerignore-lacks-.venv": (dockerignore_problems, broken(GOOD_DOCKERIGNORE, ("\n.venv\n", "\n"))),
    "dockerignore-lets-.env-back": (dockerignore_problems, GOOD_DOCKERIGNORE + "!.env\n"),
    "dockerignore-lets-everything-back": (dockerignore_problems, GOOD_DOCKERIGNORE + "!**\n"),
    "dockerignore-exception-before-data": (dockerignore_problems, GOOD_DOCKERIGNORE + "data/*\n"),
    "gitattributes-lacks-output": (gitattributes_problems, broken(GOOD_GITATTRIBUTES, ("output/** -text\n", ""))),
    "gitattributes-converts-output": (gitattributes_problems, broken(GOOD_GITATTRIBUTES, ("output/** -text", "output/** text"))),
    "gitattributes-later-text-auto": (gitattributes_problems, GOOD_GITATTRIBUTES + "* text=auto\n"),
    "gitattributes-later-csv-crlf": (gitattributes_problems, GOOD_GITATTRIBUTES + "*.csv text eol=crlf\n"),
    "gitattributes-later-parent-folder": (gitattributes_problems, GOOD_GITATTRIBUTES + "docs/** text\n"),
    "gitattributes-later-unspecified": (gitattributes_problems, GOOD_GITATTRIBUTES + "data/vocab/** !text\n"),
}


@pytest.mark.parametrize("case", sorted(BROKEN_IGNORE_AND_ATTRIBUTES))
def test_ignore_and_attribute_checks_fail_on_broken_texts(case):
    check, text = BROKEN_IGNORE_AND_ATTRIBUTES[case]
    assert check(text), case


PUBLISHED = '      - "127.0.0.1:${HIRING_DESK_PORT:-8503}:8503"\n'
APP_SERVICE = "  app:\n    <<: *run\n"

BROKEN_COMPOSE = {
    "port-on-all-interfaces": (published_port_problems, broken(GOOD_COMPOSE, (PUBLISHED, '      - "8503:8503"\n'))),
    "port-on-0.0.0.0": (published_port_problems, broken(GOOD_COMPOSE, ("127.0.0.1:${", "0.0.0.0:${"))),
    "port-variable-only": (published_port_problems, broken(GOOD_COMPOSE, ('"127.0.0.1:${', '"${'))),
    "port-in-flow-style": (published_port_problems, broken(
        GOOD_COMPOSE, ("    ports:\n" + PUBLISHED, '    ports: ["8503:8503"]\n'))),
    "port-in-long-syntax": (published_port_problems, broken(
        GOOD_COMPOSE, (PUBLISHED, "      - target: 8503\n        published: 8503\n"))),
    "port-on-a-second-service": (published_port_problems, GOOD_COMPOSE + '  qwen:\n    ports: ["8000:8000"]\n'),
    "host-network": (published_port_problems, GOOD_COMPOSE + "  qwen:\n    network_mode: host\n"),
    "data-folder-is-all-of-data": (data_folder_problems, broken(
        GOOD_COMPOSE, ("    - ./data/datapack:/app/data\n    - ./logs", "    - ./data:/app/data\n    - ./logs"))),
    "data-folder-read-only": (data_folder_problems, broken(
        GOOD_COMPOSE, ("      - ./data/datapack:/app/data\n", "      - ./data/datapack:/app/data:ro\n"))),
    "app-overrides-the-anchor-volumes": (data_folder_problems, broken(
        GOOD_COMPOSE, (APP_SERVICE, APP_SERVICE + "    volumes:\n      - ./logs:/app/logs\n"))),
    "tools-without-the-data-folder": (data_folder_problems, broken(
        GOOD_COMPOSE, ("      - ./data/datapack:/app/data\n", ""))),
    "anchor-without-the-data-folder": (data_folder_problems, broken(
        GOOD_COMPOSE, ("    - ./data/datapack:/app/data\n    - ./logs", "    - ./logs"))),
}


@pytest.mark.parametrize("case", sorted(BROKEN_COMPOSE))
def test_compose_checks_fail_on_broken_texts(case):
    check, text = BROKEN_COMPOSE[case]
    assert check(text), case
