"""Contract checks for the Docker files (plan 2026-10-27-docker, 01-design-spec.md §2, §5-§7).

Plain text: no Docker and no YAML library. Each rule is a function of a file's text that returns
the problems it finds. One test applies it to the committed file; the negative cases apply it to
small broken texts to show that it fails when its rule is broken (AGENTS testing rule 2).

Dockerfile syntax understood: comments, `\\` continuations, ENV and ARG (`k=v`, and the legacy
`ENV k v`), COPY/ADD flags, CMD and the JSON form; no heredocs. Compose: block (`- item`) and flow
(`[a, b]`) sequences, a string `command:`, block and flow mappings under `healthcheck:`,
`<<: *anchor` merges. The guide (`docs/docker.md`): its "Configuration reference" section's Markdown
tables, first column; fenced code skipped.
"""
from __future__ import annotations

import ast
import fnmatch
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

#: The image people run must not download anything at run time (AGENTS: no downloads at startup),
#: and onnxruntime (imported by GLiNER) must not write or send telemetry (S4).
APP_OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "UV_OFFLINE": "1",
               "ORT_DISABLE_TELEMETRY": "1"}
#: Streamlit's flags that keep it off the network (S4): no usage statistics, no lookup of the host's
#: external IP (it prints localhost instead), no browser to open.
STREAMLIT_FLAGS = {"--server.headless": "true", "--browser.serverAddress": "localhost",
                   "--browser.gatherUsageStats": "false"}
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


def command_flags(command: list[str]) -> dict[str, str]:
    """`--name=value` and `--name value` options of a command; a later one wins, as in Streamlit."""
    flags = {}
    for index, word in enumerate(command):
        if word.startswith("--"):
            name, equals, value = word.partition("=")
            if not equals:
                following = command[index + 1:index + 2]
                value = following[0] if following and not following[0].startswith("--") else ""
            flags[name] = value
    return flags


def streamlit_flag_problems(command: list[str] | None, where: str) -> list[str]:
    """A command that runs Streamlit passes the flags that keep it off the network."""
    if not command or "streamlit" not in command:
        return [f"{where} does not run streamlit"]
    flags = command_flags(command)
    return [f"{where} does not pass {name}={value}" for name, value in STREAMLIT_FLAGS.items()
            if flags.get(name) != value]


def app_command(dockerfile: str) -> list[str] | None:
    """The app stage's own CMD (its last), as words; None when it sets none."""
    commands = [arguments for keyword, arguments in stages(dockerfile).get("app", []) if keyword == "CMD"]
    if not commands:
        return None
    return json.loads(commands[-1]) if commands[-1].startswith("[") else shlex.split(commands[-1])


def app_command_problems(dockerfile: str) -> list[str]:
    return streamlit_flag_problems(app_command(dockerfile), "the app stage's CMD")


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
                      "logs", ".env", ".streamlit/secrets.toml", "compose.override.y*ml")
#: One machine's settings and secrets (Docker's .env and compose override, Streamlit's secrets)
#: are never committed.
GITIGNORE_LINES = (".env", "compose.override.yaml", "compose.override.yml", ".streamlit/secrets.toml")
#: The files a glob line is there for; a literal line is there for its own path.
COVERS = {"compose.override.y*ml": ("compose.override.yaml", "compose.override.yml")}
#: Their SHA-256 digests enter cache fingerprints, so Git never converts their line endings.
BYTE_EXACT = ("docs/data/**", "output/**", "data/vocab/**")


def _rules(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


def _lets_back(later: str, pattern: str) -> bool:
    """Whether a later `!` line re-includes what `pattern` excludes: the same pattern, a catch-all,
    or a glob matching a file the pattern is there for."""
    if not later.startswith("!"):
        return False
    target = later[1:].lstrip("/").removesuffix("/**").rstrip("/")
    return (target in {pattern.rstrip("/"), "*", "**", "**/*"}
            or any(fnmatch.fnmatchcase(name, target) for name in COVERS.get(pattern, (pattern.rstrip("/"),))))


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
        problems += [f"{later} lets {pattern} back into the build context"
                     for later in lines[last[pattern] + 1:] if _lets_back(later, pattern)]
    return problems


def gitignore_problems(text: str) -> list[str]:
    """One machine's settings and secrets stay out of commits, and nothing later lets them back."""
    lines = _rules(text)
    problems = [f".gitignore lacks {line}" for line in GITIGNORE_LINES if line not in lines]
    last = {line: index for index, line in enumerate(lines)}
    for pattern in GITIGNORE_LINES:
        if pattern in last:
            problems += [f"{later} lets {pattern} back into commits"
                         for later in lines[last[pattern] + 1:] if _lets_back(later, pattern)]
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


def service_environment(compose: str, service: str) -> dict[str, str]:
    """A service's own `environment:`: a block mapping, a `- KEY=value` list or a flow mapping."""
    lines = (_service_block(compose, service) or "").splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^\s*environment:\s*(.*)$", line)
        if not match:
            continue
        flow = _strip_comment(match.group(1))
        entries = flow.strip("{}").split(",") if flow.startswith("{") else _block(lines, index)
        result = {}
        for entry in entries:
            entry = _strip_comment(entry.strip())
            if not entry:
                continue
            if entry.startswith("- "):
                key, _, value = _unquote(entry[2:]).partition("=")
            else:
                key, _, value = entry.partition(":")
            result[_unquote(key)] = _unquote(value)
        return result
    return {}


def tools_settings_problems(compose: str) -> list[str]:
    """The tools service tells `datapack` where its folders are: SCREENER_DATAPACK_DEST is the
    data folder's mount (/app/data), SCREENER_DATAPACK_SOURCE the container side of a read-only
    mount (the source is never written)."""
    env = service_environment(compose, "tools")
    problems = []
    if env.get("SCREENER_DATAPACK_DEST") != DATA_FOLDER[1]:
        problems.append(f"the tools service sets SCREENER_DATAPACK_DEST={env.get('SCREENER_DATAPACK_DEST')!r}, "
                        f"not {DATA_FOLDER[1]}")
    source = env.get("SCREENER_DATAPACK_SOURCE")
    mounts = [split_mount(item) for item in service_volumes(compose, "tools")]
    if not source or not any(container == source.rstrip("/") and "ro" in mode.split(",")
                             for _, container, mode in mounts):
        problems.append(f"the tools service sets SCREENER_DATAPACK_SOURCE={source!r}, which is not "
                        "the container side of one of its read-only mounts")
    return problems


def service_command(compose: str, service: str) -> list[str] | None:
    """A service's own `command:` as words (a sequence, or a string split as a shell would); None
    when it sets none."""
    block = _service_block(compose, service)
    if block is None:
        return None
    found = sequences(block, "command")
    if found:
        return found[0]
    match = re.search(r"^\s*command:\s*(\S.*)$", block, re.MULTILINE)
    return shlex.split(_unquote(_strip_comment(match.group(1)))) if match else None


def dev_command_problems(compose: str) -> list[str]:
    """compose.dev.yaml replaces the image's command (polling, rerun on save); the replacement keeps
    the image's Streamlit flags."""
    return streamlit_flag_problems(service_command(compose, "app"), "the app service's command")


def _healthcheck(compose: str, service: str) -> tuple[str, list[str]] | None:
    """A service's own `healthcheck:` as (flow text, nested lines); None when it sets none."""
    lines = (_service_block(compose, service) or "").splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)healthcheck:\s*(.*)$", line)
        if match:
            return _strip_comment(match.group(2)), _block(lines, index)
    return None


def healthcheck_disabled(compose: str, service: str) -> bool:
    """Whether a service sets `healthcheck: disable: true` (block or flow mapping)."""
    found = _healthcheck(compose, service)
    if found is None:
        return False
    flow, block = found
    entries = re.split(r",(?![^\[]*\])", flow.strip("{}")) if flow.startswith("{") else block
    for entry in entries:
        key, _, value = _strip_comment(entry.strip()).partition(":")
        if key.strip() == "disable" and _unquote(value.strip()).lower() == "true":
            return True
    return False


def healthcheck_test(compose: str, service: str) -> list[str] | None:
    """A service's own healthcheck `test:` as words (a sequence, or a string as one word); None
    when it sets none. Block or flow mapping."""
    found = _healthcheck(compose, service)
    if found is None:
        return None
    flow, block = found
    if flow.startswith("{"):
        match = re.search(r"\btest:\s*(\[[^\]]*\]|[^,}]+)", flow)
        if not match:
            return None
        found_test = sequences("test: " + match.group(1), "test")
        return found_test[0] if found_test else [_unquote(match.group(1))]
    text = "\n".join(block)
    found_test = sequences(text, "test")
    if found_test:
        return found_test[0]
    match = re.search(r"^\s*test:\s*(\S.*)$", text, re.MULTILINE)
    return [_unquote(_strip_comment(match.group(1)))] if match else None


#: The qwen service's own check: the model server's model list on its loopback port.
QWEN_PROBE = "127.0.0.1:8000/v1/models"
#: The image's check: Streamlit's health endpoint on the app's port.
STREAMLIT_PROBES = ("/_stcore/health", ":8503")


def healthcheck_problems(compose: str) -> list[str]:
    """tools runs no server: it disables the image's healthcheck, which probes Streamlit. qwen runs
    the model server in app's network namespace, where the image's probe would report the app's
    health: it has a check of its own, enabled, that asks the model server and not Streamlit."""
    problems = []
    if not healthcheck_disabled(compose, "tools"):
        problems.append("the tools service keeps the image's Streamlit healthcheck (set healthcheck: "
                        "disable: true)")
    test = healthcheck_test(compose, "qwen")
    words = " ".join(test or [])
    if healthcheck_disabled(compose, "qwen") or (test and test[0].upper() == "NONE"):
        problems.append("the qwen service disables its healthcheck (give it its own check of "
                        f"{QWEN_PROBE})")
    elif not test:
        problems.append("the qwen service has no healthcheck test of its own, so it keeps the image's "
                        f"Streamlit probe (add one asking {QWEN_PROBE})")
    elif any(probe in words for probe in STREAMLIT_PROBES):
        problems.append("the qwen service's healthcheck probes Streamlit, which reports the app's "
                        "health, not the model server's")
    elif QWEN_PROBE not in words:
        problems.append(f"the qwen service's healthcheck does not ask the model server ({QWEN_PROBE})")
    return problems


# ------------------------------------------------------------------ the guide's settings table

GUIDE = "docs/docker.md"
#: A setting's name. A prefix ("SCREENER_QWEN_") or a longer word containing one
#: ("CANDIDATE_SCREENER_WORKER:") is not a name.
SETTING = re.compile(r"(?:HR_PORTAL|SCREENER)_[A-Z0-9]+(?:_[A-Z0-9]+)*")
CONFIGURATION_SECTION = "configuration reference"


def settings_read(sources: dict[str, str]) -> dict[str, list[str]]:
    """Every `HR_PORTAL_*` and `SCREENER_*` name a Python source holds as a whole string literal
    (an environment read, its default, or a constant that names it; comments do not count), with
    the sources that hold it."""
    found: dict[str, set[str]] = {}
    for path, text in sources.items():
        for node in ast.walk(ast.parse(text)):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and SETTING.fullmatch(node.value)):
                found.setdefault(node.value, set()).add(path)
    return {name: sorted(paths) for name, paths in sorted(found.items())}


def configuration_names(guide: str) -> set[str] | None:
    """The backticked names in the first column of every table in the guide's "Configuration
    reference" section, which ends at the next heading of its level or above; None when the guide
    has no such section. Lines in fenced code are neither headings nor table rows."""
    level, names, fenced = None, set(), False
    for line in guide.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced:
            continue
        heading = re.match(r"^ {0,3}(#{1,6})\s+(.*)$", line)
        if heading:
            depth = len(heading.group(1))
            if level is not None and depth <= level:
                break
            if level is None and CONFIGURATION_SECTION in heading.group(2).lower():
                level = depth
            continue
        if level is not None and stripped.startswith("|"):
            first_cell = stripped.strip("|").split("|")[0]
            names.update(re.findall(r"`([^`]+)`", first_cell))
    return names if level is not None else None


def undocumented_settings(sources: dict[str, str], guide: str) -> list[str]:
    """Every setting read under src/ has a row in the guide's configuration table."""
    documented = configuration_names(guide)
    if documented is None:
        return [f"{GUIDE} has no section headed 'Configuration reference'"]
    return [f"{name}, read in {', '.join(paths)}, has no row in the configuration table of {GUIDE}"
            for name, paths in settings_read(sources).items() if name not in documented]


def source_texts() -> dict[str, str]:
    """The package's Python sources, by path."""
    return {path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted((ROOT / "src").rglob("*.py"))}


# ---------------------------------------------------------------------- the committed files

def test_ports_are_published_on_host_loopback_only():
    compose = read("compose.yaml")
    assert published_port_problems(compose) == []
    assert published_port_problems(read("compose.dev.yaml")) == []
    ports = [item for items in sequences(compose, "ports") for item in items]
    assert "127.0.0.1:${HIRING_DESK_PORT:-8503}:8503" in ports


def test_app_stage_runs_offline_as_a_non_root_user():
    assert app_stage_problems(read("Dockerfile")) == []


def test_streamlit_runs_with_its_network_flags_in_the_image_and_the_dev_compose():
    assert app_command_problems(read("Dockerfile")) == []
    assert dev_command_problems(read("compose.dev.yaml")) == []
    # The commands are found, so an empty parse cannot pass.
    assert app_command(read("Dockerfile"))[:2] == ["streamlit", "run"]
    assert service_command(read("compose.dev.yaml"), "app")[:2] == ["streamlit", "run"]


def test_qwen_has_its_own_healthcheck_and_tools_disables_the_image_one():
    compose = read("compose.yaml")
    assert healthcheck_problems(compose) == []
    assert not healthcheck_disabled(compose, "app")
    # The check is found, so an empty parse cannot pass.
    assert healthcheck_test(compose, "qwen")[:2] == ["CMD", "python"]


def test_models_stage_copies_only_the_five_asset_files():
    assert models_stage_problems(read("Dockerfile")) == []


def test_frontend_is_copied_after_the_backend():
    assert frontend_order_problems(read("Dockerfile")) == []


def test_dockerignore_keeps_data_environments_logs_and_secrets_out():
    assert dockerignore_problems(read(".dockerignore")) == []


def test_gitignore_keeps_this_machines_settings_and_secrets_out():
    assert gitignore_problems(read(".gitignore")) == []


def test_gitattributes_keeps_digested_files_byte_exact():
    assert gitattributes_problems(read(".gitattributes")) == []


def test_c2_download_matches_the_code_pin():
    assert c2_pin_problems(read("Dockerfile"), read("src/candidate_screener/rerank/cross.py")) == []


def test_compose_mounts_the_data_folder_read_write():
    assert data_folder_problems(read("compose.yaml")) == []
    assert data_folder_problems(read("compose.dev.yaml"), services=()) == []


def test_tools_service_points_datapack_at_its_mounts():
    assert tools_settings_problems(read("compose.yaml")) == []


def test_every_setting_read_under_src_is_in_the_guide():
    sources = source_texts()
    # The scan finds settings known to be read, so an empty scan cannot pass.
    assert {"HR_PORTAL_CHAT_VERSION", "HR_PORTAL_EMBEDDING_BACKEND", "SCREENER_MODELS_ROOT",
            "SCREENER_DATAPACK_SOURCE", "SCREENER_DATAPACK_DEST"} <= set(settings_read(sources))
    assert undocumented_settings(sources, read(GUIDE)) == []


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
ENV ORT_DISABLE_TELEMETRY=1
USER 1000:1000
HEALTHCHECK --interval=30s \\
    CMD ["python", "-c", "print('ok')"]
CMD ["streamlit", "run", "app.py", "--server.headless=true", "--browser.serverAddress=localhost", "--browser.gatherUsageStats=false"]
"""
GOOD_CROSS = 'QWEN ={"model": "Qwen/Qwen3-Reranker-0.6B", "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473"}\n'
GOOD_DOCKERIGNORE = "\n".join(DOCKERIGNORE_LINES) + "\n"
GOOD_GITIGNORE = "data/*\n!data/README.md\nlogs/\n" + "\n".join(GITIGNORE_LINES) + "\n"
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
  qwen:
    <<: *run
    network_mode: "service:app"
    healthcheck:
      test: ["CMD", "python", "-c", "open('http://127.0.0.1:8000/v1/models')"]
      start_period: 120s
  tools:
    <<: *run
    volumes:
      - ./data/datapack:/app/data
      - "${SCREENER_DATA_SOURCE:-./data}:/source:ro"
    environment:
      SCREENER_DATAPACK_SOURCE: /source
      SCREENER_DATAPACK_DEST: /app/data
    healthcheck: {disable: true}
"""
DEV_COMMAND = """\
    command:
      - streamlit
      - run
      - src/candidate_screener/app/streamlit_app.py
      - --server.headless=true
      - --browser.serverAddress=localhost
      - --browser.gatherUsageStats=false
      - --server.runOnSave=true
"""
GOOD_DEV_COMPOSE = "services:\n  app:\n    volumes:\n      - ./src:/app/src:ro\n" + DEV_COMMAND


def broken(text: str, *replacements: tuple[str, str]) -> str:
    """`text` with each (old, new) applied; old must occur exactly once."""
    for old, new in replacements:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text


def test_every_check_passes_on_the_good_texts():
    assert app_stage_problems(GOOD_DOCKERFILE) == []
    assert app_command_problems(GOOD_DOCKERFILE) == []
    assert models_stage_problems(GOOD_DOCKERFILE) == []
    assert frontend_order_problems(GOOD_DOCKERFILE) == []
    assert c2_pin_problems(GOOD_DOCKERFILE, GOOD_CROSS) == []
    assert dockerignore_problems(GOOD_DOCKERIGNORE) == []
    assert gitignore_problems(GOOD_GITIGNORE) == []
    assert gitattributes_problems(GOOD_GITATTRIBUTES) == []
    assert published_port_problems(GOOD_COMPOSE) == []
    assert data_folder_problems(GOOD_COMPOSE) == []
    assert tools_settings_problems(GOOD_COMPOSE) == []
    assert healthcheck_problems(GOOD_COMPOSE) == []
    for other_form in ("    healthcheck:\n      test:\n        - CMD\n        - python\n"
                       "        - http://127.0.0.1:8000/v1/models\n",
                       '    healthcheck:\n      test: "CMD-SHELL python probe.py http://127.0.0.1:8000/v1/models"\n',
                       "    healthcheck: {test: [CMD, python, 'http://127.0.0.1:8000/v1/models'], start_period: 120s}\n"):
        assert healthcheck_problems(broken(GOOD_COMPOSE, (QWEN_HEALTHCHECK, other_form))) == []
    assert dev_command_problems(GOOD_DEV_COMPOSE) == []
    for other_form in ('    environment:\n      - SCREENER_DATAPACK_SOURCE=/source\n'
                       '      - "SCREENER_DATAPACK_DEST=/app/data"\n',
                       "    environment: {SCREENER_DATAPACK_SOURCE: /source, SCREENER_DATAPACK_DEST: /app/data}\n"):
        assert tools_settings_problems(broken(GOOD_COMPOSE, (TOOLS_SETTINGS, other_form))) == []
    shell_form = ("CMD streamlit run app.py --server.headless true --browser.serverAddress localhost "
                  "--browser.gatherUsageStats=false\n")
    assert app_command_problems(broken(GOOD_DOCKERFILE, (APP_CMD, shell_form))) == []
    flags = "--server.headless=true, --browser.serverAddress=localhost, --browser.gatherUsageStats=false"
    for other_form in (f"    command: [streamlit, run, app.py, {flags}]\n",
                       f'    command: "streamlit run app.py {flags.replace(",", "")}"\n'):
        assert dev_command_problems(broken(GOOD_DEV_COMPOSE, (DEV_COMMAND, other_form))) == []


APP_OFFLINE_ENV = "ENV SCREENER_MODELS_ROOT=/opt/models \\\n    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1\n"
APP_CMD = ('CMD ["streamlit", "run", "app.py", "--server.headless=true", "--browser.serverAddress=localhost", '
           '"--browser.gatherUsageStats=false"]\n')
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
    "app-without-ORT_DISABLE_TELEMETRY": (app_stage_problems, broken(GOOD_DOCKERFILE, ("ENV ORT_DISABLE_TELEMETRY=1\n", ""))),
    "app-with-ORT_DISABLE_TELEMETRY=0": (app_stage_problems, broken(
        GOOD_DOCKERFILE, ("ORT_DISABLE_TELEMETRY=1", "ORT_DISABLE_TELEMETRY=0"))),
    # the app stage's Streamlit command
    "cmd-without-headless": (app_command_problems, broken(GOOD_DOCKERFILE, ('"--server.headless=true", ', ""))),
    "cmd-without-serverAddress": (app_command_problems, broken(
        GOOD_DOCKERFILE, ('"--browser.serverAddress=localhost", ', ""))),
    "cmd-without-gatherUsageStats": (app_command_problems, broken(
        GOOD_DOCKERFILE, (', "--browser.gatherUsageStats=false"', ""))),
    "cmd-gathers-usage-stats": (app_command_problems, broken(
        GOOD_DOCKERFILE, ("gatherUsageStats=false", "gatherUsageStats=true"))),
    "cmd-flag-overridden-later": (app_command_problems, broken(
        GOOD_DOCKERFILE, ('gatherUsageStats=false"]', 'gatherUsageStats=false", "--server.headless=false"]'))),
    "cmd-shell-form-without-a-flag": (app_command_problems, broken(
        GOOD_DOCKERFILE, (APP_CMD, "CMD streamlit run app.py --server.headless=true --browser.gatherUsageStats=false\n"))),
    "cmd-space-separated-wrong-value": (app_command_problems, broken(GOOD_DOCKERFILE, (
        APP_CMD, "CMD streamlit run app.py --server.headless true --browser.serverAddress 0.0.0.0 "
                 "--browser.gatherUsageStats false\n"))),
    "app-stage-has-only-the-healthcheck-cmd": (app_command_problems, broken(GOOD_DOCKERFILE, (APP_CMD, ""))),
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
    "dockerignore-lets-the-compose-override-back": (dockerignore_problems, GOOD_DOCKERIGNORE + "!compose.override.yaml\n"),
    **{f"gitignore-lacks-{line}": (gitignore_problems, broken(GOOD_GITIGNORE, (f"\n{line}\n", "\n")))
       for line in GITIGNORE_LINES},
    "gitignore-lets-secrets-back": (gitignore_problems, GOOD_GITIGNORE + "!.streamlit/secrets.toml\n"),
    "gitignore-lets-the-override-back-by-glob": (gitignore_problems, GOOD_GITIGNORE + "!compose.override.*\n"),
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
TOOLS_SETTINGS = "    environment:\n      SCREENER_DATAPACK_SOURCE: /source\n      SCREENER_DATAPACK_DEST: /app/data\n"
APP_SERVICE = "  app:\n    <<: *run\n"
QWEN_HEALTHCHECK = ("    healthcheck:\n      test: [\"CMD\", \"python\", \"-c\", "
                    "\"open('http://127.0.0.1:8000/v1/models')\"]\n      start_period: 120s\n")
QWEN_PROBE_URL = "http://127.0.0.1:8000/v1/models"

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
    "tools-without-the-datapack-settings": (tools_settings_problems, broken(GOOD_COMPOSE, (TOOLS_SETTINGS, ""))),
    "tools-datapack-dest-elsewhere": (tools_settings_problems, broken(
        GOOD_COMPOSE, ("SCREENER_DATAPACK_DEST: /app/data", "SCREENER_DATAPACK_DEST: /app/data/datapack"))),
    "tools-datapack-source-not-mounted": (tools_settings_problems, broken(
        GOOD_COMPOSE, ("SCREENER_DATAPACK_SOURCE: /source", "SCREENER_DATAPACK_SOURCE: /src"))),
    "tools-source-mounted-read-write": (tools_settings_problems, broken(GOOD_COMPOSE, (":/source:ro", ":/source"))),
    "tools-settings-on-another-service": (tools_settings_problems, broken(
        GOOD_COMPOSE, (TOOLS_SETTINGS, ""), (APP_SERVICE, APP_SERVICE + TOOLS_SETTINGS))),
    # the image's healthcheck probes Streamlit
    "qwen-keeps-the-image-healthcheck": (healthcheck_problems, broken(GOOD_COMPOSE, (QWEN_HEALTHCHECK, ""))),
    "qwen-disables-its-healthcheck": (healthcheck_problems, broken(
        GOOD_COMPOSE, (QWEN_HEALTHCHECK, "    healthcheck:\n      disable: true\n"))),
    "qwen-disables-in-flow-form": (healthcheck_problems, broken(
        GOOD_COMPOSE, (QWEN_HEALTHCHECK, "    healthcheck: {disable: true}\n"))),
    "qwen-check-plus-disable": (healthcheck_problems, broken(
        GOOD_COMPOSE, ("      start_period: 120s\n", "      start_period: 120s\n      disable: true\n"))),
    "qwen-test-NONE": (healthcheck_problems, broken(
        GOOD_COMPOSE, (QWEN_HEALTHCHECK, '    healthcheck:\n      test: ["NONE"]\n'))),
    "qwen-probes-streamlit": (healthcheck_problems, broken(
        GOOD_COMPOSE, (QWEN_PROBE_URL, "http://127.0.0.1:8503/_stcore/health"))),
    "qwen-probes-another-port": (healthcheck_problems, broken(
        GOOD_COMPOSE, (QWEN_PROBE_URL, "http://127.0.0.1:8001/v1/models"))),
    "qwen-healthcheck-without-test": (healthcheck_problems, broken(
        GOOD_COMPOSE, (QWEN_HEALTHCHECK, "    healthcheck:\n      start_period: 120s\n"))),
    "qwen-check-on-app-not-qwen": (healthcheck_problems, broken(
        GOOD_COMPOSE, (QWEN_HEALTHCHECK, ""), (APP_SERVICE, APP_SERVICE + QWEN_HEALTHCHECK))),
    "tools-keeps-the-healthcheck": (healthcheck_problems, broken(GOOD_COMPOSE, ("    healthcheck: {disable: true}\n", ""))),
    "tools-disable-false": (healthcheck_problems, broken(
        GOOD_COMPOSE, ("    healthcheck: {disable: true}\n", "    healthcheck:\n      disable: false\n"))),
    # compose.dev.yaml's Streamlit command
    "dev-without-headless": (dev_command_problems, broken(GOOD_DEV_COMPOSE, ("      - --server.headless=true\n", ""))),
    "dev-without-serverAddress": (dev_command_problems, broken(
        GOOD_DEV_COMPOSE, ("      - --browser.serverAddress=localhost\n", ""))),
    "dev-without-gatherUsageStats": (dev_command_problems, broken(
        GOOD_DEV_COMPOSE, ("      - --browser.gatherUsageStats=false\n", ""))),
    "dev-gathers-usage-stats": (dev_command_problems, broken(
        GOOD_DEV_COMPOSE, ("gatherUsageStats=false", "gatherUsageStats=true"))),
    "dev-flag-overridden-later": (dev_command_problems, GOOD_DEV_COMPOSE + "      - --browser.serverAddress=0.0.0.0\n"),
    "dev-flow-command-without-a-flag": (dev_command_problems, broken(GOOD_DEV_COMPOSE, (
        DEV_COMMAND, "    command: [streamlit, run, app.py, --server.headless=true, --browser.gatherUsageStats=false]\n"))),
    "dev-string-command-without-a-flag": (dev_command_problems, broken(GOOD_DEV_COMPOSE, (
        DEV_COMMAND, "    command: streamlit run app.py --server.headless=true --browser.serverAddress=localhost\n"))),
    "dev-command-on-another-service": (dev_command_problems, broken(GOOD_DEV_COMPOSE, (DEV_COMMAND, ""))
                                       + "  tools:\n" + DEV_COMMAND),
}


@pytest.mark.parametrize("case", sorted(BROKEN_COMPOSE))
def test_compose_checks_fail_on_broken_texts(case):
    check, text = BROKEN_COMPOSE[case]
    assert check(text), case


GOOD_SOURCES = {
    "pkg/ui.py": 'import os\n\nVERSION = os.environ.get("HR_PORTAL_CHAT_VERSION", "v1").lower()\n',
    "pkg/folder.py": ('SOURCE_ENV, DEST_ENV = "SCREENER_DATAPACK_SOURCE", "SCREENER_DATAPACK_DEST"\n'
                      'PREFIX = "SCREENER_QWEN_"  # a prefix, not a setting\n'
                      'MARKER = "CANDIDATE_SCREENER_WORKER:"  # a protocol marker, not a setting\n'
                      "# SCREENER_OLD_SETTING, named in a comment, is not read\n"),
}
TABLE_HEAD = "| Name | Docker default | Effect | Safe to change |\n|---|---|---|---|\n"
CHAT_ROW = "| `HR_PORTAL_CHAT_VERSION` | `v1` | The chat version | Yes |\n"
FOLDER_ROW = ("| `SCREENER_DATAPACK_SOURCE`, `SCREENER_DATAPACK_DEST` | `/source`, `/app/data` "
              "| The data folder commands' folders | No |\n")
GOOD_GUIDE = (
    "# Guide\n\n## 7. Accounts\n\n`HR_PORTAL_ACCOUNTS`, named in prose outside the section.\n\n"
    "## 8. Configuration reference\n\n"
    "```yaml\n# a comment in fenced code is not a heading\nservices: {}\n```\n\n"
    + TABLE_HEAD + CHAT_ROW
    + "\n### Models and the data folder\n\n" + TABLE_HEAD + FOLDER_ROW
    + "\n## 9. Backups\n\n| Name | Notes |\n|---|---|\n| `SCREENER_BACKUP_ONLY` | in another section |\n")


def test_settings_check_passes_on_the_good_texts():
    assert settings_read(GOOD_SOURCES) == {"HR_PORTAL_CHAT_VERSION": ["pkg/ui.py"],
                                           "SCREENER_DATAPACK_DEST": ["pkg/folder.py"],
                                           "SCREENER_DATAPACK_SOURCE": ["pkg/folder.py"]}
    assert configuration_names(GOOD_GUIDE) == {"HR_PORTAL_CHAT_VERSION", "SCREENER_DATAPACK_SOURCE",
                                               "SCREENER_DATAPACK_DEST"}
    assert undocumented_settings(GOOD_SOURCES, GOOD_GUIDE) == []


BROKEN_SETTINGS = {
    "guide-lacks-a-row": (GOOD_SOURCES, broken(GOOD_GUIDE, (CHAT_ROW, ""))),
    "guide-lacks-one-name-of-a-row": (GOOD_SOURCES, broken(GOOD_GUIDE, (", `SCREENER_DATAPACK_DEST`", ""))),
    "name-misspelt": (GOOD_SOURCES, broken(
        GOOD_GUIDE, ("`HR_PORTAL_CHAT_VERSION`", "`HR_PORTAL_CHAT_VERSIONS`"))),
    "name-only-in-prose": (GOOD_SOURCES, broken(
        GOOD_GUIDE, (CHAT_ROW, "Set `HR_PORTAL_CHAT_VERSION` in `.env`.\n"))),
    "name-only-in-a-later-column": (GOOD_SOURCES, broken(
        GOOD_GUIDE, (CHAT_ROW, "| The chat version | `HR_PORTAL_CHAT_VERSION` | `v1` | Yes |\n"))),
    "row-inside-fenced-code": (GOOD_SOURCES, broken(
        GOOD_GUIDE, (CHAT_ROW, ""), ("services: {}\n", "services: {}\n" + CHAT_ROW))),
    "row-in-another-section": (GOOD_SOURCES, broken(
        GOOD_GUIDE, (CHAT_ROW, ""), ("| `SCREENER_BACKUP_ONLY` |", "| `HR_PORTAL_CHAT_VERSION` |"))),
    "section-ended-by-a-heading": (GOOD_SOURCES, broken(
        GOOD_GUIDE, ("```\n\n| Name", "```\n\n## 8a. Other settings\n\n| Name"))),
    "no-configuration-section": (GOOD_SOURCES, broken(
        GOOD_GUIDE, ("## 8. Configuration reference", "## 8. Settings"))),
    "source-reads-a-new-setting": (
        {**GOOD_SOURCES, "pkg/new.py": 'import os\n\nPATH = os.environ["SCREENER_NEW_PATH"]\n'}, GOOD_GUIDE),
    "source-reads-through-getenv": (
        {**GOOD_SOURCES, "pkg/new.py": 'import os\n\nFLAG = os.getenv("HR_PORTAL_NEW_FLAG", "false")\n'}, GOOD_GUIDE),
}


@pytest.mark.parametrize("case", sorted(BROKEN_SETTINGS))
def test_settings_check_fails_on_broken_texts(case):
    sources, guide = BROKEN_SETTINGS[case]
    assert undocumented_settings(sources, guide), case


# --------------------------------------------- the Qwen3.5-4B image and its compose service
# Plan 2026-10-28-gguf-cross-platform, 02 §3 (owner: a separate image and snapshot-pinned Ubuntu
# packages); since plan 2026-10-31-v3-default §3 the service is part of compose.yaml, with no profile.

from candidate_screener.llm import gguf_assets  # noqa: E402  (standard library and config only)

#: TARGETARCH -> gguf_assets's runtime key: exactly the two architectures Docker builds for.
LLM4B_ARCHES = {"amd64": "linux-x64", "arm64": "linux-arm64"}
LLM4B_DESTINATION = "/opt/models/chatbot/qwen35"
LLM4B_MANIFEST_COPY = "/opt/qwen35-manifest/manifest.json"
LLM4B_PROBE = "127.0.0.1:8001/v1/models"
#: The volume's name carries the pins, so a pin change gets a fresh manifest copy.
LLM4B_VOLUME = f"qwen35-manifest-{gguf_assets.REVISION[:7]}-{gguf_assets.RELEASE}"
DIGEST_PINNED = re.compile(r"^[\w./-]+:[\w.-]+@sha256:[0-9a-f]{64}$")


def stage_bases(dockerfile: str) -> dict[str, str]:
    """Each named stage's base image, ARG defaults substituted."""
    args, result = build_args(dockerfile), {}
    for keyword, arguments in instructions(dockerfile):
        if keyword == "FROM":
            words = [word for word in arguments.split() if not word.startswith("--")]
            if len(words) == 3 and words[1].upper() == "AS":
                result[words[2].lower()] = re.sub(r"\$\{?(\w+)\}?", lambda m: args.get(m.group(1), m.group(0)),
                                                  words[0])
    return result


def _words(arguments: str) -> list[str]:
    return json.loads(arguments) if arguments.startswith("[") else shlex.split(arguments)


def gguf_pin_literals() -> set[str]:
    """gguf_assets's pins; the Dockerfile calls the module instead of repeating them."""
    pins = {gguf_assets.REVISION, gguf_assets.SHA256, gguf_assets.FILENAME}
    for runtime in gguf_assets.RUNTIMES.values():
        pins |= {runtime.asset, runtime.sha256}
    return pins


def llm4b_stage_problems(dockerfile: str) -> list[str]:
    """The llm4b stage: a digest-pinned base, snapshot-pinned packages, TARGETARCH mapped onto
    exactly amd64 and arm64, downloads only by gguf_assets in a RUN, no copy of its pins, and a
    non-root, offline run time with the facade's own healthcheck and command."""
    stage = stages(dockerfile).get("llm4b")
    if stage is None:
        return ["no stage named llm4b"]
    problems = []
    base = stage_bases(dockerfile).get("llm4b", "")
    if not DIGEST_PINNED.match(base):
        problems.append(f"the llm4b base {base!r} is not pinned by digest")
    elif not base.startswith("ubuntu:24.04@"):
        problems.append(f"the llm4b base {base!r} is not Ubuntu 24.04 (the arm64 runtime needs glibc 2.38)")
    runs = [arguments for keyword, arguments in stage if keyword == "RUN"]
    text = "\n".join(runs)
    # Ubuntu packages: a dated snapshot, every package pinned by version.
    snapshot = build_args(dockerfile).get("UBUNTU_SNAPSHOT", "")
    if not re.fullmatch(r"\d{8}T\d{6}Z", snapshot) or "APT::Snapshot=$UBUNTU_SNAPSHOT" not in text:
        problems.append("apt does not read a dated snapshot (ARG UBUNTU_SNAPSHOT, APT::Snapshot)")
    # snapshot.ubuntu.com serves no ubuntu-ports (arm64), so the snapshot applies on amd64 only;
    # arm64 reads the live ports archive with the same version pins (owner, 28 Oct 2026, E5).
    if not re.search(r'if \[ "\$TARGETARCH" = amd64 \]; then[^;]*APT::Snapshot=\$UBUNTU_SNAPSHOT', text):
        problems.append("the apt snapshot is not limited to amd64 (snapshot.ubuntu.com has no ubuntu-ports)")
    for match in re.finditer(r"install --yes --no-install-recommends ([^;]*)", text):
        packages = [word for word in match.group(1).split() if not word.startswith("-")]
        problems += [f"the Ubuntu package {name} is not pinned by version" for name in packages if "=" not in name]
    if "install --yes" not in text:
        problems.append("the llm4b stage installs no Ubuntu packages (python3, ca-certificates, libgomp1)")
    # TARGETARCH -> runtime key.
    if not any(keyword == "ARG" and arguments.split("=")[0] == "TARGETARCH" for keyword, arguments in stage):
        problems.append("the llm4b stage does not declare ARG TARGETARCH")
    case = re.search(r'case\s+"?\$\{?TARGETARCH\}?"?\s+in\s+(.*?)\s+esac', text, re.DOTALL)
    arms = dict(re.findall(r"([\w*|-]+)\)\s*key=([\w-]+)\s*;;", case.group(1))) if case else {}
    if arms != LLM4B_ARCHES:
        problems.append(f"TARGETARCH maps {arms} onto runtime keys, not exactly {LLM4B_ARCHES}")
    problems += [f"{key} is not a gguf_assets runtime key" for key in arms.values() if key not in gguf_assets.RUNTIMES]
    if not case or not re.search(r"\*\)[^;]*;\s*exit 1\s*;;", case.group(1)):
        problems.append("an unknown TARGETARCH does not stop the build")
    # Downloads: only gguf_assets.py, only in a RUN, for the mapped key.
    for keyword, arguments in stage:
        if keyword == "ADD":
            problems.append(f"the llm4b stage uses ADD {arguments}")
        elif keyword in {"CMD", "ENTRYPOINT", "HEALTHCHECK"} and ("gguf_assets" in arguments
                                                                  or re.search(r"https?://(?!127\.0\.0\.1)", arguments)):
            problems.append(f"the llm4b {keyword} can download at run time: {arguments}")
    if re.search(r"\b(curl|wget)\b", text):
        problems.append("the llm4b stage downloads with curl or wget, not gguf_assets")
    if not re.search(r'(gguf_assets\.py|\$assets|\$\{assets\})"?\s+--platform\s+"?\$key"?\s+--destination\s+"?\$dest"?', text) \
            or "gguf_assets.py" not in text \
            or f"dest={LLM4B_DESTINATION};" not in text:
        problems.append(f"gguf_assets.py is not run with --platform $key --destination {LLM4B_DESTINATION}")
    if not re.search(rf"cp\s+\S*manifest\.json\S*\s+{re.escape(LLM4B_MANIFEST_COPY)}", text):
        problems.append(f"the stage keeps no manifest copy at {LLM4B_MANIFEST_COPY}")
    problems += [f"the Dockerfile repeats gguf_assets's pin {pin}" for pin in sorted(gguf_pin_literals())
                 if pin in dockerfile]
    # Run time.
    env = stage_env(stage)
    if env.get("HF_HUB_OFFLINE") != "1":
        problems.append("the llm4b stage does not set HF_HUB_OFFLINE=1")
    users = [arguments for keyword, arguments in stage if keyword == "USER"]
    if not users or users[-1].split(":")[0].strip() in {"root", "0"}:
        problems.append("the llm4b stage runs as root")
    checks = [arguments for keyword, arguments in stage if keyword == "HEALTHCHECK"]
    if not checks:
        problems.append(f"the llm4b stage has no healthcheck (ask {LLM4B_PROBE})")
    else:
        check = checks[-1]
        if LLM4B_PROBE not in check or "ProxyHandler({})" not in check:
            problems.append(f"the llm4b healthcheck does not ask {LLM4B_PROBE} without a proxy")
        period = re.search(r"--start-period=(\d+)s", check)
        if not period or int(period.group(1)) < 300:
            problems.append("the llm4b healthcheck's start period is under 300s (the model's hash and load)")
    commands = [_words(arguments) for keyword, arguments in stage if keyword == "CMD"]
    command = commands[-1] if commands else []
    if (command[1:3] != ["-m", "candidate_screener.llm.gguf_server"] or "--launch-backend" not in command
            or command_flags(command).get("--manifest") != f"{LLM4B_DESTINATION}/manifest.json"):
        problems.append("the llm4b CMD is not the facade with --launch-backend on the image's manifest")
    return problems


def lock_closure(lock_text: str, root: str) -> dict[str, dict]:
    """`root` and everything it depends on in a uv.lock, by name."""
    import tomllib
    packages = {package["name"]: package for package in tomllib.loads(lock_text)["package"]}
    result, pending = {}, [root]
    while pending:
        name = pending.pop()
        if name not in result and name in packages:
            result[name] = packages[name]
            pending += [dependency["name"] for dependency in packages[name].get("dependencies", [])]
    return result


def facade_dependency_problems(dockerfile: str, lock_text: str) -> list[str]:
    """The llm4b stage installs exactly requests and its dependencies, at uv.lock's versions and
    with hashes uv.lock holds for them, by uv (no pip)."""
    stage = stages(dockerfile).get("llm4b")
    if stage is None:
        return ["no stage named llm4b"]
    text = "\n".join(arguments for keyword, arguments in stage if keyword == "RUN")
    pinned = {name.lower(): (version, hashes) for name, version, hashes in
              re.findall(r"'([A-Za-z0-9_.-]+)==([^\s']+)((?:\s+--hash=sha256:[0-9a-f]{64})*)'", text)}
    expected = lock_closure(lock_text, "requests")
    problems = []
    if not expected:
        return ["uv.lock has no requests"]
    for name in sorted(set(expected) | set(pinned)):
        if name not in pinned:
            problems.append(f"the llm4b stage does not install {name} (a dependency of requests in uv.lock)")
            continue
        if name not in expected:
            problems.append(f"the llm4b stage installs {name}, which requests does not need")
            continue
        version, hashes = pinned[name]
        if version != expected[name]["version"]:
            problems.append(f"the llm4b stage pins {name}=={version}; uv.lock has {expected[name]['version']}")
        known = {entry["hash"] for entry in expected[name].get("wheels", []) + [expected[name].get("sdist", {})]
                 if entry.get("hash")}
        found = set(re.findall(r"sha256:[0-9a-f]{64}", hashes))
        if not found:
            problems.append(f"the llm4b stage installs {name} without a hash")
        problems += [f"{name}'s hash {digest} is not uv.lock's" for digest in sorted(found - known)]
    if "--require-hashes" not in text:
        problems.append("the llm4b stage does not install with --require-hashes")
    if re.search(r"(?<!uv )\bpip3?\s+install", text):
        problems.append("the llm4b stage installs with pip, not uv")
    return problems


def _named_volume(compose: str, key: str) -> str | None:
    """The `name:` of a top-level named volume."""
    lines = compose.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^volumes:\s*(#.*)?$", line):
            block = _block(lines, index)
            for position, entry in enumerate(block):
                if re.match(rf"^\s+{re.escape(key)}:\s*(#.*)?$", entry):
                    for nested in _block(block, position):
                        match = re.match(r"^\s*name:\s*(\S+)", nested)
                        if match:
                            return _unquote(_strip_comment(match.group(1)))
    return None


def guide_chat_defaults(guide: str) -> list[str]:
    """The Docker-default cell of every `HR_PORTAL_CHAT_VERSION` row in the guide's configuration
    reference section (same section rules as configuration_names)."""
    level, defaults, fenced = None, [], False
    for line in guide.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced:
            continue
        heading = re.match(r"^ {0,3}(#{1,6})\s+(.*)$", line)
        if heading:
            depth = len(heading.group(1))
            if level is not None and depth <= level:
                break
            if level is None and CONFIGURATION_SECTION in heading.group(2).lower():
                level = depth
            continue
        if level is not None and stripped.startswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if "HR_PORTAL_CHAT_VERSION" in re.findall(r"`([^`]+)`", cells[0]) and len(cells) > 1:
                defaults.append(cells[1])
    return defaults


#: compose.yaml's chat-version default (plan 2026-10-31-v3-default §3).
CHAT_DEFAULT = "${HR_PORTAL_CHAT_VERSION:-v3}"


def llm4b_compose_problems(compose: str) -> list[str]:
    """compose.yaml (plan 2026-10-31-v3-default §3): llm4b with no profile, so plain `up -d` starts
    it; in app's network namespace, no port, its image's own healthcheck, no data folder. app gets
    SCREENER_QWEN_MANIFEST and the manifest volume read-only, named after the pins, defaults to chat
    v3 and is still built from the app target; the 0.5B qwen stays in the `llm` profile (v2)."""
    problems = []
    block = _service_block(compose, "llm4b")
    if block is None:
        return ["compose.yaml has no llm4b service"]
    if re.search(r"^\s*profiles:", block, re.MULTILINE):
        problems.append("llm4b has a profile, so plain `docker compose up -d` does not start it")
    if not re.search(r"^\s*network_mode:\s*[\"']?service:app[\"']?\s*(#.*)?$", block, re.MULTILINE):
        problems.append("llm4b does not use network_mode: service:app")
    if sequences(block, "ports") or re.search(r"^\s*ports:", block, re.MULTILINE):
        problems.append("llm4b publishes a port")
    if not re.search(r"^\s*target:\s*llm4b\s*$", block, re.MULTILINE):
        problems.append("llm4b is not built from the Dockerfile's llm4b target")
    if healthcheck_disabled(compose, "llm4b"):
        problems.append("llm4b disables its image's healthcheck")
    test = healthcheck_test(compose, "llm4b")
    if test and LLM4B_PROBE not in " ".join(test):
        problems.append(f"llm4b's own healthcheck does not ask {LLM4B_PROBE}")
    if not re.search(r"^\s*init:\s*true\s*$", block, re.MULTILINE):
        problems.append("llm4b does not run under init")
    volumes = [split_mount(item) for item in service_volumes(compose, "llm4b")]
    if any(host.startswith((".", "/", "$")) for host, _, _ in volumes):
        problems.append("llm4b mounts a host folder (it needs no data)")
    if ("qwen35-manifest", "/opt/qwen35-manifest") not in [(host, container) for host, container, _ in volumes]:
        problems.append("llm4b does not mount the manifest volume at /opt/qwen35-manifest")
    app_env = service_environment(compose, "app")
    if app_env.get("SCREENER_QWEN_MANIFEST") != LLM4B_MANIFEST_COPY:
        problems.append(f"app does not set SCREENER_QWEN_MANIFEST={LLM4B_MANIFEST_COPY}")
    if app_env.get("HR_PORTAL_CHAT_VERSION") != CHAT_DEFAULT:
        problems.append(f"app's HR_PORTAL_CHAT_VERSION is {app_env.get('HR_PORTAL_CHAT_VERSION')!r}, "
                        f"not {CHAT_DEFAULT}")
    app_volumes = [split_mount(item) for item in service_volumes(compose, "app")]
    if ("qwen35-manifest", "/opt/qwen35-manifest", "ro") not in app_volumes:
        problems.append("app does not mount the manifest volume read-only at /opt/qwen35-manifest")
    if _named_volume(compose, "qwen35-manifest") != LLM4B_VOLUME:
        problems.append(f"the manifest volume is not named {LLM4B_VOLUME} (the pins)")
    qwen = _service_block(compose, "qwen")
    if qwen is None or sequences(qwen, "profiles") != [["llm"]]:
        problems.append("the 0.5B qwen service is not in the `llm` profile (v2 only) alone")
    if not re.search(r"^\s*target:\s*app\s*$", _service_block(compose, "app") or "", re.MULTILINE):
        problems.append("compose.yaml's app is no longer built from the app target")
    return problems


def test_llm4b_stage_follows_the_design():
    assert llm4b_stage_problems(read("Dockerfile")) == []
    # The stage is found and is not the default.
    assert "llm4b" in stages(read("Dockerfile")) and list(stages(read("Dockerfile")))[-1] == "app"


def test_llm4b_facade_dependencies_equal_the_lock():
    assert facade_dependency_problems(read("Dockerfile"), read("uv.lock")) == []
    assert set(lock_closure(read("uv.lock"), "requests")) == {"requests", "certifi", "charset-normalizer",
                                                              "idna", "urllib3"}


def test_llm4b_is_part_of_compose_yaml_and_v3_is_the_default():
    compose = read("compose.yaml")
    assert llm4b_compose_problems(compose) == []
    assert published_port_problems(compose) == []
    # One way to run (plan 2026-10-31-v3-default, Q4): the override file and its COMPOSE_FILE line are gone.
    assert not (ROOT / "compose.llm4b.yaml").exists()
    assert not re.search(r"^\s*#?\s*COMPOSE_FILE=", read(".env.example"), re.MULTILINE)
    # .env.example shows compose.yaml's default.
    assert re.findall(r"^HR_PORTAL_CHAT_VERSION=(\S+)$", read(".env.example"), re.MULTILINE) == ["v3"]
    defaults = guide_chat_defaults(read(GUIDE))
    assert defaults and set(defaults) == {"`v3`"}, defaults


GOOD_LLM4B = """\
ARG UBUNTU_IMAGE=ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c
FROM python:3.12-slim AS base
FROM ${UBUNTU_IMAGE} AS llm4b
ARG TARGETARCH
ARG UBUNTU_SNAPSHOT=20261027T000000Z
RUN apt="apt-get"; if [ "$TARGETARCH" = amd64 ]; then apt="$apt -o APT::Snapshot=$UBUNTU_SNAPSHOT"; fi; $apt update; \\
    $apt install --yes --no-install-recommends python3=3.12.3-0ubuntu2.1 libgomp1=14.2.0-4ubuntu2~24.04.1; \\
    rm -rf /var/lib/apt/lists/*
RUN printf '%s\\n' \\
        'requests==2.34.2 --hash=sha256:2a0d60c172f83ac6ab31e4554906c0f3b3588d37b5cb939b1c061f4907e278e0' \\
        'idna==3.19 --hash=sha256:815e7be7a7806d54abb586dc943addc79e8b2ee16915059658cbeff4b1b43bf4' \\
        > /tmp/r.txt; \\
    uv pip install --python /opt/gguf-venv/bin/python --require-hashes --no-deps -r /tmp/r.txt
RUN case "$TARGETARCH" in \\
        amd64) key=linux-x64 ;; \\
        arm64) key=linux-arm64 ;; \\
        *) echo "no runtime" >&2; exit 1 ;; \\
    esac; \\
    dest=/opt/models/chatbot/qwen35; \\
    python3 /app/src/candidate_screener/llm/gguf_assets.py --platform "$key" --destination "$dest"; \\
    cp "$dest/manifest.json" /opt/qwen35-manifest/manifest.json
ENV HF_HUB_OFFLINE=1 HOME=/tmp
USER 1000:1000
HEALTHCHECK --interval=30s --start-period=300s \\
    CMD ["python", "-c", "import urllib.request as r; r.build_opener(r.ProxyHandler({})).open('http://127.0.0.1:8001/v1/models')"]
CMD ["python", "-m", "candidate_screener.llm.gguf_server", "--manifest", "/opt/models/chatbot/qwen35/manifest.json", "--launch-backend"]
FROM base AS app
"""
GOOD_LOCK = """\
[[package]]
name = "requests"
version = "2.34.2"
dependencies = [{ name = "idna" }]
sdist = { url = "x", hash = "sha256:f288924cae4e29463698d6d60bc6a4da69c89185ad1e0bcc4104f584e960b9ed" }
wheels = [{ url = "y", hash = "sha256:2a0d60c172f83ac6ab31e4554906c0f3b3588d37b5cb939b1c061f4907e278e0" }]

[[package]]
name = "idna"
version = "3.19"
wheels = [{ url = "z", hash = "sha256:815e7be7a7806d54abb586dc943addc79e8b2ee16915059658cbeff4b1b43bf4" }]

[[package]]
name = "torch"
version = "2.2.2"
"""
GOOD_LLM4B_COMPOSE = f"""\
x-run: &run
  image: example
  volumes:
    - ./data/datapack:/app/data
services:
  app:
    <<: *run
    build:
      context: .
      target: app
    ports:
      - "127.0.0.1:8503:8503"
    environment:
      HR_PORTAL_CHAT_VERSION: "{CHAT_DEFAULT}"
      SCREENER_QWEN_MANIFEST: /opt/qwen35-manifest/manifest.json
    volumes:
      - ./data/datapack:/app/data
      - qwen35-manifest:/opt/qwen35-manifest:ro
  qwen:
    <<: *run
    profiles: [llm]
    network_mode: "service:app"
  llm4b:
    image: candidate-screener/hiring-desk-llm4b:local
    build:
      context: .
      target: llm4b
    network_mode: "service:app"
    init: true
    volumes:
      - qwen35-manifest:/opt/qwen35-manifest
volumes:
  qwen35-manifest:
    name: {LLM4B_VOLUME}
"""
CHAT_ROW_V3 = "| `HR_PORTAL_CHAT_VERSION` | `v3` | The chat version | Yes |\n"
ARM_ARM = "        arm64) key=linux-arm64 ;; \\\n"
LLM4B_HEALTH = "HEALTHCHECK --interval=30s --start-period=300s \\\n"


def test_llm4b_checks_pass_on_the_good_texts():
    assert llm4b_stage_problems(GOOD_LLM4B) == []
    assert facade_dependency_problems(GOOD_LLM4B, GOOD_LOCK) == []
    assert llm4b_compose_problems(GOOD_LLM4B_COMPOSE) == []
    assert data_folder_problems(GOOD_LLM4B_COMPOSE, services=("app",)) == []
    assert guide_chat_defaults(broken(GOOD_GUIDE, (CHAT_ROW, CHAT_ROW_V3))) == ["`v3`"]
    assert app_stage_problems(GOOD_LLM4B + "ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1 "
                              "ORT_DISABLE_TELEMETRY=1 SCREENER_MODELS_ROOT=/opt\nUSER 1000\n") == []


BROKEN_LLM4B = {
    # base and packages
    "base-by-tag-only": (llm4b_stage_problems, broken(
        GOOD_LLM4B, ("@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c", ""))),
    "base-bookworm": (llm4b_stage_problems, broken(GOOD_LLM4B, ("ubuntu:24.04@", "debian:bookworm@"))),
    "package-unpinned": (llm4b_stage_problems, broken(GOOD_LLM4B, ("libgomp1=14.2.0-4ubuntu2~24.04.1", "libgomp1"))),
    "no-snapshot": (llm4b_stage_problems, broken(GOOD_LLM4B, (" -o APT::Snapshot=$UBUNTU_SNAPSHOT", ""))),
    "snapshot-on-arm64": (llm4b_stage_problems, broken(GOOD_LLM4B, ('if [ "$TARGETARCH" = amd64 ]; then apt="$apt -o APT::Snapshot=$UBUNTU_SNAPSHOT"; fi;', 'apt="$apt -o APT::Snapshot=$UBUNTU_SNAPSHOT";'))),
    "snapshot-undated": (llm4b_stage_problems, broken(GOOD_LLM4B, ("SNAPSHOT=20261027T000000Z", "SNAPSHOT=latest"))),
    # TARGETARCH
    "arch-arm64-missing": (llm4b_stage_problems, broken(GOOD_LLM4B, (ARM_ARM, ""))),
    "arch-arm64-to-macos": (llm4b_stage_problems, broken(GOOD_LLM4B, ("key=linux-arm64", "key=macos-arm64"))),
    "arch-extra-riscv": (llm4b_stage_problems, broken(GOOD_LLM4B, (ARM_ARM, ARM_ARM + "        riscv64) key=linux-x64 ;; \\\n"))),
    "arch-unknown-continues": (llm4b_stage_problems, broken(GOOD_LLM4B, (' >&2; exit 1 ;;', ' >&2 ;;'))),
    "arch-not-declared": (llm4b_stage_problems, broken(GOOD_LLM4B, ("ARG TARGETARCH\n", ""))),
    # downloads
    "download-with-ADD": (llm4b_stage_problems, broken(
        GOOD_LLM4B, ("ENV HF_HUB_OFFLINE", "ADD https://huggingface.co/x.gguf /opt/x.gguf\nENV HF_HUB_OFFLINE"))),
    "download-with-curl": (llm4b_stage_problems, broken(GOOD_LLM4B, ("    dest=", "    curl -LO https://example.org/x; dest="))),
    "download-at-run-time": (llm4b_stage_problems, broken(
        GOOD_LLM4B, ('"--launch-backend"]', '"--launch-backend"]\nCMD ["python3", "gguf_assets.py"]'))),
    "assets-without-platform": (llm4b_stage_problems, broken(GOOD_LLM4B, ('--platform "$key" ', ""))),
    "pin-repeated": (llm4b_stage_problems, broken(
        GOOD_LLM4B, ("ARG TARGETARCH\n", f"ARG TARGETARCH\nARG MODEL_REVISION={gguf_assets.REVISION}\n"))),
    "no-manifest-copy": (llm4b_stage_problems, broken(
        GOOD_LLM4B, (' \\\n    cp "$dest/manifest.json" /opt/qwen35-manifest/manifest.json', ""))),
    # run time
    "root": (llm4b_stage_problems, broken(GOOD_LLM4B, ("USER 1000:1000", "USER 0"))),
    "online": (llm4b_stage_problems, broken(GOOD_LLM4B, ("HF_HUB_OFFLINE=1", "HF_HUB_OFFLINE=0"))),
    "healthcheck-streamlit": (llm4b_stage_problems, broken(GOOD_LLM4B, ("127.0.0.1:8001/v1/models", "127.0.0.1:8503/_stcore/health"))),
    "healthcheck-through-proxy": (llm4b_stage_problems, broken(GOOD_LLM4B, ("r.build_opener(r.ProxyHandler({})).open", "r.urlopen"))),
    "healthcheck-short-start": (llm4b_stage_problems, broken(GOOD_LLM4B, ("--start-period=300s", "--start-period=60s"))),
    "healthcheck-missing": (llm4b_stage_problems, broken(GOOD_LLM4B, (LLM4B_HEALTH, "RUN true \\\n"))),
    "cmd-without-backend": (llm4b_stage_problems, broken(GOOD_LLM4B, (', "--launch-backend"]', "]"))),
    "no-llm4b-stage": (llm4b_stage_problems, broken(GOOD_LLM4B, ("AS llm4b", "AS other"))),
    # facade dependencies
    "dep-version-differs": (lambda text: facade_dependency_problems(text, GOOD_LOCK), broken(GOOD_LLM4B, ("idna==3.19", "idna==3.18"))),
    "dep-missing": (lambda text: facade_dependency_problems(text, GOOD_LOCK), broken(
        GOOD_LLM4B, ("        'idna==3.19 --hash=sha256:815e7be7a7806d54abb586dc943addc79e8b2ee16915059658cbeff4b1b43bf4' \\\n", ""))),
    "dep-extra": (lambda text: facade_dependency_problems(text, GOOD_LOCK), broken(
        GOOD_LLM4B, ("        > /tmp/r.txt", "        'torch==2.2.2 --hash=sha256:" + "0" * 64 + "' \\\n        > /tmp/r.txt"))),
    "dep-hash-unknown": (lambda text: facade_dependency_problems(text, GOOD_LOCK), broken(GOOD_LLM4B, ("815e7be7", "915e7be7"))),
    "dep-without-hash": (lambda text: facade_dependency_problems(text, GOOD_LOCK), broken(
        GOOD_LLM4B, (" --hash=sha256:815e7be7a7806d54abb586dc943addc79e8b2ee16915059658cbeff4b1b43bf4", ""))),
    "deps-without-require-hashes": (lambda text: facade_dependency_problems(text, GOOD_LOCK), broken(GOOD_LLM4B, (" --require-hashes", ""))),
    "deps-by-pip": (lambda text: facade_dependency_problems(text, GOOD_LOCK), broken(GOOD_LLM4B, ("uv pip install --python /opt/gguf-venv/bin/python", "pip install"))),
    "lock-moved-on": (lambda text: facade_dependency_problems(GOOD_LLM4B, text), broken(GOOD_LOCK, ('version = "3.19"', 'version = "3.20"'))),
}
BROKEN_LLM4B_COMPOSE = {
    "not-in-app-namespace": broken(GOOD_LLM4B_COMPOSE, ('    network_mode: "service:app"\n    init', "    init")),
    "own-network": broken(GOOD_LLM4B_COMPOSE, ('"service:app"\n    init', "bridge\n    init")),
    "publishes-8001": broken(GOOD_LLM4B_COMPOSE, ("    init: true\n", '    init: true\n    ports:\n      - "127.0.0.1:8001:8001"\n')),
    "healthcheck-disabled": broken(GOOD_LLM4B_COMPOSE, ("    init: true\n", "    init: true\n    healthcheck: {disable: true}\n")),
    "healthcheck-other-port": broken(GOOD_LLM4B_COMPOSE, ("    init: true\n", "    init: true\n    healthcheck:\n      test: [CMD, python, 'http://127.0.0.1:8000/v1/models']\n")),
    "without-init": broken(GOOD_LLM4B_COMPOSE, ("    init: true\n", "")),
    "default-target": broken(GOOD_LLM4B_COMPOSE, ("target: llm4b", "target: app")),
    "mounts-the-data-folder": broken(GOOD_LLM4B_COMPOSE, ("      - qwen35-manifest:/opt/qwen35-manifest\nvolumes",
                                                          "      - qwen35-manifest:/opt/qwen35-manifest\n      - ./data/datapack:/app/data\nvolumes")),
    "llm4b-without-the-manifest-volume": broken(GOOD_LLM4B_COMPOSE, ("    volumes:\n      - qwen35-manifest:/opt/qwen35-manifest\nvolumes", "volumes")),
    "app-without-the-manifest-setting": broken(GOOD_LLM4B_COMPOSE, ("      SCREENER_QWEN_MANIFEST: /opt/qwen35-manifest/manifest.json\n", "")),
    "app-manifest-elsewhere": broken(GOOD_LLM4B_COMPOSE, ("SCREENER_QWEN_MANIFEST: /opt/qwen35-manifest/manifest.json", "SCREENER_QWEN_MANIFEST: /opt/models/chatbot/qwen35/manifest.json")),
    "app-without-the-manifest-volume": broken(GOOD_LLM4B_COMPOSE, ("      - qwen35-manifest:/opt/qwen35-manifest:ro\n", "")),
    "app-volume-read-write": broken(GOOD_LLM4B_COMPOSE, (":/opt/qwen35-manifest:ro", ":/opt/qwen35-manifest")),
    "app-default-v1": broken(GOOD_LLM4B_COMPOSE, (CHAT_DEFAULT, "${HR_PORTAL_CHAT_VERSION:-v1}")),
    "app-chat-version-unset": broken(GOOD_LLM4B_COMPOSE, (f'      HR_PORTAL_CHAT_VERSION: "{CHAT_DEFAULT}"\n', "")),
    "volume-name-without-pins": broken(GOOD_LLM4B_COMPOSE, (f"name: {LLM4B_VOLUME}", "name: qwen35-manifest")),
    "no-top-level-volume": broken(GOOD_LLM4B_COMPOSE, (f"volumes:\n  qwen35-manifest:\n    name: {LLM4B_VOLUME}\n", "")),
    "no-llm4b-service": broken(GOOD_LLM4B_COMPOSE, ("  llm4b:\n", "  other:\n")),
    "llm4b-with-a-profile": broken(GOOD_LLM4B_COMPOSE, ("    init: true\n", "    init: true\n    profiles: [llm]\n")),
    "llm4b-with-a-block-profile": broken(GOOD_LLM4B_COMPOSE, ("    init: true\n", "    init: true\n    profiles:\n      - llm4b\n")),
    "qwen-without-its-profile": broken(GOOD_LLM4B_COMPOSE, ("    profiles: [llm]\n", "")),
    "qwen-in-another-profile": broken(GOOD_LLM4B_COMPOSE, ("profiles: [llm]", "profiles: [v2]")),
    "app-default-target-changed": broken(GOOD_LLM4B_COMPOSE, ("      target: app\n", "      target: llm4b\n")),
}
BROKEN_GUIDE_CHAT_DEFAULT = {
    "guide-default-v1": GOOD_GUIDE,
    "guide-default-v3-in-one-row-only": broken(GOOD_GUIDE, (CHAT_ROW, CHAT_ROW_V3 + CHAT_ROW)),
    "guide-row-missing": broken(GOOD_GUIDE, (CHAT_ROW, "")),
}


@pytest.mark.parametrize("case", sorted(BROKEN_LLM4B))
def test_llm4b_dockerfile_checks_fail_on_broken_texts(case):
    check, text = BROKEN_LLM4B[case]
    assert check(text), case


@pytest.mark.parametrize("case", sorted(BROKEN_LLM4B_COMPOSE))
def test_llm4b_compose_checks_fail_on_broken_texts(case):
    assert llm4b_compose_problems(BROKEN_LLM4B_COMPOSE[case]), case


@pytest.mark.parametrize("case", sorted(BROKEN_GUIDE_CHAT_DEFAULT))
def test_guide_chat_default_check_fails_on_broken_texts(case):
    defaults = guide_chat_defaults(BROKEN_GUIDE_CHAT_DEFAULT[case])
    assert not (defaults and set(defaults) == {"`v3`"}), case
