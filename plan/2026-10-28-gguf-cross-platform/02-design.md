# 02 — Design

Built on the measurements in `01`. Choices marked **(Q n)** wait for the owner (README, open
questions).

## 1. A pinned runtime table in `llm/gguf_assets.py`

```python
RELEASE = "b11896"   # commit 9125a2ebe8e5a38f0f4232170c134a29ed730b66
RUNTIMES = {         # key: (asset, size, sha256), measured 28 Oct 2026 (01 §1)
    "macos-x64":   ("llama-b11896-bin-macos-x64.tar.gz",    12_038_557, "2ff5c5f3…6fb5"),  # unchanged
    "macos-arm64": ("llama-b11896-bin-macos-arm64.tar.gz",  16_297_217, "f7fdf253…e11b"),  # only if Q1 = yes
    "linux-x64":   ("llama-b11896-bin-ubuntu-x64.tar.gz",   22_497_065, "b02ed815…e25c"),
    "linux-arm64": ("llama-b11896-bin-ubuntu-arm64.tar.gz", 18_221_582, "68942cd6…4261"),
}
```

The code carries the full 64-character digests from `01` §1; they are shortened here only for
reading.

- **macos-x64 stays exactly as it is.** The URL, SHA-256 and size are unchanged, so are its
  on-disk names (the archive `llama-b11896-macos-x64.tar.gz`, the folder `runtime/`). The teammate's
  existing download and extracted runtime are then reused and verified, not fetched again.
  `RUNTIME_URL` and `RUNTIME_SHA256` stay as names for the macos-x64 entry, because
  `tests/test_gguf_assets.py` and the trial records refer to them.
- **Other keys** use `llama-b11896-<key>.tar.gz` and `runtime-<key>/`, so one folder can hold
  more than one platform (for example, a Linux host preparing a copy for a Mac) without
  overwriting anything.
- **Selection.** `platform_key(system=platform.system(), machine=platform.machine())`:
  - `Darwin` with `x86_64` gives `macos-x64`. Python running under Rosetta reports `x86_64`, so the
    Intel build is chosen, which also runs there.
  - `Darwin` with `arm64` gives `macos-arm64`.
  - `Linux` with `x86_64` or `amd64` gives `linux-x64`.
  - `Linux` with `aarch64` or `arm64` gives `linux-arm64`.

  Anything else raises `ValueError("No pinned llama.cpp b11896 runtime for <system>/<machine>;
  pinned: <keys>; pass --platform to choose one explicitly")`. A key without a pin (for example
  `macos-arm64` if Q1 is no) is refused the same way.
- **`--platform KEY`** overrides the detection; its choices are the pinned keys. The Dockerfile maps
  `TARGETARCH` (`amd64` → `linux-x64`, `arm64` → `linux-arm64`) and passes it.
- **Start check (new).** After `verify_runtime`, when the key matches the host, run `llama-server
  --version` with a 120 s timeout and require `build 11896` in its output. On failure, say what is
  likely missing: glibc ≥ 2.34 (x64) or ≥ 2.38 (arm64), and `libgomp1` (01 §2). With an override
  for a different platform, the check is skipped, and the skip is printed and recorded.
- **Manifest.** The current fields stay (`llama_release`, `runtime_archive_sha256`, now the chosen
  runtime's own digest, `llama_server`, the model fields). Two fields are added:
  - `runtime_platform` (the key);
  - `runtime_archive` (the asset name).

  `gguf_server.load_manifest` accepts the fields and is otherwise unchanged; `llama_server` stays an
  absolute path, as today. The model GGUF pin (`REVISION`, `SHA256`, `SIZE`, the URL) is untouched.

## 2. Destination from `config.MODELS`

`DESTINATION = MODELS / "chatbot/qwen35"`, like `qa_assets`, `medium_assets` and `qwen_assets`
(the module imports `candidate_screener.config`, standard library only). With
`SCREENER_MODELS_ROOT` unset, `MODELS` is `data/processed`, so the native default stays
`data/processed/chatbot/qwen35`. The README and AGENTS commands do not change.

## 3. Docker: a separate image, opt-in **(Q2)**

**Recommendation: a separate image, `candidate-screener/hiring-desk-llm4b:local`, built from a new
Dockerfile target `llm4b` on `ubuntu:24.04`.** The default image does not grow (it stays 7.98 GB),
and building it never downloads the 4B model.

Why not the app image? The arm64 release needs glibc 2.38 and the app's bookworm base has 2.36
(01 §2). Compiling for arm64 is the only way around that inside the app image: a separate binary,
an emulated arm64 compile of unknown length, and a weaker identity argument (01 §3). On Ubuntu
24.04 the **published binaries** start on both architectures.

The `llm4b` stage, `FROM ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c`:

1. **System packages:** `python3` (3.12.3, the same minor version as the project),
   `ca-certificates`, `libgomp1`. They are pinned by version against a dated
   `snapshot.ubuntu.com` mirror, so a rebuild gets the same packages.
2. **The facade's Python dependency:** `requests` and its four dependencies, installed by uv into
   `/opt/gguf-venv` at the versions and hashes in `uv.lock` (a contract test keeps them equal). No
   `pip`, no system `python3-requests`.
3. **Downloads at build time:** `gguf_assets.py --platform linux-$ARCH --destination
   /opt/models/chatbot/qwen35` (copied files: `__init__.py`, `config.py`, `llm/gguf_assets.py`,
   `llm/gguf_server.py`). A cache mount keyed by the pins makes re-runs copy instead of download.
   The start check runs here, on the real runtime base. The optional `hf_token` secret is the same
   as for the app.
4. **Run time:** user 1000, `HF_HUB_OFFLINE=1`, no network needed, read-only.
   `CMD python -m candidate_screener.llm.gguf_server --manifest
   /opt/models/chatbot/qwen35/manifest.json --launch-backend`: the same command as natively, facade
   on 127.0.0.1:8001 and backend on 127.0.0.1:8002.
5. **Healthcheck:** `GET http://127.0.0.1:8001/v1/models` with no proxy. The facade answers 503
   until the backend answers, so a healthy service means the model is loaded. `start_period` is
   300 s, covering the GGUF hash at start and the model load; S2 measures both.

**Compose (Q2).** A separate file, `compose.llm4b.yaml`, holds everything for the 4B:

- service `llm4b` (`network_mode: service:app`, `init`, the same user, no published port, no data
  volume);
- for `app`, `SCREENER_QWEN_MANIFEST=/opt/qwen35-manifest/manifest.json`;
- a read-only named volume, `qwen35-manifest-e87f176-b11896`, through which `app` reads the small
  manifest copy that the `llm4b` image keeps at `/opt/qwen35-manifest/`. Docker fills an empty
  named volume from the image; the volume name carries the pins, so a pin change gets a fresh
  volume.

Use it with `docker compose -f compose.yaml -f compose.llm4b.yaml up -d`, or set
`COMPOSE_FILE=compose.yaml:compose.llm4b.yaml` in `.env`. Two reasons it is a separate file rather
than a profile in `compose.yaml`: compose cannot set an environment variable on `app` only when a
profile is active, and with the file absent, `compose.yaml` and the default app service do not
change at all. (The alternative, `SCREENER_QWEN_MANIFEST: ${SCREENER_QWEN_MANIFEST:-}` on `app`
plus `profiles: [llm4b]`, works too, because an empty value means "not selected" in
`conversation_prose.py`. It needs two `.env` lines kept in step.)

**Costs:**

| Cost | Figure |
|---|---|
| Image | 2.74 GB model, 57 MB runtime, Ubuntu with Python about 150 MB: estimated about 3.0 GB, measured in S2 |
| Build | the 2.74 GB download (V8's rate gives about 2-3 min) |
| Memory | measured in S2: `llama-server` resident memory and the container's peak cgroup memory, idle and during a turn |
| The 60 s turn budget | latency of the D67 walkthrough prompts measured idle, and again while a D63 ranking runs in `app` on the same CPU (the backend uses `--threads 4`). The budget stays 60 s; this only reports how often it is hit |

## 4. What does not change

- `gguf_server.backend_command` (the flags, 4 threads, one slot, thinking off, no cache);
- the 60 s and 30 s limits;
- v3's checks and fallback;
- `conversation_prose`'s selection by `SCREENER_QWEN_MANIFEST`;
- the model pin;
- the 0.5B `qwen` service (port 8000), which can run beside it.

No training, no download at run time, no ranking or label change, no benchmark claim.
