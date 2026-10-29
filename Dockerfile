# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
# The Hiring Desk image (plan 2026-10-27-docker, 01-design-spec.md §2 and §5). The frontend
# pin above is the one S1 tested: `COPY --exclude` patterns are relative to the source path.
#
#   docker compose build                         # the `app` image (the default, last stage)
#   docker compose --profile test build test     # the `test` image
#   docker compose -f compose.yaml -f compose.llm4b.yaml build llm4b   # the optional Qwen3.5-4B image
#
# Stages: uv -> base -> deps -> test, and deps -> models -> app; llm4b (Ubuntu) borrows files from
# uv and base. Each person builds for their own machine; the base images are pinned by
# multi-architecture index digest (amd64 and arm64).
# Contract checks: tests/test_docker_contract.py.

# Base images, pinned by tag and index digest (27 Oct 2026, `docker buildx imagetools inspect`):
# Python 3.12.16 on Debian bookworm (image created 26 Oct 2026); uv 0.12.7, which wrote the locks.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:fc169880a4d941f3142c0e2e623f05f246150add19dc36fc9224df20fb19e5fd
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.7@sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945
# Ubuntu 24.04.5 (glibc 2.39), for the llm4b target only (plan 2026-10-28-gguf-cross-platform, 02 §3).
ARG UBUNTU_IMAGE=ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c

# uv: the uv binary. Rebuilt when the uv pin changes.
FROM ${UV_IMAGE} AS uv

# base: OS packages (OpenMP for torch), uv, fixed settings and the non-root user `screener`
# (uid/gid 1000; compose runs as the host's ids instead). Rebuilt when a base pin changes.
FROM ${PYTHON_IMAGE} AS base
RUN apt-get update \
 && apt-get install --yes --no-install-recommends ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN groupadd --gid 1000 screener \
 && useradd --uid 1000 --gid 1000 --create-home screener
WORKDIR /app

# deps: the three locked environments, /app/.venv (groups app and ml) and the model runtimes'
# runtime/{transformer,gliner-medium}/.venv. Rebuilt when a lock or a pyproject changes; the
# wheels stay in the uv cache mount.
FROM base AS deps
COPY --chmod=a+rX pyproject.toml uv.lock .python-version /app/
COPY --chmod=a+rX runtime/transformer/pyproject.toml runtime/transformer/uv.lock /app/runtime/transformer/
COPY --chmod=a+rX runtime/gliner-medium/pyproject.toml runtime/gliner-medium/uv.lock /app/runtime/gliner-medium/
RUN --mount=type=cache,id=screener-uv-cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --group app --group ml \
 && uv sync --frozen --project runtime/transformer \
 && uv sync --frozen --project runtime/gliner-medium \
 && chmod -R a+rX /app/.venv /app/runtime/transformer/.venv /app/runtime/gliner-medium/.venv

# test: the suite (`docker compose --profile test run --rm test`: no network, no data): deps plus
# the dev group and the repository as .dockerignore filters it (of data/, only README.md and
# vocab/). Rebuilt by any source or test change; the app image does not depend on it.
FROM deps AS test
RUN --mount=type=cache,id=screener-uv-cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --all-groups
COPY . /app/
RUN --mount=type=cache,id=screener-uv-cache,target=/root/.cache/uv \
    uv sync --frozen --all-groups
CMD ["/app/.venv/bin/python", "-m", "pytest"]

# llm4b: Qwen3.5-4B (GGUF, Q4_K_M) behind the loopback facade for chat v3 (plan
# 2026-10-28-gguf-cross-platform, 02 §3). A separate image, built only on request
# (`docker compose -f compose.yaml -f compose.llm4b.yaml build llm4b`) and run through
# compose.llm4b.yaml; it never changes the app image. Ubuntu 24.04 because the published
# linux-arm64 llama.cpp build needs glibc 2.38 (bookworm has 2.36; 01 §2). Rebuilt when the
# Ubuntu pin, the snapshot, a facade dependency, or gguf_assets.py/config.py changes; a change to
# gguf_server.py rebuilds only the last layers.
FROM ${UBUNTU_IMAGE} AS llm4b
ARG TARGETARCH
# Ubuntu packages, pinned by version (owner, Q6); only the `main` component is read. On amd64 they
# come from this dated snapshot. snapshot.ubuntu.com serves no ubuntu-ports, so arm64 reads the live
# ports archive with the same version pins (owner, 28 Oct 2026): the build fails loudly once Ubuntu
# replaces a version, and the pins are then bumped. The snapshot is HTTPS only and the base has no
# CA bundle yet, so apt borrows the base stage's (packages are verified by the archive's
# signatures, as always).
ARG UBUNTU_SNAPSHOT=20261027T000000Z
RUN --mount=type=bind,from=base,source=/etc/ssl/certs/ca-certificates.crt,target=/run/snapshot-ca.crt \
    set -eu; \
    sed -i -e 's/^Suites: noble noble-updates noble-backports$/Suites: noble noble-updates/' \
        -e 's/^Components: .*$/Components: main/' /etc/apt/sources.list.d/ubuntu.sources; \
    apt="apt-get"; \
    if [ "$TARGETARCH" = amd64 ]; then apt="$apt -o APT::Snapshot=$UBUNTU_SNAPSHOT -o Acquire::https::CaInfo=/run/snapshot-ca.crt"; fi; \
    $apt update; \
    $apt install --yes --no-install-recommends \
        python3=3.12.3-0ubuntu2.1 ca-certificates=20260816ubuntu0.24.04.1 libgomp1=14.2.0-4ubuntu2~24.04.1; \
    rm -rf /var/lib/apt/lists/*
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1
# The facade's one dependency, requests, and its four, at uv.lock's versions and wheel hashes
# (pure-Python wheels, the same on both architectures; a contract test keeps them equal to the
# lock). uv is mounted, not copied, so the image holds no package tool.
RUN --mount=type=bind,from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,id=screener-uv-cache,target=/root/.cache/uv \
    set -eu; \
    printf '%s\n' \
        'requests==2.34.2 --hash=sha256:2a0d60c172f83ac6ab31e4554906c0f3b3588d37b5cb939b1c061f4907e278e0' \
        'certifi==2026.7.22 --hash=sha256:62f22742b58a1a33014a2b6b706588a8d7e2a88ae7bd1a6ebe8c992928483775' \
        'charset-normalizer==3.5.1 --hash=sha256:6df0ec430f9a831772c23ca5a224cba36517a58a84bb32c32bb59a9fa67c47f6' \
        'idna==3.19 --hash=sha256:815e7be7a7806d54abb586dc943addc79e8b2ee16915059658cbeff4b1b43bf4' \
        'urllib3==2.7.0 --hash=sha256:9fb4c81ebbb1ce9531cce37674bbc6f1360472bc18ca9a553ede278ef7276897' \
        > /tmp/gguf-requirements.txt; \
    uv venv /opt/gguf-venv --python /usr/bin/python3; \
    uv pip install --python /opt/gguf-venv/bin/python --require-hashes --no-deps --only-binary :all: \
        -r /tmp/gguf-requirements.txt; \
    rm /tmp/gguf-requirements.txt; \
    chmod -R a+rX /opt/gguf-venv
COPY --chmod=a+rX src/candidate_screener/__init__.py src/candidate_screener/config.py /app/src/candidate_screener/
COPY --chmod=a+rX src/candidate_screener/llm/__init__.py src/candidate_screener/llm/gguf_assets.py /app/src/candidate_screener/llm/
# The runtime and the model, fetched and verified by gguf_assets.py itself (its pins only; none
# is repeated here), for this architecture's key. Its start check runs llama-server --version
# here, on the real base. Downloads persist in a cache mount, one folder per pin
# (`qwen35-<revision>-<release>-<key>`), so a rerun copies instead of downloading again.
# Docker fills the compose file's named volume from /opt/qwen35-manifest (a copy the app reads).
RUN --mount=type=cache,id=screener-gguf-downloads,target=/var/cache/screener-gguf,sharing=locked \
    set -eu; \
    case "$TARGETARCH" in \
        amd64) key=linux-x64 ;; \
        arm64) key=linux-arm64 ;; \
        *) echo "no pinned llama.cpp runtime for TARGETARCH=$TARGETARCH" >&2; exit 1 ;; \
    esac; \
    export PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1; \
    assets=/app/src/candidate_screener/llm/gguf_assets.py; dest=/opt/models/chatbot/qwen35; \
    set -- $(python3 -c 'import sys; from candidate_screener.llm import gguf_assets as g; k = sys.argv[1]; print(f"qwen35-{g.REVISION}-{g.RELEASE}-{k}", g.local_names(k)[0], g.FILENAME)' "$key"); \
    cache="/var/cache/screener-gguf/$1"; shift; \
    mkdir -p "$cache" "$dest"; \
    for name in "$@"; do if [ -f "$cache/$name" ]; then cp "$cache/$name" "$dest/$name"; fi; done; \
    python3 "$assets" --platform "$key" --destination "$dest"; \
    for name in "$@"; do if [ ! -f "$cache/$name" ]; then cp "$dest/$name" "$cache/$name.copy" && mv "$cache/$name.copy" "$cache/$name"; fi; done; \
    mkdir -p /opt/qwen35-manifest; \
    cp "$dest/manifest.json" /opt/qwen35-manifest/manifest.json; \
    chmod -R a+rX /opt/models /opt/qwen35-manifest
COPY --chmod=a+rX src/candidate_screener/llm/gguf_server.py src/candidate_screener/llm/gguf_probe.py /app/src/candidate_screener/llm/
# Run time: no downloads, no writes but /tmp, any user id (compose passes the host's).
ENV PATH=/opt/gguf-venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    HOME=/tmp
WORKDIR /app
USER 1000:1000
# The facade answers GET /v1/models only once the backend answers with the pinned model (503
# before), so healthy means loaded. No proxy for this loopback request. start_period covers the
# facade's SHA-256 of the 2.74 GB model and the model load.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD ["python", "-c", "import urllib.request as r; r.build_opener(r.ProxyHandler({})).open('http://127.0.0.1:8001/v1/models', timeout=5).read()"]
# The native command: facade on 127.0.0.1:8001, llama-server on 127.0.0.1:8002 (SIGTERM stops both).
CMD ["python", "-m", "candidate_screener.llm.gguf_server", "--manifest", "/opt/models/chatbot/qwen35/manifest.json", "--launch-backend"]

# models: the four pretrained checkpoints, nothing trained: RoBERTa-SQuAD2, GLiNER-medium and
# Qwen2.5-0.5B in /opt/models (the asset modules' own layout under SCREENER_MODELS_ROOT) and C2
# in /opt/hf (the Hugging Face cache layout). Rebuilt when deps (a lock), a pin below or one of the
# five copied files changes, never by a frontend change. Downloads persist in a cache mount, one
# folder per pin, so a re-run copies instead of downloading again; only metadata calls reach the
# Hub (`docker builder prune --filter type=exec.cachemount` clears the cache).
FROM deps AS models
ENV SCREENER_MODELS_ROOT=/opt/models \
    HF_HOME=/opt/hf
# Model pins, resolved 27 Oct 2026 with huggingface_hub metadata calls (no weights fetched):
# - QA_REVISION: deepset/roberta-base-squad2's current commit (last changed 24 Sep 2024), which is
#   also the revision recorded for the teammate's checkpoint (plan 2026-10-24-hiring-desk-v2, 02 §2).
# - GLINER_ENCODER_REVISION: the current commit of microsoft/deberta-v3-base (last changed 22 Sep
#   2022), the backbone that gliner_config.json names at the checkpoint's pinned revision. The
#   checkpoint's own pin is MODEL_REVISION in extraction/medium_assets.py, copied below.
# - QWEN_REVISION: Qwen/Qwen2.5-0.5B-Instruct, the owner's machine's manifest.
# - C2_MODEL, C2_REVISION: equal to rerank/cross.py's pin (checked by the contract tests).
ARG QA_REVISION=adc3b06f79f797d1c575d5479d6f5efe54a9e3b4
ARG GLINER_ENCODER_REVISION=8ccc9b6f36199bec6961081d44eb72fb3f7353f3
ARG QWEN_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
ARG C2_MODEL=Qwen/Qwen3-Reranker-0.6B
ARG C2_REVISION=e61197ed45024b0ed8a2d74b80b4d909f1255473
COPY --chmod=a+rX src/candidate_screener/__init__.py src/candidate_screener/config.py /app/src/candidate_screener/
COPY --chmod=a+rX src/candidate_screener/llm/qa_assets.py src/candidate_screener/llm/qwen_assets.py /app/src/candidate_screener/llm/
COPY --chmod=a+rX src/candidate_screener/extraction/medium_assets.py /app/src/candidate_screener/extraction/
# The asset modules run by file path (standard library plus `config`) on the main environment's
# huggingface_hub, each into a cache folder named after its pin that mirrors /opt/models; the
# downloader's own state (Xet, token) stays in the cache mount. An optional Hugging Face token
# comes only as the build secret `hf_token`. Then the copies are verified offline where the app
# reads them.
RUN --mount=type=cache,id=screener-model-downloads,target=/var/cache/screener-models,sharing=locked \
    --mount=type=secret,id=hf_token,required=false \
    set -eu; \
    cache=/var/cache/screener-models; py=/app/.venv/bin/python; pkg=/app/src/candidate_screener; \
    export PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 HF_HOME="$cache/hf-home" \
        HF_TOKEN_PATH=/run/secrets/hf_token HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0; \
    SCREENER_MODELS_ROOT="$cache/qa-$QA_REVISION" "$py" "$pkg/llm/qa_assets.py" --revision "$QA_REVISION"; \
    SCREENER_MODELS_ROOT="$cache/gliner-medium-$GLINER_ENCODER_REVISION" \
        "$py" "$pkg/extraction/medium_assets.py" --encoder-revision "$GLINER_ENCODER_REVISION"; \
    SCREENER_MODELS_ROOT="$cache/qwen-$QWEN_REVISION" "$py" "$pkg/llm/qwen_assets.py" --revision "$QWEN_REVISION"; \
    HF_HUB_CACHE="$cache/c2-$C2_REVISION" "$py" -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))' "$C2_MODEL" "$C2_REVISION"; \
    mkdir -p /opt/models /opt/hf/hub; \
    cp -a "$cache/qa-$QA_REVISION/." "$cache/gliner-medium-$GLINER_ENCODER_REVISION/." "$cache/qwen-$QWEN_REVISION/." /opt/models/; \
    cp -a "$cache/c2-$C2_REVISION/." /opt/hf/hub/; \
    chmod -R a+rX /opt/models /opt/hf; \
    export HF_HUB_OFFLINE=1; \
    "$py" "$pkg/llm/qa_assets.py" --verify; \
    "$py" "$pkg/extraction/medium_assets.py" --verify; \
    "$py" "$pkg/llm/qwen_assets.py" --revision "$QWEN_REVISION"; \
    HF_HUB_CACHE=/opt/hf/hub "$py" -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True))' "$C2_MODEL" "$C2_REVISION"
# transformers 4.48 (the model runtimes) writes this cache-version marker on first use; the cache
# is read-only at run time, so the marker is written here (the S4 audit saw the failed write).
RUN printf '1' > /opt/hf/hub/version.txt && chmod a+r /opt/hf/hub/version.txt

# app (the default, last stage): the image people run. The committed records and the backend
# code; one RUN installs the project editable, writes GLiNER-medium's runtime-versions.json from
# the image's runtime (data/README.md, "Medium-v2.5 sentence-aware extraction") and compiles the
# backend; the frontend (`app/`, `.streamlit/`) comes last, so a frontend change rebuilds only
# the final layers (V7). A backend change re-runs from the src copy, never the model downloads.
FROM models AS app
COPY --chmod=a+rX output/ /app/output/
COPY --chmod=a+rX docs/data/ /app/docs/data/
COPY --chmod=a+rX --exclude=candidate_screener/app src/ /app/src/
RUN --mount=type=cache,id=screener-uv-cache,target=/root/.cache/uv \
    set -eu; \
    uv sync --frozen --no-dev --group app --group ml; \
    PYTHONPATH=src uv run --project runtime/gliner-medium --no-sync python -c 'import json; from candidate_screener.extraction.medium_model import runtime_versions; from candidate_screener.extraction.medium_assets import MODEL_DIR; (MODEL_DIR/"runtime-versions.json").write_text(json.dumps(runtime_versions(),indent=2,sort_keys=True)+"\n")'; \
    /app/.venv/bin/python -m compileall -q -j 0 /app/src; \
    find /app /opt \( -type d ! -perm -o=rx -o -type f ! -perm -o=r \) -exec chmod a+rX {} +
COPY --chmod=a+rX src/candidate_screener/app/ /app/src/candidate_screener/app/
COPY --chmod=a+rX .streamlit/ /app/.streamlit/
# Run time (01-design-spec.md §5): no downloads (models and environments are in the image);
# writes only to the data folder, logs/ and /tmp, as any user id. ORT_DISABLE_TELEMETRY stops
# onnxruntime (imported by GLiNER) from writing telemetry files and contacting Microsoft (S4).
ENV ORT_DISABLE_TELEMETRY=1 \
    SCREENER_MODELS_ROOT=/opt/models \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    UV_OFFLINE=1 \
    HOME=/tmp \
    UV_CACHE_DIR=/tmp/uv-cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/app/.venv/bin:$PATH
USER 1000:1000
EXPOSE 8503
# Streamlit's own health endpoint; it verifies nothing else. No curl in the image, and no proxy
# for this loopback request.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request as r; r.build_opener(r.ProxyHandler({})).open('http://127.0.0.1:8503/_stcore/health', timeout=5).read()"]
# browser.serverAddress: Streamlit prints http://localhost:8503 instead of looking up the host's
# external IP (an HTTP request to checkip.amazonaws.com, S4).
CMD ["streamlit", "run", "src/candidate_screener/app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8503", "--server.headless=true", "--browser.serverAddress=localhost", "--browser.gatherUsageStats=false", "--server.fileWatcherType=none"]
# Last, so a new commit changes only the label, never a cached layer.
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
