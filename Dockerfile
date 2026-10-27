# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
# The Hiring Desk image (plan 2026-10-27-docker, 01-design-spec.md §2 and §5). The frontend
# pin above is the one S1 tested: `COPY --exclude` patterns are relative to the source path.
#
#   docker compose build                         # the `app` image (the default, last stage)
#   docker compose --profile test build test     # the `test` image
#
# Stages: uv -> base -> deps -> test, and deps -> models -> app. Each person builds for their own
# machine; the base images are pinned by multi-architecture index digest (amd64 and arm64).
# Contract checks: tests/test_docker_contract.py.

# Base images, pinned by tag and index digest (27 Oct 2026, `docker buildx imagetools inspect`):
# Python 3.12.16 on Debian bookworm (image created 26 Oct 2026); uv 0.12.7, which wrote the locks.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:fc169880a4d941f3142c0e2e623f05f246150add19dc36fc9224df20fb19e5fd
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.7@sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945

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
