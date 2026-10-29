# 01 — Measurements before design (28 Oct 2026)

Cheap checks only: the four CPU runtime archives of llama.cpp release `b11896` (12-23 MB each),
no model download. Archives in the session scratchpad (`gguf-dl/`); the source-build log is
`logs/gguf/src-build-amd64.log`. Host: Ubuntu 24.04 in WSL2 (glibc 2.39), 22 cores, Docker Engine
27.5.1, the owner's QEMU arm64 registration still installed.

## 1. What `b11896` publishes

GitHub release API, tag `b11896`, published 2026-10-26T10:17:02Z, commit
`9125a2ebe8e5a38f0f4232170c134a29ed730b66`. CPU builds relevant here (37 assets in all; CUDA,
ROCm, SYCL, Vulkan, OpenVINO, Android, s390x and Windows builds ignored):

| Key (proposed) | Asset | Size (bytes, measured) | SHA-256 (measured with `sha256sum`) | Members |
|---|---|---:|---|---:|
| `macos-x64` (pinned today) | `llama-b11896-bin-macos-x64.tar.gz` | 12,038,557 | `2ff5c5f3b5ba8a12a4a868c290510799416ad78c61528e9d7754404aa6fd6fb5` | 57 |
| `macos-arm64` (optional) | `llama-b11896-bin-macos-arm64.tar.gz` | 16,297,217 | `f7fdf25367d2a403a91ec91729322a3c6a3a80e95da15844aa2c7be9dab3e11b` | 62 |
| `linux-x64` | `llama-b11896-bin-ubuntu-x64.tar.gz` | 22,497,065 | `b02ed815efe497c0e818aadaae1b2538309f25424034555dcc1b7211c783e25c` | 62 |
| `linux-arm64` | `llama-b11896-bin-ubuntu-arm64.tar.gz` | 18,221,582 | `68942cd6067dd632831f12e0bcdb4f8edbd91882e633a4e8593e0690f0624261` | 56 |

The macos-x64 download matches today's pin byte for byte (size and SHA-256). Each measured SHA-256
also equals the digest the release API lists; the pins come from our own download, not from that
page.

## 2. What the Linux builds need

`readelf -d` / `objdump -T` over every ELF file in each archive:

| | `linux-x64` (`ubuntu-x64`) | `linux-arm64` (`ubuntu-arm64`) |
|---|---|---|
| Built with | GNU 11.4.0 (an Ubuntu 22.04 toolchain) | GNU 14.2.0 (Ubuntu 24.04) |
| Highest symbol versions | `GLIBC_2.34`, `GLIBCXX_3.4.30`, `CXXABI_1.3.13` | `GLIBC_2.38`, `GLIBCXX_3.4.32`, `CXXABI_1.3.15` |
| External libraries | libc, libm, libstdc++, libgcc_s, **libgomp**, libssl 3, libcrypto 3, the loader | the same |
| Bundled | `libllama*`, `libggml*`, `libmtmd`, and CPU back ends loaded at run time: 15 x86 variants (`libggml-cpu-{sse42,sandybridge,haswell,skylakex,icelake,zen4,…}.so`) | 8 Arm variants (`libggml-cpu-armv8.0_1` … `armv9.2_2`) |
| Library lookup | `RUNPATH $ORIGIN` on the binaries: no `LD_LIBRARY_PATH` needed | the same |
| Unpacked | 56.9 MB | 42.6 MB |

`llama-server --version` in containers (the image's own base, `python:3.12-slim-bookworm`,
glibc 2.36, GCC 12 libstdc++; and `ubuntu:24.04` index
`sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c`, glibc 2.39):

| Runtime | bookworm (the app image) | Ubuntu 24.04 |
|---|---|---|
| `linux-x64` | fails: `libgomp.so.1: cannot open shared object file`. **Starts** once Debian's `libgomp1` (12.2.0-14+deb12u1) is installed: `version: 0.7.0-dev (build 11896, commit 9125a2ebe)` | **starts** with `libgomp1` 14.2.0-4ubuntu2~24.04.1 installed (libssl3t64 is already there) |
| `linux-arm64` (QEMU emulation) | **fails**: `GLIBC_2.38 not found`, `GLIBCXX_3.4.32 not found`, `CXXABI_1.3.15 not found` (20 of its files need them, `libllama-server-impl.so` among them) | **starts** under emulation (`built with GNU 14.2.0 for Linux aarch64`); `--help` exits 0 |

Natively on this machine (Ubuntu 24.04, `libgomp1` present) the `linux-x64` server starts as is.

## 3. The alternative: build from source

For the arm64 problem on bookworm: `git checkout 9125a2ebe…` (the commit tag `b11896` names), then
`cmake -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DBUILD_SHARED_LIBS=ON
-DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF`, target `llama-server`, in `python:3.12-slim-bookworm`:

| amd64, measured | |
|---|---|
| apt (`build-essential cmake git libgomp1`) | 12 s |
| clone (blob-less) and checkout | 9 s |
| compile, 22 cores | **158 s** (3 min 8 s for the whole run) |
| result | `llama-server` reports `build 11896, commit 9125a2ebe`, GNU 12.2.0; `build/bin` 42.0 MB, 31 files |

arm64 was not compiled: under QEMU it would take far longer (S11 measured model inference about
300 times slower; a compile is likely 10-30 times slower, **not measured**), and on a real Apple
Silicon machine it builds natively. A source build is our own binary, not the published release,
which weakens the identity argument in `03` (same source, different compiler and flags).

## 4. Consequences for the design

1. `linux-x64` from the release works on the app's bookworm base with one extra package
   (`libgomp1`).
2. `linux-arm64` from the release needs glibc ≥ 2.38: it cannot run in the app image. It runs on
   Ubuntu 24.04. Hence the recommendation in `02` §3: a **separate image on `ubuntu:24.04`** for the
   GGUF service, using the release binaries on both architectures, rather than compiling.
3. Natively, the Linux runtimes need a host with glibc ≥ 2.34 (x64) or ≥ 2.38 (arm64) and
   `libgomp1`; `provision` checks this by starting `llama-server --version` and says what is missing.
4. The x86 Linux build picks one of 15 CPU kernels at start-up by the host's CPU features, and the
   macOS builds are separate compiles. Generation can therefore differ across machines in the last
   bits of a logit, which matters for `03`.
