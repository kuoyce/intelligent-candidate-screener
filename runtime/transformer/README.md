# Isolated transformer inference runtime

Sync with `uv sync --project runtime/transformer`. The main project retains its
existing package versions. This runtime pins PyTorch 2.2.2 (the last official
Intel macOS series), Transformers 4.48.3 and GLiNER 0.2.21 with NumPy 1.x. The
separate lockfile resolves compatible Intel macOS wheels, including ONNX Runtime.

On Linux the lock takes torch's CPU build from the PyTorch CPU index (`2.2.2+cpu` on
x86_64, `2.2.2` on aarch64), without the CUDA packages; the macOS and Windows entries
are unchanged. The Docker image syncs this environment ([guide](../../docs/docker.md)).

The package starts a persistent local worker through uv, passes documents over
local pipes, and receives span records. The worker loads only downloaded assets,
sets offline mode and uses four CPU threads. No remote text inference occurs.
Notebook execution and project tests continue to use the main environment. Tests
mock neural outputs and do not require this runtime or model weights.
