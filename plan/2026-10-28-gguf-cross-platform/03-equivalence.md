# 03 — Equivalence across platforms (registered before any run)

D67 requires that model identity and tokenizer/generation equivalence be preserved. This bar is
committed **before** any cross-platform generation is run, like V10 in the Docker plan. Anything
seen earlier is labelled exploratory.

## Probe

A new script, `python -m candidate_screener.llm.gguf_probe --manifest M --out OUT.json`. It is
read-only and uses invented text only. It talks to an already running service and starts
nothing: the facade (8001) for `/v1/models`, and the llama.cpp backend (8002) for `/apply-template`,
`/tokenize` and `/completion` with `n_probs`, because the facade allows only the app's own
requests.

- **Prompts:** 6 invented prompts, committed with the script. 4 follow the D67 walkthrough's shape
  (a degree question, an ongoing Master's, a comparison of two invented candidates, a "why is 1
  better" premise), and 2 are short plain prompts.
- **For each prompt, recorded:**
  - the `/apply-template` text, with `enable_thinking: false`;
  - the `/tokenize` ids of that text;
  - a greedy completion: `temperature 0`, `top_k 1`, `seed 0`, `max_tokens 64`, `n_probs 2`. That
    gives the generated token ids and, at each step, the top-2 log-probabilities.
- **Also recorded:**
  - the facade's `/v1/models` metadata (model, revision, GGUF SHA-256);
  - `runtime_platform` and `runtime_archive_sha256` from the manifest;
  - the CPU back end `llama-server` loaded (from its start-up log line);
  - the host CPU model and the thread count.

## The bar

| Check | Pass |
|---|---|
| E1 Identity | `/v1/models` model, revision and GGUF SHA-256 identical on every platform and equal to the pin |
| E2 Template and tokenizer | the `/apply-template` text and the `/tokenize` ids **identical** for all 6 prompts on every platform |
| E3 Same binary, two environments (linux-x64 natively vs linux-x64 in Docker, same machine) | generated token ids **identical** for all 6 prompts |
| E4 Across builds (macos-x64 on the teammate's Intel Mac vs linux-x64) | for each prompt, the token ids are identical, **or** they first differ at a step where the reference run's top-2 log-probability gap is ≤ 0.05 nats (a near tie). Nothing after the first difference is compared |
| E5 linux-arm64 under emulation (S11-style, slow) | E1 and E2. Plus, as a long job, prompt 1 with `max_tokens 8` under the E4 rule. If the emulated load or generation exceeds 2 h, record E5's generation part as "not run (emulation)", as S11 did |
| E6 macos-arm64 (only if Q1 = yes, on an Apple Silicon Mac) | E1 and E2, and the E4 rule against linux-x64 |

**A stop condition:** any E1/E2 failure, or an E3 failure. Do not adjust the bar afterwards;
report the difference and decide with the owner.

## What it can and cannot prove

**It can show:**

- every platform loads the same pinned weights;
- every platform applies the same chat template and tokenizer (E1, E2);
- the Docker packaging changes nothing for the same binary (E3);
- on six prompts, greedy output of different builds agrees, apart from near ties (E4).

**It cannot show:**

- agreement on other prompts or longer outputs. Greedy decoding amplifies the first tiny logit
  difference;
- answer quality or v3 acceptance rates;
- speed on a machine we have not measured;
- behaviour on a CPU that loads a different ggml kernel variant. The x86 build picks one of 15 at
  start-up (01 §4), so a second x86 machine can pass E3 here and still differ in the last bits.

It is a development check, never a held-out or model-selection result.
