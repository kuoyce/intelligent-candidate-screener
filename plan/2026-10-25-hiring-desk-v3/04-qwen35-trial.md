# Approved inference-only Qwen3.5-4B trial

Owner instruction: “try qwen 3.5-4B, improve the reasoningof the chatbot”. This
authorizes downloading and running the frozen replacement locally, evaluating it
on invented owner-flow examples, and enabling it when useful. Stay on the current
v3 branch. No training, benchmark labels or remote applicant processing.

## Design and implementation

1. Provision Unsloth's Q4_K_M GGUF conversion of `Qwen/Qwen3.5-4B` into ignored
   `data/processed/chatbot/qwen35/`. Pin source commit
   `e87f176479d0855a907a41277aca2f8ee7a09523`, file
   `Qwen3.5-4B-Q4_K_M.gguf`, 2,740,937,888 bytes, SHA-256
   `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
2. Use official llama.cpp Intel-macOS binary release `b11896`, archive SHA-256
   `2ff5c5f3b5ba8a12a4a868c290510799416ad78c61528e9d7754404aa6fd6fb5`.
   Keep its binaries in this private artifact directory, without upgrading the
   existing Torch 2.2.2 / Transformers 4.48 QA runtime or installing global packages.
3. Run backend on loopback 8002, 4096-token context, CPU, one slot, no persistent
   applicant prompt cache, thinking disabled. Add a bounded facade on loopback8001
   providing the existing attested model/tokenizer/chat contract. Count tokens
   with the same backend chat template and thinking setting used for generation.
4. Keep v3 support checks, source ownership, scope/history guards and fallback.
   First compare the same previously seen invented degree/status walkthrough;
   add the owner happy/negative regressions. Report genuine generated acceptance
   separately from routing/fallback success. This is exploratory development,
   not a held-out ranking/model-selection result.
5. Switch portal to the pinned model only after actual generation and source checks.
   Preserve the previous service on8000 for rollback. Record measured latency and
   failures. Initial native testing found three of four turns exceeded30seconds
   on this4-core,16GB Intel Mac. The selected Qwen3.5 path therefore uses a bounded
   60-second shared turn limit; extraction remains capped at30seconds, and the
   existing0.5B path stays at30seconds. No unbounded retry or budget increase.

## Resources and open questions

Download is approximately2.75GB including binaries. Four-bit weights are a memory
choice, not evidence of correctness. Background provisioning writes one log and
exit marker; do not restart/poll a long download. Independent adapter review and
focused tests precede activation. Broad independent human review remains pending.

Sources: [Qwen model](https://huggingface.co/Qwen/Qwen3.5-4B),
[GGUF conversion](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF),
[llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b11896).

## Verification record, 27 October

Provisioning completed with verified SHA-256 for both artifacts. Download log:
`logs/qwen35-provision.log` (exit0). The facade's native model metadata matches the
pinned model and the llama.cpp properties/tokenizer endpoints work on Intel macOS.
Runtime extraction is verified against the pinned archive on reuse; altered
binaries/libraries, changed symlinks, missing and extra files are rejected.

Before-fix regressions: `qwen35-selection-before.log` proves the explicit manifest
was previously ignored; `qwen35-degree-before.log` proves a correct possessive
degree phrase was rejected; `v3-4b-deadline-before.log` proves the larger-model
budget did not propagate. All now have focused regression tests. Latest adapter,
worker, service and history regressions:115passed (`logs/v3-4b-deadline-tests.log`).
Provisioning/adapter tests:65passed (`logs/v3-gguf-focused-tests.log`).

The initial real-model walkthrough (`logs/qwen35-walkthrough.json`) accepted0/4:
one correct answer failed grammatical validation and three hit the30-second
deadline. The bounded60-second rerun is recorded separately, preserving this
failure evidence. These are seen synthetic development cases, never held-out
model accuracy or ranking results.

### Activation and final limits

The portal was activated with the explicit Qwen3.5 manifest on27Oct2026 at
`http://127.0.0.1:8503`; the portal health and pinned model metadata endpoints both
returned200. Launch identity: `logs/hiring-desk-v3-launch.json`; backend/facade
identity: `logs/qwen35-service-launch.json`. Existing model service8000 remains
available for rollback.

The60-second walkthrough accepted1/4 before the final prompt/grammar changes
(`logs/qwen35-60s-walkthrough.json`). The final two-case native prompt check returned
answers in approximately48and39seconds. It correctly described the master's as
ongoing, but both outputs were rejected by the then-current finite grammar. The
comparison still asserted a general superiority premise; that remains rejected.
No claim of general accuracy improvement or passed promotion gate is made.

The prompt now includes two bounded examples forbidding credential verification
and Bachelor-status inference from an ongoing Master. The validator accepts a
narrow ongoing-degree explanation only when that source/owner is established.
A fallback ordering bug that repeated an advantage and omitted candidate2's
qualification was also fixed. Both regressions failed before correction
(`logs/qwen35-final-grammar-before.log`); the latest113focused tests passed
(`logs/qwen35-final-tests.log`). This is development refinement, not new held-out
or post-refinement live-generation evidence.

Full suite:2077passed,48skipped,19failed in238seconds
(`logs/qwen35-full-pytest.log`). Failures:2baseline-golden drift,11existing
extractive display/context expectations,6existing UI action/context expectations.
The failing extractor/UI modules and their tests are unchanged by this upgrade.
No new-model tests failed. The central record-check registry selected no records
for the changed inference paths (`logs/qwen35-affected-checks.log`: `{}`).
No cold training, research scoring, model fitting, push or weight commit occurred.

The model is a local experimental upgrade. On this4-core Intel Mac, factual
generation takes tens of seconds;60seconds is a hard shared turn bound. Checked
fallbacks remain necessary. Broader independent human review and reliable
freeform comparison quality remain open; activation does not close those gates.

28 Oct 2026: the trial was extended to macOS and Linux (x64, arm64) and to a separate Docker
image; see [`plan/2026-10-28-gguf-cross-platform/`](../2026-10-28-gguf-cross-platform/README.md).
