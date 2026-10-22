# Model bundle and hosting: the reported system's weights, portable and fetchable

Status: **draft, on hold (owner, 22 Oct 2026): pending the new chatbot a teammate is building.**
Not approved; nothing implemented. Resume once the chatbot's host, entry point and system choice
are known (open questions below).

## Why

The owner wants to be able to hand the application over, for example hosted on Streamlit from
GitHub. Today the reported system (D64: L1 on F12, trained on `chat-3`, ranks the 500-CV pool, and
its top 50 are reordered by L1 + 0.25 × C2) cannot run anywhere but this machine:

- its fitted parts live under `data/`, which git ignores;
- one part is a 123 MB Python pickle, which breaks across library versions, is unsafe to load from
  someone else, and exceeds GitHub's 100 MB file limit;
- two feature parts are **refitted at start-up** from the raw Djinni data and the dev pools
  (`a1_weights.build_scorer` calls `dev_pools.build()`, a > 4.5 GB step);
- the recruiter app (`app/streamlit_app.py` via `pipeline.screen`) runs its own TF-IDF draft ranker,
  not the reported system.

## Why it waits for the chatbot

The chatbot (design spec `plan/2026-09-26-top-down-reset/02-design-spec.md` §8, the conversational
UI; a teammate is building a new one) decides what the bundle must serve:

- the entry point and the scoring interface it calls;
- which system it runs: D64 (L1 + C2) or D63 (L1 alone);
- the host, hence the RAM and CPU budget, which decides whether C2 can run there at all;
- whether it brings its own models or API keys (secrets handling).

## Inventory (measured 22 Oct 2026)

| Part | Used for | Now | Size | Portable form |
|---|---|---|---|---|
| L1 (chat-3) | stage 2 | refitted from chat-3 features at run time | a few KB | JSON: columns, means, SDs, coefficients, intercept |
| Role-family model (`role-model-<fp>.pkl`) | X2's `role_match` (F12) | pickle under `data/processed/extraction/role-family/` | 123 MB | arrays (npz) + JSON, or ONNX |
| SBERT weak (`sbert-weak-<fp>/`) | the `sbert` feature (F12) | sentence-transformers folder under `data/processed/matching/` | 88 MB | safetensors + config and tokenizer (as is) |
| TF-IDF (`lexical.Tfidf("v2")`) | `tfidf_scaled` (F12) | refitted at start-up from A2 train text | small once fitted | vocabulary + IDF (npz / JSON) |
| Concept document frequencies (`DocumentFrequency`) | coverage features (F12) | refitted at start-up from 20k train CVs minus dev-pool CVs | small once fitted | parquet / JSON |
| Concept vocabulary and extraction rules | F12 | code and vocabulary files | small | as is (check where the vocabulary file lives) |
| C2, Qwen3-Reranker-0.6B | D64's reorder of the top 50 | Hugging Face download, revision pinned in `rerank/cross.py` | 1.2 GB (bf16 on disk; ~2.5 GB RAM in float32) | not bundled: Qwen's repo at the pinned revision |

## Design (to confirm when resumed)

1. **Export.** `candidate_screener.bundle --export` writes one versioned folder: `l1.json`,
   `role_model/`, `sbert_weak/`, `tfidf.npz`, `concept_df.parquet`, and `manifest.json` (per file:
   path, SHA-256, bytes; plus the source commit, Python and library versions, the feature-set and
   config fingerprints, and C2's repo and revision). No pickle, no CV or JD text.
2. **Load.** `bundle.load(path)` returns a scorer implementing `matching.base.System` that rebuilds
   F12 and L1 (and C2, if enabled) from the bundle alone: no raw dataset, no dev pools.
3. **Host the bundle on the Hugging Face Hub** (a model repo, private by default), pinned by commit
   hash in a committed `bundle-manifest.json`. Fallback: GitHub Releases (files < 2 GB). Not git and
   not Git LFS (LFS bandwidth quotas; Streamlit Cloud re-clones the repo).
4. **Fetch in the app.** `huggingface_hub.snapshot_download(repo, revision=…)` at start-up, every
   file's SHA-256 checked against the committed manifest, cached with `st.cache_resource`; a token
   in Streamlit secrets if private.
5. **C2 on a CPU host** (owner's decision, below). C2 costs ~5-7 s per pair even on the project
   laptop, so ~5 min per JD for 50 pairs, slower on a shared CPU. Options:
   - D63's L1 alone by default, C2 as an opt-in toggle;
   - C2 scores pre-computed for the demo data;
   - C2 on a hosted GPU endpoint.
6. **Dependencies.** `uv.lock` pins every version; Python 3.12. Check the host honours the CPU torch
   index in `pyproject.toml` (otherwise it installs the ~2 GB CUDA build); fallback `uv export` to a
   `requirements.txt` with the CPU index URL.

## Gates before any upload

- **Privacy (AGENTS data rule 4).** A TF-IDF or role-model vocabulary fitted on CVs can hold names,
  emails or phone numbers as tokens. Filter the exported vocabularies (drop anything
  `annotation.redact` would mask, and rare tokens below a document-frequency floor), or switch those
  parts to a hashed vectoriser; a test fails if an exported token matches a redaction pattern.
- **Licence.** The role model and SBERT weak are trained on Djinni data; check the dataset's terms
  (`docs/data/cards/`) before any public upload. Private by default.

## Verification

- **Round trip, registered as a `--check`.** The system loaded from the bundle reproduces the
  committed records exactly: Recall@10 0.484413 for L1 over the pools (`chat3-l1-pool.json`) and the
  D64 test figures (`c2-validation-test.json`), on the 60 final-set JDs.
- **Cross-platform smoke test.** Load the bundle on a clean machine (Linux x86 and macOS ARM, or a
  container) and score one demo JD; deterministic parts must match exactly, SBERT and C2 within a
  small tolerance (re-scoring moves their last bits; see the D55 cold-run record of 21 Oct 2026).
- **Unit tests (no data):** manifest and hash checks, the PII filter, the loader on a tiny synthetic
  bundle.

## Open questions (owner and teammate)

1. The chatbot's host (Streamlit Community Cloud or other) and its RAM / CPU limits.
2. Which system it calls: D64 (L1 + C2) or D63 (L1 alone); and the C2 policy on a CPU host.
3. Public or private weights (licence, privacy).
4. Whose Hugging Face account or organisation holds the bundle.
5. Does the chatbot need other models or API keys, and how are secrets passed?

## Cost (estimate)

About a day of agent work for the export, loader, checks and tests. The bundle is ~220 MB plus C2's
1.2 GB from Qwen's repo. No training; D63 and D64 do not change.
