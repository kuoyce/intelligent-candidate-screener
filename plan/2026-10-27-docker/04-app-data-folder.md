# 04 — The Docker data folder (`data/datapack/`)

**Decided (owner, 27 Oct 2026, third round):**

- Development stays **local and unchanged**: native runs keep reading `data/`, experiments
  included. Docker and the data folder are updated only once a change is settled.
- `data/datapack/` is **the Docker app's data folder**: the project's own files (copied from the
  local `data/`) plus the Docker deployment's workspace (accounts, reviews) and caches.
- **Pretrained model weights are downloaded when the image is built** (no fine-tuning involved):
  RoBERTa-SQuAD2, GLiNER-medium, Qwen2.5-0.5B and C2. They are not in the data folder.
- Upgrades: purge and rebuild; keep the workspace when the change is not breaking; on a breaking
  change purge it too, optionally copying the local workspace again (`create --with-workspace`).

This is option A of the earlier comparison (B, which switched native runs to the folder, is
dropped), with the model weights moved into the image.

## 1. What goes where

**In the image** (downloaded at build time at pinned revisions; found through
`SCREENER_MODELS_ROOT=/opt/models`, plus the Hugging Face cache for C2):

| Model | Used for | Size | Fine-tuned by the project? |
|---|---|---:|---|
| RoBERTa-SQuAD2 | v1 chat answers (extractive) | ~0.5 GB (est.) | no |
| GLiNER-medium v2.5 | entity comparison | ~0.8 GB (est.) | no |
| Qwen2.5-0.5B | v2/v3 chat wording | 954 MB | no |
| Qwen3-Reranker-0.6B (C2) | D64 "Refine the top 50" | 1.2 GB | no (zero-shot) |

**In the data folder** (sizes on this machine; step 1's run-time audit confirms the list):

| Item | Path under `data/datapack/` | Size | Copied from |
|---|---|---:|---|
| CSV-backed benchmark reviews | `processed/phase2-rerank-output/a2-{cv-extraction,jd-extraction,x2-scores}.csv` | 0.25 MB | committed `output/matching/` (identical bytes) |
| Source text; the `a2-final-cvs` pool | `processed/matching/a2-pairs-v2-539ddf32117c.parquet` | 0.6 MB | `data/` |
| A2 train-region CVs (background pool) and JDs | `processed/matching/a2-train-{cv,jd}-v2-20000-0.parquet` | 35 MB | `data/` |
| Role-family model (**trained by the project**) | `processed/extraction/role-family/` | 123 MB | `data/` |
| SBERT weak (**fine-tuned by the project**) | `processed/matching/sbert-weak-3255167debe9/` | 88 MB | `data/` |
| Inference parts (fitted) | `processed/inference/parts/` | 0.1 MB | `data/` |
| Skill vocabulary | `vocab/` | 1.3 MB | committed `data/vocab/` |
| `pack-manifest.json` | the folder's root | — | written by `create` |

Inputs total about 250 MB. The app also writes into the folder while it runs:

| Written by the app | Path | Kept on upgrade? |
|---|---|---|
| Workspace: accounts, saved reviews and chat history, JD drafts, mail captures, entity cache | `processed/hr-portal/` | **yes**, unless the change is breaking |
| Caches: SBERT embeddings, C2 scores, concept and title caches | `processed/embeddings/`, `processed/rerank/cross/`, `processed/extraction/{concept-cache,jd-title}/` | no: cleared on refresh, rebuilt on demand (the first ranking after a refresh takes ~18 s longer) |

## 2. How it works

`pack-manifest.json` lists each input file with its size and SHA-256, where it came from, the
fingerprints the code expects (e.g. `3255167debe9` for SBERT weak), the source commit and the date.
Four commands (`python -m candidate_screener.datapack …`), all run **by hand**, never at app start:

| Command | What it does |
|---|---|
| `create` | (Re)builds the inputs from a source: the local `data/` (following its symlinks) or a received handover copy (`--source`). Replaces old inputs, clears the caches, **keeps an existing workspace**. Writes the manifest |
| `create --with-workspace` | Also copies the source's `processed/hr-portal/` (the owner's local accounts and reviews). Refuses to overwrite a workspace already in the folder unless `--replace-workspace` |
| `verify` | Compares the folder with its manifest and with what the code expects; prints every missing, changed or stale file, and the workspace's format version where the code records one. Run after an update, not at start-up |
| `handover DEST` | A copy for passing on privately: inputs and manifest only, **no workspace, no caches**; verifies the copy |
| `purge --yes` | Empties the folder (breaking changes). Without `--yes` it only lists what it would delete |

The owner can run them natively (`uv run python -m candidate_screener.datapack create …`, main
environment only: file copies, no torch) or in Docker (`docker compose run --rm tools create …`).

## 3. Who runs what, and upgrades

`create` always does the same thing: it fills `data/datapack/` from a **source folder that has the
project's files**. Only the source differs. Anyone with their own full `data/` (the owner today)
just runs `create --with-workspace` and never needs a handover. Someone who cloned the repository
has an empty `data/` (git-ignored; the trained models come from runs on the owner's machine), so
they get the project's files from a copy made with `handover`, which leaves out the owner's
accounts, reviews and caches.

| Person | First time | Upgrade, not breaking | Upgrade, breaking |
|---|---|---|---|
| Has a full local `data/` (owner) | `create --with-workspace` (files plus local accounts and reviews) | `create` (fresh files and caches; the Docker accounts and reviews are kept) | `purge --yes`, then `create --with-workspace` (or `create` to start empty) |
| Has no project data (e.g. a new maintainer) | `create --source <received copy>` | receive a new copy; `create --source <new copy>` (their accounts and reviews are kept) | `purge --yes`, then `create --source <new copy>` |

Every upgrade also rebuilds the image (`git pull`, `docker compose build`; remove the old one with
`docker compose down --rmi local`), runs `verify`, then `docker compose up -d`. Who decides
"breaking": the developer, when the change is settled, and says so in the commit and the plan
record. `verify` helps by naming stale inputs and an unexpected workspace format.

## 4. Code changes this needs

- None for the project's data: the folder is mounted where the code already looks (`/app/data`).
- The model folders move out of `data/` in the image, so `config.py` gains
  `MODELS = SCREENER_MODELS_ROOT or PROCESSED` (about five lines; native default unchanged), and
  three asset modules build their folders from it: `llm/qa_assets.py`,
  `extraction/medium_assets.py`, `llm/qwen_assets.py` (which today spells
  `data/processed/chatbot/qwen` out). Every other path in the 141 modules the app can load already
  derives from `config.DATA`, so nothing else moves (static survey, 27 Oct 2026).
- Because `config.py` is read by every record, the check runner selects every check once
  (~70 min warm, approved by the owner on 27 Oct 2026). With the variable unset nothing changes.
