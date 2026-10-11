# Text-derived screening and multilingual interview submission

Status: approved by the user on 11 October 2026; implementation in progress. The user requested this plan and a bounded
French-video notebook on 11 October 2026. The notebook is an explicitly requested
exploratory prototype; live pipeline/UI changes are authorized by this approval.
Existing uncommitted entity-weighted work is preserved, not silently adopted.

## Requirements and evidence

Read `docs/ownership/01_project_map.md` through `04_questions_for_owner.md`.
Their central product finding is verified against current code: `pipeline.py`
selects metadata + TF-IDF for demos, keyword equality produces Fit calls, uploads
use SBERT v1, and the five-label GLiNER adapter is absent from the live path.
P9 in `plan/2026-10-06-extraction-and-rerank/` remains unfinished. The ownership
documents are an audit, not permission to adopt a new research winner or reopen
annotation. D19, D42, D46, D47 and D53 continue to apply.

Requested outcome: applicants submit a written CV or an audio/video introduction
through a conversational interface; the backend produces reviewed English text,
extracts relevant entities and presents ranked, source-linked evidence to the
recruiter. The existing recruiter chat remains template based.

The supplied French MP4 is 45 seconds, 1280×720 H.264 with mono AAC audio,
852,197 bytes. Container-header inspection supplies these values, not FFprobe.
SHA-256: `5896fa7a0093e90d4ac242bf1ecfeb24fc829fe5258e3deaaabb37c44b3c351f`.
No transcript, translation reference or permission to publish the recording is
assumed. Media bytes and complete transcripts stay local and uncommitted.

`A0205189B_MarianMT_en_ROMANCE.ipynb` teaches tokenisation, attention masks,
`MarianMTModel.generate`, decoding, greedy/beam comparisons and n-best outputs.
Its English→French/ROMANCE checkpoints cannot translate French into English.
Use `Helsinki-NLP/opus-mt-fr-en`, with French ASR before text translation.

## Scope and delivery sequence

1. Deliver `16-french-interview-translation.ipynb` and reusable prototype code.
   Run the supplied 45-second clip once with a multilingual tiny Whisper model
   on CPU, explicitly French and `task=transcribe`. Compare Marian greedy and
   beam-4 decoding on at most five ASR segments; translate every segment with
   beam 4 for a complete local transcript. No training or neural ranking grid.
   Models/dependencies require a one-time download; no hosted transcription API.
   Record actual durations, versions, hashes, configuration and completion status.
   Exclude setup/download time from inference timings and report it separately.
2. After plan approval, create `feat/multimodal-entity-pipeline` and commit this
   plan separately. Do not commit unrelated unfinished model implementation.
3. Complete the text-only P9 adapter and entity evidence contract first.
4. Add media ingestion/review and applicant chat states, then source-aware
   recruiter answers. Synthetic tests precede model-dependent smoke tests.
5. Review, run the full suite once and applicable registered checks. D55 cold
   reproduction is mandatory before a model/training commit or completion claim.
   Missing prerequisites are recorded as missing, not a passed check.
6. Update README, P9 progress and implementation records. Update the source card,
   acquisition manifest and catalog together if this clip becomes a registered
   project data source; the prototype is user-supplied local media, not benchmark
   data. Preserve ownership audits as historical documents with dated follow-up.

## Pipeline design

`ApplicantDocument`: opaque applicant ID, English scoring text, source kind
(`written_cv`, `interview`, `combined`), source language, review state, provenance.
`MediaSegment`: segment ID, audio start/end seconds, French ASR text, English MT
text and optional applicant revision. Translation offsets belong to translated
text; they are never passed off as offsets in the French speech.

`EntityBundle`: five label collections (`KNOWLEDGE`, `SKILL`, `TITLE`, `DEGREE`,
`EXPERIENCE_YEARS`), validated text-relative offsets and original surfaces,
context status, completeness/failure state and model/runtime provenance. Distinguish
successful zero predictions from unavailable or partially processed extraction.
Cache by text digest, side, model revision/weights, labels, threshold, chunk/context
policy, vocabulary and extraction code. Never treat stale or partial cache as empty.

`screen(..., entity_extractor=None, jd_title=None)` consumes CV text and allowed
JD text/title only. Remove metadata-triggered ranking and Fit calls. Demo loading
uses `text_v2`; CV Position, keyword and experience metadata never enter inference.
Default draft baseline: TF-IDF v2 for a reproducible classical path; expose X2 and
entity-weighted X2 only as explicitly named experimental choices with required
weights and train-only IDF. SBERT v2 can be selected explicitly when available.
No silent scorer switch, implicit training or claim that stage 1 has been adopted.
Scores/percentiles are relative within a JD, not probabilities or three-class calls.
Complete P9's shared title normalisation and shared `text_v2` CV builder as focused
refactors with equivalence tests; do not maintain independent demo/benchmark builders.

GLiNER SKILL/KNOWLEDGE feed affirmative deduplicated concept evidence and the
already registered experimental scorer. TITLE and DEGREE remain contextual
evidence, not hard gates. EXPERIENCE_YEARS spans show duration claims; numeric
compatibility uses tested `experience.jd_experience` / `cv_experience` semantics.
Do not sum overlapping jobs, turn company age into applicant experience, or infer
zero from a missing span. Replace the UI's crude min/max duration display.
Context filtering prevents negated/learning mentions from being called matched.
Semantic support remains labelled a similarity-based suggestion, not proficiency.
Broad vocabulary repairs and fairness conclusions are separate registered work.

## Media and conversation design

Upload → validate type/size/duration/audio track → decode mono 16 kHz PCM →
French ASR with timestamps → token-budget-safe French→English Marian generation →
applicant reviews/edits transcript → confirmation → entity extraction → screening.

Proposed operational limits, requiring approval: audio/video 20 MB and 120 seconds,
one applicant recording at a time, French/English initially. Reject empty audio,
unsupported languages, unreadable containers and exhausted processing budgets with
actionable retry messages. Bound subprocess and model work; invoke FFmpeg with an
argument list, private temporary directory and no shell interpolation. Do not use
remote-media URLs. Decode audio only; pixels, apparent gender, voice quality and
accent never become ranking features.

Chat states: `uploaded`, `processing`, `transcribed`, `translated`, `reviewed`,
`ready`, `failed`. The applicant chooses standalone interview or CV supplement;
the unanswered mode preference is tracked below. UI offers side-by-side transcript
review, editable English text, download and deletion. Screening is enabled only
after explicit transcript confirmation. A short introduction may omit qualifications;
missing interview evidence is not proof of No Fit. In combined mode preserve
separate source sections and avoid credit for repeated skills across sources.

Recruiter questions such as “what supports this skill?” and “what experience was
stated?” cite reviewed English spans and their source segment/timestamps. Mark
translated evidence clearly; English output is not a verbatim French audio quote.
Replace filename display with anonymous applicant IDs. Redact every displayed
document and chat message; the current redactor does not remove names, so do not
claim complete anonymisation. Committed notebook outputs contain aggregate metrics
only unless a separately reviewed safe preview is available.

Default retention proposal: temporary uploads/audio deleted after processing;
raw and translated transcripts retained only in the session until applicant delete
or session end. A durable application database is out of this iteration's scope.

## Notebook design and acceptance

The notebook imports package code; no inline model definition or fitting. It reports
media provenance, method/direction, prerequisite status, actual ASR/MT segment counts,
greedy-versus-beam differences, runtime and the integration boundary. Full transcripts
are saved only under ignored `data/processed/multimodal/`. Every displayed text is
redacted inside the notebook; each text output is ≤5 records / 2,000 characters.
No audio/video embedding exposes personal data in committed outputs.

French transcription uses `transcribe`, not Whisper's direct English translation,
so Marian actually performs the translation. Tokenisation must not silently truncate
long segments; split oversized segments with mapping to source timestamps, preserving
every source token. Record incomplete processing and do not score partial transcripts.
Model log scores do not mean translation accuracy. The original English clip is not
an independently verified reference; no WER/BLEU/fit metric is reported from it.

For entity preservation, report extracted translated-text entities and explicitly
review duration numbers, technology spellings and negation against French segments.
If GLiNER is unavailable, show an explicit unavailable stage rather than substitute
another extractor under its name. This one-clip demonstration (n=1) proves an ingestion
path, not multilingual accuracy, fairness or improved retrieval.

## Validation and research boundary

Synthetic invariants: metadata edits leave inputs/scores unchanged; no metadata Fit
call; negation and repetitions add no affirmative coverage; exact entity offsets;
missing extraction differs from zero; company-age/overlapping-job duration guards;
French→English direction; segmentation/timestamps survive batching; no truncation;
empty/silent/oversized/unsupported media errors; no untranslated input enters English
scoring; review required; filenames/voice/video attributes cannot affect features;
explicit prerequisite/model failures. Show regression tests fail corresponding mutants.

Use injected ASR/MT/extractor/ranker doubles so tests need no data or model downloads.
Run one real clip smoke test separately, with provenance and timings. Register any new
`--check` in `checks.py`. Do not rescore the final set for this UI/ingestion prototype.
Recall@50/Recall@10 and within-family AUC require registered train-region experiments,
followed by frozen final evaluation. Three-class inference awaits an independently
labelled training protocol; no labels or new annotation wave are created here.

## Open questions

- Pending user preference: standalone spoken CV, supplement, or both. Proposed both
  with required applicant review; do not implement the dependent UI choice silently.
- Proposed default draft TF-IDF v2 plus named experimental options; this is a product
  configuration, not adoption of a research winner. Approve with this plan or amend.
- Proposed 20 MB / 120-second limits and session-only retention need approval.
- Prototype uses French tiny-Whisper ASR and Marian fr→en, no fine-tuning. Any expansion
  in languages, recordings, cloud services or compute requires a measured budget.
- Existing entity-weighted validation and missing role-model restoration remain
  outstanding; media integration does not resolve them or justify adoption.

## Primary references

- [MarianMT API](https://huggingface.co/docs/transformers/model_doc/marian)
- [French→English checkpoint](https://huggingface.co/Helsinki-NLP/opus-mt-fr-en)
- [Whisper transcription implementation](https://github.com/openai/whisper)
- The supplied Marian notebook and four ownership documents cited above.

## Approval resolutions

User approval accepts the proposed draft TF-IDF v2 default, both standalone and supplementary interview submissions, applicant review gate, 20 MB/120-second limits and session-only upload/transcript retention. A small Streamlit chatbox is explicitly requested as an illustration; registered research adoption remains separate.
