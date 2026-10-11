# French interview prototype implementation record

Status: notebook experiment implemented and executed; live pipeline/UI design remains
proposed pending approval. This record does not mark P9 or entity-weighted evaluation
complete. No training, final-set scoring or new annotation was performed.

## Measured demonstration

Source: the user-supplied French MP4, media SHA-256 in the plan and public aggregate
record. Decoded audio is 45.0026875 seconds at 16 kHz mono. Whisper tiny produced
seven French speech segments, taking 4.53 seconds for transcription. Marian
`Helsinki-NLP/opus-mt-fr-en` translated all seven with beam width 4 in 11.68 seconds.
Greedy decoding on the first five segments took 2.04 seconds; one top translation
differed from beam 4. These are single-run timings, not a controlled speed comparison
between beams, because their populations differ. Download/load took an additional
13.89 seconds for Whisper and 56.03 seconds for Marian. No reference-quality metrics
or confidence probabilities are reported.

GLiNER then processed the complete English transcript at threshold 0.5 with 32-word
overlap and exact original-text span checks. It took 2.74 seconds, returning
KNOWLEDGE=3, EXPERIENCE_YEARS=1, TITLE=0, DEGREE=0 and SKILL=0. Complete processing
does not imply complete or accurate extraction. Source text remains unreviewed.

The technical-stack sentence retained Python, Java and SQL. Other output was
incoherent: the introduction translated to “a genius in logic development”, and
the education sentence to “I have a computer tarpaulin…”. This demonstrates error
propagation from French ASR into translation and extraction; neither a degree nor
the intended title is established by this run. An applicant must correct the text
before it becomes scoring input. The numeric duration sanity check retained a
3-year expression but is not a semantic or total-experience check.

## Files and boundaries

- `multimodal/interview_demo.py`: bounded local audio decoding, timestamped French
  transcription, full English translation, greedy/beam subset comparison and
  provenance. Uses direct Marian tokenise/generate/decode as in the supplied notebook.
- `multimodal/entity_bridge.py`: translation-to-audio links, mandatory review
  confirmation, validated five-label extraction, and distinct unavailable/zero
  states. Applicant edits invalidate automatic audio alignment. Preview never
  returns a screening-ready document or a fit call.
- `multimodal/interview_entities.py`: measured existing GLiNER adapter on the
  unreviewed translation, publishing only counts/provenance.
- `notebooks/16-french-interview-translation.ipynb`: thin report of the actual
  records, safe examples, quality limits and intended integration.
- `output/multimodal/french-interview-demo.json`: measured aggregate/provenance.
- `output/multimodal/french-interview-preview.json`: three manually reviewed display
  excerpts without identity information. This review permits displaying excerpts;
  it does not confirm an applicant submission or certify linguistic correctness.

Full transcripts, entities and media are not committed. Local transcripts and entity
preview live under ignored `data/processed/multimodal/<media hash>/`. The notebook
redacts displayed excerpts inside its cells. It does not display the identity-bearing
introduction or embed the recording. No protected attribute, voice characteristic or
video-pixel feature reaches extraction/matching.

The project base environment stays classical; the existing isolated transformer
runtime supplies compatible torch/transformers, with ephemeral uv extras. The latest
Numba attempted an unsupported Intel-Mac source build; pinning Numba 0.60.0 resolved
setup. Runtime instructions are in `runtime/transformer/INTERVIEW_DEMO.md`.
The original inference-source SHA was not captured; the record does not retroactively
claim today's source produced it. Model hashes, resolved Marian revision, dependencies,
media digest and the entity-preview producing code hashes are recorded.

## Validation

Full suite once: `logs/multimodal-pytest.log`: 748 passed, 9 skipped, 3 failures.
Two failures are the previously observed frozen A1 baseline vocabulary/model record
mismatches. The third was the missing registry entry for the earlier entity-weighted
runner, corrected in `checks.py` after discovery. Narrow verification after the
registry/review-gate fixes: 29 passed (`logs/multimodal-final-focused.log`). Synthetic
tests cover direction, token preservation, unfinished generation, nonfinite/out-of-
range timestamps, mandatory confirmation and exact entity spans/audio links. Chunk-
budget and span-validation mutants were shown to fail their invariant tests.

Registry check: `checks --cold entity_weighted` passed its prerequisite/record audit
(`logs/multimodal-checks.log`); its record remains blocked with no retrieval scores.
This audit is not cold reproduction of a trained matching model. Cold reproduction
of all affected research records is still required before committing model changes
or claiming the older experiment complete. No model implementation is committed here.

## Remaining implementation and decisions

Approval of the accompanying plan is needed for the new branch/plan commit and live
text-only P9 integration, entity evidence UI and applicant upload/review workflow.
Pending choices: both standalone and supplementary interviews; proposed default
TF-IDF v2 with named experimental rankers; 20 MB/120-second limits; session-only
retention. The prototype does not depend on those product choices.
