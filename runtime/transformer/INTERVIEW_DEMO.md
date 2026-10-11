# Optional French interview prototype

Run from the repository root. The prototype uses the existing isolated GLiNER
runtime's torch/transformers versions. Ephemeral extras do not change its lockfile:

```bash
PYTHONPATH=src UV_CACHE_DIR=/tmp/candidate-uv-cache uv run --project runtime/transformer --with openai-whisper --with imageio-ffmpeg --with sacremoses --with 'numba==0.60.0' python -m candidate_screener.multimodal.interview_demo
```

Numba 0.60.0 has an Intel macOS wheel; newer releases attempted an unsupported
source build on the development machine. SentencePiece must be available in the
runtime for Marian tokenization. First execution downloads multilingual Whisper
tiny and `Helsinki-NLP/opus-mt-fr-en`. Media and full transcripts stay local.

The whole clip is translated with beam width 4. Greedy/beam comparison covers at
most the first five speech segments. The aggregate record separates inference
timing from loading/downloading and records model weights/revision hashes.

This is an explicitly bounded notebook experiment, not production ingestion.
English text requires applicant review before any future screening integration.
Video pixels, gender, voice quality and accent are never ranking features.

After translation, run the existing five-label GLiNER adapter on the unreviewed
English transcript as an exploratory evidence preview:

```bash
UV_CACHE_DIR=/tmp/candidate-uv-cache uv run python -m candidate_screener.multimodal.interview_entities
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/16-french-interview-translation.ipynb
```

The entity preview does not confirm an applicant submission or run screening.
If the translation prototype is rerun, regenerate its entity preview too.
