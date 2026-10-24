# GLiNER medium v2.5 runtime

This independent uv environment preserves the running small-v2.1 runtime. It pins
GLiNER 0.2.21, Transformers 4.48.3, Torch 2.2.2 and spaCy 3.8.7; the lock records
all transitive versions. Click 8.1.8 is pinned explicitly because spaCy's CLI
imports it while the resolved Typer distribution does not supply it transitively.
Sentence segmentation uses `spacy.blank("en")` and the
sentencizer, so no statistical spaCy model is downloaded.

```sh
uv sync --project runtime/gliner-medium
PYTHONPATH=src uv run --project runtime/gliner-medium --no-sync python -m candidate_screener.extraction.medium_assets
PYTHONPATH=src uv run --project runtime/gliner-medium --no-sync python -m candidate_screener.extraction.medium_assets --verify
```

The acquisition command downloads only the float32 `pytorch_model.bin`, its
configuration, supplied tokenizer assets and model card at revision
`88c3b98b57ad5e7d66fb209ed61c53f4b1fd05da`. The encoder's configuration is pinned
separately; backbone weights are unnecessary because the GLiNER checkpoint
includes them. Every artifact has a SHA-256 in the ignored local model manifest.

Offline loading uses the checkpoint tokenizer, checks its entity token and
vocabulary, and loads every state entry strictly. The DeBERTa backbone config
reserves 128100 embeddings whereas this checkpoint has 128003; the loader sets
the constructed encoder vocabulary to the verified checkpoint size before loading.
Runtime package versions are recorded separately for campaign provenance.
