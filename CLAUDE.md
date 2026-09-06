# OCR Gauntlet contributor instructions

Standalone demo: plain lazy engine functions, shared benchmark runner, thin notebooks.
Python 3.12 + uv. Optional model SDKs must not become base dependencies.

- Preserve one JSONL row per planned engine/document, including failures/skips.
- Never grade incompatible outputs or unreviewed references silently.
- Keep raw outputs, exact model/configuration, hashes, completion and usage evidence.
- No implicit downloads, paid calls or credential loading in tests/notebook defaults.
- Reuse local pipelines; no service/database/registry layer is needed.
- Add a focused regression check for changed behavior, not mirrored implementation tests.
- Format with Ruff and run `uv run pytest`, `uv run ruff check .`.

See docs/evaluation.md for task metrics and the evaluation protocol.
