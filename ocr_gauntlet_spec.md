# OCR Gauntlet evaluation contract

The old implementation spec duplicated obsolete source and invented leaderboard
numbers. The authoritative implementation is now `src/ocr_gauntlet/`; see
[README](README.md) and [evaluation protocol](docs/evaluation.md).

One planned engine/document pair produces one append-only JSONL result. References
are hash-verified and explicitly reviewed. Failures, unsupported outputs and
unknown costs stay visible. A teaching smoke test never becomes a model ranking.
