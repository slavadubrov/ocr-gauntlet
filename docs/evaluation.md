# Evaluation protocol

## Scope and architecture

`engines.py` owns optional SDK/pipeline calls and completion validation.
`utils.py` validates explicit manifests and input hashes. `metrics.py` owns
pure task metrics. `benchmark.py` executes a sequential plan and appends one
result per pair. `visualize.py` plots records. Notebooks contain no alternate
engine implementations, invented prices or precomputed winners.

Single-page images keep page denominators unambiguous. `run_docling()` also
accepts PDFs for conversion exploration, but the benchmark manifest deliberately
rejects multipage images and does not silently mix whole PDFs and pages. Render
PDF pages with a pinned renderer/DPI before importing; record that preprocessing.
Born-digital PDF text extraction is a separate baseline, not currently implemented.

## Dataset choices

| Dataset | Role | Implemented path / boundary |
|---|---|---|
| [CORD](https://github.com/clovaai/cord) | Real receipts, word geometry, semantic fields | Pinned streaming importer; separate visible words/semantic targets; reviewed region recipe |
| [IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) via [Teklia mirror](https://huggingface.co/datasets/Teklia/IAM-line) | Handwritten line recognition | Pinned optional importer; source research terms apply despite mirror MIT metadata |
| [OmniDocBench](https://github.com/opendatalab/OmniDocBench) | Heterogeneous pages, tables, formulas, order | Use the upstream evaluator for TEDS/CDM/block matching; this demo does not reimplement it |
| [FUNSD](https://guillaumejaume.github.io/FUNSD/) | Form entity/link extraction | Removed from automatic five-page download; concatenated word lists are not reviewed reading-order gold |

CORD pin: `7f0115a4b758a71d6473b8d085751692da2fef98`.
IAM pin: `fbdad97500ce54635c0d1ba306bf535cb40656cf`.
OmniDocBench dataset revision observed on 2026-09-06:
`aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec`. Pin the **evaluator commit as well** for
an official benchmark run; its matching/evaluation versions change independently.
The official evaluator handles structure in ways that CER cannot approximate.

For article-scale experiments, use a held-out, declared slice, report every
selected page, and stratify receipts/forms/handwriting/language/layout rather than
averaging unrelated tasks into an overall winner. Select pages before seeing
predictions. Keep development/calibration pages separate from final evaluation.
Public datasets may be in model training; this demo cannot establish absence of
contamination. A one-region check verifies integration, not generalization.

## Reference review

The importer saves a JSON reference and marks `reviewed=false`. Check the source
image, annotation coverage, reading order, punctuation, case and numeric strings.
Do not include semantic field names in transcription unless visible on the page.
For unreadable/redacted regions, use a declared crop or exclude the sample from
text scoring, retaining it for operational review. Do not invent a transcription.

After editing a reference, update its `reference_sha256` and add a review note;
set `reviewed=true` only after this review. Each resulting manifest is a new
protocol artifact. The original source revision/index/hash remains recorded.
`scripts/prepare_reviewed_cord.py` demonstrates this process for one fixed region;
it does not mark arbitrary downloaded receipts reviewed.

Custom manifests use schema version 1 and a nonempty `samples` array. Each sample
requires unique `id`, relative `image` and `reference` paths, their SHA-256 hashes,
`protocol`, and a boolean `reviewed`. Add dataset/revision/split/license and
preprocessing. References require `text` and normally `output_format: "text"`;
optional `fields`, `table` and `reading_order` declare additional targets.

## Metrics and output contracts

- Text uses NFC, case-sensitive whitespace normalization, CER/WER and exact match.
  CER/WER can exceed one. Nonempty predictions against empty gold have infinite
  error: persisted ratios use JSON null plus edit counts and
  `empty_reference_insertion=true`, never nonstandard JSON Infinity.
- Corpus error is total edit count divided by total reference units on scored
  rows. It is conditional; coverage and completion accompany it. Empty-reference
  insertions contribute edits when the corpus has nonempty references.
- All-planned text quality averages `1 - edit_distance/max(lengths,1)` over the
  planned denominator. Failure/skips/incompatible/unreviewed outputs contribute
  zero. It is a demo operational diagnostic, not a standard benchmark metric.
- Field exact match compares key/value pairs (including explicit null). Missing
  differs from absent. Extra keys reduce precision; nested values are exact values,
  not flattened CORD benchmark entities. No parsing from gold or substring search.
- Table scoring compares cells at exact row/column positions, including extra
  cells in the denominator. It is **not TEDS** and does not model merged-cell trees.
- Reading order uses pair order of unique, externally matched block IDs. Missing
  blocks lose affected pairs. This is not a detection/matching metric; extra
  blocks need a separate detection evaluation. Upstream OmniDocBench is preferred
  for full-page structural claims.
- ANLS is available for VQA answers only: best accepted reference, lowercased
  whitespace normalization, `distance < 0.5` cutoff, then average across questions.
  See [Biten et al., ICCV 2019, Eq. 2](https://openaccess.thecvf.com/content_ICCV_2019/html/Biten_Scene_Text_Visual_Question_Answering_ICCV_2019_paper.html).
  Public prose descriptions differ at equality; this implementation declares and
  tests the strict equation convention. Empty/empty is explicitly 1.

Markdown conversion output is retained as Markdown and needs a Markdown reference
for text comparison. Docling optionally supports `output_format="text"`: native
ordered text blocks and row-major table cells, with merged cells repeated. This
projection is declared in metadata and preserves raw Markdown/JSON. The currently
installed native `export_to_text()` ignores its deprecated strict-text option, so
the adapter does not use it. The scorer never strips syntax with a regex.
Docling exposes a single detected table as a cell grid for direct table scoring;
multiple tables require independent matching and remain native outputs. Docling
JSON/tables and Paddle native blocks remain available for independently matched
structured targets. A text adapter without a field/table/order prediction
reports that task as unsupported. The third notebook demonstrates how good CER
can coexist with a wrong total, wrong table row, or inverted block order.

`review_required` is a gold-based evaluation diagnostic, including unsupported
requested tasks. It is not a production confidence threshold. Confidence scores
from providers require calibration on a separate held-out set before auto-accept.

## Runtime and model identity

Docling standard OCR options belong to `PdfPipelineOptions`, installed on both
image/PDF format options. The VLM path uses `VlmPipeline` and an explicit
Granite-Docling model/revision. Constructors are tested against installed Docling
without downloading weights. Tesseract, PaddleOCR and the standard Docling pipeline
were also exercised locally; other optional inference paths need deployment-specific checks.
Converters are initialized once and cached; `init_ms` records model startup,
`latency_ms` conversion plus export. Cache hits have zero initialization time.
This demo runs sequentially; GPU throughput needs a separate concurrency study.

Paddle selects `PP-OCRv5`, `lang=en`, orientation/unwarping off. It uses the current
3.x `predict()` API and saves the native result with boxes and recognition scores.
It is a recognition baseline, distinct from PP-Structure/PaddleOCR-VL. See the
[official 3.x quick start](https://www.paddleocr.ai/v3.3.0/en/quick_start.html).
Weight caches are not automatically proven immutable; record hashes of deployed
weights for published performance comparisons, as well as runtime versions.

For dots.ocr the client requires `model_revision` and `runtime` in settings. The
original model pin verified on 2026-09-06 is
`c0111ce6bc07803dbc267932ffef0ae3a51dc951`. The maintained
[official repository](https://github.com/studio-dots-ai/dots.ocr) now also documents
dots.mocr and its distinct parsing prompts. Follow the deployment instructions
for the **exact model** you serve; record the model SHA and runtime image digest.
A generic plain-text prompt here does not reproduce its published parsing score.
No minimum VRAM is claimed: record measured peak memory, dtype, image resolution,
context length and batch size on the actual server. A language-component parameter
count must not be presented as total model memory.

Gemini uses `google-genai`, exact requested/returned identities, high media
resolution and low thinking by default. Both can affect speed/cost. The
[dated standard paid-tier estimate](https://ai.google.dev/gemini-api/docs/pricing)
is $0.75/M input and $3.75/M output including thinking through 2026-12-31;
the currently published 2027 rates are $1.50/$7.50. Free-tier billing, discounts,
caching and tools are not assumed. Unknown models/usage have unknown estimates.
Mistral OCR 4.1 uses its [published $4/1,000-page rate](https://docs.mistral.ai/models/ocr-4-1)
for the standard unannotated request; other models are not priced by analogy.
These are estimates, never invoices. Reverify prices before paid experiments.

Remote timeouts are bounded, implicit retries disabled. A transport exception can
have incurred spend; the runner keeps cost unknown. Truncation/refusal/empty outputs
retain response usage and raw data but are not successful pages. Local API fee
zero excludes hardware, energy, staffing and review. Monthly planning separately
models provisioned hours, utilization, capacity, attempts/page, completion and
review fraction. `$0.50 × 24 × 30 = $360` before other costs.

## Reproducibility and publication

Use `uv sync --locked`, preserve the manifest and JSONL, and record the Git diff
when `git_dirty=true`. Publish a fresh run only with its actual IDs and settings.
No automatic cloud upload happens in tests. Raw outputs can contain document text;
keep private results local. The old misleading notebook outputs remain only in
Git history and are not reused as measured evidence.
