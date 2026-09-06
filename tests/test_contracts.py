"""Regression checks for comparison validity; no external OCR calls or weights."""

import json
import math
import runpy
from datetime import date
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
from PIL import Image

from ocr_gauntlet import engines
from ocr_gauntlet.benchmark import run_benchmark, score_output, summarize
from ocr_gauntlet.metrics import (
    anls,
    cer,
    cost_scenario,
    field_metrics,
    normalize_text,
    reading_order_accuracy,
    table_metrics,
    wer,
)
from ocr_gauntlet.utils import load_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_metric_boundaries_and_task_separation():
    assert cer("abc", "abc") == wer("a b", "a b") == 0
    assert cer("abc", "a") == 2
    assert math.isinf(cer("x", "")) and math.isinf(wer("x", ""))
    assert anls("abcde", "abcxx") == pytest.approx(0.6)
    assert anls("abcd", "abxx") == 0  # distance exactly .5
    assert anls("abcde", "abxxx") == 0
    assert anls("HELLO", ["other", "hello"]) == 1
    assert anls("", "") == 1 and anls("x", "") == 0
    with pytest.raises(ValueError):
        anls("x", [])
    assert normalize_text(" E\u0301  A\n") == "É A"
    assert not field_metrics({}, {"tax": None})["all_fields_exact"]
    assert field_metrics({"tax": None}, {"tax": None})["all_fields_exact"]
    assert field_metrics({"x": "1", "extra": "2"}, {"x": "1"})["field_precision"] == 0.5
    assert not table_metrics([["1", "2"]], [["2", "1"]])["table_exact"]
    assert reading_order_accuracy(["b", "a"], ["a", "b"]) == 0
    assert reading_order_accuracy(["a"], ["a", "b"]) == 0
    assert cost_scenario(1000, hourly_usd=0.50)["monthly_usd"] == 360
    assert not cost_scenario(1_000_000, hourly_usd=0.5)["capacity_sufficient"]
    with pytest.raises(ValueError):
        cost_scenario(1, utilization=0)


def test_cord_never_serializes_semantic_keys_into_gold():
    reference_for = runpy.run_path(str(ROOT / "scripts/download_samples.py"))[
        "reference_for"
    ]
    ref = reference_for(
        "cord",
        {
            "ground_truth": json.dumps(
                {
                    "gt_parse": {"total_price": "8.00"},
                    "valid_line": [{"words": [{"text": "Total"}, {"text": "8.00"}]}],
                }
            )
        },
    )
    assert ref["text"] == "Total 8.00"
    assert ref["fields"] == {"total_price": "8.00"}
    with pytest.raises(ValueError):
        reference_for("cord", {"ground_truth": '{"gt_parse": {}}'})


def test_failure_denominators_and_hashes(tmp_path):
    make = runpy.run_path(str(ROOT / "scripts/make_fixture.py"))["make_fixture"]
    manifest = make(tmp_path / "data")
    calls = []

    def runner(name, sample, settings):
        calls.append(name)
        if sample["id"] == "letter":
            return "partial", {
                "status": "error",
                "reason": "truncated",
                "cost_usd": 0.2,
                "output_format": "text",
            }
        return sample["target"]["text"], {
            "status": "success",
            "cost_usd": 0.1,
            "output_format": "text",
        }

    output = tmp_path / "results.jsonl"
    rows = run_benchmark(manifest, ["tesseract", "gemini"], output, runner=runner)
    assert len(rows) == 4 and calls == ["tesseract", "tesseract"]
    summary = summarize(rows)
    assert summary[0]["completion_rate"] == 0.5
    assert summary[0]["all_planned_text_quality"] == 0.5
    assert summary[0]["usd_per_success"] == pytest.approx(0.3)
    assert summary[1]["attempted"] == 0 and summary[1]["completion_rate"] == 0
    assert len(output.read_text().splitlines()) == 4
    rows[1]["cost_usd"] = None
    assert summarize(rows)[0]["usd_per_success"] is None
    with pytest.raises(FileExistsError):
        run_benchmark(manifest, ["tesseract"], output, runner=runner)
    sample = load_manifest(manifest)[0]
    meta = {"status": "success", "output_format": "markdown"}
    assert score_output("# OCR DEMO", meta, sample)["evaluation_status"] == "not_scored"
    sample["reviewed"] = False
    assert (
        score_output("x", {**meta, "output_format": "text"}, sample)[
            "evaluation_status"
        ]
        == "not_scored"
    )
    Path(sample["reference"]).write_text("tampered")
    with pytest.raises(ValueError, match="Hash mismatch"):
        load_manifest(manifest)


def test_exception_and_empty_results_are_persisted(tmp_path):
    make = runpy.run_path(str(ROOT / "scripts/make_fixture.py"))["make_fixture"]
    manifest = make(tmp_path / "data")

    def runner(name, sample, settings):
        if sample["id"] == "receipt":
            raise RuntimeError("SECRET must not be written")
        return "", {"status": "success", "cost_usd": 0.03}

    output = tmp_path / "out.jsonl"
    rows = run_benchmark(manifest, ["tesseract"], output, runner=runner)
    assert all(r["status"] == "error" for r in rows)
    assert "SECRET" not in output.read_text()
    assert rows[1]["cost_usd"] == 0.03


def test_gemini_price_identity_and_thinking():
    usage = {
        "prompt_token_count": 1000,
        "candidates_token_count": 100,
        "thoughts_token_count": 200,
    }
    cost, basis = engines.gemini_cost(engines.GEMINI_MODEL, usage, date(2026, 9, 6))
    assert cost == pytest.approx((1000 * 0.75 + 300 * 3.75) / 1_000_000)
    assert basis["model"] == "gemini-3.8-flash"
    assert engines.gemini_cost("other", usage)[0] is None
    assert engines.gemini_cost(engines.GEMINI_MODEL, None)[0] is None
    assert engines.gemini_cost(engines.GEMINI_MODEL, usage, date(2027, 1, 1))[
        0
    ] == pytest.approx(cost * 2)


def test_gemini_real_sdk_response_contract(monkeypatch):
    genai = pytest.importorskip("google.genai")
    from google.genai import types

    response = types.GenerateContentResponse(
        model_version="gemini-3.8-flash-version-test",
        response_id="test-id",
        candidates=[
            types.Candidate(
                finish_reason="STOP",
                content=types.Content(parts=[types.Part(text="Hello")]),
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100, candidates_token_count=10, thoughts_token_count=20
        ),
    )
    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content.return_value = response
    monkeypatch.setattr(genai, "Client", lambda **kwargs: client)
    text, meta = engines.run_gemini(
        Image.new("RGB", (5, 5)), api_key="stub", media_resolution="medium"
    )
    assert text == "Hello" and meta["status"] == "success"
    assert meta["model_returned"] == response.model_version
    assert meta["request_id"] == "test-id"
    call = client.models.generate_content.call_args.kwargs
    assert call["model"] == "gemini-3.8-flash"
    assert call["config"].media_resolution.value == "MEDIA_RESOLUTION_MEDIUM"
    response.candidates[0].finish_reason = types.FinishReason.MAX_TOKENS
    assert (
        engines.run_gemini(Image.new("RGB", (5, 5)), api_key="stub")[1]["status"]
        == "error"
    )
    response.candidates = []
    assert (
        engines.run_gemini(Image.new("RGB", (5, 5)), api_key="stub")[1]["status"]
        == "error"
    )
    with pytest.raises(ValueError):
        engines.run_gemini(
            Image.new("RGB", (5, 5)), api_key="stub", media_resolution="ignored"
        )


def test_docling_real_constructor_without_models(monkeypatch):
    module = pytest.importorskip("docling.document_converter")
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        TesseractCliOcrOptions,
    )
    from docling.pipeline.vlm_pipeline import VlmPipeline

    initialized = []
    monkeypatch.setattr(
        module.DocumentConverter,
        "initialize_pipeline",
        lambda self, fmt: initialized.append(fmt),
    )
    engines._docling_converter.cache_clear()
    t = engines._docling_converter("tesseract", False)
    e = engines._docling_converter("easyocr", False)
    v = engines._docling_converter("tesseract", True)
    for fmt in [InputFormat.IMAGE, InputFormat.PDF]:
        assert isinstance(
            t.format_to_options[fmt].pipeline_options.ocr_options,
            TesseractCliOcrOptions,
        )
        assert isinstance(
            e.format_to_options[fmt].pipeline_options.ocr_options, EasyOcrOptions
        )
        assert v.format_to_options[fmt].pipeline_cls is VlmPipeline
    assert engines._docling_converter("tesseract", False) is t
    assert len(initialized) == 6
    with pytest.raises(ValueError):
        engines._docling_converter("wrong", False)
    engines._docling_converter.cache_clear()


def test_docling_partial_table_failure(monkeypatch):
    table = NS(export_to_dataframe=MagicMock(side_effect=ValueError("broken table")))
    doc = NS(
        tables=[table],
        export_to_markdown=lambda: "text",
        export_to_dict=lambda: {},
        iterate_items=lambda: [(NS(text="text"), 0)],
    )
    result = NS(document=doc, errors=[], status="success")
    converter = NS(convert=lambda *args, **kwargs: result)
    wrapped = MagicMock(return_value=converter)
    wrapped.cache_info.return_value = NS(hits=0)
    monkeypatch.setattr(engines, "_docling_converter", wrapped)
    text, meta = engines.run_docling("test.png")
    assert meta["status"] == "error" and "table export" in meta["reason"]
    assert meta["tables_found"] == 1 and meta["tables"] == [] and text == "text"


def test_docling_placeholder_is_not_success(monkeypatch):
    doc = NS(
        tables=[],
        export_to_markdown=lambda: "<!-- image -->",
        export_to_dict=lambda: {},
        iterate_items=lambda: [],
    )
    wrapped = MagicMock(
        return_value=NS(
            convert=lambda *a, **k: NS(document=doc, errors=[], status="success")
        )
    )
    wrapped.cache_info.return_value = NS(hits=0)
    monkeypatch.setattr(engines, "_docling_converter", wrapped)
    assert engines.run_docling("test.png")[1]["status"] == "error"


def test_docling_text_and_table_projection(monkeypatch):
    grid = [[NS(text="Item"), NS(text="Price")], [NS(text="Tea"), NS(text="8")]]
    table = NS(
        data=NS(grid=grid),
        export_to_dataframe=lambda **kw: NS(to_json=lambda **kw: "{}"),
    )
    doc = NS(
        tables=[table],
        export_to_markdown=lambda: "|Item|Price|",
        export_to_dict=lambda: {},
        iterate_items=lambda: [(NS(text="Receipt"), 0), (table, 0)],
    )
    wrapped = MagicMock(
        return_value=NS(
            convert=lambda *a, **k: NS(document=doc, errors=[], status="success")
        )
    )
    wrapped.cache_info.return_value = NS(hits=0)
    monkeypatch.setattr(engines, "_docling_converter", wrapped)
    text, meta = engines.run_docling("test.png", output_format="text")
    assert text == "Receipt\nItem Price\nTea 8"
    assert meta["table"] == [["Item", "Price"], ["Tea", "8"]]
    assert meta["markdown"] == "|Item|Price|" and meta["status"] == "success"


def test_paddle_predict_retains_native_blocks(monkeypatch):
    pytest.importorskip("numpy")
    data = {"res": {"rec_texts": ["Hello", "world"], "rec_scores": [0.8, 0.9]}}
    wrapped = MagicMock(return_value=NS(predict=lambda _: [NS(json=data)]))
    wrapped.cache_info.return_value = NS(hits=0)
    monkeypatch.setattr(engines, "_paddle", wrapped)
    text, meta = engines.run_paddle(Image.new("RGB", (5, 5)))
    assert text == "Hello\nworld" and meta["blocks"][0]["rec_scores"] == [0.8, 0.9]


def test_mistral_preserves_incomplete_spend(monkeypatch):
    mistralai = pytest.importorskip("mistralai")
    from mistralai.models import OCRResponse

    response = OCRResponse(
        model="mistral-ocr-4-1", pages=[], usage_info={"pages_processed": 1}
    )
    client = MagicMock()
    client.__enter__.return_value = client
    client.ocr.process.return_value = response
    monkeypatch.setattr(mistralai, "Mistral", lambda **kwargs: client)
    _, meta = engines.run_mistral_ocr(Image.new("RGB", (5, 5)), api_key="stub")
    assert meta["status"] == "error" and meta["cost_usd"] == 0.004
    assert client.ocr.process.call_args.kwargs["model"] == "mistral-ocr-4-1"
    assert client.ocr.process.call_args.kwargs["retries"] is None


def test_dots_truncation_and_identity(monkeypatch):
    openai = pytest.importorskip("openai")
    from openai.types.chat import ChatCompletion

    response = ChatCompletion(
        id="stub",
        created=0,
        model="served-model",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "length",
                "message": {"role": "assistant", "content": "partial"},
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    client = MagicMock()
    client.__enter__.return_value = client
    client.chat.completions.create.return_value = response
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    text, meta = engines.run_dots_ocr(
        Image.new("RGB", (5, 5)), model_revision="abc", runtime="vllm-test"
    )
    assert text == "partial" and meta["status"] == "error"
    assert (
        meta["model_returned"] == "served-model" and meta["usage"]["total_tokens"] == 15
    )
    with pytest.raises(ValueError, match="model_revision"):
        engines.run_dots_ocr(Image.new("RGB", (5, 5)))


def test_adapter_metadata_cannot_replace_sample_identity(tmp_path):
    make = runpy.run_path(str(ROOT / "scripts/make_fixture.py"))["make_fixture"]
    manifest = make(tmp_path / "data")

    def runner(name, sample, settings):
        return "text", {
            "status": "success",
            "output_format": "text",
            "engine": "wrong",
            "document": {"native": True},
        }

    rows = run_benchmark(
        manifest, ["tesseract"], tmp_path / "rows.jsonl", runner=runner
    )
    assert [r["document"] for r in rows] == ["receipt", "letter"]
    assert all(r["engine"] == "tesseract" for r in rows)
