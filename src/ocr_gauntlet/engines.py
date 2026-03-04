"""
OCR engine wrappers. Each function takes a PIL Image and returns:
    (extracted_text: str, metadata: dict)

metadata contains: latency_ms, engine_name, and engine-specific fields
(tokens, cost_estimate, etc.)
"""

from __future__ import annotations
import base64
import io
import os
import time
from pathlib import Path
from PIL import Image


# ─────────────────────────────────────────────
# Tier 1: Tesseract (Traditional, CPU, Free)
# ─────────────────────────────────────────────


def run_tesseract(image: Image.Image, lang: str = "eng") -> tuple[str, dict]:
    """Run Tesseract OCR on a PIL Image.

    Requires: pip install pytesseract
    System dep: tesseract-ocr (apt install tesseract-ocr / brew install tesseract)
    """
    import pytesseract

    start = time.perf_counter()
    text = pytesseract.image_to_string(image, lang=lang)
    elapsed = (time.perf_counter() - start) * 1000

    return text.strip(), {
        "engine": "Tesseract",
        "latency_ms": round(elapsed, 1),
        "cost_per_1k_pages": 0.0,
    }


# ─────────────────────────────────────────────
# Tier 2: Docling + Tesseract (Framework, CPU, Free)
# ─────────────────────────────────────────────


def run_docling(
    image_path: str | Path,
    ocr_backend: str = "tesseract",
    use_vlm: bool = False,
) -> tuple[str, dict]:
    """Run Docling document converter on an image or PDF.

    Requires: pip install docling

    Returns markdown text (Docling's native structured output).
    Docling adds layout understanding, reading order, and table structure
    on top of the raw OCR engine.

    Args:
        image_path: Path to image or PDF file.
        ocr_backend: "tesseract" or "easyocr".
        use_vlm: If True, use Granite-Docling-258M VLM mode (~8GB VRAM).
    """
    from docling.document_converter import DocumentConverter, ImageFormatOption

    start = time.perf_counter()

    if use_vlm:
        converter = DocumentConverter()
    else:
        from docling.datamodel.pipeline_options import TesseractCliOcrOptions

        ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
        image_options = ImageFormatOption(
            ocr_options=ocr_options,
            do_table_structure=True,
        )
        converter = DocumentConverter(format_options={"image": image_options})

    result = converter.convert(str(image_path))
    markdown_text = result.document.export_to_markdown()
    elapsed = (time.perf_counter() - start) * 1000

    tables = []
    for table in result.document.tables:
        try:
            tables.append(table.export_to_dataframe())
        except Exception:
            pass

    return markdown_text.strip(), {
        "engine": "Docling (VLM)" if use_vlm else f"Docling + {ocr_backend}",
        "latency_ms": round(elapsed, 1),
        "cost_per_1k_pages": 0.0,
        "tables_found": len(tables),
        "tables": tables,
        "output_format": "markdown",
    }


# ─────────────────────────────────────────────
# Tier 3: dots.ocr (Lightweight VLM, 1.7B, Free)
# ─────────────────────────────────────────────


def run_dots_ocr(
    image: Image.Image,
    base_url: str = "http://localhost:8000/v1",
    model: str = "rednote-hilab/dots.ocr",
    api_key: str = "token-abc123",
) -> tuple[str, dict]:
    """Run dots.ocr via a vLLM OpenAI-compatible server.

    Requires:
      - A running vLLM server: vllm serve rednote-hilab/dots.ocr
      - pip install openai

    dots.ocr is a 1.7B VLM achieving SOTA on OmniDocBench.
    Supports 100+ languages.
    """
    from openai import OpenAI

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    client = OpenAI(base_url=base_url, api_key=api_key)

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract all text from this document image. "
                            "Return only the extracted text, preserving the original "
                            "layout and reading order."
                        ),
                    },
                ],
            }
        ],
        max_tokens=4096,
        temperature=0.0,
    )
    elapsed = (time.perf_counter() - start) * 1000
    text = response.choices[0].message.content or ""

    usage = response.usage
    return text.strip(), {
        "engine": "dots.ocr (1.7B)",
        "latency_ms": round(elapsed, 1),
        "cost_per_1k_pages": 0.0,
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
    }


# ─────────────────────────────────────────────
# Tier 4: Mistral OCR v3 (Dedicated OCR API)
# ─────────────────────────────────────────────


def run_mistral_ocr(
    image: Image.Image,
    api_key: str | None = None,
) -> tuple[str, dict]:
    """Run Mistral OCR v3 via the dedicated /v1/ocr endpoint.

    Requires: pip install mistralai, env MISTRAL_API_KEY

    Mistral OCR is a specialized OCR model (not a general VLM).
    Excels at forms, handwriting (88.9%), and tables (96.6%).
    Cost: $2/1K pages ($1/1K with batch API).
    """
    from mistralai import Mistral

    api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set")

    client = Mistral(api_key=api_key)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    start = time.perf_counter()
    response = client.ocr.process(
        model="mistral-ocr-2512",
        document={"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
    )
    elapsed = (time.perf_counter() - start) * 1000

    text_parts = [page.markdown for page in response.pages]
    full_text = "\n".join(text_parts)

    return full_text.strip(), {
        "engine": "Mistral OCR v3",
        "latency_ms": round(elapsed, 1),
        "cost_per_1k_pages": 2.0,
        "output_format": "markdown",
    }


# ─────────────────────────────────────────────
# Tier 5: Gemini 3 Flash (Frontier VLM API)
# ─────────────────────────────────────────────


def run_gemini(
    image: Image.Image,
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
    media_resolution: str = "medium",
) -> tuple[str, dict]:
    """Run Gemini 3 Flash via Google Generative AI SDK.

    Requires: pip install google-generativeai, env GEMINI_API_KEY

    #1 on OCR Arena (ELO 1770, 77.2% win rate).
    Cost: $0.50/M input, $3.00/M output tokens.

    media_resolution: Google says OCR quality saturates at "medium".
    """
    import google.generativeai as genai

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model)

    prompt = (
        "Extract all text from this document image. "
        "Return only the extracted text, preserving the original "
        "layout and reading order. "
        "Do not add any explanations or formatting beyond what is "
        "in the original document."
    )

    start = time.perf_counter()
    response = gen_model.generate_content(
        [prompt, image],
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=8192,
        ),
    )
    elapsed = (time.perf_counter() - start) * 1000

    text = response.text or ""

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else 0
    output_tokens = usage.candidates_token_count if usage else 0
    cost = (input_tokens * 0.50 + output_tokens * 3.00) / 1_000_000

    return text.strip(), {
        "engine": "Gemini 3 Flash",
        "latency_ms": round(elapsed, 1),
        "cost_per_1k_pages": round(cost * 1000, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_this_page": round(cost, 6),
    }


# ─────────────────────────────────────────────
# Availability checker
# ─────────────────────────────────────────────


def check_available_engines() -> dict[str, bool]:
    """Check which engines can run on this system."""
    available = {}

    # Tesseract
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        available["tesseract"] = True
    except Exception:
        available["tesseract"] = False

    # Docling
    try:
        import docling  # noqa: F401

        available["docling"] = True
    except ImportError:
        available["docling"] = False

    # dots.ocr (check if vLLM server is reachable)
    try:
        from openai import OpenAI

        client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")
        client.models.list()
        available["dots_ocr"] = True
    except Exception:
        available["dots_ocr"] = False

    # Mistral OCR
    available["mistral_ocr"] = bool(os.environ.get("MISTRAL_API_KEY"))

    # Gemini
    available["gemini"] = bool(os.environ.get("GEMINI_API_KEY"))

    return available
