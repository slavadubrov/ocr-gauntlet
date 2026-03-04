"""Visualization helpers for comparing OCR results."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TIER_COLORS = {
    "Tesseract": "#6B7280",
    "Docling + tesseract": "#3B82F6",
    "dots.ocr (1.7B)": "#8B5CF6",
    "Mistral OCR v3": "#F59E0B",
    "Gemini 3 Flash": "#10B981",
}


def results_heatmap(df: pd.DataFrame, metric: str = "cer"):
    """Heatmap: tiers (rows) x documents (columns), colored by metric."""
    pivot = df.pivot(index="engine", columns="document", values=metric)
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = "RdYlGn_r" if metric in ("cer", "wer") else "RdYlGn"
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, linewidths=0.5, ax=ax)
    ax.set_title(f"OCR {metric.upper()} by Engine x Document")
    plt.tight_layout()
    return fig


def quality_cliff_chart(df: pd.DataFrame, metric: str = "cer"):
    """Line chart showing where traditional OCR falls apart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for engine in df["engine"].unique():
        edf = df[df["engine"] == engine]
        color = TIER_COLORS.get(engine, "#999")
        ax.plot(
            edf["document"],
            edf[metric],
            marker="o",
            label=engine,
            color=color,
            linewidth=2,
        )
    ax.set_xlabel("Document (easy → hard)")
    ax.set_ylabel(metric.upper())
    ax.set_title("The Quality Cliff: Where Traditional OCR Breaks Down")
    ax.legend(loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


def speed_vs_quality(df: pd.DataFrame, metric: str = "cer"):
    """Scatter: latency (x, log) vs quality (y). Points sized by cost."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for engine in df["engine"].unique():
        edf = df[df["engine"] == engine]
        color = TIER_COLORS.get(engine, "#999")
        cost = edf["cost_per_1k_pages"].iloc[0]
        size = max(50, cost * 100) if cost > 0 else 50
        ax.scatter(
            edf["latency_ms"],
            edf[metric],
            label=engine,
            color=color,
            s=size,
            alpha=0.7,
            edgecolors="white",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms, log scale)")
    ax.set_ylabel(metric.upper())
    ax.set_title("Speed vs Quality Tradeoff")
    ax.legend()
    plt.tight_layout()
    return fig


def cost_table(pages_per_month: int, avg_metrics: dict[str, float]) -> pd.DataFrame:
    """Cost comparison table for a given monthly volume."""
    tiers = [
        ("Tesseract", 0.0, "CPU only"),
        ("Docling + tesseract", 0.0, "CPU only"),
        ("dots.ocr (1.7B)", 0.0, "~$0.50/hr GPU"),
        ("Mistral OCR v3", 2.0, "$2/1K pages"),
        ("Gemini 3 Flash", 0.8, "~$0.50/M input tokens"),
    ]
    rows = []
    for name, cost_1k, note in tiers:
        monthly = (pages_per_month / 1000) * cost_1k
        avg_cer = avg_metrics.get(name, float("nan"))
        rows.append(
            {
                "Engine": name,
                "Avg CER": f"{avg_cer:.3f}" if avg_cer == avg_cer else "N/A",
                "Monthly Cost": f"${monthly:,.0f}" if monthly > 0 else "Free",
                "Pricing": note,
            }
        )
    return pd.DataFrame(rows)
