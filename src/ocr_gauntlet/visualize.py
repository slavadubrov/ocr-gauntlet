"""Plots from recorded results only. Missing metrics are never fabricated."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def results_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{**r, **r.get("metrics", {})} for r in records])


def results_heatmap(df: pd.DataFrame, metric: str = "cer"):
    fig, ax = plt.subplots(figsize=(10, 5))
    if df.empty or metric not in df or df[metric].notna().sum() == 0:
        ax.text(
            0.5, 0.5, "No compatible scored outputs; inspect status/reason", ha="center"
        )
        ax.set_axis_off()
    else:
        pivot = df.pivot(index="engine", columns="document", values=metric)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis_r", ax=ax)
        ax.set_title(
            f"Conditional {metric.upper()} — missing cells are unscored, not zero"
        )
    fig.tight_layout()
    return fig


def completion_chart(summary: list[dict]):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([r["engine"] for r in summary], [r["completion_rate"] for r in summary])
    ax.set(
        ylim=(0, 1),
        ylabel="Successful / all planned pages",
        title="Completion before quality",
    )
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig
