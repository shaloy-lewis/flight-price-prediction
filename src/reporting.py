"""Portable JSON reports and headless, labeled scientific plots."""

import json
from pathlib import Path

import numpy as np


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8"
    )


def save_figure(figure, path: Path, source: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.text(0.01, 0.01, source, fontsize=8, color="dimgray")
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=140)
    plt.close(figure)


def error_plots(y, prediction, directory: Path, seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), min(5000, len(y)), replace=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y[idx], prediction[idx], s=5, alpha=0.25)
    limits = [min(y.min(), prediction.min()), max(y.max(), prediction.max())]
    axes[0].plot(limits, limits, "--", color="black")
    axes[0].set(
        xlabel="Observed fare (INR)",
        ylabel="Estimated fare (INR)",
        title="Locked holdout: observed vs estimated",
    )
    axes[1].scatter(prediction[idx], prediction[idx] - y[idx], s=5, alpha=0.25)
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set(
        xlabel="Estimated fare (INR)",
        ylabel="Estimate - observed (INR)",
        title="Locked holdout residuals",
    )
    save_figure(
        figure,
        directory / "holdout_errors.png",
        "Source: original Kaggle raw files; deterministic holdout sample, at most 5,000 offers.",
    )
