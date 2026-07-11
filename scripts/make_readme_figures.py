"""Generate README figures from committed evaluation summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
ASSETS = ROOT / "assets"


def load_json(name: str) -> dict:
    with (OUTPUTS / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def add_labels(ax, bars, fmt="{:.3f}"):
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            fmt.format(bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def model_comparison() -> None:
    metrics = {name: load_json(f"{name}_metrics.json") for name in ("mlp", "cnn")}
    names = ["MLP", "CNN"]
    auc = [metrics[name.lower()]["roc_auc"] for name in names]
    accuracy = [metrics[name.lower()]["accuracy"] for name in names]
    yes_f1 = [metrics[name.lower()]["classification_report"]["Yes"]["f1-score"] for name in names]

    x = np.arange(len(names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars1 = ax.bar(x - width, auc, width, label="ROC AUC", color="#2563EB")
    bars2 = ax.bar(x, accuracy, width, label="Accuracy", color="#0F766E")
    bars3 = ax.bar(x + width, yes_f1, width, label="Rain F1", color="#D97706")
    for bars in (bars1, bars2, bars3):
        add_labels(ax, bars)
    ax.set(title="Validation performance", ylabel="Score", xticks=x, xticklabels=names, ylim=(0, 1.08))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(ASSETS / "model-comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def application_examples() -> None:
    pricing = load_json("taxi_pricing_demo_summary.json")["by_risk"]
    dispatch = load_json("taxi_dispatch_demo_summary.json")["by_risk"]
    labels = [item["rain_risk_level"].replace("_", " ").title() for item in pricing]
    price_increment = [item["avg_rain_increment_pct"] for item in pricing]
    reward_gain = [item["avg_rain_policy_reward_gain"] for item in dispatch]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = ["#93C5FD", "#60A5FA", "#2563EB", "#1E3A8A"]
    bars = axes[0].bar(labels, price_increment, color=colors)
    axes[0].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    axes[0].set(title="Illustrative pricing response", ylabel="Mean fare increment (%)")

    bars = axes[1].bar(labels, reward_gain, color=colors)
    axes[1].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axes[1].set(title="Illustrative dispatch reward gain", ylabel="Mean reward gain (AUD)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Rain-probability downstream demos (scenario assumptions, not causal estimates)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "application-examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    model_comparison()
    application_examples()
    print(f"Figures written to {ASSETS}")


if __name__ == "__main__":
    main()
