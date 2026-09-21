"""Visualization helpers shared by src/experiments/ and notebooks/."""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _save(fig, save_path: Optional[str]) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_accuracy_over_rounds(
    histories: Dict[str, List[float]],
    title: str = "",
    save_path: Optional[str] = None,
    ylim: Tuple[float, float] = (0.0, 1.0),
):
    """Plots accuracy curves per round for one or more series (e.g. different
    systems or attack types)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, accs in histories.items():
        ax.plot(range(1, len(accs) + 1), accs, marker="o", markersize=3, label=label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_bar_comparison(
    values: Dict[str, float],
    title: str = "",
    ylabel: str = "",
    save_path: Optional[str] = None,
):
    """Plots a bar chart comparing one scalar value per category (e.g. final
    accuracy per system/defense strategy)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(list(values.keys()), list(values.values()))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


if __name__ == "__main__":
    plot_accuracy_over_rounds({"fedavg": [0.5, 0.6, 0.7], "fltrust": [0.5, 0.65, 0.75]}, title="demo")
    plot_bar_comparison({"fedavg": 0.7, "fltrust": 0.75, "median": 0.72}, title="demo", ylabel="accuracy")
    print("Figures generated (not saved — pass save_path to persist).")
