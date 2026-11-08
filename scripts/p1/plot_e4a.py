"""
Draws the per-round curves of E4a from results/p1_e4a/analysis/curves.csv, one panel per alpha.

Usage: python scripts/p1/plot_e4a.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = "results/p1_e4a/analysis"
ARMS = [("greedy", "G", "#2a78d6"), ("fltrust", "FLTrust", "#eb6834"), ("trimmed_mean", "Trimmed mean", "#1baf7a"),
        ("median", "Median", "#eda100"), ("krum", "Krum", "#e87ba4")]
PHASES = [(1, 15, "sign_flipping"), (16, 30, "gaussian_noise"), (31, 45, "trim_attack")]


def main():
    curves = pd.read_csv(f"{OUT}/curves.csv")
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    for ax, alpha in zip(axes, (0.1, 0.5)):
        for arm, label, color in ARMS:
            d = curves[(curves.alpha == alpha) & (curves.arm == arm)]
            ax.plot(d["round"], d.accuracy, color=color, label=label, linewidth=2.4 if arm == "greedy" else 1.4,
                    zorder=3 if arm == "greedy" else 2)
        for first, last, attack in PHASES:
            ax.axvspan(last - 4, last, color="#d9d8d3", alpha=0.45, linewidth=0, zorder=0)
            ax.text((first + last) / 2, 1.01, attack, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=9, color="#52514e")
            if last < 45:
                ax.axvline(last + 0.5, color="#52514e", linewidth=0.8, linestyle=":")
        ax.set_ylabel(f"Test accuracy, %  (α = {alpha})")
        ax.grid(axis="y", color="#e5e4df", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("Round (shaded: the last 5 rounds of each phase, used by the phase score)")
    axes[1].set_ylim(70, 95)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", frameon=False, fontsize=9, ncol=5)
    fig.savefig(f"{OUT}/curves.png", dpi=200)
    fig.savefig(f"{OUT}/curves.pdf")


if __name__ == "__main__":
    main()
