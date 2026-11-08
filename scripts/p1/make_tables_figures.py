"""
P-1: Tables 1-5 (LaTeX) and Figures 1-3 (PNG) of the P1 manuscript,
regenerated from the raw per-seed CSVs through the audit pipeline
(scripts/p1/audit_mnist.py), instead of the summary/verdict CSVs read by
scripts/gen_paper_tables_figures.py.

--source published: fixed rules from the published (unseeded) raw files, with
  the seed composition of the manuscript (Table 1 at roots 3000/50 mixes 10
  FLTrust seeds with 3 seeds of the other rules; AUDIT.md, A8). Reproduces
  the submitted tables and figures.
--source seeded (canonical): fixed rules from the seeded rerun in
  results/p1_invariance/ (10 seeds for every rule and root; AUDIT.md, B5-B7).
The adaptive systems always come from the published exp10 runs.

Writes results/p1_tables/<source>/tables.tex and fig{1,2,3}_*.png.

Usage: venv/bin/python scripts/p1/make_tables_figures.py [--source seeded|published]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))

import audit_mnist as audit  # noqa: E402
from analyze_invariance import load_seeded  # noqa: E402

ALPHAS, ATTACKS, D_R = audit.ALPHAS, audit.ATTACKS, audit.D_R

# ---------------------------------------------------------------- style (as in gen_paper_tables_figures.py)
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, SEC_INK, MUTED, GRID, BASELINE, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})
ATTACK_LABEL = {a: a.replace("_", " ") for a in ATTACKS}
ATTACK_LABEL["low_mag_backdoor"] = "low-mag backdoor"
RULE_LABEL = {"fltrust": "FLTrust", "median": "median", "trimmed_mean": "trimmed mean", "krum": "Krum"}
RULE_COLOR = {"fltrust": BLUE, "median": AQUA, "trimmed_mean": ORANGE, "krum": YELLOW}
SYS_LABEL = {"Random": "Random", "Oracle (decoupled)": "Oracle", "GRADF": "GRADF",
             "FedStrategist": "FedStrat.", "AdaAggRL": "AdaAggRL"}
SYSTEMS_T3 = ["Random", "Oracle (decoupled)", "GRADF", "FedStrategist", "AdaAggRL"]
DEPLOYABLE = ["GRADF", "FedStrategist", "AdaAggRL"]
ORDER_T45 = ["AdaAggRL", "Oracle (decoupled)", "GRADF", "FedStrategist"]


def a_str(alpha):
    return f"{alpha:.2f}".rstrip("0").rstrip(".")


def fmt_p(p):
    if p < 0.0001:
        mantissa, exp = f"{p:.1e}".split("e")
        return f"${mantissa}\\times10^{{{int(exp)}}}$"
    return f"{p:.4f}"


def signed(x):
    return f"{'+' if x >= 0 else ''}{x:.3f}"


def load_fixed(source):
    """{root label: long DataFrame of the D_R rules} for Tables 1-2, plus root 100."""
    if source == "published":
        f = audit.load_fixed()
        return {"3000": f["3000_as_published"], "100": f["100"], "50": f["50_as_published"]}
    df = load_seeded()
    return {str(r): df[df.root == r].drop(columns="root") for r in (3000, 100, 50)}


def systems_by_cell(sy):
    """(cell) x system mean accuracy, with O(c) = best D_R rule mean as 'best fixed'."""
    return audit.cell_means(sy, "system")


# ---------------------------------------------------------------- tables

def table1_tex(fixed):
    t = audit.table1(fixed).set_index("root")
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{Win count per fixed rule (FLTrust, median, trimmed mean, Krum) over the 21 cells of the grid (3 heterogeneity levels $\times$ 7 attack types), by root dataset size. The last column counts cells where the winner is not FLTrust and the margin over the runner-up exceeds the combined standard deviation across seeds.}",
           r"\label{tab:root-size-wins}", r"\begin{tabular}{lccccc}", r"\toprule",
           r"Root dataset & FLTrust & Median & Trimmed mean & Krum & Non-FLTrust (robust) \\",
           r"\midrule"]
    for r in ("3000", "100", "50"):
        x = t.loc[r]
        out.append(f"{r} & {x['fltrust']}/21 & {x['median']}/21 & {x['trimmed_mean']}/21 & {x['krum']}/21 & "
                   f"{x['non_fltrust_robust']}/21 \\\\")
    return out + [r"\bottomrule", r"\end{tabular}", r"\end{table}"], t


def table2_tex(fixed):
    t = audit.table2(fixed, roots=("3000", "100"))
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{FLTrust wins per heterogeneity level $\alpha$, over the 7 attack-type cells of each row, contrasting the inflated root (3000) with the canonical root (100).}",
           r"\label{tab:fltrust-by-alpha}", r"\begin{tabular}{lcc}", r"\toprule",
           r"$\alpha$ & Root = 3000 & Root = 100 \\", r"\midrule"]
    for _, x in t.iterrows():
        out.append(f"{a_str(x.alpha)} & {int(x['3000'])}/7 & {int(x['100'])}/7 \\\\")
    return out + [r"\bottomrule", r"\end{tabular}", r"\end{table}"], t


def table3_tex(t3):
    out = [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\caption{Full grid under canonical root (100 examples): mean accuracy (10 seeds) per system, and the best fixed rule for the cell (among FLTrust, median, trimmed mean, Krum). Bold marks cells where a deployable adaptive system (GRADF, FedStrategist, or AdaAggRL) beats the best fixed rule.}",
           r"\label{tab:full-grid-root100}", r"\begin{tabular}{llccccccl}", r"\toprule",
           r"$\alpha$ & Attack & Random & Oracle & GRADF & FedStrat. & AdaAggRL & Best fixed & Rule \\",
           r"\midrule"]
    for alpha in ALPHAS:
        for i, atk in enumerate(ATTACKS):
            x = t3.loc[(alpha, atk)]
            cells = []
            for s in SYSTEMS_T3:
                v = f"{x[s]:.3f}"
                cells.append(r"\textbf{" + v + "}" if s in DEPLOYABLE and x[s] > x.best_fixed else v)
            out.append(f"{a_str(alpha) if i == 0 else ''} & {ATTACK_LABEL[atk]} & " + " & ".join(cells)
                       + f" & {x.best_fixed:.3f} & {RULE_LABEL[x.rule]} \\\\")
        if alpha != ALPHAS[-1]:
            out.append(r"\addlinespace")
    return out + [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]


def table4_tex(t4):
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{Paired $t$-test by seed, pooled over the seven attack types under canonical root, against the Random-selector. Mean accuracy difference, $p$-value, and Cohen's $d$ ($n=10$ seeds).}",
           r"\label{tab:pooled-vs-random}", r"\begin{tabular}{lccc}", r"\toprule",
           r"System & $\Delta$ mean & $p$ & Cohen's $d$ \\", r"\midrule"]
    for s in ORDER_T45:
        x = t4[s]
        out.append(f"{SYS_LABEL[s]} & {signed(x['delta'])} & {fmt_p(x['p'])} & {x['d']:.2f} \\\\")
    return out + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]


def table5_tex(t5):
    out = [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\caption{Paired $t$-test by seed against the Random-selector, broken down by attack type under canonical root ($n=10$ seeds each). $\Delta$: mean accuracy difference; $d$: Cohen's $d$.}",
           r"\label{tab:per-attack-vs-random}", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{l" + "ccc" * len(ORDER_T45) + "}", r"\toprule",
           "Attack" + "".join(f" & \\multicolumn{{3}}{{c}}{{{SYS_LABEL[s]}}}" for s in ORDER_T45) + r" \\",
           " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(ORDER_T45))),
           " & " + " & ".join([r"$\Delta$ & $p$ & $d$"] * len(ORDER_T45)) + r" \\", r"\midrule"]
    for atk in ATTACKS:
        cells = [f"{signed(t5[s][atk]['delta'])} & {fmt_p(t5[s][atk]['p'])} & {t5[s][atk]['d']:.2f}"
                 for s in ORDER_T45]
        out.append(f"{ATTACK_LABEL[atk]} & " + " & ".join(cells) + r" \\")
    return out + [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]


# ---------------------------------------------------------------- figures

def style_bars(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure1(fixed, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2), sharey=True)
    for ax, (title, r) in zip(axes, [("Inflated root (3000 examples)", "3000"),
                                     ("Canonical root (100 examples)", "100")]):
        w = audit.winners(fixed[r]).set_index(["attack_type", "alpha"]).winner
        for i, atk in enumerate(ATTACKS):
            for j, alpha in enumerate(ALPHAS):
                ax.add_patch(plt.Rectangle((j, len(ATTACKS) - 1 - i), 1, 1,
                                           facecolor=RULE_COLOR[w[(atk, alpha)]],
                                           edgecolor=SURFACE, linewidth=2))
        ax.set_xlim(0, len(ALPHAS))
        ax.set_ylim(0, len(ATTACKS))
        ax.set_xticks([0.5, 1.5, 2.5])
        ax.set_xticklabels([f"$\\alpha={a}$" for a in ALPHAS], fontsize=9)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10.5, color=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks([len(ATTACKS) - 1 - i + 0.5 for i in range(len(ATTACKS))])
    axes[0].set_yticklabels([ATTACK_LABEL[a] for a in ATTACKS], fontsize=9)
    handles = [mpatches.Patch(facecolor=RULE_COLOR[r], label=RULE_LABEL[r]) for r in RULE_COLOR]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure2_counts(fx100, sy):
    """Headroom criterion of the manuscript (Fig. 2): cells with O(c) > B(c),
    B = d*, the global best fixed rule; a system captures a cell when it beats
    B(c) there. The first bar ("Ceiling") is the number of headroom cells.
    scripts/gen_paper_tables_figures.py used the Table 3 bold criterion
    (beats O(c)) instead, which does not match the submitted figure."""
    h = audit.headroom(fx100, sy)
    return {"oracle": h["headroom_cells"], "deployable": h["deployable_any"],
            "discrete": h["discrete_any"], "continuous": h["captured"]["AdaAggRL"],
            "dstar": h["dstar"]}


def figure2(counts, path):
    labels = ["Ceiling\n(Oracle)", "Captured\n(deployable)", "Discrete selection\n(GRADF+FedStrat.)",
              "Continuous weighting\n(AdaAggRL)"]
    values = [counts["oracle"], counts["deployable"], counts["discrete"], counts["continuous"]]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = range(len(labels))
    ax.bar(x, values, color=[VIOLET, MUTED, ORANGE, YELLOW], width=0.6, edgecolor=SURFACE, linewidth=1)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.3, f"{v}/21", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9.5, color=INK)
    ax.set_ylabel("Grid cells (out of 21)", fontsize=10, color=SEC_INK)
    ax.set_ylim(0, 23)
    style_bars(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure3(S, path):
    avg = S.groupby(level="attack_type").mean()
    sys3, colors = ["GRADF", "FedStrategist", "AdaAggRL"], {"GRADF": ORANGE, "FedStrategist": AQUA,
                                                            "AdaAggRL": YELLOW}
    fig, ax = plt.subplots(figsize=(10.5, 5))
    bar_w, x = 0.22, list(range(len(ATTACKS)))
    for i, s in enumerate(sys3):
        offs = (i - (len(sys3) - 1) / 2) * bar_w
        ax.bar([xi + offs for xi in x], [avg.loc[a, s] for a in ATTACKS], width=bar_w,
               color=colors[s], label=s, edgecolor=SURFACE, linewidth=0.8)
    group_w = bar_w * len(sys3)
    for xi, atk in zip(x, ATTACKS):
        rnd = avg.loc[atk, "Random"]
        ax.plot([xi - group_w / 2 - 0.03, xi + group_w / 2 + 0.03], [rnd, rnd],
                color=BLUE, linewidth=2, linestyle=(0, (4, 2)), zorder=5)
    ax.plot([], [], color=BLUE, linewidth=2, linestyle=(0, (4, 2)), label="Random (reference)")
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_LABEL[a] for a in ATTACKS], fontsize=9.5, rotation=15, ha="right")
    ax.set_ylabel("Mean accuracy (10 seeds $\\times$ 3 $\\alpha$)", fontsize=10, color=SEC_INK)
    ax.set_ylim(0, 1.0)
    style_bars(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


TIKZ_COLOR = {"fltrust": "cFLT", "median": "cMed", "trimmed_mean": "cTM", "krum": "cKrum"}


def figure_data_tex(fixed, counts, S):
    """The data blocks of the manuscript's TikZ figures (Fig1.tex, Fig2.tex,
    Fig3.tex), in their own syntax, to paste in place of the typed-in ones."""
    out = []
    for title, r in (("Fig1.tex, left panel (root 3000)", "3000"), ("Fig1.tex, right panel (root 100)", "100")):
        w = audit.winners(fixed[r]).set_index(["attack_type", "alpha"]).winner
        rows = [f"      {i}/{{{','.join(TIKZ_COLOR[w[(atk, a)]] for a in ALPHAS)}}}"
                for i, atk in enumerate(ATTACKS)]
        out += [f"% {title}", "  \\foreach \\r/\\cols in {", ",\n".join(rows) + "}{", ""]
    out += ["% Fig2.tex"] + [f"  \\addplot[fill={c}] coordinates {{({i},{counts[k]})}};" for i, (c, k) in
                             enumerate((("cOracle", "oracle"), ("cCapt", "deployable"),
                                        ("cDisc", "discrete"), ("cCont", "continuous")), 1)] + [""]
    avg = S.groupby(level="attack_type").mean()
    out.append("% Fig3.tex")
    for s, c in (("GRADF", "cGRADF"), ("FedStrategist", "cFed"), ("AdaAggRL", "cAda")):
        pts = " ".join(f"({i},{avg.loc[a, s]:.3f})" for i, a in enumerate(ATTACKS, 1))
        out.append(f"  \\addplot[fill={c}] coordinates {{\n    {pts}}};")
    pairs = ", ".join(f"{i}/{avg.loc[a, 'Random']:.3f}" for i, a in enumerate(ATTACKS, 1))
    out.append(f"  \\foreach \\x/\\r in {{{pairs}}}{{")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("seeded", "published"), default="seeded")
    source = ap.parse_args().source
    out = f"results/p1_tables/{source}"
    os.makedirs(out, exist_ok=True)

    fixed = load_fixed(source)
    sy = audit.load_systems()
    S = systems_by_cell(sy)
    t3 = audit.table3(fixed["100"], sy).rename(
        columns={"Attack-aware": "Oracle (decoupled)", "FedStrat.": "FedStrategist"})
    rnd = audit.per_seed(sy, "system")["Random"]
    t4 = audit.vs_reference(sy, rnd)
    t5 = audit.vs_reference(sy, rnd, by="attack_type")

    tex1, t1 = table1_tex(fixed)
    tex2, t2 = table2_tex(fixed)
    parts = [("Table 1: win counts by root size", tex1), ("Table 2: FLTrust wins by alpha", tex2),
             ("Table 3: full grid under canonical root", table3_tex(t3)),
             ("Table 4: pooled paired test vs Random", table4_tex(t4)),
             ("Table 5: per-attack-type paired test vs Random", table5_tex(t5))]
    with open(f"{out}/tables.tex", "w") as f:
        f.write(f"% Generated by scripts/p1/make_tables_figures.py --source {source} from the raw\n"
                "% per-seed CSVs; do not hand-edit.\n\n")
        for title, lines in parts:
            f.write(f"% ---- {title} ----\n" + "\n".join(lines) + "\n\n")

    figure1(fixed, f"{out}/fig1_winner_heatmap_by_root.png")
    counts = figure2_counts(fixed["100"], sy)
    figure2(counts, f"{out}/fig2_captured_headroom.png")
    figure3(S, f"{out}/fig3_systems_by_attack.png")
    with open(f"{out}/figure_data.tex", "w") as f:
        f.write(f"% Generated by scripts/p1/make_tables_figures.py --source {source}; data blocks\n"
                "% of docs/P1/fgcs_src/Fig{1,2,3}.tex. Do not hand-edit.\n\n")
        f.write("\n".join(figure_data_tex(fixed, counts, S)) + "\n")

    print("Wrote", out)
    print(t1.to_string())
    print(t2.to_string(index=False))
    print("Fig. 2 counts:", counts)


if __name__ == "__main__":
    main()
