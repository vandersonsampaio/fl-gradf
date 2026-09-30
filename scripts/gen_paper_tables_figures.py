"""
Gera as Tabelas 1-5 (LaTeX) e Figuras 1-3 (PNG) sugeridas em
docs/paperA/paper1_completo_secoes_1a9.md, a partir dos CSVs reais
produzidos por src/experiments/exp9_dominance_grid.py e
src/experiments/exp10_selector_comparison.py.

Nenhum número é inventado: tudo vem dos CSVs em results/tables/.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT = "/home/notebook/workspace-personal/fl-fedmed"
TAB = os.path.join(ROOT, "results/tables")
FIG_OUT = os.path.join(ROOT, "results/figures")
TEX_OUT = os.path.join(ROOT, "docs/paperA/tabelas_geradas.tex")
os.makedirs(FIG_OUT, exist_ok=True)
os.makedirs(os.path.dirname(TEX_OUT), exist_ok=True)

# ---------------------------------------------------------------- palette
BLUE = "#2a78d6"      # slot 1 - FLTrust / Random
ORANGE = "#eb6834"    # slot 2 - trimmed_mean / GRADF
AQUA = "#1baf7a"       # slot 3 - median / FedStrategist
YELLOW = "#eda100"     # slot 4 - krum / AdaAggRL
VIOLET = "#4a3aa7"      # slot 7 - Oracle
INK = "#0b0b0b"
SEC_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

ATTACKS = ["fltrust_aligned", "gaussian_noise", "krum_collusion", "label_flipping",
           "low_mag_backdoor", "sign_flipping", "trim_attack"]
ATTACK_LABEL = {
    "fltrust_aligned": "fltrust aligned",
    "gaussian_noise": "gaussian noise",
    "krum_collusion": "krum collusion",
    "label_flipping": "label flipping",
    "low_mag_backdoor": "low-mag backdoor",
    "sign_flipping": "sign flipping",
    "trim_attack": "trim attack",
}
ALPHAS = [0.5, 0.1, 0.05]
RULE_LABEL = {"fltrust": "FLTrust", "median": "median", "trimmed_mean": "trimmed mean", "krum": "Krum"}
RULE_COLOR = {"fltrust": BLUE, "median": AQUA, "trimmed_mean": ORANGE, "krum": YELLOW}

def esc(s):
    return str(s).replace("_", "\\_")

def fmt_p(p):
    if p < 0.0001:
        mantissa, exp = f"{p:.1e}".split("e")
        exp = int(exp)
        return f"${mantissa}\\times10^{{{exp}}}$"
    return f"{p:.4f}"

# ================================================================= LOAD
v3000 = pd.read_csv(os.path.join(TAB, "exp9_dominance_grid_10seeds_root3000_verdict.csv"))
v100 = pd.read_csv(os.path.join(TAB, "exp9_dominance_grid_10seeds_root100_verdict.csv"))
v50 = pd.read_csv(os.path.join(TAB, "exp9_dominance_grid_10seeds_root50_verdict.csv"))

variantb_summary = pd.read_csv(os.path.join(TAB, "exp10_FULL_variantb_summary.csv"))
best_fixed_cell = pd.read_csv(os.path.join(TAB, "exp10_FULL_best_fixed_rule_per_cell.csv"))
pooled_vs_random = pd.read_csv(os.path.join(TAB, "exp10_FULL_vs_random_significance_pooled.csv"))
per_attack_vs_random = pd.read_csv(os.path.join(TAB, "exp10_FULL_vs_random_significance.csv"))

# ================================================================= TABLE 1
def win_counts(v):
    counts = {r: 0 for r in RULE_LABEL}
    for r in v["winner"]:
        counts[r] += 1
    non_fltrust_solid = int(((v["winner"] != "fltrust") & (v["gap_exceeds_std"])).sum())
    return counts, non_fltrust_solid

rows_t1 = []
for root_size, v in [(3000, v3000), (100, v100), (50, v50)]:
    counts, nfs = win_counts(v)
    rows_t1.append((root_size, counts["fltrust"], counts["median"], counts["trimmed_mean"], counts["krum"], nfs))

tex_t1 = []
tex_t1.append(r"\begin{table}[t]")
tex_t1.append(r"\centering")
tex_t1.append(r"\caption{Win count per fixed rule (FLTrust, median, trimmed mean, Krum) over the 21 cells of the grid (3 heterogeneity levels $\times$ 7 attack types), by root dataset size. The last column counts cells where the winner is not FLTrust and the margin over the runner-up exceeds the combined standard deviation across seeds.}")
tex_t1.append(r"\label{tab:root-size-wins}")
tex_t1.append(r"\begin{tabular}{lccccc}")
tex_t1.append(r"\toprule")
tex_t1.append(r"Root dataset & FLTrust & Median & Trimmed mean & Krum & Non-FLTrust (robust) \\")
tex_t1.append(r"\midrule")
for root_size, f, m, ta, k, nfs in rows_t1:
    tex_t1.append(f"{root_size} & {f}/21 & {m}/21 & {ta}/21 & {k}/21 & {nfs}/21 \\\\")
tex_t1.append(r"\bottomrule")
tex_t1.append(r"\end{tabular}")
tex_t1.append(r"\end{table}")

# ================================================================= TABLE 2
rows_t2 = []
for alpha in ALPHAS:
    w3000 = int(((v3000["alpha"] == alpha) & (v3000["winner"] == "fltrust")).sum())
    w100 = int(((v100["alpha"] == alpha) & (v100["winner"] == "fltrust")).sum())
    rows_t2.append((alpha, w3000, w100))

tex_t2 = []
tex_t2.append(r"\begin{table}[t]")
tex_t2.append(r"\centering")
tex_t2.append(r"\caption{FLTrust wins per heterogeneity level $\alpha$, over the 7 attack-type cells of each row, contrasting the inflated root (3000) with the canonical root (100).}")
tex_t2.append(r"\label{tab:fltrust-by-alpha}")
tex_t2.append(r"\begin{tabular}{lcc}")
tex_t2.append(r"\toprule")
tex_t2.append(r"$\alpha$ & Root = 3000 & Root = 100 \\")
tex_t2.append(r"\midrule")
for alpha, w3000, w100 in rows_t2:
    a_str = f"{alpha:.2f}".rstrip("0").rstrip(".")
    tex_t2.append(f"{a_str} & {w3000}/7 & {w100}/7 \\\\")
tex_t2.append(r"\bottomrule")
tex_t2.append(r"\end{tabular}")
tex_t2.append(r"\end{table}")

# ================================================================= TABLE 3
SYSTEMS_T3 = ["Random", "Oracle (decoupled)", "GRADF", "FedStrategist", "AdaAggRL"]
SYS_LABEL_T3 = {"Random": "Random", "Oracle (decoupled)": "Oracle", "GRADF": "GRADF",
                "FedStrategist": "FedStrat.", "AdaAggRL": "AdaAggRL"}
IMPLEMENTABLE = ["GRADF", "FedStrategist", "AdaAggRL"]

pivot = variantb_summary.pivot_table(index=["alpha", "attack_type"], columns="system", values="accuracy_mean")
bf = best_fixed_cell.set_index(["alpha", "attack_type"])

tex_t3 = []
tex_t3.append(r"\begin{table*}[t]")
tex_t3.append(r"\centering")
tex_t3.append(r"\small")
tex_t3.append(r"\caption{Full grid under canonical root (100 examples): mean accuracy (10 seeds) per system, and the best fixed rule for the cell (among FLTrust, median, trimmed mean, Krum). Bold marks cells where a deployable adaptive system (GRADF, FedStrategist, or AdaAggRL) beats the best fixed rule.}")
tex_t3.append(r"\label{tab:full-grid-root100}")
tex_t3.append(r"\begin{tabular}{llccccccl}")
tex_t3.append(r"\toprule")
tex_t3.append(r"$\alpha$ & Attack & Random & Oracle & GRADF & FedStrat. & AdaAggRL & Best fixed & Rule \\")
tex_t3.append(r"\midrule")

def fmt_cell(val, bold):
    s = f"{val:.3f}"
    return r"\textbf{" + s + "}" if bold else s

for alpha in ALPHAS:
    a_str = f"{alpha:.2f}".rstrip("0").rstrip(".")
    for i, atk in enumerate(ATTACKS):
        key = (alpha, atk)
        row_vals = pivot.loc[key]
        bf_rule = bf.loc[key, "best_fixed_rule"]
        bf_acc = bf.loc[key, "best_fixed_rule_accuracy"]
        cells = []
        for sysname in SYSTEMS_T3:
            v = row_vals[sysname]
            bold = (sysname in IMPLEMENTABLE) and (v > bf_acc)
            cells.append(fmt_cell(v, bold))
        alpha_col = a_str if i == 0 else ""
        tex_t3.append(f"{alpha_col} & {ATTACK_LABEL[atk]} & " + " & ".join(cells) +
                       f" & {bf_acc:.3f} & {RULE_LABEL[bf_rule]} \\\\")
    if alpha != ALPHAS[-1]:
        tex_t3.append(r"\addlinespace")
tex_t3.append(r"\bottomrule")
tex_t3.append(r"\end{tabular}")
tex_t3.append(r"\end{table*}")

# ================================================================= TABLE 4
order4 = ["AdaAggRL", "Oracle (decoupled)", "GRADF", "FedStrategist"]
p4 = pooled_vs_random.set_index("system")
tex_t4 = []
tex_t4.append(r"\begin{table}[t]")
tex_t4.append(r"\centering")
tex_t4.append(r"\caption{Paired $t$-test by seed, pooled over the seven attack types under canonical root, against the Random-selector. Mean accuracy difference, $p$-value, and Cohen's $d$ ($n=10$ seeds).}")
tex_t4.append(r"\label{tab:pooled-vs-random}")
tex_t4.append(r"\begin{tabular}{lccc}")
tex_t4.append(r"\toprule")
tex_t4.append(r"System & $\Delta$ mean & $p$ & Cohen's $d$ \\")
tex_t4.append(r"\midrule")
for sysname in order4:
    row = p4.loc[sysname]
    label = SYS_LABEL_T3.get(sysname, sysname)
    sign = "+" if row["mean_diff"] >= 0 else ""
    tex_t4.append(f"{label} & {sign}{row['mean_diff']:.3f} & {fmt_p(row['p_value'])} & {row['cohens_d']:.2f} \\\\")
tex_t4.append(r"\bottomrule")
tex_t4.append(r"\end{tabular}")
tex_t4.append(r"\end{table}")

# ================================================================= TABLE 5
order5 = ["AdaAggRL", "Oracle (decoupled)", "GRADF", "FedStrategist"]
p5 = per_attack_vs_random.set_index(["attack_type", "system"])
tex_t5 = []
tex_t5.append(r"\begin{table*}[t]")
tex_t5.append(r"\centering")
tex_t5.append(r"\small")
tex_t5.append(r"\caption{Paired $t$-test by seed against the Random-selector, broken down by attack type under canonical root ($n=10$ seeds each). $\Delta$: mean accuracy difference; $d$: Cohen's $d$.}")
tex_t5.append(r"\label{tab:per-attack-vs-random}")
tex_t5.append(r"\resizebox{\textwidth}{!}{%")
tex_t5.append(r"\begin{tabular}{l" + "ccc" * len(order5) + "}")
tex_t5.append(r"\toprule")
header_top = "Attack"
for sysname in order5:
    header_top += " & \\multicolumn{3}{c}{" + SYS_LABEL_T3.get(sysname, sysname) + "}"
tex_t5.append(header_top + r" \\")
cmid = " ".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(order5)))
tex_t5.append(cmid)
header_sub = " & " + " & ".join([r"$\Delta$ & $p$ & $d$"] * len(order5))
tex_t5.append(header_sub + r" \\")
tex_t5.append(r"\midrule")
for atk in ATTACKS:
    cells = []
    for sysname in order5:
        row = p5.loc[(atk, sysname)]
        sign = "+" if row["mean_diff"] >= 0 else ""
        cells.append(f"{sign}{row['mean_diff']:.3f} & {fmt_p(row['p_value'])} & {row['cohens_d']:.2f}")
    tex_t5.append(f"{ATTACK_LABEL[atk]} & " + " & ".join(cells) + r" \\")
tex_t5.append(r"\bottomrule")
tex_t5.append(r"\end{tabular}%")
tex_t5.append(r"}")
tex_t5.append(r"\end{table*}")

# ================================================================= write tex
with open(TEX_OUT, "w") as f:
    f.write("% Tables generated automatically from the CSVs in results/tables/\n")
    f.write("% Source: src/experiments/exp9_dominance_grid.py and src/experiments/exp10_selector_comparison.py\n")
    f.write("% Generated by scripts/gen_paper_tables_figures.py -- do not hand-edit, regenerate from the CSVs.\n\n")
    f.write("% ---- Table 1: win counts by root size ----\n")
    f.write("\n".join(tex_t1) + "\n\n")
    f.write("% ---- Table 2: FLTrust wins by alpha ----\n")
    f.write("\n".join(tex_t2) + "\n\n")
    f.write("% ---- Table 3: full grid under canonical root ----\n")
    f.write("\n".join(tex_t3) + "\n\n")
    f.write("% ---- Table 4: pooled paired test vs Random ----\n")
    f.write("\n".join(tex_t4) + "\n\n")
    f.write("% ---- Table 5: per-attack-type paired test vs Random ----\n")
    f.write("\n".join(tex_t5) + "\n")

print("Wrote", TEX_OUT)

# =========================================================== FIGURE 1
fig, axes = plt.subplots(1, 2, figsize=(9, 5.2), sharey=True)
for ax, (title, v) in zip(axes, [("Inflated root (3000 examples)", v3000), ("Canonical root (100 examples)", v100)]):
    vv = v.set_index(["attack_type", "alpha"])
    for i, atk in enumerate(ATTACKS):
        for j, alpha in enumerate(ALPHAS):
            winner = vv.loc[(atk, alpha), "winner"]
            color = RULE_COLOR[winner]
            rect = plt.Rectangle((j, len(ATTACKS) - 1 - i), 1, 1, facecolor=color,
                                  edgecolor=SURFACE, linewidth=2)
            ax.add_patch(rect)
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

legend_handles = [mpatches.Patch(facecolor=RULE_COLOR[r], label=RULE_LABEL[r]) for r in RULE_COLOR]
fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(os.path.join(FIG_OUT, "figura1_heatmap_dominancia_root.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Wrote figura1_heatmap_dominancia_root.png")

# =========================================================== FIGURE 2
# Definicao operacional: uma celula tem "headroom capturado" por um sistema/familia
# quando a acuracia media desse sistema supera a da melhor regra fixa da celula
# (mesmo criterio de negrito usado na Tabela 3).
best_fixed_map = bf["best_fixed_rule_accuracy"]
oracle_acc = pivot["Oracle (decoupled)"]
gradf_acc = pivot["GRADF"]
fs_acc = pivot["FedStrategist"]
ada_acc = pivot["AdaAggRL"]

n_oracle = int((oracle_acc > best_fixed_map).sum())
n_discrete = int(((gradf_acc > best_fixed_map) | (fs_acc > best_fixed_map)).sum())
n_continuous = int((ada_acc > best_fixed_map).sum())
n_implementable = int(((gradf_acc > best_fixed_map) | (fs_acc > best_fixed_map) | (ada_acc > best_fixed_map)).sum())
TOTAL_CELLS = 21

bars_labels = ["Ceiling\n(Oracle)", "Captured\n(deployable)", "Discrete selection\n(GRADF+FedStrat.)", "Continuous weighting\n(AdaAggRL)"]
bars_values = [n_oracle, n_implementable, n_discrete, n_continuous]
bars_colors = [VIOLET, MUTED, ORANGE, YELLOW]

fig, ax = plt.subplots(figsize=(7, 4.6))
x = range(len(bars_labels))
bars = ax.bar(x, bars_values, color=bars_colors, width=0.6, edgecolor=SURFACE, linewidth=1)
for xi, v in zip(x, bars_values):
    ax.text(xi, v + 0.3, f"{v}/{TOTAL_CELLS}", ha="center", va="bottom", fontsize=10, color=INK)
ax.set_xticks(list(x))
ax.set_xticklabels(bars_labels, fontsize=9.5, color=INK)
ax.set_ylabel("Grid cells (out of 21)", fontsize=10, color=SEC_INK)
ax.set_ylim(0, TOTAL_CELLS + 2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(BASELINE)
ax.spines["bottom"].set_color(BASELINE)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(FIG_OUT, "figura2_headroom_capturado.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Wrote figura2_headroom_capturado.png")

# =========================================================== FIGURE 3
# Media sobre os 3 niveis de alpha, por tipo de ataque e sistema
avg = variantb_summary.groupby(["attack_type", "system"])["accuracy_mean"].mean().unstack()
sys3 = ["GRADF", "FedStrategist", "AdaAggRL"]
colors3 = {"GRADF": ORANGE, "FedStrategist": AQUA, "AdaAggRL": YELLOW}

fig, ax = plt.subplots(figsize=(10.5, 5))
n_sys = len(sys3)
bar_w = 0.22
x = list(range(len(ATTACKS)))
for i, sysname in enumerate(sys3):
    offs = (i - (n_sys - 1) / 2) * bar_w
    vals = [avg.loc[atk, sysname] for atk in ATTACKS]
    ax.bar([xi + offs for xi in x], vals, width=bar_w, color=colors3[sysname], label=sysname,
           edgecolor=SURFACE, linewidth=0.8)

# Random como linha de referencia por grupo
group_w = bar_w * n_sys
for xi, atk in zip(x, ATTACKS):
    rnd = avg.loc[atk, "Random"]
    ax.plot([xi - group_w / 2 - 0.03, xi + group_w / 2 + 0.03], [rnd, rnd],
            color=BLUE, linewidth=2, linestyle=(0, (4, 2)), zorder=5)

ax.plot([], [], color=BLUE, linewidth=2, linestyle=(0, (4, 2)), label="Random (reference)")
ax.set_xticks(x)
ax.set_xticklabels([ATTACK_LABEL[a] for a in ATTACKS], fontsize=9.5, rotation=15, ha="right")
ax.set_ylabel("Mean accuracy (10 seeds $\\times$ 3 $\\alpha$)", fontsize=10, color=SEC_INK)
ax.set_ylim(0, 1.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(BASELINE)
ax.spines["bottom"].set_color(BASELINE)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG_OUT, "figura3_barras_sistemas_por_ataque.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Wrote figura3_barras_sistemas_por_ataque.png")

print("\nResumo numerico para conferencia:")
print("Table1 rows:", rows_t1)
print("Table2 rows:", rows_t2)
print("Fig2 counts: oracle", n_oracle, "implementable", n_implementable, "discrete", n_discrete, "continuous", n_continuous)
