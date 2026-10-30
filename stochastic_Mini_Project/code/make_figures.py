"""make_figures.py -- builds all report figures from ../results/*.  Run after run_experiments.py"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from microgrid_sp import Params

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(RES, "results.json")))
A = np.load(os.path.join(RES, "arrays.npz"))
P = Params()
H = np.arange(24)

BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
    "savefig.dpi": 200, "savefig.bbox": "tight", "axes.titlelocation": "left"})
POL = R["policies"]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig)


# ---- Fig 1: scenario bands ------------------------------------------------
S, D = A["S_train"], A["D_train"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for a, M, col, ttl in ((ax[0], S, ORANGE, "Solar generation (kW)"), (ax[1], D, BLUE, "Campus demand (kW)")):
    a.fill_between(H, np.quantile(M, .05, 0), np.quantile(M, .95, 0), color=col, alpha=.18, lw=0, label="5-95% band")
    for i in range(5):
        a.plot(H, M[i], color=col, alpha=.35, lw=.8)
    a.plot(H, M.mean(0), color=col, lw=2, label="mean")
    a.set_title(ttl); a.set_xlabel("hour of day"); a.set_xlim(0, 23); a.legend(loc="upper left")
fig.tight_layout(); save(fig, "fig1_scenarios.png")

# ---- Fig 2: model structure (two-stage) ---------------------------------
fig, ax = plt.subplots(figsize=(9, 2.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(-0.5, 3)


def box(x, y, w, h, txt, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.6, color=INK)


box(0.1, 0.8, 2.6, 1.5, "STAGE 1  (day-ahead)\nchoose purchase plan x_t\nfor t = 1..24\nbefore weather is known", "#e7f0fb", BLUE)
box(3.7, 1.55, 2.4, 1.1, "Uncertainty revealed\nsolar S_t, demand D_t\n(N scenarios, prob 1/N)", "#f3f2ee", GRAY)
box(7.0, 0.8, 2.9, 1.5, "STAGE 2  (recourse, per scenario)\nbattery c_t, d_t, SoC b_t\nreal-time buy g_t / export e_t\nminimise imbalance cost", "#fdebe3", ORANGE)
for (x0, y0, x1, y1) in ((2.75, 1.55, 3.65, 2.0), (6.15, 2.0, 6.95, 1.6)):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.2))
ax.add_patch(FancyArrowPatch((8.4, 0.75), (1.4, 0.75), arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.2,
                             connectionstyle="arc3,rad=-0.12", linestyle="--"))
ax.text(4.9, -0.18, "expected recourse cost feeds back into the choice of x_t", ha="center", fontsize=8.4, color=INK2)
save(fig, "fig2_structure.png")

# ---- Fig 3: day-ahead plans ---------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.2))
cols = {0: GRAY, 1: ORANGE, 2: BLUE}
for i in (0, 1, 2):
    ax.step(H, A[f"plan_{i}"], where="mid", color=cols[i], lw=2 if i == 2 else 1.6,
            ls="--" if i == 0 else "-", label=POL[i])
ax.set_xlabel("hour of day"); ax.set_ylabel("day-ahead purchase x_t (kWh)"); ax.set_xlim(0, 23)
ax.set_title("Day-ahead purchase plans chosen by each policy"); ax.legend(loc="upper left", fontsize=8)
save(fig, "fig3_plans.png")

# ---- Fig 4: recourse schedules on three representative days -------------
fig, axs = plt.subplots(3, 3, figsize=(10, 7), sharex=True)
xsp = A["plan_2"]
for j, name in enumerate(["overcast", "partly cloudy", "clear"]):
    Sx, Dx = A[f"rep_{name}_S"], A[f"rep_{name}_D"]
    a = axs[0, j]
    a.plot(H, Dx, color=INK2, lw=1.6, label="demand"); a.plot(H, Sx, color=ORANGE, lw=1.6, label="solar")
    a.step(H, xsp, where="mid", color=BLUE, lw=1.8, label="day-ahead plan")
    a.set_title(f"{name} day  (cost INR {A[f'rep_{name}_cost'][0]:,.0f})")
    if j == 0: a.set_ylabel("kW")
    a = axs[1, j]
    a.bar(H, A[f"rep_{name}_g"], color=ORANGE, width=.8, label="real-time purchase")
    a.bar(H, -A[f"rep_{name}_e"], color=AQUA, width=.8, label="surplus export")
    a.axhline(0, color="#c3c2b7", lw=.8)
    if j == 0: a.set_ylabel("kWh (buy + / sell -)")
    a = axs[2, j]
    a.plot(H, A[f"rep_{name}_b"], color=BLUE, lw=1.8)
    a.set_ylim(0, 1050); a.set_xlabel("hour of day")
    if j == 0: a.set_ylabel("battery SoC (kWh)")
h1, l1 = axs[0, 0].get_legend_handles_labels(); h2, l2 = axs[1, 0].get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(); save(fig, "fig4_schedules.png")

# ---- Fig 5: cost distribution (ECDF) ------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.4))
series = [("Deterministic (EV)", A["cost_1"], ORANGE, "-"), ("Stochastic (risk-neutral)", A["cost_2"], BLUE, "-"),
          ("Stochastic (mean-CVaR, lam=0.5)", A["cost_3"], AQUA, "-"), ("Perfect information (WS)", A["ws_cost"], INK2, "--")]
for lab, c, col, ls in series:
    c = np.sort(c); ax.plot(c / 1000, np.linspace(0, 1, len(c)), color=col, lw=1.9, ls=ls, label=lab)
ax.set_xlabel("daily operating cost (INR thousand)"); ax.set_ylabel("cumulative probability")
ax.set_title("Out-of-sample distribution of daily cost (500 test days)"); ax.legend(loc="lower right", fontsize=8)
save(fig, "fig5_cost_ecdf.png")

# ---- Fig 6: Monte-Carlo SAA repetitions ---------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3))
bp = ax[0].boxplot([A["mc_ev"], A["mc_sp"]], tick_labels=["Deterministic\n(EV)", "Stochastic\n(SAA, N=200)"], widths=.5,
                   patch_artist=True, medianprops=dict(color=INK, lw=1.6))
for patch, col in zip(bp["boxes"], [ORANGE, BLUE]):
    patch.set_facecolor(col); patch.set_alpha(.55)
ax[0].set_ylabel("mean out-of-sample daily cost (INR)")
ax[0].set_title("30 independent training samples")
ax[1].plot(H, A["mc_plans_ev"].std(0, ddof=1), color=ORANGE, lw=1.8, label="Deterministic (EV)")
ax[1].plot(H, A["mc_plans_sp"].std(0, ddof=1), color=BLUE, lw=1.8, label="Stochastic (SAA)")
ax[1].set_xlabel("hour of day"); ax[1].set_ylabel("std. dev. of x_t across 30 runs (kWh)")
ax[1].set_title("Stability of the day-ahead plan"); ax[1].legend(fontsize=8)
fig.tight_layout(); save(fig, "fig6_montecarlo.png")

# ---- Fig 7: SAA convergence ---------------------------------------------
cv = R["convergence"]
fig, ax = plt.subplots(figsize=(7, 3.3))
ax.errorbar(cv["N"], cv["ins_mean"], yerr=cv["ins_sd"], color=ORANGE, marker="o", lw=1.7, capsize=3, label="in-sample optimum (optimistic)")
ax.errorbar(cv["N"], cv["oos_mean"], yerr=cv["oos_sd"], color=BLUE, marker="s", lw=1.7, capsize=3, label="out-of-sample cost of the plan")
ax.set_xscale("log"); ax.set_xticks(cv["N"]); ax.set_xticklabels(cv["N"])
ax.set_xlabel("number of scenarios N in the SAA"); ax.set_ylabel("expected daily cost (INR)")
ax.set_title("Sample-average-approximation convergence (mean ± sd over repeats)"); ax.legend(fontsize=8)
save(fig, "fig7_convergence.png")

# ---- Fig 8: risk frontier -----------------------------------------------
fr = R["frontier"]
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.plot(fr["mean"], fr["cvar90"], color=BLUE, lw=1.6, marker="o", ms=7)
offs = {0.0: (6, 5), 0.25: (6, 5), 0.5: (-8, -14), 0.75: (-6, 9), 0.9: (4, -14), 1.0: (8, 6)}
for l, m, c in zip(fr["lam"], fr["mean"], fr["cvar90"]):
    ax.annotate(f"λ={l}", (m, c), textcoords="offset points", xytext=offs[l], fontsize=8, color=INK2)
ax.set_xlabel("expected daily cost (INR)"); ax.set_ylabel("CVaR$_{90}$ of daily cost (INR)")
ax.set_title("Risk-return frontier (out-of-sample)")
save(fig, "fig8_frontier.png")

# ---- Fig 9: sensitivity (VSS top row, EVPI bottom row) ------------------
sens = R["sensitivity"]
fig, axs = plt.subplots(2, 3, figsize=(10, 5), sharex="col")
specs = [("rt_premium", "real-time price multiplier"), ("p_over", "P(overcast day)"), ("batt_kwh", "battery capacity (kWh)")]
for j, (key, lab) in enumerate(specs):
    xs = [float(k) for k in sens[key]]
    vs = [v["vss"] for v in sens[key].values()]; ev = [v["evpi"] for v in sens[key].values()]
    axs[0, j].plot(xs, vs, color=BLUE, marker="o", lw=1.8); axs[1, j].plot(xs, ev, color=ORANGE, marker="s", lw=1.8)
    axs[1, j].set_xlabel(lab)
    if j == 0:
        axs[0, j].set_ylabel("VSS (INR/day)"); axs[1, j].set_ylabel("EVPI (INR/day)")
axs[0, 1].set_title("Value of the stochastic solution (VSS)"); axs[1, 1].set_title("Expected value of perfect information (EVPI)")
fig.tight_layout(); save(fig, "fig9_sensitivity.png")
print("figures written to", FIG)
