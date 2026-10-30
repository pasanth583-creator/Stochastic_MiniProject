"""Builds ../notebook/Microgrid_Stochastic_Optimization.ipynb (no nbformat needed)."""
import json, os
cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)})

md("""# Two-Stage Stochastic Optimization of a Campus Solar + Battery Microgrid
**21AIE321T Stochastic Decision Making - Mini Project (Assignment IV)**

This notebook is a short, runnable walk-through of the model in the report (smaller sample sizes so it runs in about a minute).
The full experiments are in `code/run_experiments.py`.

**Problem.** Commit an hourly day-ahead grid purchase `x_t` *before* solar `S_t` and demand `D_t` are known; then adapt with a battery, real-time purchases `g_t` (price = 1.6 x day-ahead) and surplus export `e_t`.
All scenarios are **synthetic** (see `gen_scenarios`).""")
code("""import sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, "../code")
from microgrid_sp import *
P = Params()
print(P)""")
md("## 1. Scenarios (training and independent test sets)")
code("""S, D = gen_scenarios(100, seed=1, P=P)          # training scenarios for SAA
St, Dt = gen_scenarios(300, seed=99, P=P)       # unseen test days
h = np.arange(24)
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
for a, M, t in ((ax[0], S, "Solar (kW)"), (ax[1], D, "Demand (kW)")):
    a.fill_between(h, np.quantile(M, .05, 0), np.quantile(M, .95, 0), alpha=.25)
    a.plot(h, M.mean(0)); a.set_title(t); a.set_xlabel("hour")
plt.show()""")
md("## 2. Solve: stochastic (SAA), deterministic expected-value, and naive plans")
code("""sp = solve_sp(S, D, P)                  # two-stage SAA LP (risk neutral)
x_sp = sp["x"]; x_ev = ev_plan(S, D, P); x_nv = naive_plan(S, D, P)
print("SAA optimum (in-sample): %.0f INR/day" % sp["obj"])
pd.DataFrame({"naive": x_nv, "EV": x_ev, "stochastic": x_sp}).round(0).T""")
md("## 3. Out-of-sample evaluation, VSS and EVPI\n`evaluate` fixes the day-ahead plan and lets every test scenario re-optimise its recourse.")
code("""c_sp, c_ev, c_nv = (evaluate(x, St, Dt, P) for x in (x_sp, x_ev, x_nv))
c_ws = wait_and_see(St, Dt, P)           # perfect information: x chosen per scenario
tab = pd.DataFrame({"mean": [c.mean() for c in (c_nv, c_ev, c_sp, c_ws)],
                    "std": [c.std(ddof=1) for c in (c_nv, c_ev, c_sp, c_ws)],
                    "CVaR90": [cvar(c, .9) for c in (c_nv, c_ev, c_sp, c_ws)]},
                   index=["naive", "deterministic (EV)", "stochastic (RP)", "wait-and-see (WS)"]).round(0)
print(tab)
print("\\nVSS  = EEV - RP = %.0f INR/day" % (c_ev.mean() - c_sp.mean()))
print("EVPI = RP - WS  = %.0f INR/day" % (c_sp.mean() - c_ws.mean()))""")
md("## 4. Risk-averse extension (mean-CVaR)")
code("""rows = []
for lam in (0.0, 0.25, 0.5):
    x = solve_sp(S, D, P, lam=lam, alpha=0.9)["x"]; c = evaluate(x, St, Dt, P)
    rows.append((lam, c.mean(), cvar(c, .9)))
pd.DataFrame(rows, columns=["lambda", "mean cost", "CVaR90"]).round(0)""")
md("## 5. Sensitivity to the real-time price multiplier (kappa)")
code("""from dataclasses import replace
rows = []
for k in (1.0, 1.6, 2.5):
    Pk = replace(P, rt_premium=k)
    Sk, Dk = gen_scenarios(100, 1, Pk); Sv, Dv = gen_scenarios(300, 99, Pk)
    a = evaluate(solve_sp(Sk, Dk, Pk)["x"], Sv, Dv, Pk).mean()
    b = evaluate(ev_plan(Sk, Dk, Pk), Sv, Dv, Pk).mean()
    rows.append((k, a, b, b - a))
pd.DataFrame(rows, columns=["kappa", "stochastic", "deterministic", "VSS"]).round(0)""")
md("""## Notes
* Results here use smaller samples than the report, so numbers differ slightly (sampling noise); the report uses N = 200 training / 500 test scenarios, 30 repeats and averaged sensitivity runs.
* With kappa = 1.0 (no imbalance penalty) VSS and EVPI are zero - a useful sanity check of the model.""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebook", "Microgrid_Stochastic_Optimization.ipynb")
json.dump(nb, open(out, "w"), indent=1); print("notebook written")
