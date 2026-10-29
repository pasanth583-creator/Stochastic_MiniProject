"""
run_experiments.py -- reproduces every number, table and figure-data in the report.
Run:  python run_experiments.py      (takes ~5-8 minutes on a laptop)
Outputs: ../results/results.json, ../results/arrays.npz
"""
import json, time, sys, os
import numpy as np
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from microgrid_sp import (Params, gen_scenarios, solve_sp, evaluate, ev_plan, naive_plan,
                          wait_and_see, cvar, _solve)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)
P = Params()
SEED_TRAIN, SEED_TEST = 1, 99
N_TRAIN, N_TEST = 200, 500
R, A = {}, {}          # results (json) and arrays (npz)
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def summ(c, alpha=0.9):
    return {"mean": float(c.mean()), "std": float(c.std(ddof=1)), "p05": float(np.quantile(c, .05)),
            "p95": float(np.quantile(c, .95)), "cvar90": cvar(c, alpha), "max": float(c.max())}


def ci95(d):
    se = d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)]


# ------------------------------------------------------------------ E1 base case
log("E1: base case")
S, D = gen_scenarios(N_TRAIN, SEED_TRAIN, P)
St, Dt = gen_scenarios(N_TEST, SEED_TEST, P)
A.update(S_train=S, D_train=D, S_test=St, D_test=Dt)

sp0 = solve_sp(S, D, P)
sp_cv = solve_sp(S, D, P, lam=0.5, alpha=0.9)
plans = {"Naive rule": naive_plan(S, D, P), "Deterministic (EV)": ev_plan(S, D, P),
         "Stochastic (risk-neutral)": sp0["x"], "Stochastic (mean-CVaR, lam=0.5)": sp_cv["x"]}
full, costs = {}, {}
for k, x in plans.items():
    full[k] = _solve(St, Dt, P, x_fixed=x)
    costs[k] = full[k]["total_cost"]
ws = wait_and_see(St, Dt, P)
A["ws_cost"] = ws
for i, (k, x) in enumerate(plans.items()):
    A[f"plan_{i}"] = x
    A[f"cost_{i}"] = costs[k]
R["policies"] = list(plans.keys())
R["policy_stats"] = {k: summ(c) for k, c in costs.items()}
R["wait_and_see"] = summ(ws)
rp = costs["Stochastic (risk-neutral)"]
eev = costs["Deterministic (EV)"]
R["in_sample"] = {"RP_obj": float(sp0["obj"]),
                  "EEV_in_sample": float(evaluate(plans["Deterministic (EV)"], S, D, P).mean()),
                  "EV_problem_obj": float(_solve(S.mean(0, keepdims=True), D.mean(0, keepdims=True), P)["obj"])}
R["VSS_oos"] = float(eev.mean() - rp.mean())
R["VSS_ci95"] = ci95(eev - rp)
R["VSS_pct"] = float(100 * (eev.mean() - rp.mean()) / eev.mean())
R["EVPI_oos"] = float(rp.mean() - ws.mean())
R["EVPI_ci95"] = ci95(rp - ws)
R["EVPI_pct"] = float(100 * (rp.mean() - ws.mean()) / rp.mean())
R["SP_vs_naive_saving"] = float(costs["Naive rule"].mean() - rp.mean())
R["frac_scen_SP_better_than_EV"] = float((rp < eev).mean())

# operational / sustainability metrics (per day, test set)
solar_mean = St.sum(1).mean()
R["ops"] = {}
for k in plans:
    f = full[k]
    R["ops"][k] = {"rt_purchase_kWh": float(f["g"].sum(1).mean()),
                   "export_kWh": float(f["e"].sum(1).mean()),
                   "net_grid_import_kWh": float(f["x"].sum() + f["g"].sum(1).mean() - f["e"].sum(1).mean()),
                   "battery_throughput_kWh": float(f["d"].sum(1).mean()),
                   "day_ahead_kWh": float(f["x"].sum()),
                   "imbalance_share_pct": float(100 * f["g"].sum(1).mean() / Dt.sum(1).mean())}
R["scenario_stats"] = {"solar_kWh_mean": float(solar_mean), "solar_kWh_sd": float(St.sum(1).std(ddof=1)),
                       "demand_kWh_mean": float(Dt.sum(1).mean()), "demand_kWh_sd": float(Dt.sum(1).std(ddof=1)),
                       "solar_kWh_min": float(St.sum(1).min()), "solar_kWh_max": float(St.sum(1).max())}
# representative test days for schedule figure
order = np.argsort(St.sum(1))
idx = {"overcast": int(order[int(0.05 * N_TEST)]), "partly cloudy": int(order[int(0.5 * N_TEST)]),
       "clear": int(order[int(0.95 * N_TEST)])}
R["rep_days"] = idx
for name, i in idx.items():
    f = full["Stochastic (risk-neutral)"]
    for key in ("c", "d", "b", "g", "e"):
        A[f"rep_{name}_{key}"] = f[key][i]
    A[f"rep_{name}_S"] = St[i]; A[f"rep_{name}_D"] = Dt[i]
    A[f"rep_{name}_cost"] = np.array([f["total_cost"][i]])
log(f"  RP={rp.mean():.0f}  EEV={eev.mean():.0f}  WS={ws.mean():.0f}  VSS={R['VSS_oos']:.0f}  EVPI={R['EVPI_oos']:.0f}")

# ------------------------------------------------------------------ E2 Monte-Carlo SAA repetitions
log("E2: 30 independent SAA repetitions")
reps = 30
mc = {"sp": [], "ev": [], "plans_sp": [], "plans_ev": [], "insample": []}
for i in range(reps):
    Si, Di = gen_scenarios(N_TRAIN, 1000 + i, P)
    r = solve_sp(Si, Di, P)
    xe = ev_plan(Si, Di, P)
    mc["sp"].append(evaluate(r["x"], St, Dt, P).mean())
    mc["ev"].append(evaluate(xe, St, Dt, P).mean())
    mc["plans_sp"].append(r["x"]); mc["plans_ev"].append(xe); mc["insample"].append(r["obj"])
for k in mc:
    mc[k] = np.array(mc[k])
vss_i = mc["ev"] - mc["sp"]
R["mc"] = {"reps": reps, "sp_mean": float(mc["sp"].mean()), "sp_sd": float(mc["sp"].std(ddof=1)),
           "ev_mean": float(mc["ev"].mean()), "ev_sd": float(mc["ev"].std(ddof=1)),
           "vss_mean": float(vss_i.mean()), "vss_sd": float(vss_i.std(ddof=1)),
           "vss_min": float(vss_i.min()), "vss_max": float(vss_i.max()),
           "frac_sp_better": float((vss_i > 0).mean()),
           "insample_mean": float(mc["insample"].mean()),
           "plan_sp_sd_mean": float(mc["plans_sp"].std(0, ddof=1).mean()),
           "plan_ev_sd_mean": float(mc["plans_ev"].std(0, ddof=1).mean())}
A.update(mc_sp=mc["sp"], mc_ev=mc["ev"], mc_plans_sp=mc["plans_sp"], mc_plans_ev=mc["plans_ev"])
log(f"  SP {mc['sp'].mean():.0f}+-{mc['sp'].std(ddof=1):.0f}  EV {mc['ev'].mean():.0f}+-{mc['ev'].std(ddof=1):.0f}  VSS {vss_i.mean():.0f}")

# ------------------------------------------------------------------ E3 SAA convergence in N
log("E3: SAA convergence")
Ns = [5, 10, 25, 50, 100, 200, 400]
conv = {"N": Ns, "oos_mean": [], "oos_sd": [], "ins_mean": [], "ins_sd": []}
for N in Ns:
    nrep = 10 if N < 400 else 6
    oo, ii = [], []
    for j in range(nrep):
        Si, Di = gen_scenarios(N, 5000 + 31 * N + j, P)
        r = solve_sp(Si, Di, P)
        oo.append(evaluate(r["x"], St, Dt, P).mean()); ii.append(r["obj"])
    conv["oos_mean"].append(float(np.mean(oo))); conv["oos_sd"].append(float(np.std(oo, ddof=1)))
    conv["ins_mean"].append(float(np.mean(ii))); conv["ins_sd"].append(float(np.std(ii, ddof=1)))
    log(f"  N={N}: in-sample {np.mean(ii):.0f}  out-of-sample {np.mean(oo):.0f} +- {np.std(oo, ddof=1):.0f}")
R["convergence"] = conv

# ------------------------------------------------------------------ E4 risk frontier
log("E4: mean-CVaR frontier")
lams = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
fr = {"lam": lams, "mean": [], "cvar90": [], "std": [], "p95": []}
frplans = []
for lam in lams:
    r = solve_sp(S, D, P, lam=lam, alpha=0.9)
    c = evaluate(r["x"], St, Dt, P)
    fr["mean"].append(float(c.mean())); fr["cvar90"].append(cvar(c, 0.9))
    fr["std"].append(float(c.std(ddof=1))); fr["p95"].append(float(np.quantile(c, .95)))
    frplans.append(r["x"])
    log(f"  lam={lam}: mean {c.mean():.0f} cvar90 {cvar(c, .9):.0f}")
R["frontier"] = fr
A["frontier_plans"] = np.array(frplans)

# ------------------------------------------------------------------ E5 sensitivity
log("E5: sensitivity sweeps")


def sens_point(Pm, seeds=(7, 8, 9, 10), n_train=150):
    """Average over several independent training samples to damp SAA noise."""
    Sv, Dv = gen_scenarios(N_TEST, SEED_TEST, Pm)
    cws = wait_and_see(Sv, Dv, Pm)
    sp_l, ev_l = [], []
    for sd in seeds:
        Si, Di = gen_scenarios(n_train, sd, Pm)
        sp_l.append(evaluate(solve_sp(Si, Di, Pm)["x"], Sv, Dv, Pm).mean())
        ev_l.append(evaluate(ev_plan(Si, Di, Pm), Sv, Dv, Pm).mean())
    sp_m, ev_m, ws_m = float(np.mean(sp_l)), float(np.mean(ev_l)), float(cws.mean())
    vs = np.array(ev_l) - np.array(sp_l)
    return {"sp": sp_m, "ev": ev_m, "ws": ws_m, "vss": float(vs.mean()), "vss_sd": float(vs.std(ddof=1)),
            "evpi": sp_m - ws_m, "vss_pct": 100 * (ev_m - sp_m) / ev_m, "evpi_pct": 100 * (sp_m - ws_m) / sp_m}


sens = {}
sens["rt_premium"] = {str(v): sens_point(replace(P, rt_premium=v)) for v in [1.0, 1.2, 1.4, 1.6, 2.0, 2.5]}
log("  rt_premium done")
sens["p_over"] = {str(v): sens_point(replace(P, p_over=v)) for v in [0.05, 0.15, 0.25, 0.40, 0.55]}
log("  p_over done")
sens["batt_kwh"] = {str(v): sens_point(replace(P, batt_kwh=float(v), batt_kw=v / 4.0)) for v in [0, 500, 1000, 2000]}
log("  battery done")
R["sensitivity"] = sens

# ------------------------------------------------------------------ E6 distribution shift
log("E6: distribution shift (train p_over vs test p_over)")
grid = [0.05, 0.25, 0.50]
shift = {"grid": grid, "sp": [], "ev": []}
tests = {pt: gen_scenarios(N_TEST, SEED_TEST, replace(P, p_over=pt)) for pt in grid}
for ptr in grid:
    Pm = replace(P, p_over=ptr)
    Si, Di = gen_scenarios(150, 7, Pm)
    xs, xe = solve_sp(Si, Di, Pm)["x"], ev_plan(Si, Di, Pm)
    shift["sp"].append([float(evaluate(xs, *tests[pt], replace(P, p_over=pt)).mean()) for pt in grid])
    shift["ev"].append([float(evaluate(xe, *tests[pt], replace(P, p_over=pt)).mean()) for pt in grid])
R["shift"] = shift

R["runtime_s"] = time.time() - t0
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(R, f, indent=2)
np.savez_compressed(os.path.join(OUT, "arrays.npz"), **A)
log("DONE")
