"""build_report.py -- assembles the PDF report from results.json + figures (all numbers are read from results)."""
import json, os
import numpy as np
import matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
                                Image, KeepTogether, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG, OUT = (os.path.join(HERE, "..", d) for d in ("results", "figures", "report"))
os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(RES, "results.json")))
A = np.load(os.path.join(RES, "arrays.npz"))

# ---- fonts (DejaVu has Greek, math symbols and the rupee sign) ----
fd = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
for n, f in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
             ("DV-BI", "DejaVuSans-BoldOblique.ttf")):
    pdfmetrics.registerFont(TTFont(n, os.path.join(fd, f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK, INK2, BLUE, LIGHT = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#2a78d6"), colors.HexColor("#f3f2ee")
body = ParagraphStyle("body", fontName="DV", fontSize=8.6, leading=12, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=4)
small = ParagraphStyle("small", parent=body, fontSize=7.6, leading=10, alignment=TA_LEFT, textColor=INK2)
cap = ParagraphStyle("cap", parent=small, alignment=TA_CENTER, spaceBefore=1, spaceAfter=6)
h1 = ParagraphStyle("h1", parent=body, fontName="DV-B", fontSize=11, leading=14, spaceBefore=7, spaceAfter=3, textColor=BLUE, alignment=TA_LEFT)
h2 = ParagraphStyle("h2", parent=body, fontName="DV-B", fontSize=9.2, leading=12, spaceBefore=4, spaceAfter=2, alignment=TA_LEFT)
title = ParagraphStyle("title", parent=body, fontName="DV-B", fontSize=14.5, leading=18, alignment=TA_LEFT, spaceAfter=2)
bul = ParagraphStyle("bul", parent=body, leftIndent=11, bulletIndent=2, spaceAfter=2)
eqs = ParagraphStyle("eq", parent=body, alignment=TA_LEFT, fontSize=8.6, leading=12, spaceAfter=0)
tcell = ParagraphStyle("tcell", parent=body, fontSize=7.7, leading=9.6, alignment=TA_LEFT, spaceAfter=0)
tcellb = ParagraphStyle("tcellb", parent=tcell, fontName="DV-B")


def P(t, s=body): return Paragraph(t, s)
def B(t): return Paragraph(t, bul, bulletText="•")
def fm(x, d=0): return f"{x:,.{d}f}"
def inr(x): return "₹" + fm(x)


def eq(t, n):
    tb = Table([[P(t, eqs), P(f"({n})", eqs)]], colWidths=[15.2 * cm, 1.6 * cm])
    tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                            ("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("TOPPADDING", (0, 0), (-1, -1), 2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("LEFTPADDING", (0, 0), (0, 0), 10)]))
    return [tb, Spacer(1, 2)]


def table(data, widths, hdr=True, zebra=True, align_right_from=1):
    rows = [[P(str(c), tcellb if (hdr and i == 0) else tcell) for c in r] for i, r in enumerate(data)]
    t = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1 if hdr else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c3c2b7")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if hdr: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7f0fb")))
    if zebra:
        for i in range(2, len(rows), 2): st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(st)); return t


def img(name, w_cm):
    p = os.path.join(FIG, name); im = PILImage.open(p); r = im.size[1] / im.size[0]
    return Image(p, width=w_cm * cm, height=w_cm * r * cm)


def fig(name, w_cm, caption):
    return KeepTogether([img(name, w_cm), P(caption, cap)])


# ------------------------------------------------------------------ numbers
pol = R["policies"]; ps = R["policy_stats"]
naive, ev, sp, cv = (ps[k] for k in pol)
ws = R["wait_and_see"]; mc = R["mc"]; ops = R["ops"]; sc = R["scenario_stats"]; fr = R["frontier"]
sens = R["sensitivity"]; conv = R["convergence"]; shift = R["shift"]
vss, vss_ci, evpi, evpi_ci = R["VSS_oos"], R["VSS_ci95"], R["EVPI_oos"], R["EVPI_ci95"]
fr_i = lambda lam: fr["lam"].index(lam)
c25 = fr_i(0.25)
cvar_gain25 = 100 * (fr["cvar90"][0] - fr["cvar90"][c25]) / fr["cvar90"][0]
mean_pen25 = 100 * (fr["mean"][c25] - fr["mean"][0]) / fr["mean"][0]
cvar_gain50 = 100 * (sp["cvar90"] - cv["cvar90"]) / sp["cvar90"]
mean_pen50 = 100 * (cv["mean"] - sp["mean"]) / sp["mean"]
batt_val = sens["batt_kwh"]["0"]["sp"] - sens["batt_kwh"]["1000"]["sp"]
rt = sens["rt_premium"]; po = sens["p_over"]; bt = sens["batt_kwh"]
sp_plan, ev_plan_, nv_plan = A["plan_2"], A["plan_1"], A["plan_0"]

# ------------------------------------------------------------------ story
st = []
st += [P("Stochastic Optimization of Day-Ahead Energy Procurement for a Campus Solar + Battery Microgrid", title),
       P("Mini Project (Assignment IV, CLA-2) · 21AIE321T Stochastic Decision Making · B.Tech AI, Semester V · SRM Institute of Science and Technology, Tiruchirappalli · AY 2026-27", small),
       P("<b>Student:</b> [Your name] · <b>Register No.:</b> [Your register number] · <b>Faculty:</b> Dr. Himanshu Rai · <b>SDG 9 (Target 9.5)</b> · Code: <i>code/</i> folder (microgrid_sp.py, run_experiments.py, notebook)", small),
       Spacer(1, 4)]

st += [P("Abstract", h1),
       P(f"A campus with solar PV and a battery must commit an hourly grid-purchase schedule one day ahead, before the weather and the load are known. "
         f"Shortfalls are settled at a penalty price and surplus is sold cheaply, so the cost of a wrong commitment is asymmetric. I model this as a "
         f"<b>two-stage stochastic linear program</b> solved by Sample Average Approximation (SAA) with N = 200 scenarios, extend it with a "
         f"<b>mean-CVaR</b> risk term, and test every plan on 500 independent unseen days. The stochastic plan costs {inr(sp['mean'])}/day against "
         f"{inr(ev['mean'])} for the deterministic (expected-value) plan and {inr(naive['mean'])} for a rule-of-thumb plan. The Value of the Stochastic "
         f"Solution is {inr(vss)}/day (95% CI {inr(vss_ci[0])}-{inr(vss_ci[1])}); the Expected Value of Perfect Information is {inr(evpi)}/day "
         f"({R['EVPI_pct']:.1f}% of cost). The stochastic advantage rises to {rt['2.5']['vss_pct']:.1f}% when the real-time price multiplier is 2.5. "
         f"<b>Data note:</b> all weather and demand scenarios are synthetic (generated from stated assumptions), not measured campus data.")]

# ---- 1 problem
st += [P("1. Real-world problem and why it is stochastic", h1),
       P("<b>Setting.</b> An Indian university campus (peak load ≈ 0.7 MW) owns an 800 kWp solar plant and a 1 MWh / 250 kW battery, and buys the "
         "remaining energy from the grid. Following the day-ahead scheduling and deviation-settlement structure used in power markets, the campus "
         "must declare how much it will buy in each hour of tomorrow <i>today</i>. Energy bought in real time to cover a shortfall is charged at a "
         "premium (assumed 1.6× the day-ahead price), and surplus power is exported at a low price (₹3/kWh)."),
       P("<b>Uncertainty.</b> (i) Solar output depends on cloud cover and the monsoon (bimodal: clear versus overcast days); (ii) demand differs between "
         "working days and holidays. Because the cost of buying too little (premium price) differs from the cost of buying too much (cheap export), "
         "the best commitment is not the one that matches average conditions - a newsvendor-type asymmetry. This is exactly the case where "
         "replacing random inputs by their averages is expected to be sub-optimal, and the project measures by how much."),
       P("<b>Link to SDG 9 (Target 9.5).</b> Better scheduling of renewable energy plus storage is a technology-upgrading measure for industrial and "
         "campus infrastructure; Section 8 defines measurable indicators (imbalance energy, net grid import, battery value).")]

# ---- 2 formulation
st += [P("2. Formal stochastic optimization model", h1),
       P("<b>Sets and data.</b> Hours t ∈ {1,…,24}; scenarios ω ∈ Ω = {1,…,N}, each with probability 1/N. Random parameters: solar S<sub>t</sub><super>ω</super> and demand "
         "D<sub>t</sub><super>ω</super> (kW = kWh per hour). Prices: day-ahead π<sub>t</sub> (time-of-day tariff), real-time π<sub>t</sub><super>RT</super> = κπ<sub>t</sub> (κ = 1.6), "
         "export price π<super>X</super> = 3, battery wear cost c<super>d</super> = 1 ₹/kWh. Battery: capacity B<super>max</super> = 1000 kWh, floor B<super>min</super> = 100 kWh, initial B<sub>0</sub> = 500 kWh, "
         "power limit P<super>max</super> = 250 kW, one-way efficiency η = 0.92. Contract limit X<super>max</super> = 1000 kW."),
       P("<b>Decision variables.</b> <i>First stage</i> (chosen before ω is known, identical for all scenarios - this is the non-anticipativity requirement): "
         "x<sub>t</sub> ≥ 0, the day-ahead purchase. <i>Second stage / recourse</i> (chosen after ω is revealed): battery charge c<sub>t</sub><super>ω</super>, discharge d<sub>t</sub><super>ω</super>, "
         "state of charge b<sub>t</sub><super>ω</super>, real-time purchase g<sub>t</sub><super>ω</super> ≥ 0 and exported surplus e<sub>t</sub><super>ω</super> ≥ 0.")]
st += eq("<b>min</b> Σ<sub>t</sub> π<sub>t</sub> x<sub>t</sub> + (1/N) Σ<sub>ω</sub> Σ<sub>t</sub> [ π<sub>t</sub><super>RT</super> g<sub>t</sub><super>ω</super> − π<super>X</super> e<sub>t</sub><super>ω</super> + c<super>d</super> d<sub>t</sub><super>ω</super> ]", 1)
st += eq("x<sub>t</sub> + S<sub>t</sub><super>ω</super> + d<sub>t</sub><super>ω</super> + g<sub>t</sub><super>ω</super> = D<sub>t</sub><super>ω</super> + c<sub>t</sub><super>ω</super> + e<sub>t</sub><super>ω</super>  &nbsp;&nbsp;(energy balance, ∀t, ω)", 2)
st += eq("b<sub>t</sub><super>ω</super> = b<sub>t−1</sub><super>ω</super> + η c<sub>t</sub><super>ω</super> − d<sub>t</sub><super>ω</super>/η, &nbsp; b<sub>0</sub> = B<sub>0</sub>, &nbsp; B<super>min</super> ≤ b<sub>t</sub><super>ω</super> ≤ B<super>max</super>, &nbsp; b<sub>24</sub><super>ω</super> ≥ B<sub>0</sub>", 3)
st += eq("0 ≤ c<sub>t</sub><super>ω</super>, d<sub>t</sub><super>ω</super> ≤ P<super>max</super>, &nbsp; g<sub>t</sub><super>ω</super>, e<sub>t</sub><super>ω</super> ≥ 0, &nbsp; 0 ≤ x<sub>t</sub> ≤ X<super>max</super>", 4)
st += [P("Constraint (2) is the hourly power balance; (3) is battery dynamics with losses, the minimum charge and a terminal condition that stops the model from "
         "emptying the battery at the end of the day. Because g<sub>t</sub><super>ω</super> is unrestricted above, every first-stage plan is feasible in every scenario "
         "(<i>relatively complete recourse</i>), so no scenario can make the problem infeasible. The expectation in (1) is replaced by the average over N sampled "
         "scenarios (<b>Sample Average Approximation</b>, Kleywegt et al., 2002). The resulting extensive-form LP has 24 + 120N variables and 48N equality "
         "constraints (24,024 variables and 9,600 constraints for N = 200).")]
st += [P("<b>Risk-averse variant.</b> With Z<sup>ω</sup> the total daily cost in scenario ω, I minimise (1−λ)·E[Z] + λ·CVaR<sub>α</sub>(Z) with α = 0.9, using the "
         "Rockafellar-Uryasev linearisation", body)]
st += eq("CVaR<sub>α</sub>(Z) = min<sub>τ,u</sub> { τ + 1/((1−α)N) Σ<sub>ω</sub> u<sup>ω</sup> : u<sup>ω</sup> ≥ Z<sup>ω</sup> − τ, u<sup>ω</sup> ≥ 0 }", 5)
st += [P("<b>Benchmarks from stochastic programming theory.</b> RP = expected cost of the stochastic plan x*; EV plan = optimum of the problem with S, D replaced by their means; "
         "EEV = expected cost of that EV plan when real uncertainty is faced (with optimal recourse); WS = expected cost if each day's scenario were known in advance "
         "(a separate optimisation per scenario). Then <b>VSS = EEV − RP ≥ 0</b> is the value of modelling uncertainty and <b>EVPI = RP − WS ≥ 0</b> is the most the campus "
         "should pay for a perfect forecast. All three are estimated <i>out-of-sample</i> on 500 independent test days to avoid the optimistic bias of in-sample SAA values.")]

# ---- 3 data
st += [P("3. Data, scenario model and assumptions", h1),
       P("No metered campus data were available, so scenarios are <b>synthetically generated</b> from a transparent model (code: <i>gen_scenarios</i>). Parameters are illustrative "
         "assumptions chosen to be plausible, not values from a tariff order or a measured data set.")]
st += [table([
    ["Item", "Assumption"],
    ["Solar", "800 kWp × clear-sky shape (peak 0.88) × clearness k<sub>t</sub>. Day type: clear 35%, partly cloudy 40%, overcast 25%; day-mean k = 0.92 / 0.62 / 0.28 (+N(0,0.05)); hourly AR(1) noise, ρ = 0.8, σ = 0.03 / 0.18 / 0.08"],
    ["Demand", "Campus base profile (240-720 kW) × exp(N(0,0.08)) × 0.72 on holiday-type days (prob. 2/7) × AR(1) hourly noise (ρ = 0.7, σ = 5%)"],
    ["Day-ahead tariff", "₹5.5 (22:00-06:00), ₹6.5 (06-09), ₹6.0 (09-18, solar hours), ₹9.0 (18-22 evening peak)"],
    ["Real-time / export", "Real-time = 1.6 × day-ahead; export ₹3.0/kWh; battery wear ₹1/kWh discharged"],
    ["Sample sizes", "Training N = 200 (seed 1); independent test set 500 days (seed 99); 30 independent training repeats; sensitivity points average 4 training samples"]],
    [3.0, 13.8]), Spacer(1, 4),
    P(f"Resulting daily totals (test set): solar {fm(sc['solar_kWh_mean'])} kWh (sd {fm(sc['solar_kWh_sd'])}, range {fm(sc['solar_kWh_min'])}-{fm(sc['solar_kWh_max'])}); "
      f"demand {fm(sc['demand_kWh_mean'])} kWh (sd {fm(sc['demand_kWh_sd'])}). Solar covers roughly {100 * sc['solar_kWh_mean'] / sc['demand_kWh_mean']:.0f}% of demand on average, "
      f"but with a large day-to-day spread.")]
st += [fig("fig1_scenarios.png", 15.6, "Figure 1. Hourly solar and demand scenarios: mean, 5-95% band and five sample paths. The solar band is wide because of the clear/overcast mixture."),
       ]

# ---- 4 method
st += [P("4. Choice of method and justification", h1),
       P("<b>Two-stage stochastic programming with recourse (solved by SAA)</b> was chosen because: (1) the decision genuinely has two stages - commit first, adapt later with the battery and "
         "real-time market; (2) the cost of a violation is a <i>price</i>, not a prohibition, so expected-cost recourse is more natural than chance constraints; (3) the model is a linear program, so each "
         "SAA problem is solved to proven global optimality; (4) the cost of wrong commitments is asymmetric, which an average-value model cannot capture. "
         "Alternatives considered: <i>Monte Carlo simulation alone</i> evaluates a plan but does not optimise it (I use it for out-of-sample testing); <i>metaheuristics such as simulated annealing</i> "
         "give no optimality guarantee and are slower on a 24-dimensional continuous problem that an LP solver handles in about one second; <i>robust optimisation</i> would plan for the worst day and "
         "is too conservative for an expected-cost objective. CVaR is added only to treat the large daily cost spread."),
       P("<b>Implementation.</b> Python 3 with NumPy/SciPy; the LP is assembled as a sparse matrix and solved with the HiGHS solver (<i>scipy.optimize.linprog</i>). One SAA solve with N = 200 "
         f"takes about 1 s; the full experiment suite (≈ {R['runtime_s'] / 60:.0f} min) is reproducible from <i>run_experiments.py</i> with fixed seeds.")]

# ---- 5 results
st += [P("5. Results", h1), P("5.1 Day-ahead plans and recourse behaviour", h2),
       P(f"Figure 2 compares the plans. The deterministic plan treats the average solar curve as certain. The stochastic plan sits at "
         f"{fm(sp_plan[7:13].mean())} kW on average over 07:00-12:00 (deterministic: {fm(ev_plan_[7:13].mean())} kW) and commits {fm(sp_plan[17:24].mean())} kW over 17:00-23:00 "
         f"(deterministic: {fm(ev_plan_[17:24].mean())} kW), which is consistent with hedging the evening hours, where a real-time purchase is most expensive (₹{1.6 * 9.0:.1f}/kWh at the evening peak). "
         f"Figure 3 shows the second-stage response of this plan on three test days: on an overcast day the battery is pre-charged at night and the campus buys heavily in real time; on a clear day "
         f"the surplus is exported and the battery carries solar energy into the evening peak.")]
st += [fig("fig3_plans.png", 11.5, "Figure 2. Day-ahead purchase plans x<sub>t</sub> of the naive rule, the deterministic (EV) model and the stochastic model."),
       fig("fig4_schedules.png", 15.2, "Figure 3. Recourse decisions of the stochastic plan on an overcast, a partly cloudy and a clear test day (rows: demand/solar/plan; real-time buy and export; battery state of charge).")]

st += [P("5.2 Out-of-sample cost comparison (500 unseen days)", h2)]
rows = [["Policy", "Mean ₹/day", "Std dev", "95th pct", "CVaR<sub>90</sub>", "vs stochastic"]]
for k in pol:
    s = ps[k]; rows.append([k, fm(s["mean"]), fm(s["std"]), fm(s["p95"]), fm(s["cvar90"]), f"{s['mean'] - sp['mean']:+,.0f}"])
rows.append(["Perfect information (wait-and-see)", fm(ws["mean"]), fm(ws["std"]), fm(ws["p95"]), fm(ws["cvar90"]), f"{ws['mean'] - sp['mean']:+,.0f}"])
st += [table(rows, [5.6, 2.2, 1.9, 2.0, 2.0, 2.4]), Spacer(1, 3),
       P(f"<b>Table 2.</b> Daily operating cost on the same 500 test scenarios. The risk-neutral stochastic plan has the lowest mean among implementable policies; the mean-CVaR plan (λ = 0.5) gives up "
         f"{mean_pen50:.1f}% of mean cost to cut CVaR<sub>90</sub> by {cvar_gain50:.1f}% and nearly halve the standard deviation ({fm(sp['std'])} → {fm(cv['std'])}).", small),
       P(f"<b>Value of the Stochastic Solution.</b> VSS = EEV − RP = {fm(ev['mean'])} − {fm(sp['mean'])} = <b>{inr(vss)}/day</b> ({R['VSS_pct']:.2f}% of the deterministic cost); the paired 95% interval "
         f"over the test days is [{fm(vss_ci[0])}, {fm(vss_ci[1])}], so the gain is statistically positive. The stochastic plan beats the deterministic one on {100 * R['frac_scen_SP_better_than_EV']:.0f}% of individual days "
         f"(the gain is a statement about the <i>average</i>, not every day). Against the rule-of-thumb plan the stochastic model saves {inr(R['SP_vs_naive_saving'])}/day. "
         f"Over 365 days the VSS is about ₹{R['VSS_oos'] * 365 / 1e5:.2f} lakh and the saving over the naive rule about ₹{R['SP_vs_naive_saving'] * 365 / 1e5:.1f} lakh per year (indicative, assuming the test-day distribution is representative)."),
       P(f"<b>Expected Value of Perfect Information.</b> EVPI = RP − WS = {fm(sp['mean'])} − {fm(ws['mean'])} = <b>{inr(evpi)}/day</b> ({R['EVPI_pct']:.1f}% of cost; 95% CI {fm(evpi_ci[0])}-{fm(evpi_ci[1])}). "
         f"This is an <i>upper bound</i> on what any weather/load forecasting service can be worth: about ₹{evpi * 365 / 1e5:.0f} lakh per year. A real forecast is imperfect, so only a fraction of this is attainable. "
         f"The EVPI is about {evpi / vss:.0f} times the VSS, which says most of the avoidable cost is information, not modelling.")]
st += [fig("fig5_cost_ecdf.png", 11.5, "Figure 4. Empirical cumulative distribution of daily cost. The CVaR plan removes the expensive right tail; perfect information shifts the whole curve left.")]

st += [P("5.3 Variability of the solution: Monte Carlo repeats and SAA convergence", h2),
       P(f"To check that the result is not an accident of one random sample, the whole procedure was repeated 30 times with independent N = 200 training samples and scored on the same 500 test days (Figure 5). "
         f"The stochastic plan gives {fm(mc['sp_mean'])} ± {fm(mc['sp_sd'])} ₹/day (mean ± sd over repeats), the deterministic plan {fm(mc['ev_mean'])} ± {fm(mc['ev_sd'])}; the VSS is positive in "
         f"{100 * mc['frac_sp_better']:.0f}% of repeats (mean {fm(mc['vss_mean'])}, range {fm(mc['vss_min'])}-{fm(mc['vss_max'])}). "
         f"Figure 6 shows SAA behaviour as N grows: the in-sample optimum fluctuates strongly for small N, while the out-of-sample cost of the chosen plan falls and stabilises; "
         f"with N = 5 the plan is {fm(conv['oos_mean'][0] - conv['oos_mean'][-2])} ₹/day worse than with N = 200, and beyond N ≈ 50-100 the gains are small. One more point in the right panel of Figure 5: the stochastic plan moves more with the training sample than the deterministic one "
         f"(mean sd of x<sub>t</sub> {mc['plan_sp_sd_mean']:.0f} versus {mc['plan_ev_sd_mean']:.0f} kW) because it reacts to the tails of the sample, yet its <i>cost</i> is more stable (sd {fm(mc['sp_sd'])} versus {fm(mc['ev_sd'])} ₹/day), since the cost surface is flat near the optimum.")]
st += [fig("fig6_montecarlo.png", 15.2, "Figure 5. Left: out-of-sample cost over 30 independent training samples. Right: hour-wise spread of the chosen day-ahead plan across repeats."),
       fig("fig7_convergence.png", 10.5, "Figure 6. SAA convergence: in-sample (optimistic) versus out-of-sample cost as the scenario count N increases.")]

st += [P("5.4 Risk: mean-CVaR trade-off", h2),
       P(f"Daily cost has a large spread (sd ≈ {inr(sp['std'])}), driven by overcast days. Figure 7 traces the frontier as λ increases from 0 (risk-neutral) to 1. A modest weight λ = 0.25 reduces CVaR<sub>90</sub> by "
         f"{cvar_gain25:.1f}% for only {mean_pen25:.1f}% extra mean cost; λ = 0.5 reduces it by {cvar_gain50:.1f}% for {mean_pen50:.1f}%. Beyond λ ≈ 0.5 the out-of-sample CVaR stops improving "
         f"(it stays between {fm(min(fr['cvar90'][2:]))} and {fm(max(fr['cvar90'][2:]))}) because the 10% tail of a 200-scenario sample contains only 20 days, so estimates of the tail are noisy. The risk-averse plans (λ ≥ 0.5) buy almost "
         f"everything day-ahead and export surpluses at a loss; this lowers variance but is wasteful in expectation, so I recommend a small λ rather than λ ≈ 1.")]
st += [fig("fig8_frontier.png", 9.5, "Figure 7. Out-of-sample mean cost versus CVaR<sub>90</sub> for different risk weights λ.")]

# ---- 6 sensitivity
st += [P("6. Sensitivity analysis", h1),
       P("Each point in Figure 8 is the average over four independent training samples (N = 150) and is tested on 500 fresh days generated under the same changed parameters; EVPI uses the same test days.")]
rows = [["Parameter changed", "Value", "Stochastic ₹/day", "Deterministic ₹/day", "VSS ₹/day", "VSS %", "EVPI ₹/day"]]
for key, lab in (("rt_premium", "Real-time price multiplier κ"), ("p_over", "P(overcast day)"), ("batt_kwh", "Battery (kWh)")):
    for v, d in sens[key].items():
        rows.append([lab, v, fm(d["sp"]), fm(d["ev"]), fm(d["vss"]), f"{d['vss_pct']:.2f}", fm(d["evpi"])])
st += [table(rows, [4.4, 1.5, 2.6, 2.9, 1.9, 1.5, 2.0]), Spacer(1, 3),
       P("<b>Table 3.</b> Sensitivity of costs, VSS and EVPI to the price multiplier, the weather mixture and the battery size.", small),
       fig("fig9_sensitivity.png", 14.8, "Figure 8. VSS (top) and EVPI (bottom) as one parameter changes at a time; different scales, so the two measures are in separate panels."),
       P(f"<b>Findings.</b> (1) <b>Price asymmetry drives the value of stochastic modelling.</b> With κ = 1.0 (no imbalance penalty) VSS and EVPI are both zero: if shortfalls cost the same as planned purchases, "
         f"the commitment does not matter. As κ rises to 2.5 the VSS increases to {inr(rt['2.5']['vss'])}/day ({rt['2.5']['vss_pct']:.1f}%) and the EVPI to {inr(rt['2.5']['evpi'])}/day. "
         f"(2) <b>Weather mix:</b> costs rise with the overcast probability ({inr(po['0.05']['sp'])} → {inr(po['0.55']['sp'])}/day), while the VSS stays within {fm(min(v['vss'] for v in po.values()))}-{fm(max(v['vss'] for v in po.values()))} ₹/day. "
         f"(3) <b>Battery:</b> a 1 MWh battery lowers the stochastic cost by {inr(batt_val)}/day relative to no battery (≈ ₹{batt_val * 365 / 1e5:.1f} lakh/yr), and 2 MWh by a further {inr(bt['1000']['sp'] - bt['2000']['sp'])}/day (diminishing value per kWh). "
         f"The VSS grows from {inr(bt['0']['vss'])} with no battery to {inr(bt['1000']['vss'])} with 1 MWh and then plateaus ({inr(bt['2000']['vss'])}), consistent with storage supplying the recourse flexibility that the stochastic model exploits. "
         f"<b>Caution on noise:</b> the standard deviation of the VSS across training samples is {fm(min(v['vss_sd'] for d in sens.values() for v in d.values()))}-{fm(max(v['vss_sd'] for d in sens.values() for v in d.values()))} ₹/day, so small differences (for example κ = 1.2 versus 1.4) should not be over-interpreted. "
         f"The κ = 1.6 row ({inr(rt['1.6']['vss'])}) differs from the single-sample base case ({inr(vss)}) because it averages four training samples; the 30-repeat Monte Carlo average ({inr(mc['vss_mean'])}) shows the base case was a slightly low draw.")]

st += [P("Robustness to a wrong probability model.", h2)]
g = shift["grid"]
rows = [["Plan trained at P(overcast) ↓ / tested at →"] + [f"{x:.2f}" for x in g]]
for i, ptr in enumerate(g):
    rows.append([f"Stochastic, trained {ptr:.2f}"] + [fm(v) for v in shift["sp"][i]])
for i, ptr in enumerate(g):
    rows.append([f"Deterministic, trained {ptr:.2f}"] + [fm(v) for v in shift["ev"][i]])
reg = max(shift["sp"][1][j] - shift["sp"][j][j] for j in range(3))
maxreg = max(shift["sp"][i][j] - shift["sp"][j][j] for i in range(3) for j in range(3))
st += [table(rows, [6.0, 2.4, 2.4, 2.4]), Spacer(1, 3),
       P(f"<b>Table 4.</b> Expected daily cost (₹) when the training distribution and the true distribution differ. Within every column the stochastic plans cost less than the deterministic ones. "
         f"A plan trained at the base probability 0.25 is at most {inr(reg)}/day ({100 * reg / shift['sp'][2][2]:.2f}%) worse than a plan trained on the true probability, so the solution is not fragile to this misspecification. The largest regret in the table (a plan trained on the optimistic value 0.05 and tested at 0.50) is {inr(maxreg)}/day ({100 * maxreg / shift['sp'][2][2]:.1f}%), larger than the base-case VSS: a badly wrong weather model can cost more than the stochastic model gains.", small)]

# ---- 7 deterministic comparison
st += [P("7. Comparison with the deterministic baseline - value of modelling uncertainty", h1),
       P(f"The deterministic baseline replaces the random solar and demand by their means (the EV problem). It reports an optimistic cost of {inr(R['in_sample']['EV_problem_obj'])}/day, but its plan really costs {inr(ev['mean'])}/day "
         f"once scenarios occur - a gap of {inr(ev['mean'] - R['in_sample']['EV_problem_obj'])}/day that the deterministic model hides (the 'flaw of averages'). The stochastic model reports {inr(R['in_sample']['RP_obj'])} in-sample and {inr(sp['mean'])} out-of-sample, "
         f"an honest forecast of cost. The stochastic model also (a) reduces real-time purchases ({fm(ops[pol[2]]['rt_purchase_kWh'])} versus {fm(ops[pol[1]]['rt_purchase_kWh'])} kWh/day) and uses the battery less aggressively "
         f"({fm(ops[pol[2]]['battery_throughput_kWh'])} versus {fm(ops[pol[1]]['battery_throughput_kWh'])} kWh/day throughput, which also saves wear cost), (b) gives a measurable cost saving with a confidence interval, and (c) exposes risk (CVaR) that a point model cannot. "
         f"The honest limitation is that the average saving ({R['VSS_pct']:.2f}%) is small in the base case; modelling uncertainty matters most when imbalance prices are high (Table 3).")]

# ---- 8 SDG
st += [P("8. SDG 9 (Target 9.5) - impact and measurement", h1)]
rows = [["Indicator (per day, test set)", "Naive rule", "Deterministic", "Stochastic", "Mean-CVaR λ=0.5"]]
lab = [("Day-ahead purchase (kWh)", "day_ahead_kWh"), ("Real-time (imbalance) purchase (kWh)", "rt_purchase_kWh"), ("Surplus exported (kWh)", "export_kWh"),
       ("Net grid import (kWh)", "net_grid_import_kWh"), ("Imbalance share of demand (%)", "imbalance_share_pct"), ("Battery throughput (kWh)", "battery_throughput_kWh")]
for l, k in lab:
    rows.append([l] + [fm(ops[p][k], 1 if "pct" in k else 0) for p in pol])
rows.append(["Expected cost (₹)"] + [fm(ps[p]["mean"]) for p in pol])
st += [table(rows, [6.2, 2.4, 2.6, 2.6, 3.0]), Spacer(1, 3),
       P("<b>Table 5.</b> Indicators used to measure impact. Net grid import = day-ahead + real-time − exported energy.", small),
       P(f"These indicators make the benefit measurable: the stochastic plan lowers imbalance energy by {fm(ops[pol[1]]['rt_purchase_kWh'] - ops[pol[2]]['rt_purchase_kWh'])} kWh/day against the deterministic plan, lowers cost by {inr(vss)}/day, and quantifies what "
         f"a battery is worth operationally ({inr(batt_val)}/day for 1 MWh). Such numbers support infrastructure upgrade decisions (battery sizing, forecasting investment up to the EVPI) in line with Target 9.5. "
         f"The CVaR plan swaps almost all real-time purchases ({fm(ops[pol[3]]['rt_purchase_kWh'])} versus {fm(ops[pol[2]]['rt_purchase_kWh'])} kWh/day) for day-ahead purchases of {fm(ops[pol[3]]['day_ahead_kWh'])} kWh and exports {fm(ops[pol[3]]['export_kWh'])} kWh/day; net import is similar across policies, but the gross energy traded with the grid is much larger - a side effect a purely cost-based view would miss.")]

# ---- 9 limitations / conclusion
st += [P("9. Limitations and future work", h1),
       B("Scenarios are synthetic; conclusions about the <i>size</i> of VSS and EVPI depend on assumed prices and weather mixture, although the <i>direction</i> of every effect follows from the model structure. With metered campus data the same code can be re-run."),
       B("Only the day-ahead decision is stochastic; a rolling-horizon or multistage model (intra-day re-optimisation) and forecast-conditioned scenarios would raise the value of information that is captured."),
       B("Battery degradation is a linear cost; real ageing depends on depth of discharge. Prices are fixed; correlated price uncertainty is ignored. The hourly time step hides intra-hour ramps."),
       B("CVaR estimates from 200 scenarios are noisy; larger samples or variance-reduction methods (e.g. Latin hypercube) would help.")]
st += [P("10. Conclusion and recommendation", h1),
       P(f"Formulating campus energy procurement as a two-stage stochastic program yields a plan that is better than both a rule of thumb ({inr(R['SP_vs_naive_saving'])}/day) and the deterministic expected-value plan ({inr(vss)}/day, significant at 95%), "
         f"and the benefit grows strongly with the real-time price penalty (up to {rt['2.5']['vss_pct']:.1f}% at κ = 2.5). Perfect information would be worth up to {inr(evpi)}/day, which suggests that better forecasting has more potential than further refinement of the optimisation model. "
         f"<b>Recommendation:</b> use the risk-neutral stochastic plan, or a mean-CVaR plan with a small weight (λ ≈ 0.25: {cvar_gain25:.1f}% lower tail cost for {mean_pen25:.1f}% higher mean), re-solve with N ≈ 100-200 scenarios "
         f"each day, and calibrate the scenario generator on metered solar and load data before deployment.")]
st += [P("References", h1)]
refs = ["Birge, J. R. and Louveaux, F. (2011). <i>Introduction to Stochastic Programming</i>, 2nd ed. Springer.",
        "Shapiro, A., Dentcheva, D. and Ruszczyński, A. (2021). <i>Lectures on Stochastic Programming: Modeling and Theory</i>, 3rd ed. SIAM.",
        "Kleywegt, A. J., Shapiro, A. and Homem-de-Mello, T. (2002). The sample average approximation method for stochastic discrete optimization. <i>SIAM Journal on Optimization</i>, 12(2), 479-502.",
        "Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional value-at-risk. <i>Journal of Risk</i>, 2(3), 21-41.",
        "Huangfu, Q. and Hall, J. A. J. (2018). Parallelizing the dual revised simplex method (HiGHS). <i>Mathematical Programming Computation</i>, 10, 119-142.",
        "Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. <i>Nature Methods</i>, 17, 261-272.",
        "United Nations. Sustainable Development Goal 9, Target 9.5 (enhance scientific research and upgrade industrial technologies)."]
for i, r in enumerate(refs, 1):
    st.append(Paragraph(r, small, bulletText=f"[{i}]"))


# ------------------------------------------------------------------ build
def footer(c, d):
    c.saveState(); c.setFont("DV", 7); c.setFillColor(INK2)
    c.drawString(1.8 * cm, 1.0 * cm, "21AIE321T Stochastic Decision Making · Mini Project · Campus microgrid two-stage stochastic program")
    c.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Page {d.page}"); c.restoreState()


doc = BaseDocTemplate(os.path.join(OUT, "Mini_Project_Report.pdf"), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                      topMargin=1.5 * cm, bottomMargin=1.6 * cm, title="Stochastic Optimization of a Campus Solar + Battery Microgrid",
                      author="Student, SRMIST Tiruchirappalli")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=footer)])
doc.build(st)
print("report built")
