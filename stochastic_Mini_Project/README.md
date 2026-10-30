# Two-Stage Stochastic Optimization of a Campus Solar + Battery Microgrid

**Course:** 21AIE321T - Stochastic Decision Making  
**Assessment:** Continuous Learning Assessment – 2 (Assignment IV: Mini Project)  
**Programme:** B.Tech Artificial Intelligence · Semester V · Academic Year 2026–2027  
**Institution:** SRM Institute of Science and Technology, Tiruchirappalli  
**Faculty Handling:** Dr. Himanshu Rai, Assistant Professor  
**UN Sustainable Development Goal:** SDG 9 (Industry, Innovation, and Infrastructure · Target 9.5)  

---

## 📋 Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Proposed Solution & Methodology](#2-proposed-solution--methodology)
3. [Mathematical Model Formulation](#3-mathematical-model-formulation)
4. [System Parameters & Representative Dataset](#4-system-parameters--representative-dataset)
5. [Experimental Results & Evaluation](#5-experimental-results--evaluation)
6. [Sensitivity Analysis & Baseline Comparison](#6-sensitivity-analysis--baseline-comparison)
7. [Practical Recommendations & SDG 9.5 Impact](#7-practical-recommendations--sdg-95-impact)
8. [Repository Structure & Deliverables](#8-repository-structure--deliverables)
9. [How to Run and Reproduce](#9-how-to-run-and-reproduce)
10. [References](#10-references)

---

## 1. Problem Statement

### 1.1 The Real-World Operational Context
Modern institutional, commercial, and industrial facilities (such as an Indian university campus with a peak load of ~0.7 MW) increasingly operate on-site renewable energy assets—specifically a **solar photovoltaic (PV) array (800 kWp)** and a **battery energy storage system (BESS, 1 MWh / 250 kW)**—while procuring supplementary energy from the regional power grid.

Under standard deregulated electricity market frameworks and grid deviation-settlement regulations:
* The facility must submit a **binding, hourly day-ahead purchase schedule** ($x_t, t=1,\dots,24$) to the grid utility **24 hours in advance**.
* This commitment must be made **before** the next day's solar irradiance and campus electricity demands are observed.

### 1.2 Sources of Uncertainty (Stochasticity)
1. **Solar PV Generation ($S_t$):** Severely impacted by local weather regimes, moving cloud formations, and seasonal monsoon cloud covers. Realized generation exhibits high variance and bimodal distributions (clear vs. overcast/monsoon days) with temporal autocorrelation.
2. **Campus Load Demand ($D_t$):** Dictated by variable academic schedules, class hours, laboratory experiment loads, campus events, and weather-driven HVAC cooling needs, leading to significant weekday vs. weekend discrepancies and stochastic intraday spikes.

### 1.3 The Economic Asymmetry (Newsvendor Effect)
The financial consequence of misestimating tomorrow's energy requirement is fundamentally asymmetric:
* **Under-Commitment (Shortfall):** If actual demand exceeds the day-ahead schedule ($D_t - S_t > x_t$), the microgrid operator is forced to buy balance energy in the real-time spot market at an **unfavorable penalty multiplier** ($\kappa = 1.6\times$ the day-ahead tariff).
* **Over-Commitment (Surplus):** If solar generation or day-ahead procurement exceeds demand, surplus energy exported back to the utility grid receives a **low feed-in tariff** of only **₹3.0 / kWh**.

### 1.4 Why Deterministic Planning Fails (The "Flaw of Averages")
Standard deterministic models replace uncertain solar generation and load with their expected (average) values: $\bar{S}_t = \mathbb{E}[S_t]$ and $\bar{D}_t = \mathbb{E}[D_t]$. 
Due to Jensen's inequality and asymmetric cost penalties, optimizing for the average scenario leads to:
* Systematic under-hedging against cloudy, high-demand days.
* Severe exposure to real-time penalty purchases.
* Suboptimal battery dispatch and higher average operating costs.

---

## 2. Proposed Solution & Methodology

To overcome the flaw of averages, this project formulates and solves a **Two-Stage Stochastic Linear Program (2-SLP)** solved via **Sample Average Approximation (SAA)** with a **Mean-Conditional Value at Risk (Mean-CVaR)** risk-averse extension.

```
       STAGE 1 (Day-Ahead: Here-and-Now)
       Decision: Day-ahead hourly contract commitment x_t
       [Uncertainty is unknown: Solar S_t and Demand D_t are random]
                              │
                              ▼
       N Scenarios Sampled: ω = (S_{t,ω}, D_{t,ω})
                              │
                              ▼
       STAGE 2 (Real-Time: Recourse / Adaptation)
       Decisions per Scenario ω:
       ├── Battery Charge:     c_{t,ω}  (Store solar/off-peak energy)
       ├── Battery Discharge:  d_{t,ω}  (Shave evening peak demand)
       ├── Real-time Purchase: g_{t,ω}  (Cover shortfalls at 1.6x tariff)
       └── Surplus Export:     e_{t,ω}  (Sell excess power at ₹3/kWh)
```

### Key Methodological Pillars
1. **Two-Stage Stochastic Programming:** Splits the decision timeline into first-stage commitments ($x_t$) made under uncertainty, and second-stage corrective recourse actions ($c_t, d_t, g_t, e_t, b_t$) adapted to each realized scenario.
2. **Sample Average Approximation (SAA):** Replaces the continuous scenario distribution with a discrete sample of $N=200$ scenarios, transforming the problem into a large-scale, deterministic equivalent linear program solved globally to proven optimality using SciPy's HiGHS dual revised simplex solver.
3. **Mean-CVaR Tail-Risk Control:** Uses the Rockafellar-Uryasev linear formulation to explicitly minimize the conditional expectation of costs in the worst 10% of days ($\alpha = 0.90$), protecting the facility against catastrophic monsoon/outage events.
4. **Out-of-Sample Monte Carlo Validation:** All policies are evaluated on an independent, unseen test set of **500 days** to ensure honest, out-of-sample performance metrics without in-sample overfitting.

---

## 3. Mathematical Model Formulation

### 3.1 Notation & Decision Variables

| Variable | Domain | Stage | Physical Description |
| :--- | :---: | :---: | :--- |
| $x_t$ | $[0, X_{\max}]$ | Stage 1 | Day-ahead grid purchase schedule for hour $t$ (kW) |
| $c_{t,\omega}$ | $[0, P_{\text{batt}}]$ | Stage 2 | Battery charging rate in scenario $\omega$, hour $t$ (kW) |
| $d_{t,\omega}$ | $[0, P_{\text{batt}}]$ | Stage 2 | Battery discharging rate in scenario $\omega$, hour $t$ (kW) |
| $g_{t,\omega}$ | $\ge 0$ | Stage 2 | Real-time shortfall purchase from grid in scenario $\omega$, hour $t$ (kWh) |
| $e_{t,\omega}$ | $\ge 0$ | Stage 2 | Surplus solar power exported to grid in scenario $\omega$, hour $t$ (kWh) |
| $b_{t,\omega}$ | $[B_{\min}, B_{\max}]$ | Stage 2 | Battery State of Charge at start of hour $t$ in scenario $\omega$ (kWh) |

---

### 3.2 Objective Function (Risk-Neutral Two-Stage LP)

$$\min_{x \in \mathcal{X}} \; \sum_{t=1}^{24} p_t^{\text{DA}} x_t + \frac{1}{N} \sum_{n=1}^N Q(x, \omega_n)$$

where the second-stage recourse cost $Q(x, \omega)$ is:

$$Q(x, \omega) = \min_{c, d, g, e, b} \sum_{t=1}^{24} \left( p_t^{\text{RT}} g_{t,\omega} - p^{\text{export}} e_{t,\omega} + c_{\text{deg}} d_{t,\omega} \right)$$

---

### 3.3 System Constraints ($\forall t \in \{1,\dots,24\}, \forall \omega \in \Omega$)

1. **Microgrid Energy Balance:**
   $$x_t + g_{t,\omega} - e_{t,\omega} + d_{t,\omega} - c_{t,\omega} = D_{t,\omega} - S_{t,\omega}$$
2. **Battery State of Charge Dynamics:**
   $$b_{t+1,\omega} = b_{t,\omega} + \eta \cdot c_{t,\omega} - \frac{1}{\eta} \cdot d_{t,\omega}$$
   where $\eta = 0.92$ is the one-way efficiency (round-trip efficiency $\eta^2 \approx 84.6\%$).
3. **Storage Energy & Power Bounds:**
   $$B_{\min} \le b_{t,\omega} \le B_{\max}, \quad 0 \le c_{t,\omega} \le P_{\text{batt}}, \quad 0 \le d_{t,\omega} \le P_{\text{batt}}$$
   with $B_{\min} = 100\text{ kWh}$, $B_{\max} = 1000\text{ kWh}$, and $P_{\text{batt}} = 250\text{ kW}$.
4. **Terminal Energy Neutrality (Daily Cyclicality):**
   $$b_{25,\omega} \ge b_1 \quad (\text{with } b_1 = 500\text{ kWh})$$
   ensuring the battery is not depleted to artificially lower daily operating costs.

---

### 3.4 Risk-Averse Extension: Mean-CVaR LP Formulation

$$\min_{x, \zeta, u} \; (1 - \lambda) \mathbb{E}[C_\omega(x)] + \lambda \left( \zeta + \frac{1}{(1-\alpha) N} \sum_{n=1}^N u_n \right)$$
$$\text{s.t.} \quad u_n \ge C_n(x) - \zeta, \quad u_n \ge 0, \quad \forall n \in \{1,\dots,N\}$$

where $\lambda \in [0, 1]$ controls the weight assigned to risk, $\zeta$ represents $\text{VaR}_\alpha$, and $u_n$ measures tail cost exceedances above $\zeta$.

---

## 4. System Parameters & Representative Dataset

### 4.1 Campus Microgrid Asset Specifications
* **Solar PV Array:** 800 kWp nominal rating; clear-sky maximum yield of ~5,155 kWh/day.
* **Battery Energy Storage System (BESS):** 1,000 kWh capacity, 250 kW power rating, 10% depth-of-discharge reserve ($B_{\min}=100\text{ kWh}$), cell degradation cost $c_{\text{deg}} = ₹1.0/\text{kWh}$.
* **Contract Capacity Limit:** Maximum day-ahead grid line draw $X_{\max} = 1000\text{ kW}$.

### 4.2 Tariff Structure & Pricing Mechanics
* **Time-of-Day (ToD) Day-Ahead Tariffs:**
  * Night Off-Peak (22:00 – 06:00): **₹5.50 / kWh**
  * Morning Shoulder (06:00 – 09:00): **₹6.50 / kWh**
  * Solar Midday (09:00 – 18:00): **₹6.00 / kWh**
  * Evening Peak (18:00 – 22:00): **₹9.00 / kWh**
* **Real-Time Imbalance Shortfall:** $\kappa \times p_t^{\text{DA}} = 1.6 \times p_t^{\text{DA}}$ (e.g., ₹14.40/kWh during evening peak).
* **Surplus Power Export Credit:** ₹3.00 / kWh net metering tariff.

### 4.3 Realistic Scenario Generation Model
* **Solar Irradiance:** Mixture of 3 weather states: Clear ($35\%$), Partly Cloudy ($40\%$), and Overcast Monsoon ($25\%$), modeled with an $\text{AR}(1)$ Markov noise process ($\rho=0.8$) to simulate realistic cloud dynamics.
* **Campus Load:** Base academic load curve ranging from 240 kW (night baseload) to 720 kW (daytime lecture and lab peak), incorporating weekend reductions ($P(\text{weekend})=2/7$, scaling factor 0.72) and hourly log-normal noise.

---

## 5. Experimental Results & Evaluation

All policies were trained on $N = 200$ scenarios and tested out-of-sample over **500 unseen operational days**:

### 5.1 Comprehensive Policy Comparison

| Optimization Policy | Expected Daily Cost (₹) | Standard Deviation (₹) | 95th Percentile Cost (₹) | $\text{CVaR}_{90}$ Tail Risk (₹) | Real-Time Shortfall (kWh/day) | BESS Daily Cycling (kWh) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Rule-of-Thumb** | ₹57,832.8 | ₹13,734.4 | ₹85,961.4 | ₹86,667.2 | 835.6 kWh | 235.4 kWh |
| **Deterministic Baseline (EV)** | ₹57,120.5 | ₹14,219.3 | ₹86,097.8 | ₹86,583.7 | 874.9 kWh | 754.3 kWh |
| **Stochastic Program (RP, Risk-Neutral)** | **₹56,798.4** | **₹14,124.3** | **₹85,762.7** | **₹86,291.3** | **826.5 kWh** | **633.4 kWh** |
| **Mean-CVaR ($\lambda = 0.25$)** | ₹57,751.4 | ₹10,567.4 | ₹80,405.0 | ₹80,985.2 | 412.0 kWh | 512.6 kWh |
| **Mean-CVaR ($\lambda = 0.50$)** | ₹62,337.1 | ₹7,447.5 | ₹73,958.0 | **₹75,811.3** | **35.9 kWh** | 379.0 kWh |
| **Wait-and-See (WS, Perfect Information)** | ₹50,393.6 | ₹13,798.4 | ₹73,015.1 | ₹73,310.9 | 0.0 kWh | 892.1 kWh |

---

### 5.2 Key Stochastic Metrics Explained

1. **Value of the Stochastic Solution (VSS):**
   $$\text{VSS} = \mathbb{E}[\text{Cost}(\text{EV})] - \mathbb{E}[\text{Cost}(\text{SP})] = \mathbf{₹322.01 / \text{day}} \quad (95\%\text{ CI: } [₹254.99, ₹389.03])$$
   *Over a 30-run Monte Carlo bootstrap evaluation, the mean VSS is **₹384.27 / day**, with the stochastic plan outperforming the deterministic plan in **100% of Monte Carlo runs**.*
2. **Expected Value of Perfect Information (EVPI):**
   $$\text{EVPI} = \mathbb{E}[\text{Cost}(\text{SP})] - \mathbb{E}[\text{Cost}(\text{WS})] = \mathbf{₹6,404.90 / \text{day}} \quad (95\%\text{ CI: } [₹6,040.45, ₹6,769.35])$$
   *Represents **11.28%** of the total daily energy bill, quantifying the maximum monetary budget the campus should allocate toward high-precision telemetry and weather-forecasting radar.*
3. **Tail Risk Mitigation (CVaR):**
   * Incorporating risk aversion ($\lambda = 0.50$) reduces worst-10% tail risk ($\text{CVaR}_{90}$) by **₹10,480 / day (12.1% reduction)**, decreasing real-time penalty purchases from 826.5 kWh down to **35.9 kWh / day**.

---

## 6. Sensitivity Analysis & Baseline Comparison

### 6.1 Impact of Real-Time Penalty Multiplier ($\kappa \in [1.0, 2.5]$)
* **At $\kappa = 1.0$ (No Penalty):** $\text{VSS} = ₹0.0$ and $\text{EVPI} = ₹0.0$, validating the theoretical consistency of the linear model (when penalties equal base rates, timing uncertainty carries zero monetary penalty).
* **At $\kappa = 2.5$ (Severe Grid Penalties):** VSS grows exponentially to **₹2,559 / day (9.5% savings)**. This proves that stochastic optimization delivers increasing financial dividends as grid regulations tighten.

### 6.2 Battery Energy Storage Sizing Valuation
* Eliminating the battery ($B = 0$) increases daily operating cost by **₹3,700+ / day**.
* A 1 MWh battery acts as an essential physical hedge, storing excess midday solar generation and discharging during the expensive 18:00–22:00 evening peak.

### 6.3 Distributional Drift & Weather Misspecification Regret Matrix
* A regret analysis evaluated models trained under a base monsoon probability ($p_{\text{overcast}} = 0.25$) when deployed into true environments where $p_{\text{overcast}} \in [0.05, 0.50]$.
* The stochastic plan experienced a maximum regret of $<1.5\%$, proving robust resistance to climate and weather modeling errors.

---

## 7. Practical Recommendations & SDG 9.5 Impact

### 7.1 Actionable Operational Recommendations
1. **Adopt SAA Stochastic Procurement:** Transition from average-based forecasting to SAA two-stage linear programming using $N=100\text{–}200$ scenarios re-solved daily.
2. **Apply Moderate Risk-Aversion ($\lambda \approx 0.25$):** Achieves a **6.1% reduction in tail losses** with only a minor $1.6\%$ increase in expected cost, offering the best trade-off for conservative institutional CFOs.
3. **Invest in Day-Ahead Weather Telemetry:** Since $\text{EVPI} \approx ₹6,400/\text{day}$ (~₹23.3 Lakhs annually), investing in dedicated on-site sky-cameras and AI solar forecasting pays for itself within months.

### 7.2 UN SDG 9 (Target 9.5) Measurable Impact
* **Infrastructure Upgrades:** Modernizes traditional campus electrical substations into an intelligent, autonomous microgrid.
* **Imbalance Energy Reduction:** Lowers volatile grid deviation imports by **48.4 kWh/day** vs. deterministic dispatch and **>800 kWh/day** under Mean-CVaR, improving overall regional grid stability.
* **Carbon Decarbonization:** Ensures maximum utilization of the 800 kWp solar PV system, minimizing curtailment and grid reliance.

---

## 8. Repository Structure & Deliverables

```plaintext
Assignment4_MiniProject/
├── README.md                      # Comprehensive academic & technical documentation
├── requirements.txt               # Verified Python package dependencies
├── code/                          # Production-grade Python scripts
│   ├── microgrid_sp.py            # Two-stage LP formulation & SciPy HiGHS solver
│   ├── run_experiments.py         # Full experimental suite (Monte Carlo, SAA convergence, sensitivity)
│   ├── make_figures.py            # Generates all 9 high-resolution publication figures
│   ├── build_report.py            # Automated ReportLab script producing the 7-page PDF report
│   └── build_notebook.py          # Script generating the Jupyter walkthrough notebook
├── notebook/                      # Interactive experimentation
│   └── Microgrid_Stochastic_Optimization.ipynb  # Runnable Jupyter Notebook
├── report/                        # Final academic deliverables
│   └── Mini_Project_Report.pdf    # Full 7-page academic report with proofs & rubrics
├── slides/                        # Presentation slide deck
│   └── Mini_Project_Presentation.pptx # 9-slide conference/classroom PowerPoint presentation
├── figures/                       # High-resolution vector/raster graphics
│   ├── fig1_scenarios.png         # Solar & demand scenario fan charts
│   ├── fig2_structure.png         # Microgrid system schematic & power flow diagram
│   ├── fig3_plans.png             # Day-ahead hourly commitment schedules
│   ├── fig4_schedules.png         # 24-hour battery SoC and dispatch trajectories
│   ├── fig5_cost_ecdf.png         # Empirical Cumulative Distribution Functions (ECDF)
│   ├── fig6_montecarlo.png        # Bootstrap distributions of VSS and cost savings
│   ├── fig7_convergence.png       # SAA sample size convergence (in-sample vs out-of-sample)
│   ├── fig8_frontier.png          # Pareto risk frontier (Expected Cost vs CVaR90)
│   └── fig9_sensitivity.png       # Multi-parameter sensitivity sweeps
└── results/                       # Numerical experiment data
    ├── results.json               # Serialized results, metrics, and confidence intervals
    ├── policy_summary.csv         # Comparative policy summary table
    └── arrays.npz                 # Saved numpy arrays for exact numerical reproducibility
```

---

## 9. How to Run and Reproduce

### Step 1: Environment Setup
Ensure you have Python 3.10+ installed:
```bash
pip install -r requirements.txt
```

### Step 2: Run All Experiments
To execute the optimizations, Monte Carlo simulations, convergence runs, and sensitivity analyses (~5–7 minutes):
```bash
cd code
python run_experiments.py
```
This writes the numerical outputs to `results/results.json` and `results/arrays.npz`.

### Step 3: Regenerate Figures and the PDF Report
```bash
python make_figures.py
python build_report.py
```
The compiled 7-page academic report will be generated at `report/Mini_Project_Report.pdf`.

---

## 10. References
1. **Birge, J. R., & Louveaux, F.** (2011). *Introduction to Stochastic Programming* (2nd ed.). Springer Science & Business Media.
2. **Shapiro, A., Dentcheva, D., & Ruszczyński, A.** (2021). *Lectures on Stochastic Programming: Modeling and Theory* (3rd ed.). SIAM.
3. **Rockafellar, R. T., & Uryasev, S.** (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2(3), 21–41.
4. **Kleywegt, A. J., Shapiro, A., & Homem-de-Mello, T.** (2002). The sample average approximation method for stochastic discrete optimization. *SIAM Journal on Optimization*, 12(2), 479–502.
5. **Huangfu, Q., & Hall, J. A. J.** (2018). Parallelizing the dual revised simplex method (HiGHS). *Mathematical Programming Computation*, 10, 119–142.
