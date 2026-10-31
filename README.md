# PULSE: Pop-Up Unified Logistics and Supplied Emergence

**Formal model specification and code guide**

PULSE matches nontraditional ("pop-up") suppliers with requesters of an emergency item, in repeated rounds over a two-week horizon. It extends Shield-Net (Alcock, Boutilier, Siddiq, *INFORMS J. Applied Analytics*, 2022) and is set in India at district level.

PULSE has two parts, and this document keeps them apart:

- **The model** is the optimisation problem the platform solves each round to decide who gets what (Sections 5–7). Like Shield-Net's, it is a linear program over supplier–requester pairs, integral and solved exactly in about half a second.
- **The simulation** is the world we test the model in: random arrivals, matches that succeed or fail, stock that carries over, requesters that give up (Sections 8–12). It exists to evaluate the model, and none of it is part of the optimisation.

Between the two sits one piece of **platform procedure**: after each round the platform updates its estimate of how often matches succeed, and the next round's model uses that estimate.

`PULSE_MODEL.md` is a short, model-only summary. The code is Python with **PuLP** (bundled CBC solver); Section 16 says how to run it.

---

## 1. What changed and why

**Where** says which part of PULSE each change lives in: **M** = the optimisation model, **P** = the platform's update between rounds, **S** = the simulation used for evaluation.

| # | Shield-Net | PULSE | Where | Real-world reason |
|---|---|---|---|---|
| 1 | Minimise urgency-weighted shortfall + α·distance | Maximise **expected** urgency-weighted delivery: each unit is weighted by its probability of being delivered | M | Paper: only ~27% of matches succeeded, and same-state was the only significant predictor of success (odds ratio 3.55) |
| 2 | Success not considered | Success probabilities are **learned** from observed outcomes, starting from the paper's values | P | US numbers are a starting guess for India, not a fact |
| 3 | Self-reported urgency, weights 1/2/3 | Urgency level from **facility type**; weights **1/10/100** | M | Self-reports are gameable; a COVID ward must not lose to a nearby grocery store |
| 4 | Price ignored | Requester budget is a hard filter; price is a tie-breaker | M | Paper names price as a source of failed matches |
| 5 | No quality signal | Certified flag on suppliers; requesters choose **none / prefer / require** | M | Paper names quality uncertainty as a source of failed matches |
| 6 | One required product type | Requester accepts a **set** of variants | M | Many buyers would take kits or assembled shields |
| 7 | No hoarding control | Each request is split into **core** (plausible need for a facility of its type and size, minus what it already received) and **excess** (the rest). Core is served before any excess; nothing is cut | M | One 5-million-unit request had to be removed by hand |
| 8 | Capacity per round | Capacity is a **daily rate** × days since the last round, plus carried stock | M (supply), S (carry-over) | Fri→Mon is 3 days, the other gaps are 2 |
| 9 | No inventory | Unsold stock carries over with **leakage** and a **storage cap** | S | Suppliers had outside buyers; stock doesn't wait forever |
| 10 | US states, straight-line distance between state centroids | 800 Indian districts, **road-distance** matrix | M (data) | Local setting; roads, not straight lines |
| 11 | Requesters wait until served | Per-round **abandonment** | S | 52% of unsuccessful requesters sourced elsewhere |
| 12 | Matches assumed to happen | Each match succeeds or fails at random; a requester refuses stock over its budget or uncertified stock it said it required; failed units return to demand | S | Matched is not the same as delivered |
| 13 | Minimum request of 100, outside the model | Platform minimum request $`m_{min} = 50`$; **supplier minimum orders are not modelled** | — | A known limitation (Section 14) |

---

## 2. Sets and time

- $`K`$: districts. $`\sigma(k)`$: state of district $`k`$.
- $`R[k,l]`$: road distance in km from district $`k`$ (supplier side) to district $`l`$ (requester side), read from the input file (Section 2.1). $`R[k,l] = \infty`$ if the cell is blank (unreachable).
- $`R[k,k] = r^{intra}_k`$: distance within a district, drawn once per district from $`U[10, 20]`$ km with a seeded RNG, unless the file gives a positive diagonal value.
- $`R_{max}`$: the largest **finite** entry of $`R`$.
- Rounds $`\tau = 1,\dots,6`$ on days $`d_\tau \in \lbrace 1,3,5,8,10,12\rbrace`$ (Mon/Wed/Fri for two weeks). Gap $`\Delta_\tau = d_\tau - d_{\tau-1}`$ with $`d_0 = -1`$, so $`\Delta = (2,2,2,3,2,2)`$.
- $`V`$: item variants (default {assembled, kit}).
- $`I_\tau = \lbrace i : e_i \le \tau\rbrace`$: suppliers enrolled by round $`\tau`$, where $`e_i`$ is the enrolment round. Suppliers are never removed.
- $`J_\tau`$: requesters active at round $`\tau`$, meaning arrived and not yet fully served or abandoned.
- $`\mathcal{G} = \lbrace \text{same}, \text{cross}\rbrace`$: pair classes. $`g(i,j) = \text{same}`$ if $`\sigma(k(i)) = \sigma(k(j))`$, else cross.

### 2.1 Input data file

All district data comes from one Excel workbook. The code takes its path as a parameter, so the placeholder file and the real file are interchangeable.

**Sheet `districts`**, one row per district:

| Column | Required | Meaning |
|---|---|---|
| `district_id` | yes | unique ID, also used as the matrix headers |
| `name` | yes | district name |
| `state` | yes | state or UT name; defines $`\sigma(k)`$ and so the same/cross-state class |
| `population` | no | used to place requesters (Section 12.2) |

**Sheet `distances`**: a $`|K| \times |K|`$ matrix (about 800 × 800) of road distances in km. The first column holds row `district_id`s and the first row holds column `district_id`s. Row $`k`$, column $`l`$ is $`R[k,l]`$.

**The loader checks and normalises:**
- Row IDs, column IDs and the `districts` sheet contain exactly the same IDs. Rows and columns are reordered to the `districts` order, so the file's own ordering does not matter.
- Every non-blank value is a number $`\ge 0`$.
- **Diagonal** values that are 0 or blank are replaced by $`r^{intra}_k \sim U[10,20]`$ km.
- **Off-diagonal blank cells** become $`\infty`$ (unreachable), so those pairs are never eligible. They are never set to 0, which would make them look like the closest pairs.
- **Off-diagonal zeros** are an error, since two different districts cannot be 0 km apart.
- **Asymmetry** is allowed. $`R[k,l]`$ is always read as supplier district $`k`$ to requester district $`l`$.
- The number of blank cells and asymmetric pairs is logged.

**Placeholder file**, used until the real one is ready, in exactly the same format:
- 800 districts with IDs `D001`–`D800`, spread over 36 states/UTs `S01`–`S36`. District counts per state come from a Dirichlet(1) split of the 800, with at least 1 district per state.
- Each state gets a centre placed uniformly in a 3000 km × 3000 km square. Each district sits at its state centre plus Gaussian noise with $`\sigma = 150`$ km.
- Off-diagonal road distance = 1.3 × straight-line distance, rounded to whole km and symmetric. The diagonal is left 0, so the loader's $`U[10,20]`$ rule is exercised.
- Population ~ round(lognormal($`\ln(2 \times 10^6)`$, 0.6)).
- Generated from a fixed seed and saved with `openpyxl`.

---

## 3. Parameters

### 3.1 Supplier parameters (supplier i)

| Symbol | Meaning |
|---|---|
| $`k(i)`$ | district |
| $`e_i`$ | enrolment round |
| $`s_i \in \mathbb{Z}_{+}`$ | production rate, units per **day** |
| $`\pi_i > 0`$ | unit price (delivered) |
| $`q_i \in \lbrace 0,1\rbrace`$ | 1 = quality-certified |
| $`V_i \subseteq V`$ | variants offered |

### 3.2 Requester parameters (requester j)

Each requester submits **one request** at its arrival round $`\tau_j`$.

| Symbol | Meaning |
|---|---|
| $`k(j)`$ | district |
| $`\tau_j`$ | arrival round |
| $`D^0_j \in \mathbb{Z}_{+}`$ | request size, $`D^0_j \ge m_{min}`$ |
| $`\ell_j \in \lbrace 1,2,3\rbrace`$ | urgency level, from facility type (table below) |
| $`p_j = W[\ell_j]`$ | urgency weight |
| $`\bar\pi_j`$ | maximum acceptable unit price |
| $`\kappa_j \in \lbrace 0,1,2\rbrace`$ | certificate need: 0 = none, 1 = prefer certified, 2 = require certified |
| $`V_j \subseteq V`$, $`V_j \ne \emptyset`$ | acceptable variants |
| $`E_j \in \mathbb{Z}_{+}`$ | staff count (or beds), declared on the intake form |

| Facility type | Level $`\ell_j`$ | Plausible usage $`u_{\ell}`$ (units per staff per day) |
|---|---|---|
| Treating patients of the emergency (e.g. COVID ward) | 3 | 1.0 |
| Other healthcare (hospital, nursing home, clinic) | 2 | 0.5 |
| Essential non-healthcare (police, grocery, municipal) | 1 | 0.25 |

### 3.3 Global parameters

Each parameter has a **class**, which says how it is justified in the report:
- **Policy:** a design choice we defend.
- **Calibration:** taken from the paper, or an assumed norm.
- **Behavioural:** unknown, so it gets a sensitivity sweep.
- **Scenario:** the simulated "truth", which the platform does not know.
- **Switch:** turns a feature on or off (used by the ablation).

| Symbol | Code name | Meaning | Default | Class |
|---|---|---|---|---|
| $`W`$ | `W` | urgency weights for levels 1/2/3 | (1, 10, 100) | Policy |
| $`m_{min}`$ | `m_min` | platform minimum request size | 50 | Policy |
| $`H`$ | `H` | days of plausible need counted as core | 14 | Policy |
| $`\varepsilon`$ | `excess_weight` | value of an excess unit relative to the same unit as core | $`10^{-4}`$ | Policy |
| $`\tilde\alpha`$ | `alpha` | distance weight (tie-breaker) | 0.002 | Policy |
| $`\beta`$ | `beta` | price weight (tie-breaker) | 0.002 | Policy |
| $`\mu`$ | `mu` | penalty per unit of uncertified stock to a "prefer certified" requester | 0.02 | Policy |
| $`\Delta_{max}`$ | `delta_max` | optional distance cutoff (km) | $`\infty`$ | Policy |
| $`\psi_{floor}`$ | `psi_floor` | lower clip on learned $`\hat\psi`$ | 0.05 | Policy |
| $`u_{\ell}`$ | `usage_rates` | plausible usage per staff per day, levels 1/2/3 | (0.25, 0.5, 1.0) | Calibration |
| $`\psi^0_{same}, \psi^0_{cross}`$ | `psi0_same`, `psi0_cross` | prior success probability | 0.40, 0.16 | Calibration |
| $`n_0`$ | `n0` | prior strength, in pseudo-matches | 20 | Calibration |
| $`\psi^{\ast}_{same}, \psi^{\ast}_{cross}`$ | `psi_true_same`, `psi_true_cross` | **true** success probability, used only by the simulation | = prior | Scenario |
| $`\eta`$ | `eta` | fraction of unsold stock still available next round | 0.5 | Behavioural |
| $`h`$ | `h` | storage cap, in days of production | 5 | Behavioural |
| $`\omega`$ | `omega` | probability an unserved requester abandons per round | 0.10 | Behavioural |
| | `use_psi_objective`, `learn_psi`, `simulate_failures`, `price_filter`, `cert_filter`, `core_excess` | feature switches | all True | Switch |

**Required conditions:**
- **(C1)** $`\tilde\alpha + \beta + \mu < W[1]\cdot\psi_{floor}`$, checked at start-up. Defaults: $`0.024 < 0.05`$. Every core unit has positive value.
- **(C2)** $`W[\ell+1]\,\hat\psi_{cross} - (\tilde\alpha+\beta+\mu) > W[\ell]\,\hat\psi_{same}`$ for $`\ell = 1, 2`$, checked every round. Defaults: $`1.576 > 0.40`$. Urgency beats locality.
- **(C3)** $`\varepsilon \cdot W[3] < W[1]\cdot\psi_{floor} - (\tilde\alpha+\beta+\mu)`$, checked at start-up. Defaults: $`0.01 < 0.026`$. Any core unit is worth more than any excess unit.

Section 7 explains all three.

**Prior calibration.** The paper reports a 27% success rate among classified matches, 48% of matches local (same state), and an odds ratio of 3.55 for local matches. Solving $`0.48\,\psi_{same} + 0.52\,\psi_{cross} = 0.27`$ together with $`\text{odds}(\psi_{same}) = 3.55\cdot\text{odds}(\psi_{cross})`$ gives $`\psi_{same} \approx 0.40`$ and $`\psi_{cross} \approx 0.16`$.

### 3.4 Derived quantities

- Variant compatibility: $`\gamma_{ij} = \mathbf{1}[V_i \cap V_j \ne \emptyset]`$.
- Normalised distance: $`\hat c_{ij} = R[k(i),k(j)] / R_{max} \in (0,1]`$.
- Normalised price: $`\rho_{ij} = \pi_i / \bar\pi_j \in (0,1]`$ on eligible pairs.
- Believed success probability: $`\hat\psi_{ij,\tau} = \hat\psi_{g(i,j),\tau}`$ (Section 4).
- Tie-break cost: $`c_{ij} = \tilde\alpha\,\hat c_{ij} + \beta\,\rho_{ij} + \mu\,\mathbf{1}[\kappa_j = 1]\,(1 - q_i)`$.
- Genuine need over the horizon: $`N_j = u_{\ell_j} \cdot E_j \cdot H`$.

---

## 4. State carried between rounds

| State | Meaning | Initial value |
|---|---|---|
| $`r_{i,\tau-1}`$ | supplier stock carried into round $`\tau`$ | 0 |
| $`carry_{j,\tau-1}`$ | requester's outstanding demand carried into round $`\tau`$ | 0 |
| $`G_j`$ | units requester $`j`$ has received so far | 0 |
| $`S_g, F_g`$ | observed successful / failed matches in class $`g`$ | 0 |

The platform's belief in round $`\tau`$ is the Beta posterior mean, clipped below:

```math
\hat\psi_{g,\tau} = \max\left(\psi_{floor},\ \frac{n_0\,\psi^0_g + S_g}{n_0 + S_g + F_g}\right)
```

$`S_g`$ and $`F_g`$ count **matches** (one per pair with $`x_{ij} > 0`$), not units, because a match succeeds or fails as a whole. If `learn_psi` is False, $`\hat\psi_g = \psi^0_g`$ throughout.

---

## 5. Pre-solve for each round

These steps compute the numbers that go into the round's model.

1. **Enrolment and arrivals.** Add suppliers with $`e_i = \tau`$. Add requesters with $`\tau_j = \tau`$.
2. **Supply.** Fresh production $`f_{i,\tau} = s_i\,\Delta_\tau`$. Available stock $`A_{i,\tau} = f_{i,\tau} + r_{i,\tau-1}`$. Total $`A^{tot}_\tau = \sum_{i \in I_\tau} A_{i,\tau}`$.
3. **Outstanding demand.** $`D_{j,\tau} = carry_{j,\tau-1} + D^0_j\,\mathbf{1}[\tau = \tau_j]`$.
4. **Core and excess.** The core is the genuine need the requester has not yet received; the excess is the rest of what it still asks for:

```math
C_{j,\tau} = \min\Big(D_{j,\tau},\ \max\big(0,\ \lfloor N_j - G_j \rfloor\big)\Big), \qquad X_{j,\tau} = D_{j,\tau} - C_{j,\tau}
```

Nothing is cut: the requester can still receive its whole request. Excess just waits behind everyone's core. Because $`N_j`$ depends only on facility type and size, inflating a request only adds excess, never core. Receipts reduce the core over time, so a requester that has already been served comes back mostly as excess. If `core_excess` is False, $`C = D`$ and $`X = 0`$ (Shield-Net behaviour).

5. **Eligibility.**

```math
a_{ij} = \gamma_{ij}\cdot \mathbf{1}[\pi_i \le \bar\pi_j]\cdot \mathbf{1}[q_i \ge \mathbf{1}[\kappa_j = 2]]\cdot \mathbf{1}[R[k(i),k(j)] \le \Delta_{max}]\cdot\mathbf{1}[A_{i,\tau} > 0]
```

A pair with $`R[k(i),k(j)] = \infty`$ (blank in the file) is never eligible, even when $`\Delta_{max} = \infty`$. The edge set is $`E_\tau = \lbrace (i,j) : a_{ij} = 1\rbrace`$, and variables exist only on edges.

6. **Coefficients.** $`w_{ij} = p_j\,\hat\psi_{ij,\tau} - c_{ij}`$, the value of one core unit on edge $`(i,j)`$.
7. **Check (C2)** with the current $`\hat\psi`$. If it fails, the model is still valid but Property 7.2 no longer holds; the code records it as `c2_ok = False`.

---

## 6. The model for each round

### 6.1 Shield-Net's model

In the paper's notation ($`s_i`$ capacity, $`d_j`$ demand, $`p_j \in \lbrace 1,2,3\rbrace`$, $`c_{ij}`$ in km, $`\alpha = 10^{-6}`$, $`a_{ij}`$ product-type compatibility):

```math
\begin{aligned}
\min_{x,\,t}\quad & \sum_j p_j\,t_j + \alpha \sum_{i,j} c_{ij}\,x_{ij}\\
\text{s.t.}\quad & \sum_j x_{ij} \le s_i,\qquad \sum_i x_{ij} \le d_j,\qquad x_{ij} \le d_j\,a_{ij}\\
& t_j \ge d_j - \sum_i x_{ij},\qquad t_j \ge 0,\qquad x_{ij} \ge 0
\end{aligned}
```

At an optimum $`t_j = d_j - \sum_i x_{ij}`$, so dropping the constant $`\sum_j p_j d_j`$ this is the same as

```math
\max_x \sum_{i,j:\,a_{ij}=1} (p_j - \alpha\,c_{ij})\,x_{ij}\quad\text{s.t.}\quad \sum_j x_{ij} \le s_i,\quad \sum_i x_{ij} \le d_j,\quad x \ge 0 .
```

### 6.2 PULSE's model

Each allowed pair has two flows: $`x^c_{ij}`$ toward the requester's core and $`x^e_{ij}`$ toward its excess. The supplier ships $`x_{ij} = x^c_{ij} + x^e_{ij}`$ units.

```math
\begin{aligned}
\max_{x^c,\,x^e}\quad & \sum_{(i,j)\in E_\tau} w_{ij}\,\big(x^c_{ij} + \varepsilon\,x^e_{ij}\big), \qquad w_{ij} = p_j\,\hat\psi_{ij} - c_{ij}\\
\text{s.t.}\quad & \sum_{j} \big(x^c_{ij} + x^e_{ij}\big) \le A_{i,\tau} && \forall i \in I_\tau && (1)\\
& \sum_{i} x^c_{ij} \le C_{j,\tau} && \forall j \in J_\tau && (2c)\\
& \sum_{i} x^e_{ij} \le X_{j,\tau} && \forall j \in J_\tau && (2e)\\
& x^c_{ij},\ x^e_{ij} \ge 0 && \forall (i,j)\in E_\tau
\end{aligned}
```

Read it as: maximise the **expected urgency-weighted delivery of genuine need**, with excess served only from what is left, by the same priorities. The code implements this with `LpMaximize`. A flow is created only where its demand row is positive, so requesters with no excess get no excess variables.

### 6.3 Exactly what changed

Every change is to the objective coefficients, the edge set or the right-hand sides, plus one duplicated demand row. The constraint matrix keeps Shield-Net's form, which is why PULSE is still an integral LP (Section 7.3).

| Change | Shield-Net | PULSE | Improvement (Section 1) |
|---|---|---|---|
| Objective coefficient | $`p_j - \alpha\,c_{ij}`$ | $`p_j\,\hat\psi_{ij} - c_{ij}`$ (core), $`\varepsilon`$ times that (excess) | 1, 7 |
| Urgency weights | $`p_j \in \lbrace 1,2,3\rbrace`$, self-reported | $`p_j = W[\ell_j]`$, $`W = (1,10,100)`$, $`\ell_j`$ from facility type | 3 |
| Tie-break cost | $`\alpha \times`$ km, $`\alpha = 10^{-6}`$ | $`c_{ij} = \tilde\alpha\,\hat c_{ij} + \beta\,\rho_{ij} + \mu\,\mathbf{1}[\kappa_j = 1](1 - q_i)`$ on road distance | 4, 5, 10 |
| Edge set | $`a_{ij}`$ = product type matches | $`a_{ij}`$ = variant-set overlap × within budget × certified if required × reachable | 4, 5, 6, 10 |
| Supply right-hand side | $`s_i`$ | $`A_{i,\tau} = s_i\,\Delta_\tau + r_{i,\tau-1}`$ | 8 |
| Demand rows | one row, $`\le d_j`$ | two rows: core $`\le C_{j,\tau}`$ and excess $`\le X_{j,\tau}`$ | 7 |

---

## 7. Properties (each one is also a test)

### 7.1 Serving always beats not serving
For every edge, $`w_{ij} \ge W[1]\,\psi_{floor} - (\tilde\alpha+\beta+\mu) > 0`$ by (C1), so excess units have positive value too. This holds in every round, whatever the learning does, because $`\hat\psi`$ is clipped at $`\psi_{floor}`$. An optimal solution therefore never leaves an edge idle where the supplier still has stock **and** the requester still has core or excess demand.
**Test:** asserted every round in the simulation.

### 7.2 Urgency dominates across levels; locality decides within a level
A core unit's value is $`p_j\,\hat\psi_{ij}`$, up to tie-breaks. With $`W = (1,10,100)`$ and the prior $`\hat\psi`$:

| Level | same state | cross state |
|---|---|---|
| 1 | 0.40 | 0.16 |
| 2 | 4.0 | 1.6 |
| 3 | 40 | 16 |

The worst cross-state unit at a higher level beats the best same-state unit at the level below as long as (C2) holds. So a supplier never serves a less urgent requester's core while a more urgent one it can reach goes short. Within a level, same-state pairs are preferred by a factor of $`\hat\psi_{same}/\hat\psi_{cross}`$, then the certificate preference ($`\mu`$) decides, then distance and price ($`\tilde\alpha, \beta`$). The same ordering applies among excess units, scaled by $`\varepsilon`$.

With Shield-Net's weights $`W = (1,2,3)`$, (C2) fails: a level-1 same-state request (0.40) outranks a level-2 cross-state one (0.32). On the placeholder data that produces 50–80 such inversions per scarce round; with $`(1,10,100)`$ there are none.
**Test:** `test_urgency_dominates_locality`, and `c2_failures` in every run summary.

### 7.3 The model is integral
Every variable ($`x^c_{ij}`$ or $`x^e_{ij}`$) appears in exactly one supply row (1) and exactly one demand row ((2c) or (2e)). The constraint matrix is therefore the incidence matrix of a bipartite graph (suppliers on one side, core and excess rows on the other), which is totally unimodular. The right-hand sides $`A_{i,\tau}`$, $`C_{j,\tau}`$ and $`X_{j,\tau}`$ are integers ($`s_i, \Delta_\tau`$ are integers; $`r`$ and the core are floored). So every vertex is integral for **any** real objective, and $`\hat\psi`$, $`\varepsilon`$ and the tie-breaks don't affect integrality. CBC solves LPs with simplex, which returns a vertex. Every round is solved to proven optimality.
**Test:** integrality is asserted after every solve, and `test_optimal_against_scipy` re-solves random instances with an independent solver.

### 7.4 Core comes before excess
By (C3), the most valuable excess unit, $`\varepsilon\,W[3]`$, is worth less than the least valuable core unit, $`W[1]\psi_{floor} - (\tilde\alpha+\beta+\mu)`$. So if a supplier ships any excess while one of its eligible requesters still has unmet core, moving one unit from that excess to that core strictly improves the objective. An optimal solution therefore **never ships excess from a supplier that could still serve someone's core**.

Precisely what the single LP guarantees, and what it does not:
- Any difference in value between two core units that is larger than what an excess unit is worth (at most $`\varepsilon W[3] = 0.01`$ at the defaults) is respected exactly. That covers urgency levels (factors of 10), and same- vs cross-state at the prior $`\hat\psi`$ (at least 0.24).
- Smaller differences can occasionally be traded to fit in more excess. In practice these are the distance and price tie-breaks (at most 0.004 per unit). Measured at full size, the core value given up is at most about 0.002% of the core-only optimum. A strict two-stage solve (core first, then excess with core fixed) would remove this, at twice the solve time.

**Test:** asserted every round in the simulation, plus `test_core_before_excess`.

### 7.5 Shield-Net is a special case
Set:
- $`\hat\psi \equiv 1`$ (`use_psi_objective` False, `learn_psi` False) and $`W = (1,2,3)`$
- $`\beta = \mu = 0`$
- `core_excess` False, so the whole request is core, and no price or certificate filters
- $`\tilde\alpha\,\hat c_{ij}`$ equal to the paper's $`\alpha c_{ij}`$, i.e. $`\tilde\alpha = 10^{-6} R_{max}`$

The model then reduces to the paper's LP. This was checked numerically on round 1 of the placeholder data: same edge set (29,479 pairs) and the same optimal value as the paper's formulation solved directly. This configuration is step 0 of the ablation (Section 13).

---

## 8. Between rounds

Run these in order after the round's model is solved.

**Simulation (the world):**
1. **Realise matches.** Each pair with $`x^{\ast}_{ij} = x^{c\ast}_{ij} + x^{e\ast}_{ij} > 0`$ is one shipment. Draw $`B_{ij} \sim \text{Bernoulli}(\psi^{\ast}_{g(i,j)})`$ if `simulate_failures`, else $`B_{ij} = 1`$. Outcomes are drawn from the **true** $`\psi^{\ast}`$, never from $`\hat\psi`$.
**Refusals:** the requester refuses the match ($`B_{ij} = 0`$) if $`\pi_i > \bar\pi_j`$ or if $`\kappa_j = 2`$ and $`q_i = 0`$. This is part of the world, so it applies even when the model's own price and certificate filters are switched off, as in the Shield-Net baseline. With the filters on, no match is ever refused.
Delivered: $`\delta_{ij} = x^{\ast}_{ij} B_{ij}`$.

**Platform (the learning step):**

2. **Learn.** If `learn_psi`: for each match, add 1 to $`S_{g(i,j)}`$ if $`B_{ij} = 1`$, else to $`F_{g(i,j)}`$. The new $`\hat\psi`$ is used from the next round.

**Simulation (continued):**

3. **Suppliers.** Unsold stock $`\ell_i = A_{i,\tau} - \sum_j \delta_{ij}`$ (failed shipments stay with the supplier). Carry-over $`r_{i,\tau} = \min(\lfloor \eta\,\ell_i \rfloor,\ h\,s_i)`$.
4. **Requesters.** $`G_j \mathrel{+}= \sum_i \delta_{ij}`$ and $`carry_{j,\tau} = D_{j,\tau} - \sum_i \delta_{ij}`$. Remove $`j`$ if its carry is 0 (served). Only delivered units count toward $`G_j`$, so a failed match does not use up core.
5. **Abandonment.** Each remaining $`j`$ leaves with probability $`\omega`$. Its residual is logged as abandoned demand.

---

## 9. What learning can and cannot do

The learning step is what a live platform would run: it watches which matches are actually delivered and updates $`\hat\psi`$. In this project the outcomes come from the simulation, which draws them from a $`\psi^{\ast}`$ **we choose**. So:

- The simulation shows that the mechanism works: when the prior is wrong, $`\hat\psi`$ moves toward the truth (Section 13, "value of learning").
- It cannot tell us the real values for India. The district file gives locations and states, which decide each pair's class, but not whether matches succeed. Calibrating $`\psi`$ for India needs real delivery records.

On the placeholder data, learning barely changes outcomes even when the prior is wrong. Supply becomes plentiful from round 4, and urgency already fixes most of the ordering, so a better estimate rarely changes who gets served. That is a result to report, not a bug. Learning should matter more when supply stays scarce (try `--set supply_scale=0.6`).

---

## 10. Metrics

**Per round** (`rounds.csv`):
- **Supply and demand:** $`A^{tot}_\tau`$, outstanding demand, their ratio (below 1 = scarce); core and excess demand, and supply relative to core.
- **Allocation:** eligible edges, units allocated (core and excess separately), expected delivered $`\sum \hat\psi x^{\ast}`$, realised delivered $`\sum \delta`$, failed matches and units, refused matches, utilisation.
- **Geography:** share of same-state units, mean and median road distance.
- **Learning:** $`\hat\psi_{same}, \hat\psi_{cross}`$, whether (C2) holds.
- **Exits and stock:** served, abandoned, leaked stock, stock lost to the storage cap.
- **Stranded requesters:** active with no eligible edge, by first failing filter (variant, price, certificate, distance, no stock).
- **Solver:** status, objective, time.

**Per run** (`summary.json`, and one row per run in the experiment CSVs). Fill always means realised deliveries divided by an amount asked for or needed, summed over requesters.
- **Need-based fill (headline).** Each requester's genuine need over the horizon is $`\min(D^0_j,\ u_{\ell_j} E_j \cdot 14)`$, a fixed property of the facility that does not depend on the policy's $`H`$. Deliveries count only up to it.
  - `need_fill`: share of total genuine need delivered, all requesters including bulk.
  - `wneed_fill_ref`: the same, weighting each requester by fixed reference weights (1, 10, 100) for levels 1/2/3, whatever $`W`$ the run used. This is the headline metric.
  - `excess_received`: units delivered beyond genuine need.
- **Request-based fill.** `fill_overall`, `fill_nonbulk`, `fill_L1`–`fill_L3` and `wfill_ref` divide by the full request $`D^0_j`$. **Bulk requesters are reported separately** (`fill_bulk`): a single bulk request can be the size of a whole round's supply and would swamp every other number.
- `fill_L3_honest`: level-3 requesters whose request is within their genuine need.
- `exp_fill_nonbulk` replaces each match's random outcome with its true success probability (0 for a refused match): $`\sum \psi^{\ast} x / \sum D^0`$ over non-bulk requesters. It removes the luck of each individual draw. It is not completely luck-free, because which demand is left in later rounds depends on earlier outcomes.
- Also: over-askers (requests above genuine need), same-state and same-district share, mean distance, mean price paid, certified share, units abandoned, demand outstanding at the end, final $`\hat\psi`$, `c2_failures`, stranded requester-rounds.

---

## 11. Simulation loop

```
init: r_i = 0; carry_j = 0; G_j = 0; S_g = F_g = 0; random numbers drawn up front
for tau in 1..6:
    enrol suppliers with e_i == tau; add requesters with tau_j == tau
    A_i = s_i * Delta[tau] + r_i ; A_tot = sum A_i
    D_j = carry_j + D0_j * [tau == tau_j]
    C_j = min(D_j, max(0, floor(N_j - G_j))) ; X_j = D_j - C_j
    psi_hat_g = max(psi_floor, (n0*psi0_g + S_g) / (n0 + S_g + F_g)); check C2
    build edges E via a_ij; w_ij = p_j*psi_hat_ij - c_ij
    solve the LP (core flows, excess flows)        # the model
    x_ij = xc_ij + xe_ij
    B_ij = U[tau,i,j] < psi_true_g and not refused ; delta_ij = x_ij * B_ij
    update S_g, F_g                                # the platform's learning
    r_i = min(floor(eta * (A_i - sum_j delta_ij)), h * s_i)
    G_j += sum_i delta_ij ; carry_j = D_j - sum_i delta_ij ; drop served
    abandonment (omega); log metrics
```

---

## 12. Synthetic data generator (seeded)

Districts, their states, populations and the road-distance matrix come from the input file (Section 2.1). Everything below is synthetic and scaled to the default item (face shields, prices in ₹).

### 12.1 Suppliers

| Item | Default |
|---|---|
| Count | $`N_S = 450`$ (times `supply_scale`) |
| Enrolment | 70% at round 1; the rest uniform over rounds 2–6 |
| District | uniform over $`K`$ |
| Class S / M / L | 0.5 / 0.3 / 0.2 |
| Daily rate $`s_i`$ (integer uniform) | S [25, 100], M [60, 250], L [200, 800] |
| Price $`\pi_i`$ | $`U[20,150]`$ × {1.0, 0.9, 0.8} for S / M / L |
| Certified probability | 0.3 / 0.5 / 0.8 for S / M / L |
| Variants (assembled only / kit only / both) | 0.57 / 0.10 / 0.33 (paper's supplier counts) |

### 12.2 Requesters

| Item | Default |
|---|---|
| Arrivals at round $`\tau`$ | Poisson($`\lambda_\tau`$), $`\lambda_\tau = 120\,\theta^{\tau-1}`$, $`\theta = 0.7`$ (surge, then decline) |
| District | weighted by the `population` column if present, else uniform |
| Urgency level $`\ell_j`$ = 1 / 2 / 3 | 0.3 / 0.4 / 0.3 |
| True need $`n_j`$ | round(lognormal($`\ln 400`$, 1.4)), clipped to [50, 100000]; mean ≈ 1065, the paper's average request |
| Staff $`E_j`$ | $`\max\big(1, \lceil n_j\, g_j / (u_{\ell_j} \cdot 14) \rceil\big)`$, $`g_j \sim \text{lognormal}(\ln 2, 0.5)`$; so the genuine need over 14 days is $`n_j g_j`$ |
| Request $`D^0_j`$ | $`n_j`$ (normal requester) |
| Max price $`\bar\pi_j`$ | $`U[30,120] + 20(\ell_j - 1)`$ |
| $`\kappa_j`$ = 0 / 1 / 2 | $`\ell=1`$: 0.5 / 0.3 / 0.2; $`\ell=2`$: 0.3 / 0.4 / 0.3; $`\ell=3`$: 0.2 / 0.3 / 0.5 |
| Acceptable variants (assembled / kit / either) | 0.40 / 0.33 / 0.27 |
| Bulk requester | each arrival is bulk with probability $`P_{bulk} = 0.005`$; $`D^0_j = \text{round}(U[0.5,1.5] \times \bar A_1)`$, exempt from the 100000 clip; $`E_j`$ drawn from $`n_j`$ as for a normal requester, so a bulk request is almost entirely excess |

Here $`\bar A_1 = 0.7\,N_S \cdot 2 \cdot \mathbb{E}[s_i]`$ is the expected round-1 supply, about 112k units. The 14 days used to size facilities is a fixed property of the generated world; it does not change when the policy's $`H`$ is swept, so the generated instance never depends on policy.

Since $`g_j < 1`$ with probability about 8%, about 8% of normal requesters ask for more than their 14-day genuine need. Their extra is excess.

### 12.3 Calibration check (measured over 200 seeds)

| Quantity | Target | Measured |
|---|---|---|
| arrivals per round | 120, 84, 58.8, 41.2, 28.8, 20.2 | 119.7, 84.3, 57.8, 41.2, 28.8, 20.1 |
| mean normal request | ≈ 1065 | 1064 |
| share of normal requests above genuine need | ≈ 8% ($`P(g_j < 1)`$) | 7.9% |
| round-1 supply | ≈ 112k | 111.7k |
| round-1 normal demand | ≈ 128k | 127.4k |
| bulk requesters per run | ≈ 1.8 | 1.7 |

Round 1 is scarce, more than the average ratio (0.88) suggests. Request sizes are heavy-tailed, so round-1 demand ranges from about 90k to 200k between seeds, and about 45% of seeds also have a bulk request in round 1. The typical (median) round-1 supply/demand ratio in full runs is about 0.57. Supply then grows (enrolment plus carried stock) while arrivals decline, so from round 4 supply exceeds demand.

In full runs, the share of failed matches is within 0.2% of what $`\psi^{\ast}`$ predicts, abandonment is 0.098 per unserved requester-round (target 0.10), and with $`\psi^{\ast}`$ = prior the final $`\hat\psi`$ is 0.398 / 0.162 (0.40 / 0.16).

---

## 13. Experiments

Use the same seeds across settings, and at least 20 seeds for anything you report.

**Ablation ladder.** Start from Shield-Net (Section 7.5) and add one model feature at a time.

**Every step runs in the same simulated world as full PULSE**: carried stock, abandonment, random match outcomes and refusals. So the steps differ only in the platform's decisions, and the comparison is fair. Inventory and abandonment describe the world, not the platform, so they are not steps.

| Step | Adds |
|---|---|
| 0 | Shield-Net |
| 1 | expected-delivery objective, fixed prior $`\psi`$ |
| 2 | learning $`\psi`$ |
| 3 | urgency weights (1, 10, 100) |
| 4 | price and certificate filters, PULSE tie-breakers |
| 5 | core/excess split = full PULSE |

Indicative result (10 seeds, placeholder data; report your own 20-seed numbers, not these):
- Urgency-weighted genuine need met (`wneed_fill_ref`) rises from 0.49 (Shield-Net) to 0.72 (PULSE), and PULSE is higher on every seed, by 0.17 to 0.32.
- Most of that gain comes from step 1 (planning for deliveries) and step 4 (not making matches the requester will refuse). Shield-Net makes over a thousand refused matches per run.
- Step 5 (core/excess) raises genuine need met (`need_fill`) from 0.60 to 0.66, by up to 0.16 on a seed, and cuts what over-askers receive beyond their need by about 40%. Bulk requests still get served, but only after everyone's core.

**Value of learning.** Set the true $`\psi^{\ast}_{same} \in \lbrace 0.30, 0.40, 0.55\rbrace`$ and $`\psi^{\ast}_{cross} \in \lbrace 0.08, 0.16, 0.25\rbrace`$, so the prior is right or wrong in either direction. Compare `learn_psi` on vs off, and track how fast $`\hat\psi`$ converges.

**Sensitivity sweeps** (one parameter at a time, from the PULSE defaults):
- supply scale × {0.6, 1.0, 1.5} on the number of suppliers
- $`P_{bulk} \in \lbrace 0, 0.005, 0.02\rbrace`$: how robust PULSE is to giant requests
- $`H \in \lbrace 7, 14, 28\rbrace`$ days: how generous the core is
- $`\eta \in \lbrace 0, 0.5, 1\rbrace`$
- $`\omega \in \lbrace 0, 0.1, 0.2\rbrace`$
- $`W \in \lbrace (1,10,100), (1,2,3)\rbrace`$

**Headline comparisons:**
- urgency-weighted genuine need met and expected fill, Shield-Net vs PULSE
- genuine need met, and units received beyond need, with and without core/excess
- fill of honest level-3 requesters
- same-state share
- stranded and abandoned demand

**Common random numbers.** The random numbers for match outcomes and abandonment are drawn up front: one per (round, supplier, requester) and one per (round, requester). A given match or exit therefore turns out the same under every configuration, so differences on the same seed come only from different decisions. This makes paired comparisons, like the ablation steps, much less noisy. A test checks it.

---

## 14. Assumptions and limitations

1. **Supplier minimum orders are not modelled.** The paper names them, with price and quality, as a reason matches failed. PULSE addresses price and quality but not this; enforcing them would need binary variables and turn each round into an integer program.
2. **Genuine need is estimated from a usage norm.** The rates $`u_{\ell}`$ (0.25 / 0.5 / 1.0 units per staff per day) and $`H = 14`$ days are assumptions. A facility whose real use is far above its type's norm gets part of its real need treated as excess; it is still served, just after everyone's core.
3. **Excess is shared by priority, not proportionally.** Leftover stock goes to excess in the same urgency and locality order as core. An alternative is to share it in proportion to each requester's core (equal days of buffer stock); that needs a second, max-min-fair LP.
4. **One request per requester.** Repeat requests are not modelled, so receipts reduce core only within a request's life ($`G_j`$), not across separate requests.
5. **One item, pooled capacity.** A supplier's daily rate is shared across variants; real lines produce kits and assembled units at different rates.
6. **Declared data is truthful and fixed.** Capacity, price, certification and staff count don't change after enrolment, and suppliers don't reprice. Core relies on staff counts being truthful; checking them is outside the model.
7. **Districts are points.** All locations in a district are one point; the distance within a district is a random $`U[10,20]`$ km draw, not measured.
8. **No transport model.** No lead times, vehicle capacity or delivery scheduling. The price is a delivered price; who pays for shipping is outside the model.
9. **Learning is coarse, and only demonstrated in simulation.**
   - The prior comes from US data, and India's values need field data (Section 9).
   - There are only two classes (same / cross state), so distance within a state is ignored.
   - Learning sees only the matches the model chose to make. If few cross-state matches happen, $`\hat\psi_{cross}`$ updates slowly (selection bias).
10. **Match outcomes are all-or-nothing** and become known before the next round. A refusal (over budget, or uncertified when required) is certain, and the platform cannot tell it apart from any other failure, so in configurations without the filters it also lowers $`\hat\psi`$.
11. **Behavioural parameters are constant over time.** $`\eta, \omega`$ do not change over the horizon. Panic, media coverage and government action would make them vary; we sweep them instead of modelling that.
12. **Urgency weights are a value judgement.** Under $`W = (1,10,100)`$, urgency always outranks locality across levels.
13. **The two-week horizon truncates demand.** Demand still outstanding after round 6 is reported, not resolved.
14. **The synthetic data is not validated** against real Indian demand or supplier data. Only the request-size mean, the supplier variant mix and $`\psi`$ are anchored to the paper.

---

## 15. Implementation notes

- **PuLP version:** pinned to **3.3.2**. PuLP 4.x (what a plain `pip install pulp` now gives) changed the modelling API and no longer bundles the CBC solver, so this code would not run on it unchanged.
- **Solver:** CBC, bundled with PuLP 3.3.2. `--set solver=highs` switches to HiGHS (needs `pip install highspy`); the model is unchanged.
- **Eligibility** is built with vectorised NumPy operations over (enrolled suppliers × active requesters): about 15–32 thousand allowed pairs per round on the placeholder data. A core variable is created on a pair only if the requester has core left, and an excess variable only if it has excess.
- **Fast model building:** the objective and row sums use `pulp.LpAffineExpression` from prebuilt lists, not nested `lpSum`, which is slow at this size.
- **Integers everywhere:** $`s_i`$, $`A_{i,\tau}`$, $`D_{j,\tau}`$, $`C_{j,\tau}`$, $`X_{j,\tau}`$ and $`r_{i,\tau}`$. The LP solution is asserted integral, then rounded.
- **Reproducibility:** one master seed, with separate `numpy.random.Generator` streams for supplier generation, requester generation, match outcomes and abandonment. Changing a policy parameter never changes the generated instance or the random numbers. Runs are exactly reproducible.
- **Checks every round** (`check_invariants`, on by default): supply, core and excess limits; no idle usable edge (7.1); integrality (7.3); core before excess (7.4); every edge value positive (C1).
- **Tests** (`python -m pytest`, about 5 seconds):
  - model: integrality and no idle edge, optimality against an independent scipy solve, urgency dominance, core before excess, excess by priority once core is met
  - loader: placeholder loads, diagonal in [10, 20], file order doesn't matter, blank pairs never matched, bad files rejected, cache round-trip
  - simulation: conservation of demand, core and excess add up to demand, core/excess off treats all demand as core, inflating a request gains no core, the instance is independent of policy, learning moves toward the truth, learning off keeps the prior, (C1) rejected, reproducibility, common random numbers, refused matches never deliver

---

## 16. Running the code

### Setup (once)

Python 3.10 or newer. In PowerShell (Windows):

```powershell
cd "C:\path\to\MSO-103_PULSE_Matching"
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. On Linux or macOS, use `python3 -m venv .venv` and `source .venv/bin/activate` instead.

In every new terminal after that: `cd` to the folder and activate the venv again.

### Project layout

| Path | What it is |
|---|---|
| `PULSE_MODEL.md` | short, model-only summary: the LP and how it fixes each Shield-Net issue |
| `pulse/config.py` | every parameter with its README default |
| `pulse/districts.py` | Excel loader and validation (Section 2.1) |
| `pulse/make_districts.py` | placeholder workbook generator |
| `pulse/generator.py` | synthetic suppliers and requesters (Section 12) |
| `pulse/model.py` | the round LP in PuLP (Section 6) |
| `pulse/simulation.py` | the six-round loop (Sections 5 and 8) |
| `pulse/metrics.py` | per-round records and run summaries (Section 10) |
| `run.py` | one simulation, printed round by round |
| `experiments.py` | ablation, sweeps and learning suites (Section 13) |
| `plots.py` | figures from the experiment results |
| `tests/` | the Section 7 properties and loader checks |

### Commands

```bash
# one run (creates data/districts.xlsx on first use if it is missing)
python run.py                          # seed 1
python run.py --seed 3
python run.py --set H=7 --set "W=(1,2,3)" --set omega=0   # override parameters
python run.py --out results/run2       # keep this run's output separately

# experiment suites; results go to results/
python experiments.py --suite ablation --seeds 5
python experiments.py --suite sweeps   --seeds 5
python experiments.py --suite learning --seeds 5
python experiments.py --suite all      --seeds 20 --jobs 4   # the full set for the report

# figures from whatever is in results/
python plots.py

# tests
python -m pytest
```

`--set KEY=VALUE` works on both `run.py` and `experiments.py` and accepts any field of `pulse/config.py` (the "Code name" column in Section 3.3). Numbers, `True`/`False`, `inf`, `None` and tuples such as `"(1,2,3)"` are all understood.

### Using your real district file

Put the workbook anywhere, in the format of Section 2.1, and point to it:

```bash
python run.py --data path/to/real_districts.xlsx
python experiments.py --suite all --data path/to/real_districts.xlsx
```

The first load of an 800 × 800 sheet takes a few seconds. After that it is read from `real_districts.xlsx.cache.npz`, which is rebuilt automatically whenever the workbook changes. If the file has a problem (mismatched IDs, a non-numeric cell, an off-diagonal zero), the loader stops with a message naming the cell.

### Runtime

Measured on a 2-core Linux machine:

| Step | Time |
|---|---|
| generate the placeholder workbook | ~8 s (once) |
| first load of an 800 × 800 workbook | ~6 s, then ~0.4 s from the cache |
| one run | 3.1 s on average (2.7–3.8 s over 10 seeds) |
| `--suite ablation`, per seed | ~30 s |
| `--suite sweeps`, per seed | ~50 s |
| `--suite learning`, per seed | ~55 s |
| `--suite all --seeds 20` | ~45 min with `--jobs 1`; divide by the number of jobs |
| `python -m pytest` | ~5 s |

Most of a run's time is PuLP writing each round's model to a file and starting CBC, not the solve itself. Use `--jobs` up to your number of cores minus one.

### Outputs

- `results/single/rounds.csv` and `summary.json`: from `run.py`. Add `--matches` to also get every individual match.
- `results/<suite>_runs.csv`: one row per configuration and seed.
- `results/<suite>_rounds.csv`: one row per configuration, seed and round.
- `results/<suite>_table.csv`: mean and standard deviation per configuration; the table for the report.
- `results/figures/*.png`: from `plots.py`.

**Use many seeds.** Most matches fail (the success probabilities are 0.16–0.40) and a single match can be thousands of units, so one seed's realised fill is noisy. Differences between configurations at 1–2 seeds are mostly luck. Use at least 20 seeds for anything you report, and look at `exp_fill_nonbulk` alongside the realised fills.

### Cleaning up

```powershell
Remove-Item -Recurse -Force results, .pytest_cache -ErrorAction SilentlyContinue   # outputs only
Remove-Item data\*.cache.npz                                                         # distance cache only
```

`.gitignore` already excludes `.venv`, `results`, `__pycache__` and cache files.
