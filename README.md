# PULSE: Pop-Up Unified Logistics and Supplied Emergence

**Formal model specification**

PULSE matches nontraditional ("pop-up") suppliers with requesters of an emergency item, in repeated rounds over a two-week horizon. It extends Shield-Net (Alcock, Boutilier, Siddiq, *INFORMS J. Applied Analytics*, 2022) and is set in India at district level.

The document is written to be implemented directly in Python with **PuLP** (bundled CBC solver). Every quantity that enters a constraint is integer, and every random draw comes from a seeded RNG.

---

## 1. What changed and why

| # | Shield-Net | PULSE | Real-world reason |
|---|---|---|---|
| 1 | Face shields only | Generic item with variants; requester accepts a **set** of variants | Many buyers would take kits or assembled |
| 2 | US states | Indian districts, grouped into states | Local setting |
| 3 | Euclidean distance between state centroids | Road-distance matrix with a nonzero intra-district distance | Roads, not straight lines |
| 4 | 2 rounds/week, capacity per round | 3 rounds/week (Mon/Wed/Fri); capacity is a **daily rate** × days since the last round | Fri→Mon is 3 days, the other gaps are 2 |
| 5 | No quality signal | Certified flag on suppliers; requesters choose **none / prefer / require** | Under shortage, many buyers accept uncertified stock rather than nothing |
| 6 | Price ignored | Supplier price; requester budget as a hard filter; price as a tie-breaker | Paper names price as a source of failed matches |
| 7 | Minimum request of 100, outside the model | **Strict** supplier minimum order + a **no-tiny-remainder** rule | Paper: suppliers refused small orders |
| 8 | No inventory | Unsold stock carries over with **leakage** η and a **storage cap** of h days | Suppliers had outside buyers; stock doesn't wait forever |
| 9 | No hoarding control | **Plausibility flag** at arrival (request vs. size); flagged requesters are **always capped** each round until verified | Hoarding is about what is asked for, not what the optimiser allocated |
| 10 | Minimise weighted shortfall + α·distance | Minimise **expected** urgency-weighted shortfall, using a match-success probability | Paper: same-state was the only significant predictor of success (OR 3.55); only ~27% of matches succeeded |
| 11 | Success assumed | Success probability is **learned** from observed outcomes (Beta-Bernoulli), starting from the paper's values as a prior | US numbers are a starting guess for India, not a fact |
| 12 | Self-reported urgency, weights 1/2/3 | Urgency level from **facility type**; weights **1/10/100**, so urgency dominates across levels and locality decides within a level | Self-reports are gameable; a COVID ward must not lose to a nearby grocery store |
| 13 | Requesters wait until served | Per-round **abandonment** ω | 52% of unsuccessful requesters sourced elsewhere |
| 14 | Matches assumed to happen | Simulation draws match success; failed units return to demand | Matched is not the same as delivered |

---

## 2. Sets and time

- $`K`$: districts. $`\sigma(k)`$: state of district $`k`$.
- $`R[k,l]`$: road distance in km from district $`k`$ (supplier side) to district $`l`$ (requester side), read from the input file (Section 2.1). $`R[k,l] = \infty`$ if the cell is missing (unreachable).
- $`R[k,k] = r^{intra}_k`$: intra-district distance, drawn once per district from $`U[10, 20]`$ km with a seeded RNG, unless the file gives a positive diagonal value.
- $`R_{max}`$: the largest **finite** entry of $`R`$.
- Rounds $`\tau = 1,\dots,6`$ on days $`d_\tau \in \lbrace 1,3,5,8,10,12\rbrace`$. Gap $`\Delta_\tau = d_\tau - d_{\tau-1}`$ with $`d_0 = -1`$, so $`\Delta = (2,2,2,3,2,2)`$.
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
| `state` | yes | state or UT name; defines $`\sigma(k)`$ |
| `population` | no | used to place requesters (Section 11.2) |

**Sheet `distances`**: an $`|K| \times |K|`$ matrix (about 800 × 800) of road distances in km. The first column holds row `district_id`s and the first row holds column `district_id`s. Row $`k`$, column $`l`$ is $`R[k,l]`$.

**The loader checks and normalises:**
- Row IDs, column IDs and the `districts` sheet contain exactly the same IDs. Rows and columns are reordered to the `districts` order, so the file's own ordering does not matter.
- Every non-blank value is a number $`\ge 0`$.
- **Diagonal** values that are 0 or blank are replaced by $`r^{intra}_k \sim U[10,20]`$ km.
- **Off-diagonal blank cells** become $`\infty`$ (unreachable), so those pairs are never eligible. They are never set to 0, which would make them look like the closest pairs.
- **Off-diagonal zeros** are an error, since two different districts cannot be 0 km apart.
- **Asymmetry** is allowed. $`R[k,l]`$ is always read as supplier district $`k`$ to requester district $`l`$.
- The number of blank cells and asymmetric pairs is logged.

**Placeholder file**, used until the real one is ready, in exactly the same format:
- 800 districts with IDs `D001`–`D800`, spread over 36 states/UTs `S01`–`S36`. District counts per state are drawn from a Dirichlet(1) split of the 800, with at least 1 district per state.
- Each state gets a centre placed uniformly in a 3000 km × 3000 km square. Each district sits at its state centre plus Gaussian noise with $`\sigma = 150`$ km.
- Off-diagonal road distance = 1.3 × straight-line distance, rounded to whole km and symmetric. The diagonal is left 0, so the loader's $`U[10,20]`$ rule is exercised.
- Population ~ round(lognormal($`\ln(2 \times 10^6)`$, 0.6)).
- Generated from a fixed seed and saved with `openpyxl`.

## 3. Parameters

### 3.1 Supplier $`i`$

| Symbol | Meaning |
|---|---|
| $`k(i)`$ | district |
| $`e_i`$ | enrolment round |
| $`s_i \in \mathbb{Z}_{+}`$ | production rate, units per **day** |
| $`m_i \in \mathbb{Z}_{+}`$ | minimum order per shipment, declared by the supplier, with $`m_i \ge m_{min}`$ |
| $`\pi_i > 0`$ | unit price (delivered) |
| $`q_i \in \lbrace 0,1\rbrace`$ | 1 = quality-certified |
| $`V_i \subseteq V`$ | variants offered |

### 3.2 Requester $`j`$

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
| $`\text{size}_j \in \mathbb{Z}_{+}`$ | staff count (or beds), declared on the intake form |

| Facility type | Level $`\ell_j`$ |
|---|---|
| Treating patients of the emergency (e.g. COVID ward) | 3 |
| Other healthcare (hospital, nursing home, clinic) | 2 |
| Essential non-healthcare (police, grocery, municipal) | 1 |

### 3.3 Global parameters

Each parameter has a **class**, which says how it is justified in the report:
- **Policy:** a design choice we defend.
- **Calibration:** taken from the paper.
- **Behavioural:** unknown, so it gets a sensitivity sweep.
- **Scenario:** the simulated "truth", which the model does not know.
- **Switch:** turns a feature on or off.

| Symbol | Meaning | Default | Class |
|---|---|---|---|
| $`W`$ | urgency weights for levels 1/2/3 | (1, 10, 100) | Policy |
| $`m_{min}`$ | platform minimum request size | 50 | Policy |
| $`c`$ | share cap for flagged requesters, fraction of round supply | 0.05 | Policy |
| $`\tilde\alpha`$ | distance weight (tie-breaker) | 0.002 | Policy |
| $`\beta`$ | price weight (tie-breaker) | 0.002 | Policy |
| $`\mu`$ | penalty per unit for uncertified stock to a "prefer certified" requester | 0.02 | Policy |
| $`\Delta_{max}`$ | optional distance cutoff (km) | $`\infty`$ | Policy |
| $`\psi^0_{same}, \psi^0_{cross}`$ | prior mean success probability | 0.40, 0.16 | Calibration |
| $`n_0`$ | prior strength, in pseudo-matches | 20 | Calibration |
| $`\psi_{floor}`$ | lower clip on learned $`\hat\psi`$ | 0.05 | Policy |
| $`\psi^{\ast}_{same}, \psi^{\ast}_{cross}`$ | **true** success probability, used only to draw outcomes | = prior | Scenario |
| $`u`$ | plausible usage, units per staff per day | 0.5 | Behavioural |
| $`T^{cov}`$ | days of coverage a request may plausibly cover | 14 | Policy |
| $`\nu`$ | probability a flagged requester is verified (unflagged) per round | 0.25 | Behavioural |
| $`\eta`$ | fraction of unsold stock still available next round | 0.5 | Behavioural |
| $`h`$ | storage cap, in days of production | 5 | Behavioural |
| $`\omega`$ | probability an unserved requester abandons per round | 0.10 | Behavioural |
| `enforce_min_shipment` | Mode A (False, LP) / Mode B (True, MILP) | per run | Switch |
| `simulate_failures` | draw match success from $`\psi^{\ast}`$ (True), or all succeed (False) | True | Switch |
| `learn_psi` | update $`\hat\psi`$ from outcomes (requires `simulate_failures`) | True | Switch |

**Required conditions:**
- (C1) $`\tilde\alpha + \beta + \mu < W[1]\cdot\psi_{floor}`$. Defaults: $`0.024 < 0.05`$.
- (C2) $`\hat\psi_{same} < 10 \hat\psi_{cross} - (\tilde\alpha+\beta+\mu)`$, checked every round. Defaults: $`0.40 < 1.576`$.

Section 7 explains both.

**Prior calibration.** The paper reports a 27% success rate among classified matches, 48% of matches local (same state), and an odds ratio of 3.55 for local matches. Solving $`0.48 \psi_{same} + 0.52 \psi_{cross} = 0.27`$ together with $`\text{odds}(\psi_{same}) = 3.55\cdot\text{odds}(\psi_{cross})`$ gives $`\psi_{same} \approx 0.40`$ and $`\psi_{cross} \approx 0.16`$.

### 3.4 Derived per-pair quantities

- Variant compatibility: $`\gamma_{ij} = \mathbf{1}[V_i \cap V_j \ne \emptyset]`$.
- Normalised distance: $`\hat c_{ij} = R[k(i),k(j)] / R_{max} \in (0,1]`$.
- Normalised price: $`\rho_{ij} = \pi_i / \bar\pi_j \in (0,1]`$ on eligible pairs.
- Believed success probability: $`\hat\psi_{ij,\tau} = \hat\psi_{g(i,j),\tau}`$ (Section 4).
- Tie-break cost: $`c_{ij} = \tilde\alpha \hat c_{ij} + \beta \rho_{ij} + \mu \mathbf{1}[\kappa_j = 1] (1 - q_i)`$.
- Plausible need: $`P_j = u \cdot \text{size}_j \cdot T^{cov}`$.

---

## 4. State carried between rounds

| State | Meaning | Initial value |
|---|---|---|
| $`r_{i,\tau-1}`$ | supplier stock carried into round $`\tau`$ | 0 |
| $`carry_{j,\tau-1}`$ | requester's outstanding demand carried into round $`\tau`$ | 0 |
| $`z_j \in \lbrace 0,1\rbrace`$ | plausibility flag | set at arrival (Section 5, step 4) |
| $`S_g, F_g`$ | observed successful / failed matches in class $`g`$ | 0 |

The belief used in round $`\tau`$ is the Beta posterior mean, clipped:

```math
\hat\psi_{g,\tau} = \max\left(\psi_{floor},\ \frac{n_0 \psi^0_g + S_g}{n_0 + S_g + F_g}\right)
```

$`S_g`$ and $`F_g`$ count **matches** (one per $`x_{ij} > 0`$), not units, because a match succeeds or fails as a whole. If `learn_psi` is False, $`\hat\psi_g = \psi^0_g`$ throughout.

---

## 5. Pre-solve for round $`\tau`$

1. **Enrolment and arrivals.** Add suppliers with $`e_i = \tau`$. Add requesters with $`\tau_j = \tau`$.
2. **Supply.** Fresh production $`f_{i,\tau} = s_i \Delta_\tau`$. Available stock $`A_{i,\tau} = f_{i,\tau} + r_{i,\tau-1}`$. Total $`A^{tot}_\tau = \sum_{i \in I_\tau} A_{i,\tau}`$.
3. **Outstanding demand.** $`D_{j,\tau} = carry_{j,\tau-1} + D^0_j \mathbf{1}[\tau = \tau_j]`$.
4. **Plausibility flag (new arrivals only).** $`z_j = \mathbf{1}[D^0_j > P_j]`$.
5. **Share cap.** $`\text{cap}_\tau = \max(\lfloor c A^{tot}_\tau \rfloor,\ m_{min})`$ and

   ```math
   \tilde D_j = \begin{cases}
   \min(D_{j,\tau},\ \text{cap}_\tau) & \text{if } z_j = 1\\
   D_{j,\tau} & \text{otherwise}
   \end{cases}
   ```

   The cap applies to every flagged requester in every round until verification, whatever the aggregate supply. Demand above the cap is **deferred, not dropped**: it stays in $`D_{j,\tau}`$. A flag means *unverified*, not *guilty*.

6. **Eligibility.**

   ```math
   a_{ij} = \gamma_{ij}\cdot \mathbf{1}[\pi_i \le \bar\pi_j]\cdot \mathbf{1}[q_i \ge \mathbf{1}[\kappa_j = 2]]\cdot \mathbf{1}[R[k(i),k(j)] \le \Delta_{max}]
   ```

   A pair with $`R[k(i),k(j)] = \infty`$ (missing in the file) is never eligible, even when $`\Delta_{max} = \infty`$. In Mode B, also require $`\min(A_{i,\tau}, \tilde D_j) \ge m_i`$, because otherwise no valid shipment exists. The edge set is $`E_\tau = \lbrace (i,j) : a_{ij} = 1\rbrace`$, and variables exist only on edges.

7. **Coefficients and bounds.** $`w_{ij} = p_j \hat\psi_{ij,\tau} - c_{ij}`$ and $`\bar x_{ij} = \min(A_{i,\tau}, \tilde D_j)`$.
8. **Check (C2)** with the current $`\hat\psi`$. Log a warning if it fails; the model is still valid, but Property 7.2 no longer holds.

---

## 6. Optimisation model for round $`\tau`$

### 6.1 Objective

Minimise expected urgency-weighted shortfall plus tie-break costs:

```math
\min\ \sum_{j\in J_\tau} p_j\Big(\tilde D_j - \sum_{i} \hat\psi_{ij} x_{ij}\Big) \quad +\quad  \sum_{(i,j)\in E_\tau} c_{ij} x_{ij}
```

Dropping the constant $`\sum_j p_j \tilde D_j`$, this is the same as

```math
\max\ \sum_{(i,j)\in E_\tau} w_{ij} x_{ij}, \qquad w_{ij} = p_j \hat\psi_{ij} - c_{ij} > 0 .
```

Implement the second form (`LpMaximize`).

### 6.2 Mode A (LP, no minimum order)

```math
\begin{aligned}
&\sum_{j} x_{ij} \le A_{i,\tau} && \forall i \in I_\tau && (1)\\
&\sum_{i} x_{ij} \le \tilde D_j && \forall j \in J_\tau && (2)\\
&x_{ij} \ge 0 && \forall (i,j)\in E_\tau
\end{aligned}
```

### 6.3 Mode B (MILP, strict minimum order, no tiny remainder)

Add $`y_{ij} \in \lbrace 0,1\rbrace`$ per edge (1 = shipment happens) and $`w_j \in \lbrace 0,1\rbrace`$ per requester (1 = fully closed this round). $`x_{ij} \in \mathbb{Z}_{+}`$.

```math
\begin{aligned}
&(1),\ (2)\\
&m_i y_{ij} \le x_{ij} \le \bar x_{ij} y_{ij} && \forall (i,j)\in E_\tau && (3)\\
&\sum_i x_{ij} \ge D_{j,\tau} w_j && \forall j && (4)\\
&\sum_i x_{ij} \le D_{j,\tau} - m_{min} (1 - w_j) && \forall j && (5)
\end{aligned}
```

- (3) is the strict minimum order: every shipment from $`i`$ is either 0 or at least $`m_i`$. There is no closing exception.
- (4)–(5) say that after this round the requester's remainder $`D_{j,\tau} - \sum_i x_{ij}`$ is either **0** or **at least $`m_{min}`$**. So a remainder can always still be served by a smallest-class supplier, and no request is left with a fragment nobody will ship.
- Preprocessing: if $`\tilde D_j < D_{j,\tau}`$ (capped), fix $`w_j = 0`$, since (2) makes $`w_j = 1`$ infeasible.

| Mode | Constraints | Variables | Problem class |
|---|---|---|---|
| A | (1)–(2) | $`x`$ continuous | LP (integral, Section 7.3) |
| B | (1)–(5) | $`x`$ integer, $`y, w`$ binary | MILP |

---

## 7. Properties (each one is also a test)

### 7.1 Serving always beats not serving
For every edge, $`w_{ij} \ge W[1] \psi_{floor} - (\tilde\alpha+\beta+\mu) > 0`$ by (C1). This holds every round, whatever the learning does, because $`\hat\psi`$ is clipped at $`\psi_{floor}`$. In **Mode A**, an optimal solution therefore never leaves an edge where the supplier still has stock **and** the requester still has matchable demand.
**Test:** after solving Mode A, assert there is no edge with $`A_{i,\tau} - \sum_j x_{ij} > 0`$ and $`\tilde D_j - \sum_i x_{ij} > 0`$.

### 7.2 Urgency dominates across levels; locality decides within a level
A unit's value is $`p_j \hat\psi_{ij}`$, up to tie-breaks. With $`W = (1,10,100)`$ and the prior $`\hat\psi`$:

| Level | same state | cross state |
|---|---|---|
| 1 | 0.40 | 0.16 |
| 2 | 4.0 | 1.6 |
| 3 | 40 | 16 |

The worst cross-state unit at a higher level beats the best same-state unit at the level below as long as $`10 \hat\psi_{cross} - (\tilde\alpha+\beta+\mu) > \hat\psi_{same}`$, which is (C2). The gap between levels is a factor of 10 and the worst level-boundary case is level 1 vs level 2, so (C2) covers every pair of levels.

Within a level, same-state pairs are preferred by a factor of $`\hat\psi_{same}/\hat\psi_{cross}`$. Then come the certificate preference ($`\mu`$), then distance and price ($`\tilde\alpha, \beta`$).

The **"locality-first"** variant $`W = (1,2,3)`$ is run as a sensitivity case. Under it, a level-1 same-state request (0.40) outranks a level-2 cross-state one (0.32).

### 7.3 Mode A is integral
Constraints (1)–(2) form the incidence matrix of a bipartite graph, which is totally unimodular. The right-hand sides $`A_{i,\tau}`$ and $`\tilde D_j`$ are integers ($`s_i, \Delta_\tau`$ are integers, $`r`$ is floored, the cap is floored). So every vertex is integral for **any** real objective, and $`\hat\psi`$ and the tie-breaks don't affect integrality. PuLP's CBC solves LPs with the CLP simplex, which returns a vertex.
**Test:** round, and assert $`|x - \mathrm{round}(x)| < 10^{-6}`$.

### 7.4 Mode B is always feasible
$`x = y = w = 0`$ is feasible if and only if $`D_{j,\tau} \ge m_{min}`$ for every active $`j`$. This invariant holds by induction:
- **Base:** $`D^0_j \ge m_{min}`$.
- **Step:** the next carry is $`(D_{j,\tau} - \sum_i x_{ij}) + \sum_i (\text{failed } x_{ij})`$. The first term is $`0`$ or $`\ge m_{min}`$ by (4)–(5). Each failed shipment is $`\ge m_i \ge m_{min}`$ by (3), because a match fails as a whole. A sum of terms that are each $`0`$ or $`\ge m_{min}`$ is itself $`0`$ or $`\ge m_{min}`$.
- The cap keeps $`\tilde D_j \ge m_{min}`$ because $`\text{cap}_\tau \ge m_{min}`$.

**Test:** assert the invariant on every carry in Mode B.

### 7.5 Bounds on Mode B
Mode B only adds constraints to Mode A, so $`Z^{\ast}_B \le Z^{\ast}_A`$. A tighter bound is $`Z^{LP}_B`$, Mode B with all variables made continuous. For each round, report:

```math
\text{gap}_\tau = \frac{Z^{LP}_B - Z^{found}_B}{Z^{LP}_B}
```

This is an upper bound on how far a time-limited Mode B solution can be from optimal. Separately, report $`Z^{\ast}_A - Z^{\ast}_B`$ as the **cost of minimum orders**.

### 7.6 Shield-Net is a special case
Set:
- $`\hat\psi \equiv 1`$ with `learn_psi` False, and $`W = (1,2,3)`$
- $`\beta = \mu = 0`$, $`\eta = 0`$, $`\omega = 0`$
- all $`z_j = 0`$ (no cap), and no price or certificate filters
- $`\tilde\alpha \hat c_{ij}`$ equal to the paper's $`\alpha c_{ij}`$
- Mode A

The model then reduces to the paper's LP. This is the baseline for comparisons (Section 12).

---

## 8. Post-solve updates for round $`\tau`$

Run these in order.

1. **Realise matches.** For each $`x^{\ast}_{ij} > 0`$, draw $`B_{ij} \sim \text{Bernoulli}(\psi^{\ast}_{g(i,j)})`$ if `simulate_failures`, else $`B_{ij} = 1`$. Delivered: $`\delta_{ij} = x^{\ast}_{ij} B_{ij}`$. Outcomes are drawn from the **true** $`\psi^{\ast}`$, never from $`\hat\psi`$.
2. **Learn.** If `learn_psi`: for each match, add 1 to $`S_{g(i,j)}`$ if $`B_{ij} = 1`$, else to $`F_{g(i,j)}`$. The new $`\hat\psi`$ takes effect next round.
3. **Suppliers.** Unsold stock $`\ell_i = A_{i,\tau} - \sum_j \delta_{ij}`$ (failed shipments stay with the supplier). Carry-over $`r_{i,\tau} = \min(\lfloor \eta \ell_i \rfloor,\ h s_i)`$.
4. **Requesters.** $`carry_{j,\tau} = D_{j,\tau} - \sum_i \delta_{ij}`$. Remove $`j`$ if this is 0 (served).
5. **Abandonment.** Each remaining $`j`$ leaves with probability $`\omega`$. Log its residual as abandoned demand.
6. **Verification.** Each remaining flagged $`j`$ ($`z_j = 1`$) is unflagged with probability $`\nu`$.

---

## 9. Metrics logged per round

- **Allocation:** units allocated $`\sum x^{\ast}`$; expected delivered $`\sum \hat\psi x^{\ast}`$; realised delivered $`\sum \delta`$.
- **Fill rates:** realised fill overall and by urgency level; fill of non-flagged level-3 requesters (the group the cap protects).
- **Supply:** $`A^{tot}_\tau`$; aggregate supply/demand ratio $`A^{tot}_\tau / \sum_j D_{j,\tau}`$; utilisation $`\sum x^{\ast} / A^{tot}`$; leaked stock; stock lost to the storage cap.
- **Geography:** share of same-state matches, share of same-district matches, mean and median road distance.
- **Price and quality:** mean unit price paid; share of certified units overall and to $`\kappa = 1`$ requesters.
- **Hoarding control:** number of flagged requesters; units deferred by the cap; verifications this round.
- **Learning:** $`\hat\psi_{same}, \hat\psi_{cross}`$ and their gap to $`\psi^{\ast}`$; whether (C2) holds.
- **Failures and exits:** failed matches and units; abandoned requesters and units.
- **Stranded requesters:** active with no eligible edge, broken down by cause (variant, price, certificate, distance, minimum order).
- **Solver:** status, objective, time, number of edges and binaries; for Mode B, $`Z^{LP}_B`$ and $`\text{gap}_\tau`$.
- **End of horizon:** outstanding demand after round 6, which is truncated, not unmet forever.

---

## 10. Simulation loop

```
init: r_i = 0; carry_j = 0; S_g = F_g = 0; rng streams seeded
for tau in 1..6:
    enrol suppliers with e_i == tau; add requesters with tau_j == tau
    for new requesters: z_j = [D0_j > P_j]
    A_i = s_i * Delta[tau] + r_i ; A_tot = sum A_i
    D_j = carry_j + D0_j * [tau == tau_j]
    cap = max(floor(c * A_tot), m_min)
    D_tilde_j = min(D_j, cap) if z_j else D_j
    psi_hat_g = max(psi_floor, (n0*psi0_g + S_g) / (n0 + S_g + F_g)); check C2
    build edges E via a_ij (+ Mode B filter); w_ij = p_j*psi_hat_ij - c_ij
    solve Mode A (LP) or Mode B (MILP, plus its LP relaxation for the bound)
    draw B_ij ~ Bernoulli(psi_star_g); delta_ij = x_ij * B_ij; update S_g, F_g
    r_i = min(floor(eta * (A_i - sum_j delta_ij)), h * s_i)
    carry_j = D_j - sum_i delta_ij ; drop served
    abandonment (omega); verification (nu); log metrics
```

---

## 11. Synthetic data generator (seeded)

Districts, their states, populations and the road-distance matrix come from the input file (Section 2.1). Everything below is synthetic and scaled to the default item (face shields, prices in ₹).

### 11.1 Suppliers

| Item | Default |
|---|---|
| Count | $`N_S = 450`$ |
| Enrolment | 70% at round 1; the rest uniform over rounds 2–6 |
| District | uniform over $`K`$ (or weighted by industrial activity if available) |
| Class S / M / L | 0.5 / 0.3 / 0.2 |
| Daily rate $`s_i`$ (integer uniform) | S [25, 100], M [60, 250], L [200, 800] |
| Minimum order $`m_i`$ | S 50, M 100, L 200 (so $`m_i \le 2 s_i`$: every supplier can meet its own minimum in a 2-day round) |
| Price $`\pi_i`$ | $`U[20,150]`$ × {1.0, 0.9, 0.8} for S / M / L |
| Certified probability | 0.3 / 0.5 / 0.8 for S / M / L |
| Variants (assembled only / kit only / both) | 0.57 / 0.10 / 0.33 (paper's supplier counts) |

### 11.2 Requesters

| Item | Default |
|---|---|
| Arrivals at round $`\tau`$ | Poisson($`\lambda_\tau`$), $`\lambda_\tau = 120 \theta^{\tau-1}`$, $`\theta = 0.7`$ (surge, then decline) |
| District | weighted by the `population` column if present, else uniform |
| Urgency level $`\ell_j`$ = 1 / 2 / 3 | 0.3 / 0.4 / 0.3 |
| True need $`n_j`$ | round(lognormal($`\ln 400`$, 1.4)), clipped to [50, 100000]; mean ≈ 1065, the paper's average request |
| Size | $`\text{size}_j = \max\big(1, \lceil n_j g_j / (u T^{cov}) \rceil\big)`$, $`g_j \sim \text{lognormal}(\ln 2, 0.5)`$ |
| Request $`D^0_j`$ | $`n_j`$ (normal requester) |
| Max price $`\bar\pi_j`$ | $`U[30,120] + 20(\ell_j - 1)`$ |
| $`\kappa_j`$ = 0 / 1 / 2 | $`\ell=1`$: 0.5 / 0.3 / 0.2; $`\ell=2`$: 0.3 / 0.4 / 0.3; $`\ell=3`$: 0.2 / 0.3 / 0.5 |
| Acceptable variants (assembled / kit / either) | 0.40 / 0.33 / 0.27 |
| Bulk requester | each arrival is bulk with probability $`P_{bulk} = 0.005`$; $`D^0_j = \text{round}(U[0.5,1.5] \times \bar A_1)`$, exempt from the 100000 clip; $`\text{size}_j`$ drawn from $`n_j`$ as for a normal requester |

Here $`\bar A_1 = 0.7 N_S \cdot 2 \cdot \mathbb{E}[s_i]`$ is the expected round-1 supply, about 112k units.

**Calibration check.**
- $`\mathbb{E}[s_i] = 0.5(62.5) + 0.3(155) + 0.2(500) = 177.75`$ units per day.
- Expected round-1 supply is about $`315 \times 2 \times 177.75 \approx 112\text{k}`$. Expected round-1 demand is about $`120 \times 1065 \approx 128\text{k}`$, a ratio of about 0.88, so round 1 is mildly scarce.
- Supply grows (enrolment plus carried stock) while demand declines, so later rounds become abundant.
- Because $`g_j < 1`$ with probability about 8%, about 8% of normal requests over-ask and get flagged. Bulk requests are almost always flagged, since their size is unrelated to their request.

---

## 12. Experiments

Use the same seeds across settings and at least 20 seeds per setting.

**Ablation ladder.** Start from the Shield-Net configuration (Section 7.6) and add one feature at a time:
1. ψ objective with fixed prior
2. ψ learning
3. urgency weights (1, 10, 100)
4. price and certificate
5. minimum order (Mode B)
6. inventory ($`\eta, h`$)
7. hoarding control
8. abandonment

Run with `simulate_failures = True` throughout so every step is measured on **realised** deliveries.

**Value of learning.** Set the true $`\psi^{\ast}_{cross} \in \lbrace 0.08, 0.16, 0.25\rbrace`$ and $`\psi^{\ast}_{same} \in \lbrace 0.30, 0.40, 0.55\rbrace`$, so the prior is right or wrong in either direction. Compare `learn_psi` on vs off on realised fill, and track how fast $`\hat\psi`$ converges.

**Sensitivity sweeps:**
- supply scale × {0.6, 1.0, 1.5} on $`s_i`$
- $`P_{bulk} \in \lbrace 0, 0.005, 0.02\rbrace`$
- $`c \in \lbrace 0.02, 0.05, 0.10\rbrace`$
- $`\nu \in \lbrace 0.1, 0.25, 0.5\rbrace`$
- $`\eta \in \lbrace 0, 0.5, 1\rbrace`$
- $`\omega \in \lbrace 0, 0.1, 0.2\rbrace`$
- $`W \in \lbrace (1,10,100), (1,2,3)\rbrace`$
- Mode A vs Mode B

**Headline comparisons:**
- realised urgency-weighted fill
- fill of non-flagged level-3 requesters with and without the cap
- same-state share
- stranded and abandoned demand
- the cost of minimum orders: Mode A fill minus Mode B fill

---

## 13. Assumptions and limitations

1. **One item, pooled capacity.** A supplier's daily rate is shared across variants; real lines produce kits and assembled units at different rates.
2. **Declared data is truthful and fixed.** Capacity, price, certification and size don't change after enrolment, and suppliers don't reprice.
3. **Districts are points.** All locations in a district are represented by one point. Distance within a district is a random $`U[10,20]`$ km draw, not measured.
4. **No transport model.** There are no lead times, vehicle capacity or delivery scheduling. The price is a delivered price; who pays for shipping is outside the model.
5. **Learning is coarse.**
   - The ψ prior comes from US data.
   - It is learned for only two classes (same / cross state), so distance within a state is ignored.
   - Learning sees only the matches the model chose to make. If few cross-state matches happen, $`\hat\psi_{cross}`$ updates slowly (selection bias).
6. **Match outcomes are all-or-nothing** and become known before the next round.
7. **Verification and certification happen outside the model.** They are represented only by $`\nu`$ and $`q_i`$. Fraud and rejection after verification are not modelled.
8. **Plausibility uses a single usage rate $`u`$** for all facility types.
9. **Behavioural parameters are stationary.** $`\eta, \omega, \nu, u`$ are constant over the horizon. Panic, media coverage and government action would make them change over time; we sweep them instead of modelling that.
10. **Urgency weights are a value judgement.** Under $`W = (1,10,100)`$, urgency dominates locality across levels.
11. **The two-week horizon truncates demand.** Demand still outstanding after round 6 is reported, not resolved.
12. **The synthetic data is not validated** against real Indian demand or supplier data. Only the request-size mean and the supplier variant mix are anchored to the paper.

---

## 14. Implementation notes (PuLP)

- **Loading the distance file:** read both sheets with `pandas.read_excel(path, sheet_name=..., engine="openpyxl")`, run the checks in Section 2.1, and store $`R`$ as a NumPy float array with `np.inf` for unreachable pairs. Reading an 800 × 800 sheet takes several seconds, so cache the result as a `.npz` file next to the workbook and reuse it while the workbook's modification time is unchanged. Each supplier and requester stores its district as an integer row index into $`R`$, so $`R[k(i),k(j)]`$ is a direct array lookup.
- **Eligibility:** build $`a_{ij}`$ with vectorised NumPy boolean operations over (enrolled suppliers × active requesters), then create variables only on edges.
- **Model:** `prob = pulp.LpProblem("pulse_round", pulp.LpMaximize)`.
  - Mode A: `LpVariable(..., lowBound=0, upBound=xbar_ij, cat="Continuous")`.
  - Mode B: `cat="Integer"` for $`x`$, and `cat="Binary"` for $`y`$ and $`w`$.
  - Fix $`w_j = 0`$ for capped requesters by setting `upBound=0`.
- **Fast model building:** with ~10⁴–10⁵ edges, build the objective and the row sums with `pulp.LpAffineExpression([(var, coef), ...])` from prebuilt lists. Avoid nested `lpSum` over generators, which is slow at this size. Group edges by supplier and by requester once per round, and reuse the groups for (1), (2), (4), (5).
- **Solver:** `pulp.PULP_CBC_CMD(msg=False, timeLimit=120)` for Mode B; there is no limit needed for Mode A.
  - Check `pulp.LpStatus[prob.status]`, and use `prob.sol_status` to tell a proven optimum (1) from a time-limited feasible solution (2).
  - For the gap bound in 7.5, solve a copy of the Mode B model with every variable relaxed to continuous.
  - `pulp.HiGHS_CMD` can replace CBC if HiGHS is installed; the model is unchanged.
- **Mode B size:** about 10⁴ binaries per round after filtering. Size grows with suppliers × requesters, not with the number of districts. If a round hits the time limit, report the gap. As a last resort, prune each requester's candidate edges to the top-$`k`$ by $`w_{ij}`$ (e.g. $`k = 30`$) or set $`\Delta_{max}`$. Report either as a heuristic.
- **Integers everywhere:** $`s_i`$, $`m_i`$, $`A_{i,\tau}`$, $`D_{j,\tau}`$, $`\tilde D_j`$, cap and $`r_{i,\tau}`$. Read $`x`$ via `var.varValue` (None means unsolved) and round it after the integrality assert.
- **Reproducibility:** use one master seed and derive separate `numpy.random.Generator` streams for supplier generation, requester generation, match outcomes, abandonment and verification. That way changing a policy parameter doesn't change the generated instance or the outcome draws.
- **Tests:**
  - Properties 7.1 (Mode A, no edge left with stock and demand on both ends), 7.3 (integrality), 7.4 (Mode B carry invariant).
  - Condition (C1) at start-up, and (C2) every round.
  - Cap: $`\sum_i x_{ij} \le \text{cap}_\tau`$ for flagged $`j`$.
  - Minimum order: $`x_{ij} = 0`$ or $`x_{ij} \ge m_i`$ in Mode B.
  - Conservation: supplier and requester balances before and after each round.
  - Bound: $`Z^{found}_B \le Z^{LP}_B \le Z^{\ast}_A`$.
  - Loader: the placeholder file loads with no errors; every diagonal value is in $`[10,20]`$; shuffling the file's row and column order gives the same $`R`$; a pair blanked out in the file never appears as an edge.
  - Learning: with $`\psi^{\ast} = \psi^0`$ and many rounds, $`\hat\psi`$ stays near the prior; with $`\psi^{\ast}`$ set very differently, $`\hat\psi`$ moves toward $`\psi^{\ast}`$.