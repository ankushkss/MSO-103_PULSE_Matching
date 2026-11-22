# PULSE: what changed on `novelty-AC` relative to `main`

This document explains everything the `novelty-AC` branch adds to the original course submission on `main`. The changes come in two layers:

| Layer | Branch where it lives | What it adds |
|---|---|---|
| **1. Evaluation** | `benchmarks` (and therefore also `novelty-AC`) | paired statistics, a fair Shield-Net baseline, benchmark policies and upper bounds, a real-data oxygen case study |
| **2. Novelty** | `novelty-AC` only | **Idea A**: misreporting, audits, fairness; **Idea C**: supplier reliability learning; **overbooking**, which links the two |

```
main ── benchmarks (layer 1) ── novelty-AC (layer 2)
```

With every new switch at its default, `novelty-AC` reproduces `main`'s model and the `benchmarks` results **exactly** (checked to the last decimal), so every difference reported below comes from a feature being switched on.

Section 1 gives the complete formal models (Shield-Net, PULSE and PULSE-AC). Section 2 explains each change in words. Section 3 describes the simulation experiments, Section 4 the evaluation metrics and statistics, and Section 5 the results. Section 6 lists limitations and open questions, and Section 7 says how to reproduce everything.

---

## 1. Formal models: Shield-Net, PULSE and PULSE-AC

Three models, each a generalisation of the previous one:

| Model | Where | Decides |
|---|---|---|
| **(M0) Shield-Net** | Alcock, Boutilier & Siddiq (2022) | one LP per matching round |
| **(M1) PULSE** | `main` | one LP per round $`\tau`$, plus state updates between rounds |
| **(M2) PULSE-AC** | `novelty-AC` | M1 with declared vs true need, audits, a fairness weight, per-supplier reliability and overbooking |

Each model is written as the optimisation problem the **platform** solves. M1 and M2 also give the **between-round dynamics**: the world's random outcomes and the platform's learning. Section 1.4 states exactly when each model reduces to the previous one.

### 1.1 (M0) Shield-Net

**Sets.** $`I`$ suppliers; $`J`$ requesters.

**Parameters.**

| Symbol | Meaning |
|---|---|
| $`s_i`$ | capacity of supplier $`i`$ in the round |
| $`d_j`$ | units requested by $`j`$ |
| $`p_j \in \{1,2,3\}`$ | self-reported urgency |
| $`c_{ij}`$ | straight-line km between the centroids of the two (US) states |
| $`\alpha = 10^{-6}`$ | distance weight |
| $`a_{ij} \in \{0,1\}`$ | 1 if the product type (kit or assembled) matches |

**Decision variables.** $`x_{ij} \ge 0`$: units from $`i`$ to $`j`$. $`t_j \ge 0`$: unmet demand of $`j`$.

```math
\begin{aligned}
\textbf{(M0)}\qquad \min_{x,\,t}\quad & \sum_{j\in J} p_j\,t_j \;+\; \alpha\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} && \text{(0a)}\\
\text{s.t.}\quad & \sum_{j\in J} x_{ij} \le s_i && \forall i \in I \quad\text{(0b)}\\
& \sum_{i\in I} x_{ij} \le d_j && \forall j \in J \quad\text{(0c)}\\
& x_{ij} \le d_j\,a_{ij} && \forall i, j \quad\text{(0d)}\\
& t_j \ge d_j - \sum_{i\in I} x_{ij} && \forall j \in J \quad\text{(0e)}\\
& x_{ij} \ge 0,\quad t_j \ge 0 && \forall i, j \quad\text{(0f)}
\end{aligned}
```

At an optimum $`t_j = d_j - \sum_i x_{ij}`$, so (M0) is equivalent to

```math
\max_{x \ge 0}\ \sum_{(i,j):\,a_{ij}=1} \big(p_j - \alpha\,c_{ij}\big)\,x_{ij}
\quad\text{s.t. (0b), (0c)}. \tag{0'}
```

Each round is solved on the current lists; nothing is carried between rounds inside the model.

### 1.2 (M1) PULSE

#### Sets and indices

| Symbol | Meaning |
|---|---|
| $`\tau \in \mathcal{T} = \{1,\dots,6\}`$ | rounds, on days $`d_\tau \in \{1,3,5,8,10,12\}`$; gap $`\Delta_\tau = d_\tau - d_{\tau-1}`$, $`d_0 = -1`$ |
| $`K`$, $`\sigma(k)`$ | districts; state of district $`k`$ |
| $`I_\tau = \{i : e_i \le \tau\}`$ | suppliers enrolled by round $`\tau`$ ($`e_i`$ = enrolment round) |
| $`J_\tau`$ | requesters active in round $`\tau`$ (arrived, not yet served or abandoned) |
| $`\mathcal{G} = \{\text{same}, \text{cross}\}`$, $`g(i,j)`$ | pair class: same state if $`\sigma(k(i)) = \sigma(k(j))`$ |
| $`\ell \in \{1,2,3\}`$ | urgency level from facility type |

#### Parameters

| Symbol | Meaning |
|---|---|
| $`s_i,\ \pi_i,\ q_i,\ V_i`$ | supplier daily rate, unit price, certified flag, variants offered |
| $`\tau_j,\ D^0_j,\ \ell_j,\ \bar\pi_j,\ \kappa_j,\ V_j,\ E_j`$ | requester arrival round, request, urgency level, max price, certificate need (0 none / 1 prefer / 2 require), acceptable variants, staff count |
| $`R[k,l],\ R_{max}`$ | road distance (km) from district $`k`$ to $`l`$; largest finite entry |
| $`W = (W_1,W_2,W_3) = (1,10,100)`$ | urgency weights |
| $`u_\ell`$, $`H`$ | plausible usage per staff per day; days of need counted as core (14) |
| $`\varepsilon = 10^{-4}`$ | value of an excess unit relative to a core unit |
| $`\tilde\alpha,\ \beta,\ \mu`$ | tie-break weights for distance, price, uncertified stock |
| $`\psi^0_g,\ n_0,\ \psi_{floor}`$ | prior success probability, prior strength, lower clip |
| $`\eta,\ h,\ \omega`$ | **world:** share of unsold stock kept, storage cap (days of production), abandonment probability |
| $`\psi^{\ast}_g`$ | **world:** true success probability (unknown to the platform) |

#### Derived quantities in round $`\tau`$ (state from round $`\tau-1`$)

```math
\begin{aligned}
A_{i\tau} &= s_i\,\Delta_\tau + r_{i,\tau-1} && \text{stock available} \quad\text{(1a)}\\
D_{j\tau} &= \mathrm{carry}_{j,\tau-1} + D^0_j\,\mathbf{1}[\tau = \tau_j] && \text{outstanding request} \quad\text{(1b)}\\
N_j &= u_{\ell_j}\,E_j\,H && \text{plausible need} \quad\text{(1c)}\\
C_{j\tau} &= \min\!\big(D_{j\tau},\ \lfloor N_j - G_{j,\tau-1}\rfloor^{+}\big),\qquad X_{j\tau} = D_{j\tau} - C_{j\tau} && \text{core, excess} \quad\text{(1d)}\\
\hat\psi_{g\tau} &= \max\!\Big(\psi_{floor},\ \frac{n_0\psi^0_g + S_{g,\tau-1}}{n_0 + S_{g,\tau-1} + F_{g,\tau-1}}\Big) && \text{learned success rate} \quad\text{(1e)}\\
c_{ij} &= \tilde\alpha\,\frac{R[k(i),k(j)]}{R_{max}} + \beta\,\frac{\pi_i}{\bar\pi_j} + \mu\,\mathbf{1}[\kappa_j = 1]\,(1-q_i) && \text{tie-break cost} \quad\text{(1f)}\\
w_{ij\tau} &= W_{\ell_j}\,\hat\psi_{g(i,j),\tau} - c_{ij} && \text{value of a core unit} \quad\text{(1g)}
\end{aligned}
```

**Eligibility.** Variables exist only on

```math
E_\tau = \Big\{(i,j) \in I_\tau\times J_\tau:\ V_i\cap V_j \ne \emptyset,\ \ \pi_i \le \bar\pi_j,\ \ q_i \ge \mathbf{1}[\kappa_j = 2],\ \ R[k(i),k(j)] < \infty,\ \ A_{i\tau} > 0\Big\}. \tag{1h}
```

**Decision variables.** $`x^c_{ij} \ge 0`$, units towards $`j`$'s core; $`x^e_{ij} \ge 0`$, units towards its excess; $`x_{ij} = x^c_{ij} + x^e_{ij}`$.

#### Round problem

```math
\begin{aligned}
\textbf{(M1)}\qquad \max_{x^c,\,x^e}\quad & \sum_{(i,j)\in E_\tau} w_{ij\tau}\,\big(x^c_{ij} + \varepsilon\,x^e_{ij}\big) && \text{(1i)}\\
\text{s.t.}\quad & \sum_{j:(i,j)\in E_\tau} \big(x^c_{ij} + x^e_{ij}\big) \le A_{i\tau} && \forall i \in I_\tau \quad\text{(1j)}\\
& \sum_{i:(i,j)\in E_\tau} x^c_{ij} \le C_{j\tau} && \forall j \in J_\tau \quad\text{(1k)}\\
& \sum_{i:(i,j)\in E_\tau} x^e_{ij} \le X_{j\tau} && \forall j \in J_\tau \quad\text{(1l)}\\
& x^c_{ij},\ x^e_{ij} \ge 0 && \forall (i,j) \in E_\tau \quad\text{(1m)}
\end{aligned}
```

**Conditions on the parameters** (checked by the code):

```math
\begin{aligned}
&\text{(C1)}\quad \tilde\alpha + \beta + \mu < W_1\,\psi_{floor} && \text{every unit has positive value}\\
&\text{(C2)}\quad W_{\ell+1}\,\hat\psi_{cross} - (\tilde\alpha+\beta+\mu) > W_\ell\,\hat\psi_{same},\ \ \ell = 1,2 && \text{urgency beats locality}\\
&\text{(C3)}\quad \varepsilon\,W_3 < W_1\,\psi_{floor} - (\tilde\alpha+\beta+\mu) && \text{any core unit beats any excess unit}
\end{aligned}
```

#### Between rounds

Let $`x^{\ast}`$ be the optimal solution. For every shipment $`x^{\ast}_{ij} > 0`$:

```math
\begin{aligned}
B_{ij\tau} &\sim \mathrm{Bernoulli}\big(\psi^{\ast}_{g(i,j)}\big)\ \ \text{(world)},\qquad B_{ij\tau} = 0 \text{ if } \pi_i > \bar\pi_j \text{ or } (\kappa_j = 2 \wedge q_i = 0) && \text{(1n)}\\
\delta_{ij\tau} &= x^{\ast}_{ij}\,B_{ij\tau} && \text{delivered} \quad\text{(1o)}\\
S_{g\tau} &= S_{g,\tau-1} + \#\{(i,j): x^{\ast}_{ij}>0,\ g(i,j)=g,\ B_{ij\tau}=1\},\quad F_{g\tau}\ \text{likewise for } B = 0 && \text{learning} \quad\text{(1p)}\\
r_{i\tau} &= \min\!\Big(\big\lfloor \eta\,\big(A_{i\tau} - \textstyle\sum_j \delta_{ij\tau}\big)\big\rfloor,\ h\,s_i\Big) && \text{stock carried} \quad\text{(1q)}\\
G_{j\tau} &= G_{j,\tau-1} + \textstyle\sum_i \delta_{ij\tau},\qquad \mathrm{carry}_{j\tau} = D_{j\tau} - \textstyle\sum_i \delta_{ij\tau} && \text{requester state} \quad\text{(1r)}
\end{aligned}
```

Requester $`j`$ leaves $`J_{\tau+1}`$ when $`\mathrm{carry}_{j\tau} = 0`$ (served), or with probability $`\omega`$ if not served (abandons). Initial state: $`r_{i0} = \mathrm{carry}_{j0} = G_{j0} = S_{g0} = F_{g0} = 0`$.

### 1.3 (M2) PULSE-AC

M2 keeps the sets, eligibility (1h), costs (1f) and the stock and request recursions of M1, and changes the items below.

#### Additional parameters

| Symbol | Kind | Meaning |
|---|---|---|
| $`\mathcal{M} \subseteq J`$, $`\varphi \ge 1`$ | world | misreporting cohort (10% of non-bulk requesters); inflation factor |
| $`\theta_i \sim \mathcal{N}(0,\sigma^2)`$ | world | supplier $`i`$'s effect on the log-odds of success |
| $`q_a \in [0,1]`$, $`\mathcal{R}`$ | policy | audit probability; audit rule |
| $`\lambda \ge 0`$ | policy | fairness weight |
| $`\beta_o \ge 0`$ | policy | overbooking factor |
| $`m \in \{\text{class}, \text{supplier}, \text{eb}, \text{thompson}, \text{oracle}\}`$, $`n_s`$, $`\psi_{cap}`$ | policy | learning mode; supplier prior strength; upper clip |

#### World: declared vs true values, and true success probabilities

```math
\begin{aligned}
\hat D^0_j &= \begin{cases}\operatorname{round}(\varphi D^0_j) & j\in\mathcal{M}\\ D^0_j & j\notin\mathcal{M}\end{cases}
\qquad
\hat E_j = \begin{cases}\lceil\varphi E_j\rceil & j\in\mathcal{M}\\ E_j & j\notin\mathcal{M}\end{cases} && \text{declared} \quad\text{(2a)}\\
\psi^{\ast}_{ij} &= \operatorname{expit}\!\big(\operatorname{logit}\psi^{\ast}_{g(i,j)} + \theta_i\big) && \text{true success} \quad\text{(2b)}
\end{aligned}
```

The platform sees only $`(\hat D^0_j, \hat E_j)`$. (1b) uses $`\hat D^0_j`$. All scoring uses the true $`(D^0_j, E_j)`$.

#### Platform: need basis after audits

Each $`j`$ is audited with probability $`q_a`$. It is caught if audited and $`\hat E_j > E_j`$.

```math
\hat N_j = \begin{cases}
u_{\ell_j}\hat E_j H & \text{not caught}\\
N_j & \text{caught},\ \mathcal{R} = \texttt{truth}\\
f\,N_j & \text{caught},\ \mathcal{R} = \texttt{fraction=}f\\
0 & \text{caught},\ \mathcal{R} = \texttt{zero}
\end{cases}
\qquad
C_{j\tau} = \min\!\big(D_{j\tau},\ \lfloor \hat N_j - G_{j,\tau-1}\rfloor^{+}\big),\quad X_{j\tau} = D_{j\tau} - C_{j\tau} \tag{2c}
```

#### Platform: per-supplier success estimate

With $`S_i`$ the supplier's successful matches and $`\mathcal{E}_i`$ the successes the class estimate predicted for them (both up to round $`\tau-1`$):

```math
\begin{aligned}
\hat\rho_{i\tau} &= \begin{cases}
1 & m = \text{class}\\[2pt]
\dfrac{n + S_i}{n + \mathcal{E}_i},\ \ n = n_s & m = \text{supplier}\\[6pt]
\dfrac{n + S_i}{n + \mathcal{E}_i},\ \ n = \operatorname{clip}\!\big(1/\max(\hat\tau^2,10^{-3}),\,1,\,1000\big) & m = \text{eb}\\[6pt]
\text{a draw from } \mathrm{Gamma}\big(\text{shape } n_s + S_i,\ \text{rate } n_s + \mathcal{E}_i\big) & m = \text{thompson}
\end{cases} && \text{(2d)}\\[4pt]
\hat\tau^2 &= \operatorname{Var}_i\!\Big(\frac{S_i}{\mathcal{E}_i}\Big) - \operatorname{mean}_i\!\Big(\frac{\bar r}{\mathcal{E}_i}\Big), \qquad \bar r = \operatorname{mean}_i \frac{S_i}{\mathcal{E}_i}\quad (\text{over } i \text{ with } \mathcal{E}_i \ge 0.5) && \text{(2e)}\\[4pt]
\hat\psi_{ij\tau} &= \begin{cases}\operatorname{clip}\!\big(\psi^{\ast}_{ij},\ \psi_{floor},\ \psi_{cap}\big) & m = \text{oracle}\\ \operatorname{clip}\!\big(\hat\psi_{g(i,j),\tau}\ \hat\rho_{i\tau},\ \psi_{floor},\ \psi_{cap}\big) & \text{otherwise}\end{cases} && \text{(2f)}
\end{aligned}
```

#### Platform: values with the fairness weight

With $`f_{j\tau} = \min\big(1,\ G_{j,\tau-1}/\hat N_j\big)`$ (1 if $`\hat N_j = 0`$), the share of declared need already received:

```math
w^{c}_{ij\tau} = W_{\ell_j}\,\hat\psi_{ij\tau}\,\big(1 + \lambda(1 - f_{j\tau})\big) - c_{ij},
\qquad
w^{e}_{ij\tau} = W_{\ell_j}\,\hat\psi_{ij\tau} - c_{ij} \tag{2g}
```

#### Round problem

```math
\begin{aligned}
\textbf{(M2)}\qquad \max_{x^c,\,x^e}\quad & \sum_{(i,j)\in E_\tau} \Big( w^{c}_{ij\tau}\,x^c_{ij} + \varepsilon\,w^{e}_{ij\tau}\,x^e_{ij} \Big) && \text{(2h)}\\
\text{s.t.}\quad & \sum_{j:(i,j)\in E_\tau} \big(x^c_{ij} + x^e_{ij}\big) \le A_{i\tau} && \forall i \in I_\tau \quad\text{(2i)}\\
& \sum_{i:(i,j)\in E_\tau} x^c_{ij} \le \big\lfloor (1+\beta_o)\,C_{j\tau} \big\rfloor && \forall j \in J_\tau \quad\text{(2j)}\\
& \sum_{i:(i,j)\in E_\tau} x^e_{ij} \le \big\lfloor (1+\beta_o)\,X_{j\tau} \big\rfloor && \forall j \in J_\tau \quad\text{(2k)}\\
& x^c_{ij},\ x^e_{ij} \ge 0 && \forall (i,j) \in E_\tau \quad\text{(2l)}
\end{aligned}
```

#### Between rounds

Outcomes now use each pair's true probability, and a requester accepts at most what it still needs. Order the suppliers shipping to $`j`$ by distance, with $`k \prec i`$ meaning "$`k`$ is nearer to $`j`$ than $`i`$":

```math
\begin{aligned}
B_{ij\tau} &\sim \mathrm{Bernoulli}\big(\psi^{\ast}_{ij}\big),\ \text{refusals as in (1n)}, \qquad \delta_{ij\tau} = x^{\ast}_{ij}B_{ij\tau} && \text{(2m)}\\
\delta^{acc}_{ij\tau} &= \min\!\Big(\delta_{ij\tau},\ \big[\,D_{j\tau} - \textstyle\sum_{k\prec i}\delta^{acc}_{kj\tau}\big]^{+}\Big) && \text{acceptance} \quad\text{(2n)}\\
r_{i\tau} &= \min\!\Big(\big\lfloor \eta\,\big(A_{i\tau} - \textstyle\sum_j \delta^{acc}_{ij\tau}\big)\big\rfloor,\ h\,s_i\Big) && \text{refused surplus stays with } i \quad\text{(2o)}\\
G_{j\tau} &= G_{j,\tau-1} + \textstyle\sum_i \delta^{acc}_{ij\tau},\qquad \mathrm{carry}_{j\tau} = D_{j\tau} - \textstyle\sum_i \delta^{acc}_{ij\tau} && \text{(2p)}\\
S_i &\mathrel{+}= \textstyle\sum_j \mathbf{1}[x^{\ast}_{ij}>0]\,B_{ij\tau},\qquad \mathcal{E}_i \mathrel{+}= \textstyle\sum_j \mathbf{1}[x^{\ast}_{ij}>0]\,\hat\psi_{g(i,j),\tau} && \text{supplier learning} \quad\text{(2q)}
\end{aligned}
```

$`S_g, F_g`$ are updated as in (1p), and abandonment is unchanged. The waste caused by overbooking is $`\sum_{ij}(\delta_{ij\tau} - \delta^{acc}_{ij\tau})`$.

### 1.4 How the models nest, and what they share

**PULSE-AC reduces to PULSE** when the world is honest with identical suppliers and every new policy lever is off:

```math
\varphi = 1,\quad \sigma = 0,\quad q_a = 0,\quad \lambda = 0,\quad \beta_o = 0,\quad m = \text{class}
\ \ \Longrightarrow\ \
\hat N_j = N_j,\ \hat\rho_i = 1,\ w^c = w^e = w,\ \delta^{acc} = \delta
\ \ \Longrightarrow\ \ \text{(M2)} \equiv \text{(M1)}.
```

**PULSE reduces to Shield-Net** with $`\hat\psi \equiv 1`$ (no learning), $`W = (1,2,3)`$, $`\beta = \mu = 0`$, $`\tilde\alpha = 10^{-6}R_{max}`$, the whole request treated as core ($`C_{j\tau} = D_{j\tau}`$, $`X_{j\tau} = 0`$), and only product-type eligibility. Then (M1) is exactly (0') on (M0)'s data. This was checked numerically: same edge set and the same optimal value.

**Common structure.** In every model each variable appears in exactly one supply row and one demand row (core or excess), so the constraint matrix is the incidence matrix of a bipartite graph and is **totally unimodular**. Every right-hand side is an integer: $`s_i,\Delta_\tau`$ are integers, $`r`$, $`C`$, $`X`$ are floored, and so is $`(1+\beta_o)C`$. So every vertex is integral for any objective, including the fairness-weighted and per-supplier values. **Each round of M2 is still one exact LP with whole-unit shipments.** Because the fairness boost is $`\ge 1`$ and applies to core only, and $`\hat\psi_{ij} \ge \psi_{floor}`$, conditions (C1) and (C3) hold in M2 exactly as in M1.

---

## 2. Changes to the mathematical formulation, explained

Section 1 gives the complete models. This section explains each change in words, with the reasoning behind it.

### 2.1 Layer 1: evaluation tools (no change to PULSE itself)

#### (a) Fair baseline: Shield-Net + filters
Shield-Net's LP, with PULSE's hard eligibility rules added (price $`\pi_i \le \bar\pi_j`$, certification if required). This separates the gain from "never propose a match the requester will refuse" from the gain from PULSE's objective.

#### (b) Benchmark policies
Three simple rules replace only the LP. They see the same edges $`E`$, stock $`A_i`$ and rows $`C_j, X_j`$, so they differ from PULSE only in how stock is divided:

| Rule | Allocation |
|---|---|
| Proportional rationing | every requester first gets the same share $`\min(1, \sum_i A_i / \sum_j C_j)`$ of its core, then leftovers are handed out greedily |
| FCFS | requesters in arrival order, each served by its nearest suppliers |
| Random | random requester order, random supplier order |

All three serve all core before any excess, so the model's invariants still hold.

#### (c) Clairvoyant upper bounds
One LP over all rounds $`t = 1..T`$ with shipments $`x_{tij}`$, carried stock $`r_{it}`$ and credited need $`y_j`$. Let $`\pi_{tij}`$ be the delivered fraction of a shipped unit.

- **H2, perfect hindsight**: knows every coin flip. $`\pi_{tij} = 1`$, and a pair exists only if its pre-drawn coin flip succeeds.
- **H1, clairvoyant fluid bound**: knows future arrivals but only the success *probabilities*. $`\pi_{tij} = \psi^{\ast}_{ij}`$ on every eligible pair.

```math
\begin{aligned}
\max\quad & \textstyle\sum_j \omega_j\, y_j\\
\text{s.t.}\quad & \textstyle\sum_j x_{tij} \le s_i\Delta_t + r_{i,t-1} && \text{supply}\\
& r_{it} \le \eta\big(s_i\Delta_t + r_{i,t-1} - \textstyle\sum_j \pi_{tij}x_{tij}\big),\quad r_{it} \le h\,s_i && \text{carry-over}\\
& \textstyle\sum_i x_{tij} + \sum_{t'<t}\sum_i \pi_{t'ij}x_{t'ij} \le D^0_j && \text{ship at most what is outstanding}\\
& y_j \le \textstyle\sum_{t,i}\pi_{tij}x_{tij},\qquad 0 \le y_j \le \min(D^0_j,\ N_j) && \text{credited need}
\end{aligned}
```

Requester $`j`$ is available from its arrival until its abandonment draw fires. Both bounds relax integrality and the floor in the carry-over, so they remain valid upper bounds. Without the "ship at most what is outstanding" row, the fluid LP **overbooks**, which motivated Section 2.2(e).

### 2.2 Layer 2: the novelty (`novelty-AC` only)

#### (a) Declared vs true need: modelling misreporting (Idea A)

The world now has a **true** request and staff count $`(D^{0}_j, E_j)`$, and a **declared** pair $`(\hat D^0_j, \hat E_j)`$. The platform sees only the declared values. A fixed cohort $`\mathcal{M}`$ (10% of non-bulk requesters, chosen from its own random stream) inflates both by a factor $`\varphi \ge 1`$:

```math
\hat D^0_j = \begin{cases}\operatorname{round}(\varphi\, D^0_j) & j \in \mathcal{M}\\ D^0_j & \text{otherwise}\end{cases}
\qquad
\hat E_j = \begin{cases}\lceil \varphi\, E_j\rceil & j \in \mathcal{M}\\ E_j & \text{otherwise}\end{cases}
```

The **platform's core** uses the declared staff count, $`\hat N_j = u_{\ell_j}\hat E_j H`$, so $`C_j = \min(\hat D_j, \lfloor \hat N_j - G_j\rfloor^+)`$. **Scoring** always uses the truth: genuine need $`= \min(D^0_j,\ u_{\ell_j}E_j\cdot 14)`$, so lying can never raise its own yardstick.

Both values are inflated because honest requesters in this world usually ask for *less* than their genuine need (their core already equals their request). Inflating the staff count alone would gain nothing.

#### (b) Audits (Idea A)

Each requester is audited with probability $`q`$ (pre-drawn uniform $`U_j`$). A requester is **caught** if $`U_j < q`$ and $`\hat E_j > E_j`$. A caught requester's core basis is replaced before round 1:

```math
\hat N_j \;\leftarrow\; \begin{cases} N_j & \texttt{truth}: \text{reset to the true need}\\ f\cdot N_j & \texttt{fraction=}f: \text{proportional penalty}\\ 0 & \texttt{zero}: \text{no core, everything waits as excess}\end{cases}
```

#### (c) Fairness boost (Idea A)

Core units of requesters that have received little of their need so far are worth more. With $`f_j = \min(1, G_j / \hat N_j)`$ the share of (declared) need received:

```math
w^{c}_{ij} \;=\; p_j\,\hat\psi_{ij}\,\big(1 + \lambda\,(1 - f_j)\big) - c_{ij},
\qquad
w^{e}_{ij} \;=\; p_j\,\hat\psi_{ij} - c_{ij}
```

The objective becomes $`\sum w^c_{ij}x^c_{ij} + \varepsilon\sum w^e_{ij}x^e_{ij}`$. The boost is $`\ge 1`$ and applies to core only, so conditions (C1) "every unit has positive value" and (C3) "any core unit beats any excess unit" still hold, and the matrix is unchanged (still totally unimodular). $`\lambda = 0`$ is PULSE.

#### (d) Supplier reliability (Idea C)

**World.** Suppliers differ in how reliably they deliver. Each supplier has a hidden effect $`\theta_i \sim \mathcal{N}(0, \sigma^2)`$ on the log-odds of success:

```math
\operatorname{logit}\psi^{\ast}_{ij} \;=\; \operatorname{logit}\psi^{\ast}_{g(i,j)} + \theta_i
```

$`\sigma = 0`$ recovers the original world exactly.

**Platform.** The estimate becomes per pair, $`\hat\psi_{ij} = \operatorname{clip}\big(\hat\psi_{g(i,j)}\cdot \hat\rho_i,\ \psi_{floor},\ 0.95\big)`$, where $`\hat\rho_i`$ is the supplier's estimated **reliability ratio**. For every match a supplier makes, the platform records whether it succeeded ($`S_i`$ counts successes) and how many successes the class estimate predicted ($`\mathcal{E}_i = \sum \hat\psi_g`$ at the time of each match). Five learning modes:

| Mode | $`\hat\rho_i`$ | Idea |
|---|---|---|
| `class` | $`1`$ | original PULSE: two numbers for everyone |
| `supplier` | $`\dfrac{n_0 + S_i}{n_0 + \mathcal{E}_i}`$, fixed $`n_0 = 5`$ | Gamma–Poisson posterior mean, shrunk towards the class value |
| **`supplier_eb`** | same, with $`n_0 = 1/\hat\tau^2`$ | **empirical Bayes**: the prior strength is estimated from the data |
| `thompson` | one draw from $`\text{Gamma}(n_0 + S_i,\ \text{rate}\ n_0 + \mathcal{E}_i)`$ per round | exploration (Thompson sampling) |
| `oracle` | the true $`\psi^{\ast}_{ij}`$ is used directly | reference for regret |

**Empirical-Bayes prior.** Across suppliers, the observed ratio $`S_i/\mathcal{E}_i`$ varies for two reasons: true differences ($`\tau^2`$) and chance (Poisson noise, about $`\bar r/\mathcal{E}_i`$). So

```math
\hat\tau^2 = \operatorname{Var}_i\!\left(\frac{S_i}{\mathcal{E}_i}\right) - \operatorname{mean}_i\!\left(\frac{\bar r}{\mathcal{E}_i}\right),
\qquad n_0 = \operatorname{clip}\!\left(\frac{1}{\max(\hat\tau^2, 10^{-3})},\ 1,\ 1000\right)
```

using suppliers with $`\mathcal{E}_i \ge 0.5`$ (at least 10 of them, otherwise the fixed $`n_0`$). If suppliers look alike, $`n_0`$ is large and every supplier is treated like its class; if they clearly differ, $`n_0`$ is small and each supplier's own record dominates.

#### (e) Overbooking (links A and C)

The platform may ship up to $`(1+\beta)`$ times what a requester still needs, expecting most shipments to fail. Only the right-hand sides of the demand rows change:

```math
\sum_i x^c_{ij} \;\le\; \big\lfloor (1+\beta)\,C_j \big\rfloor,
\qquad
\sum_i x^e_{ij} \;\le\; \big\lfloor (1+\beta)\,X_j \big\rfloor
```

They stay integers, so the LP stays totally unimodular. **World rule:** a requester accepts at most its outstanding demand $`D_j`$. If several shipments succeed, the nearest are accepted first and the surplus is refused and returns to the supplier, where it leaks like any unsold stock:

```math
\delta^{\text{acc}}_{ij} = \min\Big(\delta_{ij},\ \big[D_j - \textstyle\sum_{k \prec i}\delta^{\text{acc}}_{kj}\big]^+\Big),
\qquad \text{returned} = \textstyle\sum_{ij}(\delta_{ij} - \delta^{\text{acc}}_{ij})
```

where $`\delta_{ij} = x_{ij}B_{ij}`$ is what succeeded and $`k \prec i`$ means "supplier $`k`$ is nearer to $`j`$". $`\beta = 0`$ is PULSE.

### 2.3 Summary of formulation changes

| Change | Objective | Constraints | Right-hand sides | World (simulation) | Still TU? |
|---|---|---|---|---|---|
| Declared vs true need | — | — | $`C_j`$ from $`\hat E_j`$ | cohort inflates $`D^0, E`$ | yes |
| Audits | — | — | caught: $`\hat N_j \leftarrow`$ rule | audit draws | yes |
| Fairness boost | $`w^c`$ × $`(1+\lambda(1-f_j))`$ | — | — | — | yes |
| Supplier reliability | $`\hat\psi_{ij} = \hat\psi_g\hat\rho_i`$ | — | — | $`\psi^{\ast}_{ij}`$ via $`\theta_i`$ | yes |
| Overbooking | — | — | $`\times(1+\beta)`$ | acceptance cap, returns | yes |

Every change keeps PULSE a single, exactly-solvable transportation LP per round.

---

## 3. What the simulations test

### 3.1 Design
- **Common random numbers.** Match outcomes, abandonment, the misreporting cohort, audit draws, supplier effects $`\theta_i`$ and Thompson draws each come from their own pre-drawn random stream. Two configurations on the same seed therefore face the *same* world, and differ only in the platform's decisions.
- **30 paired seeds** per configuration, on the placeholder district data (800 districts).
- A test confirms that, with all new switches at their defaults, results are identical to `main`'s model.

### 3.2 Experiment suites

| Suite | Question | Configurations | Compared against |
|---|---|---|---|
| `ablation` | which PULSE feature matters? | Shield-Net → +filters → +ψ objective → +learning → +weights → +core/excess | Shield-Net; fair Shield-Net; previous step |
| `benchmarks` | how close to the best achievable? | PULSE, H1, H2, proportional, FCFS, random, Shield-Net + filters | PULSE; share of H1 and H2 |
| `sweeps`, `learning` | sensitivity; value of learning $`\hat\psi_g`$ | one parameter at a time; 9 true-ψ worlds × learning on/off | default; learning off |
| **`misreport`** | does lying pay? do audits stop it? | factor $`\varphi \in \{1,2,3,5\}`$; audits $`q \in \{0.1, 0.25, 0.5\}`$ × 3 rules at $`\varphi = 3`$ | honest world ($`\varphi = 1`$) |
| **`overbook`** | does platform hedging help, and does it remove the incentive to lie? | $`\beta \in \{0, 0.5, 1, 2, 3\}`$ × cohort honest / lying 3× | $`\beta = 0`$; honest cohort at the same $`\beta`$ |
| **`reliability`** | how much is knowing supplier reliability worth, and which learner gets it? | $`\sigma \in \{0, 0.5, 1\}`$ × 5 learning modes | `class` (gain); `oracle` (regret) |
| **`fairness`** | does the boost reduce inequality? | $`\lambda \in \{0, 0.5, 1, 2, 3\}`$ × supply ×1.0 / ×0.6 | $`\lambda = 0`$ |
| `oxygen_case.py` | real data: how should scarce oxygen be shared? | 21 April 2021, 19 states; supply 60–100% of demand | request pro-rata; caseload; actual order |

### 3.3 Unit tests
62 tests (`python -m pytest`) cover, among others: the LP is integral and optimal; core before excess; heuristics respect all limits; **perfect hindsight ≥ every policy**; misreporting never moves the yardstick; every audit rule sets the caught liars' core exactly and leaves honest requesters untouched; the EB prior strength is large when suppliers are alike and small when they differ; with overbooking no requester ever receives more than it asked for.

---

## 4. Evaluation metrics and statistics

### 4.1 Metrics
"Genuine need" of requester $`j`$ is $`\min(D^0_j,\ u_{\ell_j}E_j\cdot 14)`$, always computed from the **true** values; "met" is deliveries counted up to it.

| Metric | Definition | Why |
|---|---|---|
| `need_fill` (**need met**) | $`\sum_j \text{met}_j / \sum_j \text{genuine}_j`$ | headline: total genuine need delivered |
| `wneed_fill_ref` | same, each requester weighted 1/10/100 by urgency level | urgency-weighted need met |
| `wneed_fill_w123` | same, weighted 1/2/3 | checks that results don't depend on PULSE's own weights |
| `fill_L3_honest` | fill of the most urgent requesters whose request is within their need | protects the most urgent |
| `exp_fill_nonbulk` | expected fill using the true $`\psi^{\ast}`$ instead of the coin flips | removes outcome luck |
| `need_fill_cohort` / `need_fill_others` | need met for the (possibly lying) cohort / everyone else | gain from lying; harm to the honest |
| `gini_need_fill`, `gini_need_fill_L3` | Gini coefficient of per-requester need met (all / level 3) | inequality |
| `share_below_half_need`, `p10_need_fill` | share of requesters with < 50% of need met; 10th percentile | how many are left behind |
| `excess_received` | units delivered beyond genuine need | hoarding |
| `units_returned` | successful shipments refused as surplus | waste caused by overbooking |
| share of H1 / H2 | policy mean ÷ bound mean | how close to the best achievable |

### 4.2 Statistics (`pulse/stats.py`)
Every comparison is **paired by seed**: the per-seed difference "configuration − reference". For each metric:

| Statistic | Meaning |
|---|---|
| mean difference with a 95% t-interval | the effect and its uncertainty; a CI that excludes 0 means a real difference |
| 95% bootstrap interval (5,000 resamples) | a check on the t-interval for skewed differences |
| Wilcoxon signed-rank p, Holm-adjusted across 5 primary metrics | significance, corrected for testing several metrics |
| $`d_z`$ = mean / sd of the differences | effect size |
| seeds won (e.g. 29/30) | how consistently the configuration beats the reference |

---

## 5. Results (30 paired seeds unless stated)

### 5.1 Layer 1: PULSE itself

**Ablation.** Need met rises from **0.368 (Shield-Net)** to **0.652 (PULSE)**. The honest headline is against the fair baseline: **PULSE vs Shield-Net + filters: +0.129 [0.100, 0.157], 30/30 seeds.** By marginal step:
- price and certificate filters: +0.176, the largest gain;
- expected-delivery objective: +0.070;
- core/excess split: +0.059, and it halves units delivered beyond need (45,068 → 21,388);
- learning $`\hat\psi_g`$: no effect;
- weights 1/10/100: trade total need for urgency.

**Benchmarks.**

| Policy | Need met | % of H1 |
|---|---|---|
| H2 perfect hindsight | 0.980 | — |
| H1 clairvoyant fluid bound | 0.674 | 100% |
| **PULSE** | **0.652** | **96.8%** |
| Proportional rationing | 0.576 | 85.5% |
| FCFS | 0.573 | 85.0% |
| Shield-Net + filters | 0.524 | 77.7% |
| Random | 0.436 | 64.8% |

- **PULSE is within about 3% of the best possible without knowing the future**, so a lookahead model is not worth building.
- The gap between H1 and H2 (0.67 vs 0.98) is the value of knowing **which matches will fail**. That is where the novelty work aims.

**Oxygen case study** (real data, 19 states, 21 April 2021). State requests ranged from 0.20× (Kerala) to 3.66× (Gujarat) their caseload need. With supply at 60/70/80% of demand, need-based rules meet **0.60/0.70/0.73** of need against **0.51/0.57/0.62** for sharing by request. Under-asking caps every request-bounded rule at 0.73; the actual order reached 0.75 by giving some states more than they asked for.

### 5.2 Layer 2: Idea A, misreporting and audits

**Lying pays.** A 10% cohort inflating its request and staff count:

| Factor $`\varphi`$ | Liars' gain in need met | Honest others |
|---|---|---|
| 2 | +0.145 [0.122, 0.169], 30/30 | −0.010 |
| 3 | +0.173 [0.148, 0.199], 29/30 | −0.022 |
| 5 | +0.183 [0.155, 0.211], 30/30 | −0.050 |

**Why it pays: a hedge against match failure.** With match failures switched off, the same lie gains −0.015 (5-seed mechanism check). An inflated request is matched with more suppliers, so the liar still collects enough when most matches fail. **Requesters overbook on their own behalf.**

**Audits are a weak defence** (liars' gain at $`\varphi = 3`$; no audit: +0.173):

| Audit rate | `truth` | `fraction=0.5` | `zero` |
|---|---|---|---|
| 10% | +0.176 | +0.170 | +0.154 |
| 25% | +0.175 | +0.167 | +0.115 |
| 50% | +0.175 | +0.153 | +0.052 [0.017, 0.087] |

`truth` deters nothing: a caught liar is reset to its true need, which exceeds an honest request in this world. Even auditing half of all requesters with the harshest rule leaves a positive gain.

### 5.3 Overbooking: the link between A and C

**Moderate overbooking improves everything at once** (honest world, vs $`\beta = 0`$):

| β | Need met | Urgency-weighted | Most-urgent fill | Units returned |
|---|---|---|---|---|
| 0 | 0.652 | 0.704 | 0.706 | 0 |
| **0.5** | **+0.038 [0.031, 0.044], 29/30** | **+0.076, 30/30** | **+0.082, 30/30** | 26,662 |
| 1 | +0.035, 27/30 | +0.116, 30/30 | +0.127, 30/30 | 48,223 |
| 2 | −0.005 (n.s.) | +0.157, 30/30 | +0.180, 30/30 | 80,866 |
| 3 | −0.046, 3/30 | +0.173, 30/30 | +0.204, 30/30 | 105,069 |

**And it removes the incentive to lie, without audits** (liars' gain at $`\varphi = 3`$):

| β | 0 | 0.5 | 1 | **2** | 3 |
|---|---|---|---|---|---|
| Liars' gain | +0.173 | +0.091 | +0.036 | **−0.012 (n.s.)** | −0.009 (n.s.) |
| Honest others | −0.022 | −0.044 | −0.062 | −0.065 | −0.067 |

When the platform hedges for everyone, lying stops being profitable, but it still harms honest requesters (about −0.065).

### 5.4 Idea C: supplier reliability learning

Gain in need met over today's class-level learning (urgency-weighted in brackets):

| How much suppliers differ (σ) | `supplier` ($`n_0`$ = 5) | **`supplier_eb`** | `thompson` | `oracle` |
|---|---|---|---|---|
| 0 | −0.011 (hurts) | **+0.002** (n.s.) | −0.026 | 0.000 |
| 0.5 | +0.003 (+0.030) | **+0.013 [0.007, 0.019]** (+0.031) | −0.014 (+0.002) | +0.015 (+0.075) |
| 1.0 | +0.024 (+0.070) | **+0.032 [0.026, 0.038], 30/30** (+0.066) | −0.005 (+0.027) | +0.038 (+0.136) |

- **Empirical Bayes is safe and near-optimal on need met.** It never hurts, and is never significantly behind the oracle (regret −0.002 and −0.006, both n.s.). It recovers about 85% of the oracle's need-met gain at σ = 1, and about half of its urgency-weighted gain.
- **Exploration does not pay.** Thompson sampling is the worst learner at every σ: over six rounds, exploring costs more than it learns.

### 5.5 Fairness
- **The fairness boost has no meaningful effect**: Gini 0.287 → 0.287 at full supply; −0.006 at 60% supply.
- **The cause is failure luck, not eligibility.** Of the ~61 requesters per run who receive nothing, 56 were matched but every match failed; none was left without an eligible supplier.
- **Overbooking and fairness:** within the most urgent level, overbooking halves inequality (Gini 0.249 → 0.124 at β = 3) by diversifying failure risk. Overall it raises inequality (0.287 → 0.398), because the hedges go to urgent requesters.

### 5.6 The story in five points
1. PULSE beats a fair Shield-Net by +0.13 need met and gets within 3% of the best achievable without knowing the future.
2. When most matches fail, **over-asking is a rational hedge**, and audits are an expensive, incomplete fix.
3. **Platform overbooking removes the incentive to lie** (β ≈ 2); **moderate overbooking (β ≈ 0.5) improves allocation on every measure**.
4. **Learning each supplier's reliability, with an empirical-Bayes prior,** is safe and near-optimal on need met; **exploration does not pay** over a short horizon.
5. Real 2021 oxygen data show that sharing by request rewards over-asking, while need-based sharing meets 9–13 points more need under shortage.

---

## 6. Limitations and open questions

- **Synthetic world.** District geography is a placeholder, and requesters and suppliers are generated. Only the request-size mean, the supplier variant mix and the success rates are anchored to the Shield-Net paper.
- **Assumed behaviour.** The 10% lying cohort and its factor, the reliability spread σ, and "refused surplus leaks like unsold stock" are modelling choices; results are shown across ranges where possible.
- **Honest requests are below genuine need** in this generator. This drives the `truth` audit result and should be stated wherever audits are discussed.
- **The β tension is unresolved.** β ≈ 0.5 maximises total need met, but β ≈ 2 is needed to remove the incentive to lie. Overbooking by urgency level or by estimated success rate is the natural next step.
- **Overbooking raises overall inequality.** A fairness-aware overbooking rule is open.
- **Oxygen case.** Need ignores case severity; the caseload norm is calibrated from the same allocation; one date and 19 states make it an illustration, not a statistical test. A supply-side network (oxygen production by state or plant) would let PULSE's distance and reliability features matter.

---

## 7. How to reproduce

```bash
conda activate pulse
python -m pytest -q                                            # 62 tests

# layer 1
python experiments.py --suite all --seeds 30 --jobs 4          # ablation, sweeps, learning, benchmarks
python scripts/oxygen_case.py                                  # oxygen case study

# layer 2 (novelty-AC only)
python experiments.py --suite misreport   --seeds 30 --jobs 4
python experiments.py --suite overbook    --seeds 30 --jobs 4
python experiments.py --suite reliability --seeds 30 --jobs 4
python experiments.py --suite fairness    --seeds 30 --jobs 4

python plots.py                                                # figures for the layer-1 suites
```

Every result file appears in `results/`: `<suite>_runs.csv` (one row per configuration and seed), `<suite>_paired.csv` (paired statistics), `<suite>_table.csv` (means and standard deviations). Any new feature can be switched on in a single run, for example:

```bash
python run.py --set overbook=0.5 --set learn_mode=supplier_eb --set reliability_sigma=1.0
```

| Code | Where |
|---|---|
| model, simulation, new switches | `pulse/model.py`, `pulse/simulation.py`, `pulse/config.py` |
| world generator (cohort, θ) | `pulse/generator.py` |
| benchmarks and bounds | `pulse/benchmarks.py` |
| metrics, statistics | `pulse/metrics.py`, `pulse/stats.py` |
| experiment suites | `experiments.py` |
| oxygen case | `scripts/oxygen_case.py`, `data/oxygen/` |
