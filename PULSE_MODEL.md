# The PULSE Model

PULSE is an improved version of the Shield-Net matching model (Alcock, Boutilier, Siddiq, *INFORMS J. Applied Analytics*, 2022). Every Monday, Wednesday and Friday for two weeks, the platform solves one linear program that decides how many units each supplier sends to each requester.

This document covers only the model: what it is, and how it fixes Shield-Net's problems. The simulation used to test it, the data and the code are described in `README.md`.

---

## 1. Shield-Net's model

Each matching round, Shield-Net solves:

```math
\begin{aligned}
\min_{x,\,t}\quad & \sum_j p_j\,t_j + \alpha \sum_{i,j} c_{ij}\,x_{ij}\\
\text{s.t.}\quad & \sum_j x_{ij} \le s_i \quad \forall i, \qquad \sum_i x_{ij} \le d_j \quad \forall j, \qquad x_{ij} \le d_j\,a_{ij} \quad \forall i,j\\
& t_j \ge d_j - \sum_i x_{ij}, \qquad t_j \ge 0, \qquad x_{ij} \ge 0
\end{aligned}
```

| Symbol | Meaning |
|---|---|
| $`x_{ij}`$ | units from supplier $`i`$ to requester $`j`$ |
| $`t_j`$ | unmet demand of requester $`j`$ |
| $`s_i`$ | supplier capacity per round |
| $`d_j`$ | units requested |
| $`p_j \in \lbrace 1,2,3\rbrace`$ | self-reported urgency |
| $`c_{ij}`$ | straight-line km between the centres of the two US states |
| $`\alpha = 10^{-6}`$ | distance weight |
| $`a_{ij}`$ | 1 if the product type (kit or assembled) matches |

Since $`t_j = d_j - \sum_i x_{ij}`$ at the optimum, this is the same as

```math
\max_x \sum_{i,j:\,a_{ij}=1} \big(p_j - \alpha\,c_{ij}\big)\,x_{ij}
\quad\text{s.t.}\quad \sum_j x_{ij} \le s_i,\quad \sum_i x_{ij} \le d_j,\quad x \ge 0 .
```

In words: send as many units as possible, most urgent first, and use distance only to break ties.

---

## 2. The PULSE model

### 2.1 The idea in one paragraph

Three times a week, the platform looks at every supplier's stock and every requester's outstanding request and decides **how many units each supplier should send to each requester**. It prefers shipments that are **urgent** (a COVID ward before a grocery store) and **likely to actually arrive** (same-state matches succeed more often). Each request is split into the part the facility **genuinely needs** and the **rest**; all genuine need is served before any of the rest. Distance, price and certification only break ties.

### 2.2 What the model decides

For every allowed supplier–requester pair $`(i, j)`$ there are two numbers to choose:

| Decision variable | Meaning |
|---|---|
| $`x^c_{ij} \ge 0`$ | units from supplier $`i`$ that go towards requester $`j`$'s **genuine need** (core) |
| $`x^e_{ij} \ge 0`$ | units from supplier $`i`$ that go towards the **rest** of $`j`$'s request (excess) |

The supplier ships $`x_{ij} = x^c_{ij} + x^e_{ij}`$ units in total.

### 2.3 What the model is given each round

| Input | Meaning | Where it comes from |
|---|---|---|
| $`A_i`$ | stock supplier $`i`$ can ship this round | daily rate × days since last round + carried stock: $`A_i = s_i\,\Delta + r_i`$ |
| $`C_j`$ | requester $`j`$'s **core**: genuine need not yet received | $`C_j = \min\big(D_j,\ \max(0,\ \lfloor N_j - G_j \rfloor)\big)`$ with $`N_j = u_{\ell_j} E_j H`$ |
| $`X_j`$ | requester $`j`$'s **excess**: the rest of its request | $`X_j = D_j - C_j`$ |
| $`E`$ | pairs allowed to match | variants overlap **and** price within budget **and** certified if required **and** reachable by road |
| $`p_j`$ | urgency weight | 1 / 10 / 100 for essential non-healthcare / other healthcare / treating the emergency |
| $`\hat\psi_{ij}`$ | chance the shipment actually arrives | 0.40 if same state, 0.16 if not (then learned from outcomes) |
| $`c_{ij}`$ | small tie-break cost | distance + price + "uncertified but certified preferred" (at most 0.024) |
| $`\varepsilon`$ | value of an excess unit relative to a core unit | $`10^{-4}`$ |

Here $`D_j`$ is the outstanding request, $`E_j`$ the staff count, $`u_{\ell}`$ the usage norm per staff per day, $`H = 14`$ days, and $`G_j`$ what the requester has already received.

### 2.4 The objective, term by term

**Value of one unit** shipped on pair $`(i,j)`$:

```math
w_{ij} \;=\; \underbrace{p_j}_{\text{how urgent}} \;\times\; \underbrace{\hat\psi_{ij}}_{\text{chance it arrives}} \;-\; \underbrace{c_{ij}}_{\text{tie-break}}
```

So $`p_j\,\hat\psi_{ij}`$ is the **expected urgency-weighted delivery** of that unit. An excess unit is worth the same, multiplied by the tiny $`\varepsilon`$.

**Objective:** maximise the total value of everything shipped,

```math
\max \quad \sum_{(i,j)\in E} w_{ij}\,x^c_{ij} \;+\; \varepsilon \sum_{(i,j)\in E} w_{ij}\,x^e_{ij}
```

The ranking this produces (prior $`\hat\psi`$, ignoring tie-breaks):

| Value of one core unit | same state ($`\hat\psi = 0.40`$) | other state ($`\hat\psi = 0.16`$) |
|---|---|---|
| Level 3: treating the emergency ($`p = 100`$) | **40** | **16** |
| Level 2: other healthcare ($`p = 10`$) | 4.0 | 1.6 |
| Level 1: essential non-healthcare ($`p = 1`$) | 0.40 | 0.16 |
| Any excess unit | at most 0.01 | |

Read down the table: **urgency always wins** (the worst level-3 unit, 16, beats the best level-2 unit, 4.0). Read across: **within a level, same-state wins** (2.5×). Read the last row: **any core unit beats any excess unit**.

### 2.5 The constraints, in words

| # | Constraint | In words |
|---|---|---|
| (1) | $`\displaystyle\sum_{j} \big(x^c_{ij} + x^e_{ij}\big) \le A_i`$ for every supplier $`i`$ | a supplier cannot ship more than it has |
| (2) | $`\displaystyle\sum_{i} x^c_{ij} \le C_j`$ for every requester $`j`$ | a requester's genuine need is filled at most once |
| (3) | $`\displaystyle\sum_{i} x^e_{ij} \le X_j`$ for every requester $`j`$ | the rest of the request is filled at most once |
| (4) | $`x^c_{ij},\ x^e_{ij} \ge 0`$, and only for allowed pairs $`(i,j) \in E`$ | no negative shipments; no shipments on forbidden pairs |

### 2.6 The full model

```math
\boxed{
\begin{aligned}
\max_{x^c,\,x^e}\quad & \sum_{(i,j)\in E} w_{ij}\,\big(x^c_{ij} + \varepsilon\,x^e_{ij}\big), \qquad w_{ij} = p_j\,\hat\psi_{ij} - c_{ij}\\[4pt]
\text{subject to}\quad & \sum_j \big(x^c_{ij} + x^e_{ij}\big) \le A_i && \forall i && \text{(1) supply}\\
& \sum_i x^c_{ij} \le C_j && \forall j && \text{(2) core}\\
& \sum_i x^e_{ij} \le X_j && \forall j && \text{(3) excess}\\
& x^c_{ij},\ x^e_{ij} \ge 0 && \forall (i,j) \in E && \text{(4)}
\end{aligned}
}
```

It has the same shape as Shield-Net (one supply row per supplier, demand rows per requester), so it stays a **transportation LP**: totally unimodular, so the solution is in whole units, solved exactly in about half a second per round (Section 4).

### 2.7 A worked example

One supplier in **Maharashtra** has $`A = 100`$ units. Three requesters, each with core only:

| Requester | State | Level | Core $`C_j`$ | Value of one unit $`w`$ |
|---|---|---|---|---|
| COVID ward | Maharashtra (same) | 3 | 60 | $`100 \times 0.40 = 40`$ |
| Clinic | Gujarat (other) | 2 | 80 | $`10 \times 0.16 = 1.6`$ |
| Grocery store | Maharashtra (same) | 1 | 50 | $`1 \times 0.40 = 0.40`$ |

The LP fills the highest values first: **COVID ward 60**, then **clinic 40** (the remaining stock), **grocery 0**. The clinic is in another state and less likely to receive the shipment, yet it still beats the nearby grocery store, because urgency dominates locality across levels. Shield-Net, with weights 1/2/3, would have scored the same-state grocery store at 0.40 vs the cross-state clinic at 0.32 if $`\psi`$ were added to it, putting the grocery store first.

### 2.8 Between rounds: learning $`\hat\psi`$

After each round the platform counts, per class $`g \in \{\text{same}, \text{cross}\}`$, the matches that succeeded ($`S_g`$) and failed ($`F_g`$), and updates its estimate (a Beta posterior mean with the paper's value as prior worth $`n_0 = 20`$ matches):

```math
\hat\psi_g \;=\; \max\left(\psi_{floor},\ \frac{\overbrace{n_0\,\psi^0_g}^{\text{prior}} + \overbrace{S_g}^{\text{successes}}}{n_0 + S_g + F_g}\right)
```

Delivered units are added to each requester's $`G_j`$ (so its core shrinks), and unserved demand carries over to the next round.

### 2.9 Default values

| Parameter | Value | Where it comes from |
|---|---|---|
| $`\psi^0_{same},\ \psi^0_{cross}`$ | 0.40, 0.16 | solved from the paper: 27% success rate, 48% local matches, odds ratio 3.55 for local |
| $`n_0`$ | 20 | how much weight the prior gets, in matches |
| $`\psi_{floor}`$ | 0.05 | keeps every pair worth serving |
| $`W`$ | (1, 10, 100) | policy choice |
| $`\tilde\alpha,\ \beta,\ \mu`$ | 0.002, 0.002, 0.02 | small, so they only break ties |
| $`u_{\ell}`$ | 0.25, 0.5, 1.0 units per staff per day, levels 1/2/3 | assumed usage norm by facility type |
| $`H`$ | 14 days | the horizon |
| $`\varepsilon`$ | $`10^{-4}`$ | any core unit (worth at least 0.026) beats any excess unit (worth at most 0.01) |

---

## 3. The issues and how PULSE handles them

### Issue 1: Most matches never turned into deliveries

**In Shield-Net.** Only about 27% of matches led to a delivery, and being in the same state was the only significant predictor of success (odds ratio 3.55). But the model counted every matched unit as if it would arrive, so a likely-to-fail match looked as good as a likely-to-succeed one.

**In PULSE.** Each unit is weighted by its chance of being delivered:

```math
p_j \;\longrightarrow\; p_j\,\hat\psi_{ij}
```

The model now maximises expected deliveries. A same-state unit (0.40) is worth 2.5 times a cross-state unit (0.16) at the same urgency.

### Issue 2: Those success rates come from the US

**In Shield-Net.** Not modelled at all. And the 0.40 and 0.16 PULSE starts from are US numbers, which may not hold in India.

**In PULSE.** They are a starting point, not a fixed assumption. After each round, $`\hat\psi_g`$ is updated with the formula in 2.8. Early on, it stays close to the paper's values. As outcomes accumulate, it moves toward what is actually observed. The model itself is unchanged; it simply reads the new $`\hat\psi`$ next round.

Real values for India need real delivery records; in this project the mechanism is demonstrated in simulation.

### Issue 3: Distance could beat urgency, and urgency could be faked

**In Shield-Net.** Urgency weights were 1, 2, 3 and self-reported. Once distance or success probability enters the objective, a nearby low-urgency requester can outrank a distant high-urgency one. For example, a grocery store in the same state (1 × 0.40 = 0.40) beats a hospital in another state (2 × 0.16 = 0.32). And anyone can claim to be urgent.

**In PULSE.**
- Urgency comes from facility type: 3 = treating patients of the emergency, 2 = other healthcare, 1 = essential non-healthcare.
- Weights are $`W = (1, 10, 100)`$, so each level is ten times the one below.
- Urgency always wins across levels as long as

```math
W[\ell+1]\,\hat\psi_{cross} - \max c_{ij} \;>\; W[\ell]\,\hat\psi_{same} \qquad (\ell = 1, 2)
```

With the defaults: $`10 \times 0.16 - 0.024 = 1.576 > 0.40`$. The code checks this every round.
- Distance and success probability now only decide between requesters at the **same** urgency level.

### Issue 4: Price was ignored

**In Shield-Net.** Price was not in the model, so a requester could be matched with a supplier it couldn't afford. The paper names this as a reason matches failed (prices ranged from 1 to 7 US dollars).

**In PULSE.** Each requester gives a maximum unit price $`\bar\pi_j`$.
- **Hard rule:** a pair is allowed only if $`\pi_i \le \bar\pi_j`$. Over-budget pairs are not in $`E`$.
- **Preference:** among affordable suppliers, cheaper is slightly better, through $`\beta\,\pi_i/\bar\pi_j`$ in $`c_{ij}`$.

### Issue 5: Quality was uncertain

**In Shield-Net.** No information about quality. The paper names this as a reason requesters didn't order.

**In PULSE.** Each supplier is certified or not ($`q_i`$). Each requester chooses $`\kappa_j`$:
- $`\kappa_j = 2`$, require certified: a pair is allowed only if $`q_i = 1`$.
- $`\kappa_j = 1`$, prefer certified: uncertified stock costs $`\mu`$ extra per unit in $`c_{ij}`$, so certified suppliers are used first.
- $`\kappa_j = 0`$, no preference.

### Issue 6: Product type was too rigid

**In Shield-Net.** Each requester wanted exactly one type: kit or assembled.

**In PULSE.** Each requester gives a **set** of acceptable variants $`V_j`$, for example "either". A pair is allowed if the supplier offers at least one of them:

```math
\gamma_{ij} = \mathbf{1}[V_i \cap V_j \ne \emptyset]
```

### Issue 7: One requester could take everything

**In Shield-Net.** No protection. One request for 5 million shields, 50 times larger than any other, had to be removed by hand. More generally, any requester that asked for more than it needed could take stock that others genuinely needed, as long as it was urgent enough.

**In PULSE.** Each request is split into what the facility plausibly needs and the rest:
- **Genuine need** over the horizon is estimated from facility type and size: $`N_j = u_{\ell_j} \cdot E_j \cdot H`$. For example, a COVID ward with 50 staff needs about $`1.0 \times 50 \times 14 = 700`$ shields over two weeks.
- **Core** is the part of the request within that need, minus what the requester has already received:

```math
C_j = \min\big(D_j,\ \max(0,\ \lfloor N_j - G_j \rfloor)\big), \qquad X_j = D_j - C_j
```

- **Core comes first.** Core and excess have separate demand rows, and an excess unit is worth only $`\varepsilon`$ times the same unit as core. By the condition

```math
\varepsilon \cdot W[3] \;<\; W[1]\,\psi_{floor} - \max c_{ij}
```

any core unit is worth more than any excess unit (defaults: $`0.01 < 0.026`$). So no supplier ever ships excess while a requester it can serve still has core left.

Three consequences:
- **Nothing is cut.** A requester can still receive its whole request; the excess just waits behind everyone's core, and is served from what is left, by the same priorities.
- **Inflating a request gains nothing.** $`N_j`$ depends only on facility type and size, so asking for more only adds excess, never core.
- **Being served shrinks the core.** Receipts $`G_j`$ count against the need, so a requester that has already been served comes back mostly as excess.

The 5-million request from the paper would have had a core equal to what its hospitals' staff could plausibly use in two weeks. Everything beyond that would have been served only from leftover stock, after every other requester's genuine need.

### Issue 8: Supply did not reflect time or leftover stock

**In Shield-Net.** Each supplier had a fixed capacity $`s_i`$ per round, whether the gap since the last round was 2 days or 3, and leftover stock was ignored.

**In PULSE.** $`s_i`$ is a **daily** rate:

```math
A_i = s_i\,\Delta + r_i
```

- $`\Delta`$ is the number of days since the last round: 3 after a weekend, 2 otherwise.
- $`r_i`$ is unsold stock carried over from earlier rounds.

### Issue 9: Distances were too coarse

**In Shield-Net.** Distance was the straight line between the centres of two US states, so every pair in the same state had distance 0, and $`\alpha = 10^{-6}`$ per km was hard to interpret.

**In PULSE.**
- Distance is the **road distance between Indian districts**, from an 800 × 800 matrix, scaled by the longest distance:

```math
\hat c_{ij} = \frac{R[k(i),\,k(j)]}{R_{max}} \in (0, 1]
```

- Within a district, the distance is 10–20 km rather than 0.
- A pair with no road distance in the data is not in $`E`$.
- "Same state" in Issue 1 is decided from each district's state in the same file.

---

## 4. What stays the same

- **Still a linear program.** Every change above alters only the objective coefficients, which pairs are allowed, the right-hand sides, and one extra demand row per requester. Each variable still appears in exactly one supply row and exactly one demand row (core or excess). The constraint matrix is therefore totally unimodular, so with integer stock, core and excess the LP's solution is automatically whole units. Each round solves exactly, in about half a second.
- **Every allowed pair is worth serving.** Every coefficient $`p_j\,\hat\psi_{ij} - c_{ij}`$ is positive, because $`\tilde\alpha + \beta + \mu = 0.024 < 1 \times \psi_{floor} = 0.05`$. So the model never leaves a supplier with stock unused next to a requester it could serve.
- **Shield-Net is a special case.** Set $`\hat\psi = 1`$, $`W = (1,2,3)`$, $`\beta = \mu = 0`$, treat the whole request as core, and drop the price and certificate rules, and PULSE gives exactly Shield-Net's answer. This was checked numerically.

---

## 5. What the model does not handle

- **Supplier minimum order quantities.** The paper names them as a problem, but enforcing them needs yes/no variables, which would make each round an integer program. Left as a stated limitation.
- **Genuine need is an estimate.** It rests on assumed usage rates per facility type and on truthful staff counts. A facility that really uses more than its type's norm is still served in full eventually; the part above the norm just waits behind others' core.
- **Excess is shared by priority, not proportionally.** Leftover stock goes to excess in the same urgency and locality order as core, rather than giving every requester the same number of days of buffer stock.
- **Outcomes are not decided by the model.** Whether a match succeeds, whether a requester refuses it, whether a requester gives up, and how much unsold stock survives are all part of the **simulation** used to test PULSE. They are not part of the optimisation; see `README.md`.
- **Never more than the request.** Each requester gets at most what it asked for. In the 2021 oxygen data some states asked for far less than their caseload suggested (Kerala about 0.2×), and no request-capped rule can correct that (README Section 13.6).

---

## 6. How well it works (summary; details in README Section 13)

| Question | Answer (30 paired seeds, placeholder data) |
|---|---|
| Better than Shield-Net? | Need met 0.37 → 0.65; better on 30/30 seeds |
| Better than Shield-Net given the same price/certificate filters? | +0.13 need met [95% CI 0.10, 0.16], 30/30 seeds |
| How close to the best achievable without knowing the future? | **96.8%** of the clairvoyant bound (need met) |
| Better than simple rules (proportional, first-come-first-served)? | +0.08 need met, 30/30 seeds |
| Does learning $`\hat\psi`$ help? | It converges to the truth (within 0.01) but changes outcomes by < 0.003 |
| Real data (oxygen, 21 April 2021)? | Need-based rules meet 0.60–0.73 of need vs 0.51–0.62 for request pro-rata when supply is 60–80% of demand |

