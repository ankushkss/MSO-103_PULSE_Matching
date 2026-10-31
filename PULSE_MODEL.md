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

### 2.1 The problem solved each round

```math
\begin{aligned}
\max_x\quad & \sum_{(i,j)\in E} \big(p_j\,\hat\psi_{ij} - c_{ij}\big)\,x_{ij}\\
\text{s.t.}\quad & \sum_j x_{ij} \le A_i && \forall i && \text{(supply)}\\
& \sum_i x_{ij} \le \tilde D_j && \forall j && \text{(demand)}\\
& x_{ij} \ge 0 && \forall (i,j) \in E
\end{aligned}
```

Same structure as Shield-Net: one variable per supplier–requester pair, one supply limit per supplier, one demand limit per requester. What changes is what goes into it:

| Term | Meaning | How it is computed |
|---|---|---|
| $`E`$ | the pairs that are allowed to match | Section 3, issues 4–6 and 9 |
| $`p_j`$ | urgency weight | $`W[\ell_j]`$ with $`W = (1, 10, 100)`$ and $`\ell_j \in \lbrace 1,2,3\rbrace`$ from facility type |
| $`\hat\psi_{ij}`$ | estimated probability that the match is actually delivered | $`\hat\psi_{same}`$ if supplier and requester are in the same state, else $`\hat\psi_{cross}`$ |
| $`c_{ij}`$ | small tie-break cost | $`\tilde\alpha\,\hat c_{ij} + \beta\,\rho_{ij} + \mu\,\mathbf{1}[\kappa_j = 1](1 - q_i)`$: distance, price, uncertified stock |
| $`A_i`$ | stock the supplier has this round | $`s_i \Delta + r_i`$: daily rate × days since the last round, plus carried stock |
| $`\tilde D_j`$ | how much the requester may receive this round | its outstanding demand, capped if flagged as a possible hoarder |

The objective is the **expected urgency-weighted number of units delivered**, minus small tie-break costs.

### 2.2 Between rounds

The platform updates its success estimates from the outcomes it has observed:

```math
\hat\psi_g = \max\left(\psi_{floor},\ \frac{n_0\,\psi^0_g + S_g}{n_0 + S_g + F_g}\right), \qquad g \in \lbrace \text{same}, \text{cross} \rbrace
```

where $`S_g`$ and $`F_g`$ count successful and failed matches so far. Unserved demand carries over to the next round, as in Shield-Net.

### 2.3 Default values

| Parameter | Value | Where it comes from |
|---|---|---|
| $`\psi^0_{same},\ \psi^0_{cross}`$ | 0.40, 0.16 | solved from the paper: 27% success rate, 48% local matches, odds ratio 3.55 for local |
| $`n_0`$ | 20 | how much weight the prior gets, in matches |
| $`\psi_{floor}`$ | 0.05 | keeps every pair worth serving |
| $`W`$ | (1, 10, 100) | policy choice |
| $`\tilde\alpha,\ \beta,\ \mu`$ | 0.002, 0.002, 0.02 | small, so they only break ties |
| $`c`$ | 0.05 | a flagged requester gets at most 5% of the round's supply |

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

**In PULSE.** They are a starting point, not a fixed assumption. After each round, $`\hat\psi_g`$ is updated with the formula in 2.2. Early on, it stays close to the paper's values. As outcomes accumulate, it moves toward what is actually observed. The model itself is unchanged; it simply reads the new $`\hat\psi`$ next round.

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

**In Shield-Net.** Price was not in the model, so a requester could be matched with a supplier it couldn't afford. The paper names this as a reason matches failed (prices ranged from \$1 to \$7).

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

**In Shield-Net.** No protection. One request for 5 million shields, 50 times larger than any other, had to be removed by hand.

**In PULSE.**
- **Flag** a request that is implausible for the facility's size:

  ```math
  z_j = \mathbf{1}\big[D^0_j > u \cdot \text{size}_j \cdot T^{cov}\big]
  ```

  That is, more than its staff could use in 14 days at 0.5 units per person per day.
- **Cap** a flagged requester's demand in the model:

  ```math
  \tilde D_j = \min\big(D_j,\ \max(\lfloor c\,A^{tot} \rfloor,\ m_{min})\big)
  ```

  It can receive at most 5% of the round's supply.
- The demand above the cap is **deferred, not deleted**: it stays outstanding, and the cap lifts once the requester is verified. A flag means "unverified", not "guilty".

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

- **Still a linear program.** Every change above alters only the objective coefficients, which pairs are allowed, or the right-hand sides. The constraint matrix keeps Shield-Net's form: each variable appears in exactly one supply row and one demand row. It is therefore totally unimodular, so with integer stock and demand the LP's solution is automatically whole units. Each round solves exactly, in about a second.
- **Every allowed pair is worth serving.** Every coefficient $`p_j\,\hat\psi_{ij} - c_{ij}`$ is positive, because $`\tilde\alpha + \beta + \mu = 0.024 < 1 \times \psi_{floor} = 0.05`$. So the model never leaves a supplier with stock unused next to a requester it could serve.
- **Shield-Net is a special case.** Set $`\hat\psi = 1`$, $`W = (1,2,3)`$, $`\beta = \mu = 0`$, no cap and no price or certificate rules, and PULSE gives exactly Shield-Net's answer. This was checked numerically.

---

## 5. What the model does not handle

- **Supplier minimum order quantities.** The paper names them as a problem, but enforcing them needs yes/no variables, which would make each round an integer program. Left as a stated limitation.
- **Outcomes are not decided by the model.** Whether a match succeeds, whether a requester refuses it, whether a requester gives up, and how much unsold stock survives are all part of the **simulation** used to test PULSE. They are not part of the optimisation; see `README.md`.
