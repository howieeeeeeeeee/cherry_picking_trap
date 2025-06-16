import streamlit as st
from app._utils.db_setup import db
from src.data_management.mongo_handler import get_simulation_results_filters
from src.analysis.tables import get_simulation_data
from src.analysis.graphs import plot_simulation_results

st.markdown(
    r"""
### Numerical Simulation Methodology

This module numerically solves for the Bayes-Nash Equilibrium of the Cherry-Picking model to facilitate comparative statics and welfare analysis. The methodology is grounded in the Structure Theorem presented in the research and leverages large-scale Monte Carlo simulation to approximate key expected values.

---
#### 1. Monte Carlo Simulation

The core of the methodology is to approximate the space of all possible outcomes. We generate a large number $(N=10,000,000)$ of policy "baskets," $B = \{x_1, ..., x_K\}$, where each policy $x_i$ is drawn i.i.d. from a specified probability distribution $\phi$. For each simulated basket, we pre-calculate and store the fundamental values:

* **Cherry-Picked Policy:** $C(B) := \max_{x \in B} \pi_P(x)$
* **Moderated Policy:** $M(B) := \max_{x \in B: \pi_R(x) \geq 0} \pi_P(x)$ (if no palatable policy is found, $M(B) = C(B)$)
* **Temptation:** The relative benefit of cherry-picking, $Q(B)$.
* **Potential Payoffs:** The payoffs for both players under moderation, $(\pi_P(M), \pi_R(M))$, and cherry-picking, $(\pi_P(C), \pi_R(C))$.

---

#### 2. Equilibrium Characterization

The simulation follows the logic of the Structure Theorem to identify the equilibrium for the given parameters $(K, \lambda, \theta, \phi)$:

**Boundary Check:** We first calculate the boundary expectations, $E[\pi_R(M(B))]$ and $E[\pi_R(C(B))]$, by averaging over all $N$ simulations.

**Case Identification:** The simulation identifies the equilibrium case using the following decision tree:

| Case | Condition | Equilibrium Outcome |
|-----------|------|-------------------|
| **Case D: Always Informed** | $\lambda = 1$ |  $q^* = \infty$, $\alpha^* = \text{undefined}$ |
| **Case A: Always Bad** | $E[\pi_R(M(B))] \leq 0$ | $q^* = \infty$, $\alpha^* = 0$ | 
| **Case B: Always Good** | $E[\pi_R(C(B))] \geq 0$ | $q^* = \frac{\lambda}{1-\lambda}$, $\alpha^* = 1$ |
| **Case C: Interesting Case** | $E[\pi_R(C(B))] < 0 < E[\pi_R(M(B))]$ | Solve for $q^*$ numerically, then $\alpha^* = \frac{\lambda}{(1-\lambda) \cdot q^*}$ or $\alpha^* = 1$ |


**Solving for the Threshold:** Only in Case C is there a strategic tension requiring a mixed strategy. The Responder's expected payoff function,

$$\Pi_R(q) := E[\pi_R(M(B)) | Q(B) \leq q] \cdot P(Q(B) \leq q) + E[\pi_R(C(B)) | Q(B) > q] \cdot P(Q(B) > q)$$

is approximated using the simulation data. A numerical root-finding algorithm (brentq) is employed to solve for the unique equilibrium threshold $q^*$ that satisfies $\Pi_R(q^*) = 0$.

**Deriving Strategies:** The uninformed Responder's equilibrium acceptance probability, $\alpha^*$, and the Proposer's equilibrium decision threshold, $q_{eq}$, are then determined based on the identified case.


---
#### 3. Ex-Ante Welfare Calculation

With the equilibrium strategies solved, the module calculates the ex-ante expected payoffs, $E[\pi_P]$ and $E[\pi_R]$. This is achieved by performing a final pass over the $N$ simulated baskets. For each basket, we first determine the Proposer's action by applying the solved equilibrium rule:

* The Proposer **moderates** if $Q(B) \leq q_{eq}$.
* The Proposer **cherry-picks** if $Q(B) > q_{eq}$.

Based on this action, the final payoffs for that basket are calculated by weighting them with the probabilities of acceptance, which critically depend on the solved $\alpha^*$ and the given $\lambda$. The average of these payoffs across all $N$ simulations yields the final ex-ante welfare results.
"""
)
