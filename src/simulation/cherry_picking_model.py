import numpy as np
from scipy.optimize import brentq
import pandas as pd
import numba
from numba import prange
import time


@numba.jit(nopython=True, parallel=True)
def process_all_simulations_numba(
    num_simulations, K, theta, dist_name, scale=1.0, lam=1.0, low=0.0, high=1.0
):
    """
    Processes all simulations in one go using Numba, including random generation.
    Correctly handles payoffs when no palatable policies exist.
    """
    # Pre-allocate all arrays
    pi_p_m = np.empty(num_simulations, dtype=np.float32)
    pi_p_c = np.empty(num_simulations, dtype=np.float32)
    pi_r_m = np.empty(num_simulations, dtype=np.float32)
    pi_r_c = np.empty(num_simulations, dtype=np.float32)
    q_values = np.zeros(num_simulations, dtype=np.float32)

    # Process simulations in parallel
    for i in prange(num_simulations):
        # Generate basket inline (more efficient than pre-generating all)
        basket = np.empty(K, dtype=np.float32)

        # Simple random generation based on distribution type
        if dist_name == 0:  # exponential
            for j in range(K):
                basket[j] = np.random.exponential(scale)
        elif dist_name == 1:  # poisson
            for j in range(K):
                basket[j] = np.random.poisson(lam)
        elif dist_name == 2:  # uniform
            for j in range(K):
                basket[j] = np.random.uniform(low, high)

        # --- Combined loop to find C(B) and M(B) in one pass ---
        c_policy = np.float32(-np.inf)
        m_policy = np.float32(-np.inf)
        found_palatable = False

        for j in range(K):
            policy = basket[j]
            if policy > c_policy:
                c_policy = policy
            if policy <= theta:
                if policy > m_policy:
                    m_policy = policy
                found_palatable = True

        if not found_palatable:
            m_policy = c_policy

        # Calculate potential payoffs
        pi_p_m[i] = m_policy
        pi_r_m[i] = theta - m_policy

        # Payoffs for cherry-picking are calculated as before
        pi_p_c[i] = c_policy
        pi_r_c[i] = theta - c_policy

        # Calculate Q value
        if m_policy > 0 and c_policy > m_policy:
            q_values[i] = (c_policy - m_policy) / m_policy

    return pi_p_m, pi_p_c, pi_r_m, pi_r_c, q_values


@numba.jit(nopython=True)
def calculate_Pi_R_numba(q_values, pi_r_m, pi_r_c, q_threshold):
    """
    Fast calculation of Pi_R for a given q threshold.
    """
    total_payoff = 0.0
    n = len(q_values)

    for i in range(n):
        if q_values[i] <= q_threshold:
            total_payoff += pi_r_m[i]
        else:
            total_payoff += pi_r_c[i]

    return total_payoff / n


@numba.jit(nopython=True, parallel=True)
def calculate_welfare_numba(
    lamb, alpha_star, q_star, q_values, pi_p_m, pi_p_c, pi_r_m, pi_r_c
):
    """
    Calculates the total expected payoffs for both players in one pass.
    """
    total_proposer_payoff = 0.0
    total_responder_payoff = 0.0
    n = len(q_values)

    if alpha_star is None:
        alpha_star = 0

    for i in prange(n):
        # Determine Proposer's action for this simulated basket
        is_moderate = q_values[i] <= q_star

        if is_moderate:
            # --- Proposer chooses to Moderate ---
            # Proposer's payoff: gets pi_p_m if Responder accepts.
            # Informed R always accepts. Uninformed R accepts with prob alpha_star.
            # Total acceptance probability = lamb * 1 + (1 - lamb) * alpha_star

            ## firstly, check if pi_r_m is non-negative
            if pi_r_m[i] >= 0:
                total_proposer_payoff += pi_p_m[i] * (lamb + (1 - lamb) * alpha_star)
                total_responder_payoff += pi_r_m[i] * (lamb + (1 - lamb) * alpha_star)
            else:
                total_proposer_payoff += pi_p_m[i] * (1 - lamb) * alpha_star
                total_responder_payoff += pi_r_m[i] * (1 - lamb) * alpha_star

        else:
            # --- Proposer chooses to Cherry-Pick ---
            # Proposer's payoff: gets pi_p_c only if R is UNINFORMED and ACCEPTS.
            # Informed R rejects (payoff = 0).

            if pi_r_c[i] >= 0:
                total_proposer_payoff += pi_p_c[i] * (lamb + (1 - lamb) * alpha_star)
                total_responder_payoff += pi_r_c[i] * (lamb + (1 - lamb) * alpha_star)
            else:
                total_proposer_payoff += pi_p_c[i] * (1 - lamb) * alpha_star
                total_responder_payoff += pi_r_c[i] * (1 - lamb) * alpha_star

    # Return the average payoff per simulation
    return total_proposer_payoff / n, total_responder_payoff / n


class CherryPickingModel:
    """
    Optimized version of the Cherry-Picking Model with improved performance.
    """

    DISTRIBUTIONS = {
        "exponential": {"func": np.random.exponential, "params": ["scale"], "id": 0},
        "poisson": {"func": np.random.poisson, "params": ["lam"], "id": 1},
        "uniform": {"func": np.random.uniform, "params": ["low", "high"], "id": 2},
    }

    def __init__(self, K, lamb, theta, distribution_name, dist_params):
        self.K = K
        self.lamb = lamb
        self.theta = theta

        if distribution_name not in self.DISTRIBUTIONS:
            raise ValueError(f"Distribution '{distribution_name}' not supported.")

        dist_info = self.DISTRIBUTIONS[distribution_name]
        self.distribution_func = dist_info["func"]
        self.dist_id = dist_info["id"]

        for param in dist_info["params"]:
            if param not in dist_params:
                raise ValueError(
                    f"Missing required parameter '{param}' for {distribution_name} distribution."
                )

        self.dist_params = dist_params
        self.simulation_arrays = None  # Store raw arrays instead of DataFrame

    def run_monte_carlo_numba(self, num_simulations=1000000):
        """
        Numba version that generates and processes everything in Numba.
        """

        # Extract distribution parameters
        scale = self.dist_params.get("scale", 1.0)
        lam = self.dist_params.get("lam", 1.0)
        low = self.dist_params.get("low", 0.0)
        high = self.dist_params.get("high", 1.0)

        # Run the entire simulation in Numba
        pi_p_m, pi_p_c, pi_r_m, pi_r_c, q_values = process_all_simulations_numba(
            num_simulations, self.K, self.theta, self.dist_id, scale, lam, low, high
        )

        # Store as raw arrays for fast access
        self.simulation_arrays = {
            "pi_p_m": pi_p_m,
            "pi_p_c": pi_p_c,
            "pi_r_m": pi_r_m,
            "pi_r_c": pi_r_c,
            "q": q_values,
        }

    def calculate_Pi_R(self, q):
        """
        Fast calculation using raw arrays and Numba.
        """
        if self.simulation_arrays is None:
            raise Exception("Please run the Monte Carlo simulation first.")

        return calculate_Pi_R_numba(
            self.simulation_arrays["q"],
            self.simulation_arrays["pi_r_m"],
            self.simulation_arrays["pi_r_c"],
            q,
        )

    def solve_for_q_star(self, bracket=[0, 100000]):
        """
        Finds the equilibrium q* using the fast Pi_R calculation.
        """
        # try:
        #     q_star = brentq(self.calculate_Pi_R, a=bracket[0], b=bracket[1])
        #     return q_star
        # except ValueError:
        #     return "Could not find a root within the given bracket. Try expanding it."
        q_star = brentq(self.calculate_Pi_R, a=bracket[0], b=bracket[1])
        return q_star

    def get_simulation_dataframe(self):
        """
        Convert simulation arrays to DataFrame only when needed.
        """
        if self.simulation_arrays is None:
            raise Exception("Please run the Monte Carlo simulation first.")

        return pd.DataFrame(self.simulation_arrays)

    def solve_and_analyze(self, num_simulations=1000000):
        """
        Orchestrates the full equilibrium solving process.
        """

        start_time = time.time()

        self.run_monte_carlo_numba(num_simulations)

        # Calculate boundary conditions using raw arrays
        E_pi_r_m = float(np.mean(self.simulation_arrays["pi_r_m"]))
        E_pi_r_c = float(np.mean(self.simulation_arrays["pi_r_c"]))

        case = None
        q_star = None
        alpha_star = None

        # Determine equilibrium case
        if self.lamb == 1:
            case = "Case D: Always Informed"
            alpha_star = None
            q_star = float("inf")
        elif E_pi_r_m <= 0:
            case = "Case A: Always Bad"
            alpha_star = 0.0
            q_star = float("inf")
        elif E_pi_r_c >= 0:
            case = "Case B: Always Good"
            alpha_star = 1.0
            q_star = float(self.lamb / (1 - self.lamb)) if self.lamb != 1 else np.inf
        else:
            case = "Case C: Interesting Case"
            q_star = float(self.solve_for_q_star())
            if self.lamb >= q_star / (1 + q_star) and q_star > 0:
                alpha_star = 1.0
            else:
                alpha_star = float(self.lamb / ((1 - self.lamb) * q_star))

        self.equilibrium_results = {
            "case": case,
            "q_star": q_star,
            "alpha_star": alpha_star,
            "E_pi_r_m": E_pi_r_m,
            "E_pi_r_c": E_pi_r_c,
            "elapsed_time": float(time.time() - start_time),
        }

        welfare_results = self.calculate_welfare()
        self.equilibrium_results.update(welfare_results)

        # print(f"Equilibrium analysis complete. Case found: {case}")
        return self.equilibrium_results

    def calculate_welfare(self):
        """
        Calculates the ex-ante expected payoffs for both players in equilibrium.
        This method uses a Numba-compiled helper for maximum performance.
        """
        if not self.equilibrium_results:
            raise Exception(
                "Please run solve_and_analyze() first to determine the equilibrium."
            )

        # Retrieve the solved equilibrium parameters
        alpha_star = self.equilibrium_results["alpha_star"]
        q_star = self.equilibrium_results["q_star"]

        # Retrieve the raw simulation data arrays
        sim = self.simulation_arrays

        # Call the fast Numba function to do the heavy lifting
        E_pi_P, E_pi_R = calculate_welfare_numba(
            self.lamb,
            alpha_star,
            q_star,
            sim["q"],
            sim["pi_p_m"],
            sim["pi_p_c"],
            sim["pi_r_m"],
            sim["pi_r_c"],
        )

        return {
            "proposer_expected_payoff": float(E_pi_P),
            "responder_expected_payoff": float(E_pi_R),
            "percent_of_moderate_policies": float(np.mean(sim["q"] <= q_star)),
        }
