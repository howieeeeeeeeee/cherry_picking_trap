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
    Process Monte Carlo simulations for the cherry-picking model using parallel computation.

    This function generates policy baskets and calculates payoffs for both moderate and
    cherry-picking strategies across multiple simulations. It handles cases where no
    palatable policies exist by setting M(B) = C(B).

    Parameters
    ----------
    num_simulations : int
        Number of Monte Carlo simulations to run.
    K : int
        Number of policies in each basket.
    theta : float
        Responder's threshold for acceptable policies. Policies ≤ theta are palatable.
    dist_name : int
        Distribution identifier (0=exponential, 1=poisson, 2=uniform).
    scale : float, default=1.0
        Scale parameter for exponential distribution.
    lam : float, default=1.0
        Lambda parameter for Poisson distribution.
    low : float, default=0.0
        Lower bound for uniform distribution.
    high : float, default=1.0
        Upper bound for uniform distribution.

    Returns
    -------
    pi_p_m : np.ndarray
        Proposer's payoff when choosing moderate strategy (shape: num_simulations,).
    pi_p_c : np.ndarray
        Proposer's payoff when choosing cherry-pick strategy (shape: num_simulations,).
    pi_r_m : np.ndarray
        Responder's payoff when proposer moderates (shape: num_simulations,).
    pi_r_c : np.ndarray
        Responder's payoff when proposer cherry-picks (shape: num_simulations,).
    q_values : np.ndarray
        Q-values measuring relative benefit of cherry-picking: (C(B) - M(B)) / M(B).

    Notes
    -----
    - Uses Numba's parallel processing for performance optimization.
    - M(B) = max{policy ∈ basket : policy ≤ theta}, or C(B) if no palatable policies.
    - C(B) = max{policy ∈ basket}.
    - Q-value is set to 0 when M(B) = 0 or C(B) = M(B).
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
    Calculate the expected responder payoff for a given Q-threshold strategy.

    This function implements the responder's expected payoff when the proposer uses
    a threshold strategy: moderate if Q ≤ q_threshold, cherry-pick otherwise.

    Parameters
    ----------
    q_values : np.ndarray
        Q-values from simulations measuring cherry-picking benefit.
    pi_r_m : np.ndarray
        Responder payoffs when proposer moderates.
    pi_r_c : np.ndarray
        Responder payoffs when proposer cherry-picks.
    q_threshold : float
        Threshold value for proposer's decision rule.

    Returns
    -------
    float
        Average expected payoff for the responder across all simulations.

    Notes
    -----
    The proposer's strategy is:
    - If Q ≤ q_threshold: choose moderate policy → responder gets pi_r_m
    - If Q > q_threshold: choose cherry-pick → responder gets pi_r_c
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
    Calculate equilibrium welfare (expected payoffs) for both players.

    Computes the expected payoffs considering the equilibrium strategies and
    information structure. Accounts for informed/uninformed responders and their
    acceptance decisions.

    Parameters
    ----------
    lamb : float
        Probability that responder is informed (knows if cherry-picking occurred).
    alpha_star : float or None
        Equilibrium acceptance probability for uninformed responder.
        None is treated as 0.
    q_star : float
        Equilibrium Q-threshold for proposer's strategy.
    q_values : np.ndarray
        Q-values from simulations.
    pi_p_m : np.ndarray
        Proposer payoffs when moderating.
    pi_p_c : np.ndarray
        Proposer payoffs when cherry-picking.
    pi_r_m : np.ndarray
        Responder payoffs when proposer moderates.
    pi_r_c : np.ndarray
        Responder payoffs when proposer cherry-picks.

    Returns
    -------
    tuple[float, float]
        (proposer_expected_payoff, responder_expected_payoff)

    Notes
    -----
    Acceptance logic:
    - Informed responder: Accepts if payoff ≥ 0 (knows the true action)
    - Uninformed responder: Accepts with probability alpha_star
    - Total acceptance prob = lamb * (informed acceptance) + (1-lamb) * alpha_star
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
    A game-theoretic model of strategic policy selection with information asymmetry.

    This model analyzes interactions between a Proposer who selects policies from a
    basket and a Responder who decides whether to accept. The Proposer can either
    moderate (choose the best acceptable policy) or cherry-pick (choose the best
    policy regardless of acceptability).

    Parameters
    ----------
    K : int
        Number of policies in each basket.
    lamb : float
        Probability that the responder is informed (0 ≤ lamb ≤ 1).
    theta : float
        Responder's threshold for acceptable policies.
    distribution_name : str
        Name of the distribution ('exponential', 'poisson', or 'uniform').
    dist_params : dict
        Parameters for the chosen distribution:
        - exponential: {'scale': float}
        - poisson: {'lam': float}
        - uniform: {'low': float, 'high': float}

    Attributes
    ----------
    simulation_arrays : dict or None
        Stores simulation results after running Monte Carlo.
    equilibrium_results : dict or None
        Stores equilibrium analysis results.

    Examples
    --------
    >>> model = CherryPickingModel(
    ...     K=5, lamb=0.3, theta=10,
    ...     distribution_name='exponential',
    ...     dist_params={'scale': 2.0}
    ... )
    >>> results = model.solve_and_analyze(num_simulations=1000000)
    >>> print(f"Equilibrium case: {results['case']}")
    >>> print(f"q* = {results['q_star']:.3f}, α* = {results['alpha_star']:.3f}")
    """

    DISTRIBUTIONS = {
        "exponential": {"func": np.random.exponential, "params": ["scale"], "id": 0},
        "poisson": {"func": np.random.poisson, "params": ["lam"], "id": 1},
        "uniform": {"func": np.random.uniform, "params": ["low", "high"], "id": 2},
    }

    def __init__(self, K, lamb, theta, distribution_name, dist_params):
        """
        Initialize the Cherry-Picking Model with game parameters.

        Parameters
        ----------
        K : int
            Number of policies in each basket.
        lamb : float
            Probability that responder is informed (0 ≤ lamb ≤ 1).
        theta : float
            Responder's threshold for acceptable policies.
        distribution_name : str
            Name of the distribution for generating policies.
        dist_params : dict
            Parameters specific to the chosen distribution.

        Raises
        ------
        ValueError
            If distribution_name is not supported or required parameters are missing.
        """
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
        Execute Monte Carlo simulations using Numba-optimized functions.

        Generates policy baskets and calculates payoffs for all possible strategies
        across the specified number of simulations. Results are stored in memory
        as raw arrays for efficient access.

        Parameters
        ----------
        num_simulations : int, default=1000000
            Number of Monte Carlo iterations to perform.

        Notes
        -----
        - Uses parallel processing via Numba for performance.
        - Stores results in self.simulation_arrays as a dictionary of arrays.
        - Keys: 'pi_p_m', 'pi_p_c', 'pi_r_m', 'pi_r_c', 'q'
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
        Calculate expected responder payoff for a given Q-threshold.

        Wrapper method that interfaces with the Numba-compiled calculation function.
        Used primarily for finding the equilibrium Q-value.

        Parameters
        ----------
        q : float
            Q-threshold value to evaluate.

        Returns
        -------
        float
            Expected responder payoff under the given threshold.

        Raises
        ------
        Exception
            If Monte Carlo simulation has not been run yet.
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
        Find the equilibrium Q-threshold using root-finding methods.

        Solves for q* where Pi_R(q*) = 0, representing the point where the responder
        is indifferent between accepting and rejecting in expectation.

        Parameters
        ----------
        bracket : list, default=[0, 100000]
            Search interval [lower, upper] for root finding.

        Returns
        -------
        float
            Equilibrium Q-threshold (q*) where responder's expected payoff equals zero.

        Notes
        -----
        Uses Brent's method (brentq) for robust root finding. The bracket should be
        chosen to ensure the zero-crossing is within the interval.
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
        Convert simulation arrays to pandas DataFrame.

        Provides a convenient DataFrame representation of simulation results for
        analysis and visualization. Only converts when explicitly requested to
        maintain performance.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'pi_p_m', 'pi_p_c', 'pi_r_m', 'pi_r_c', 'q'
            Each row represents one simulation iteration.

        Raises
        ------
        Exception
            If Monte Carlo simulation has not been run yet.
        """
        if self.simulation_arrays is None:
            raise Exception("Please run the Monte Carlo simulation first.")

        return pd.DataFrame(self.simulation_arrays)

    def solve_and_analyze(self, num_simulations=1000000):
        """
        Orchestrate complete equilibrium analysis of the model.

        Performs Monte Carlo simulation, determines equilibrium case type, solves for
        equilibrium parameters (q*, α*), and calculates welfare metrics. This is the
        main method for model analysis.

        Parameters
        ----------
        num_simulations : int, default=1000000
            Number of Monte Carlo iterations to perform.

        Returns
        -------
        dict
            Equilibrium results containing:
            - case : str
                Equilibrium type ('Case A: Always Bad', 'Case B: Always Good',
                'Case C: Interesting Case', 'Case D: Always Informed')
            - q_star : float
                Equilibrium Q-threshold
            - alpha_star : float or None
                Equilibrium acceptance probability for uninformed responder
            - E_pi_r_m : float
                Expected responder payoff when proposer moderates
            - E_pi_r_c : float
                Expected responder payoff when proposer cherry-picks
            - proposer_expected_payoff : float
                Proposer's equilibrium expected payoff
            - responder_expected_payoff : float
                Responder's equilibrium expected payoff
            - percent_of_moderate_policies : float
                Proportion of cases where proposer moderates in equilibrium
            - elapsed_time : float
                Computation time in seconds

        Notes
        -----
        Case determination logic:
        - Case A: E[π_r^m] ≤ 0 → No policies acceptable
        - Case B: E[π_r^c] ≥ 0 → All policies acceptable
        - Case C: Mixed equilibrium requiring q* solution
        - Case D: λ = 1 → Perfect information
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
        Calculate ex-ante expected payoffs for both players in equilibrium.

        Uses the solved equilibrium parameters to compute welfare metrics, including
        the expected payoffs for proposer and responder, and the frequency of
        moderation versus cherry-picking.

        Returns
        -------
        dict
            Welfare metrics containing:
            - proposer_expected_payoff : float
                Expected payoff for the proposer in equilibrium
            - responder_expected_payoff : float
                Expected payoff for the responder in equilibrium
            - percent_of_moderate_policies : float
                Proportion of simulations where proposer chooses to moderate

        Raises
        ------
        Exception
            If solve_and_analyze() has not been called to determine equilibrium.

        Notes
        -----
        Welfare calculation accounts for:
        - Information structure (informed vs uninformed responders)
        - Equilibrium strategies (q*, α*)
        - Acceptance probabilities based on payoff signs
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
