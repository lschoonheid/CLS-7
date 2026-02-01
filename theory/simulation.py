"""
Core simulation engine for timeliness criticality model.

Implements the Mean Field dynamics from Eq. 2 of Moran et al. (2024):
    τ_i(t) = [max_j A_{ij}(t-1) τ_j(t-1) - B]^+ + ε_i(t)

Uses Numba JIT compilation for performance when available.
"""

import numpy as np
from scipy.stats import linregress

# Try to import numba for JIT compilation (recommended for performance)
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a no-op decorator
    def njit(*args, **kwargs):
        """Fallback decorator when numba is not installed."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@njit(fastmath=True, cache=True)
def _simulate_core(
    N: int, 
    K: int, 
    T: int, 
    B: float, 
    seed: int, 
    return_full_delays: bool = False
):
    """
    Simulate the timeliness criticality model (Eq. 2 of paper).
    
    Implements Mean Field approximation where each node connects to K 
    uniformly random neighbors at each timestep.
    
    Parameters
    ----------
    N : int
        System size (number of nodes).
    K : int
        Connectivity (number of random neighbors per node).
    T : int
        Number of time steps to simulate.
    B : float
        Buffer size (control parameter). Critical point at Bc*.
    seed : int
        Random seed for reproducibility.
    return_full_delays : bool, optional
        If True, return final delay distribution for α measurement.
        
    Returns
    -------
    mean_delay_history : ndarray of shape (T,)
        Mean delay per node at each timestep.
    final_delays : ndarray of shape (N,) or None
        Final delay values if return_full_delays=True, else None.
        
    Notes
    -----
    The dynamics follow:
    1. Find max delay among K random neighbors
    2. Subtract buffer B and apply positive part: [max - B]^+
    3. Add exponential(1) noise
    
    In the moving phase (B < Bc*), delays grow linearly with velocity v = Bc* - B.
    In the stationary phase (B > Bc*), delays reach a stationary distribution.
    """
    np.random.seed(seed)
    
    # Initialize delays with exponential noise
    delays_current = np.zeros(N)
    delays_next = np.zeros(N)
    
    for i in range(N):
        delays_current[i] = -np.log(np.random.random())  # Exp(1) sample
    
    # History of mean delay per node
    mean_delay_history = np.zeros(T)
    mean_delay_history[0] = np.mean(delays_current)
    
    for t in range(1, T):
        sum_delays = 0.0
        
        for i in range(N):
            # Find maximum delay among K random neighbors (Mean Field)
            max_input = delays_current[np.random.randint(0, N)]
            for _ in range(K - 1):
                neighbor_delay = delays_current[np.random.randint(0, N)]
                if neighbor_delay > max_input:
                    max_input = neighbor_delay
            
            # Apply buffer (Eq. 2: [max - B]^+)
            buffered = max_input - B
            if buffered < 0.0:
                buffered = 0.0
            
            # Add exponential noise
            noise = -np.log(np.random.random())  # Exp(1) sample
            delays_next[i] = buffered + noise
            sum_delays += delays_next[i]
        
        mean_delay_history[t] = sum_delays / N
        
        # Swap buffers
        delays_current, delays_next = delays_next, delays_current
    
    if return_full_delays:
        return mean_delay_history, delays_current.copy()
    return mean_delay_history, None


def simulate_timeliness(
    N: int, 
    K: int, 
    T: int, 
    B: float, 
    seed: int, 
    return_full_delays: bool = False
):
    """
    Simulate the timeliness criticality model (Eq. 2 of paper).
    
    This is a wrapper around the JIT-compiled core function.
    See _simulate_core for full documentation.
    
    Parameters
    ----------
    N : int
        System size (number of nodes).
    K : int
        Connectivity (number of random neighbors per node).
    T : int
        Number of time steps to simulate.
    B : float
        Buffer size (control parameter).
    seed : int
        Random seed for reproducibility.
    return_full_delays : bool, optional
        If True, return final delay distribution.
        
    Returns
    -------
    mean_delay_history : ndarray of shape (T,)
    final_delays : ndarray of shape (N,) or None
    """
    assert N > 0, f"System size N must be positive, got {N}"
    assert K >= 1, f"Connectivity K must be >= 1, got {K}"
    assert T > 0, f"Time steps T must be positive, got {T}"
    assert B >= 0, f"Buffer B must be non-negative, got {B}"
    
    return _simulate_core(N, K, T, B, seed, return_full_delays)


def measure_velocity(history: np.ndarray, burn_in_fraction: float = 0.5) -> tuple:
    """
    Measure velocity (order parameter) from delay history.
    
    The velocity v = d<τ>/dt is the slope of mean delay vs time.
    - v > 0 indicates moving phase (B < Bc*)
    - v ≈ 0 indicates stationary phase (B > Bc*)
    
    Parameters
    ----------
    history : ndarray
        Mean delay history from simulation.
    burn_in_fraction : float, optional
        Fraction of initial timesteps to discard as transient.
        
    Returns
    -------
    slope : float
        Velocity estimate (order parameter).
    r_squared : float
        Goodness of fit for linear regression.
    standard_error : float
        Standard error of slope estimate.
    """
    burn_in = int(len(history) * burn_in_fraction)
    steady_state = history[burn_in:]
    
    assert len(steady_state) > 10, "Insufficient data after burn-in"
    
    t_vals = np.arange(len(steady_state))
    slope, intercept, r, p, se = linregress(t_vals, steady_state)
    
    return slope, r**2, se


def measure_alpha(
    delays: np.ndarray, 
    tau_min_percentile: float = 75, 
    min_samples: int = 100, 
    n_bootstrap: int = 200
) -> tuple:
    """
    Measure exponential tail exponent α from delay distribution.
    
    The stationary distribution has exponential tail: ψ(τ) ~ exp(-α*τ).
    Uses MLE for shifted exponential with bootstrap error estimation.
    
    Parameters
    ----------
    delays : ndarray
        Array of delay values from final simulation state.
    tau_min_percentile : float, optional
        Percentile threshold for tail fitting.
    min_samples : int, optional
        Minimum samples required for reliable estimate.
    n_bootstrap : int, optional
        Number of bootstrap samples for error estimation.
        
    Returns
    -------
    alpha_mle : float
        Maximum likelihood estimate of α.
    alpha_se : float
        Standard error from bootstrap.
        
    Notes
    -----
    Theory predicts:
    - α = α_c* = 1 - 1/Bc* for B < Bc* (critical exponent)
    - α increases with B for B > Bc*
    - Square-root singularity: α - α_c* ~ (B - Bc*)^{1/2}
    """
    if delays is None or len(delays) < min_samples:
        return np.nan, np.nan
    
    assert 0 <= tau_min_percentile <= 100, f"tau_min_percentile must be in [0,100], got {tau_min_percentile}"
    assert min_samples > 0, f"min_samples must be positive, got {min_samples}"
    assert n_bootstrap > 0, f"n_bootstrap must be positive, got {n_bootstrap}"
    
    # Use percentile to ensure we are in the tail
    tau_min = np.percentile(delays, tau_min_percentile)
    tail_data = delays[delays > tau_min]
    
    if len(tail_data) < min_samples // 4:
        return np.nan, np.nan
    
    # MLE for shifted exponential: α = 1 / <τ - τ_min>
    shifted = tail_data - tau_min
    alpha_mle = 1.0 / np.mean(shifted)
    
    # Bootstrap for error estimation
    alpha_boots = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(shifted, size=len(shifted), replace=True)
        if np.mean(sample) > 0:
            alpha_boots.append(1.0 / np.mean(sample))
            
    alpha_se = np.std(alpha_boots) if len(alpha_boots) > 1 else 0.0
    
    return alpha_mle, alpha_se


def run_single_simulation(
    N: int, 
    K: int, 
    T: int, 
    B: float, 
    seed: int, 
    burn_in_fraction: float = 0.5
) -> tuple:
    """
    Run single simulation and extract velocity and α.
    
    Convenience wrapper for parallel execution.
    
    Parameters
    ----------
    N, K, T, B, seed : simulation parameters
    burn_in_fraction : float
        Fraction of timesteps to discard.
        
    Returns
    -------
    v : float
        Velocity (order parameter).
    r2 : float
        R² of velocity fit.
    alpha : float
        Tail exponent estimate.
    """
    history, final_delays = simulate_timeliness(
        N, K, T, B, seed, return_full_delays=True
    )
    v, r2, v_se = measure_velocity(history, burn_in_fraction)
    alpha, alpha_se = measure_alpha(final_delays)
    
    return v, r2, alpha
