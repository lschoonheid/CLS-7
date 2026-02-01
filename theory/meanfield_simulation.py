"""
Mean Field Simulation Module
"""

import numpy as np
from scipy import stats


def meanfield_simulation(N, K, T, B, distribution=None, dist_params=None, return_full_history=False):
    """
    Simulates the delay evolution in a Mean Field (MF) approximation.

    Parameters:
    -----------
    N : int
        Number of nodes (system components).
    K : int
        Number of inputs/neighbors per node.
    T : int
        Number of time steps.
    B : float
        Buffer size.
    distribution : scipy.stats distribution or str, optional
        Distribution to use for noise generation. Can be:
        - A scipy.stats distribution object (e.g., stats.expon, stats.norm)
        - A string name of a scipy.stats distribution (e.g., 'expon', 'norm', 'gamma')
        - None (default): uses exponential distribution with scale=1.0
    dist_params : dict, optional
        Parameters to pass to the distribution's rvs() method.
        For example: {'scale': 1.0}, {'loc': 0, 'scale': 1}, {'a': 2, 'scale': 1}
    return_full_history : bool, optional
        If True, returns (N, T) matrix. If False, returns (T,) vector of mean delays.

    Returns:
    --------
    delays : numpy.ndarray
        Array of delays.

    Examples:
    ---------
    >>> # Using default exponential distribution
    >>> delays = meanfield_simulation(N=1000, K=3, T=100, B=1.0)

    >>> # Using normal distribution
    >>> from scipy import stats
    >>> delays = meanfield_simulation(N=1000, K=3, T=100, B=1.0,
    ...                               distribution=stats.norm,
    ...                               dist_params={'loc': 1, 'scale': 0.5})

    >>> # Using distribution name as string
    >>> delays = meanfield_simulation(N=1000, K=3, T=100, B=1.0,
    ...                               distribution='lognorm',
    ...                               dist_params={'s': -0.5, 'scale': np.exp(-0.5)})
    """
    # Set up the distribution
    if distribution is None:
        # Default: exponential with scale=1.0
        dist = stats.expon
        params = {'scale': 1.0} if dist_params is None else dist_params
    elif isinstance(distribution, str):
        # Get distribution by name from scipy.stats
        if not hasattr(stats, distribution):
            raise ValueError(f"Unknown distribution: {distribution}")
        dist = getattr(stats, distribution)
        params = {} if dist_params is None else dist_params
    else:
        # Assume it's already a scipy.stats distribution object
        dist = distribution
        params = {} if dist_params is None else dist_params

    # Helper function to generate random samples
    def generate_noise(size):
        return dist.rvs(size=size, **params)

    # Initialize state: Delays at t=0 are just the noise
    current_tau = generate_noise(size=N)

    if return_full_history:
        history = np.zeros((N, T))
        history[:, 0] = current_tau
    else:
        # If we only need the mean delay per time step
        history = np.zeros(T)
        history[0] = np.mean(current_tau)

    # Simulation loop
    for t in range(1, T):
        # 1. Select random neighbors for all N nodes (MF assumption: reshuffled every step)
        neighbor_indices = np.random.randint(0, N, size=(N, K))

        # 2. Retrieve the delays of these neighbors from the previous step
        neighbor_delays = current_tau[neighbor_indices]

        # 3. Calculate the maximum delay among neighbors
        max_neighbor_delay = np.max(neighbor_delays, axis=1)

        # 4. Apply Buffer B and Rectification (ReLU)
        buffered_delay = np.maximum(0, max_neighbor_delay - B)

        # 5. Add new intrinsic noise (epsilon)
        epsilon = generate_noise(size=N)
        current_tau = buffered_delay + epsilon

        # Store results
        if return_full_history:
            history[:, t] = current_tau
        else:
            history[t] = np.mean(current_tau)

    return history


def get_order_parameter(mean_delays_series):
    """
    Calculates the order parameter v (velocity of delay growth).
    This corresponds to the slope of the mean delay vs time.

    Parameters:
    -----------
    mean_delays_series : numpy.ndarray
        Array of mean delays over time (1D array).

    Returns:
    --------
    v : float
        Order parameter (velocity of delay growth).
    """
    # Exclude the transient initial phase (first 10% of simulation)
    start_idx = int(len(mean_delays_series) * 0.1)
    return np.mean(np.diff(mean_delays_series[start_idx:]))