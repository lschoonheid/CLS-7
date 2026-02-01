"""
Ensemble simulation framework for timeliness criticality model.

Provides parallel execution of simulation ensembles over parameter sweeps.
"""

import numpy as np
from scipy.stats import sem

# Optional dependencies with fallbacks
try:
    from joblib import Parallel, delayed, cpu_count
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    def cpu_count():
        """Fallback cpu_count."""
        import os
        return os.cpu_count() or 1

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Fallback tqdm that just returns the iterable."""
        return iterable

from .simulation import run_single_simulation


# Default random seed base for reproducibility
SEED_BASE = 42


def run_ensemble_sweep(
    N: int,
    K: int,
    T: int,
    B_values: np.ndarray,
    M_trials: int,
    burn_in_fraction: float = 0.5,
    n_jobs: int = -1,
    seed_base: int = SEED_BASE
) -> dict:
    """
    Run ensemble of simulations over buffer values.
    
    Executes M_trials independent simulations at each B value,
    parallelized across available CPU cores.
    
    Parameters
    ----------
    N : int
        System size (number of nodes).
    K : int
        Connectivity.
    T : int
        Number of timesteps per simulation.
    B_values : ndarray
        Array of buffer values to sweep.
    M_trials : int
        Number of independent trials at each B.
    burn_in_fraction : float, optional
        Fraction of timesteps to discard as transient.
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all cores minus one.
    seed_base : int, optional
        Base random seed for reproducibility.
        
    Returns
    -------
    dict
        Results dictionary containing:
        - 'B_values': buffer values
        - 'velocities': (M_trials, len(B_values)) array
        - 'alphas': (M_trials, len(B_values)) array
        - 'v_mean', 'v_std', 'v_sem': velocity statistics
        - 'alpha_mean', 'alpha_std', 'alpha_sem': α statistics
        
    Notes
    -----
    Seeds are deterministically generated from seed_base to ensure
    reproducibility while avoiding correlations between trials.
    """
    assert N > 0, f"N must be positive, got {N}"
    assert M_trials >= 1, f"M_trials must be >= 1, got {M_trials}"
    assert len(B_values) > 0, "B_values must not be empty"
    assert 0 <= burn_in_fraction < 1, f"burn_in_fraction must be in [0,1), got {burn_in_fraction}"
    
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)
    
    # Build task list with unique seeds
    tasks = []
    for b_idx, B in enumerate(B_values):
        for m in range(M_trials):
            seed = seed_base + int(N) + b_idx * 10000 + m
            tasks.append((N, K, T, B, seed, burn_in_fraction, b_idx, m))
    
    def task_wrapper(args):
        N, K, T, B, seed, burn_in, b_idx, m = args
        return run_single_simulation(N, K, T, B, seed, burn_in)
    
    # Parallel or serial execution
    if JOBLIB_AVAILABLE:
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(task_wrapper)(task) 
            for task in tqdm(tasks, desc=f"N={N}", leave=False)
        )
    else:
        # Serial fallback
        results_list = [task_wrapper(task) for task in tqdm(tasks, desc=f"N={N} (serial)")]
    
    # Organize results into arrays
    velocities = np.zeros((M_trials, len(B_values)))
    alphas = np.zeros((M_trials, len(B_values)))
    
    for idx, (v, r2, alpha) in enumerate(results_list):
        b_idx = tasks[idx][6]
        m = tasks[idx][7]
        velocities[m, b_idx] = v
        alphas[m, b_idx] = alpha
    
    return {
        'B_values': np.array(B_values),
        'velocities': velocities,
        'alphas': alphas,
        'v_mean': np.mean(velocities, axis=0),
        'v_std': np.std(velocities, axis=0),
        'v_sem': sem(velocities, axis=0),
        'alpha_mean': np.nanmean(alphas, axis=0),
        'alpha_std': np.nanstd(alphas, axis=0),
        'alpha_sem': sem(alphas, axis=0, nan_policy='omit')
    }


def run_multi_K_sweep(
    N: int,
    K_values: list,
    T: int,
    B_range_func,
    M_trials: int,
    burn_in_fraction: float = 0.5,
    n_jobs: int = -1,
    seed_base: int = SEED_BASE
) -> dict:
    """
    Run ensemble sweeps for multiple connectivity values K.
    
    Parameters
    ----------
    N : int
        System size.
    K_values : list of int
        Connectivity values to test.
    T : int
        Timesteps per simulation.
    B_range_func : callable
        Function K -> ndarray that returns B values to sweep for given K.
    M_trials : int
        Trials per B value.
    burn_in_fraction : float, optional
        Transient fraction.
    n_jobs : int, optional
        Parallel jobs.
    seed_base : int, optional
        Base seed.
        
    Returns
    -------
    dict
        Dictionary mapping K -> ensemble results.
    """
    results = {}
    
    for K in tqdm(K_values, desc="Multi-K sweep"):
        B_values = B_range_func(K)
        results[K] = run_ensemble_sweep(
            N=N, K=K, T=T, B_values=B_values,
            M_trials=M_trials, burn_in_fraction=burn_in_fraction,
            n_jobs=n_jobs, seed_base=seed_base + K * 100000
        )
    
    return results


def run_fss_sweep(
    system_sizes: list,
    K: int,
    T: int,
    B_values: np.ndarray,
    M_trials: int,
    burn_in_fraction: float = 0.5,
    n_jobs: int = -1,
    seed_base: int = SEED_BASE
) -> dict:
    """
    Run finite-size scaling sweep over multiple system sizes.
    
    Parameters
    ----------
    system_sizes : list of int
        System sizes N to test.
    K : int
        Connectivity.
    T : int
        Timesteps per simulation.
    B_values : ndarray
        Buffer values to sweep.
    M_trials : int
        Trials per B value.
    burn_in_fraction : float, optional
        Transient fraction.
    n_jobs : int, optional
        Parallel jobs.
    seed_base : int, optional
        Base seed.
        
    Returns
    -------
    dict
        Dictionary mapping N -> ensemble results.
    """
    results = {}
    
    for N in tqdm(system_sizes, desc="FSS sweep (varying N)"):
        results[N] = run_ensemble_sweep(
            N=N, K=K, T=T, B_values=B_values,
            M_trials=M_trials, burn_in_fraction=burn_in_fraction,
            n_jobs=n_jobs, seed_base=seed_base
        )
    
    return results
