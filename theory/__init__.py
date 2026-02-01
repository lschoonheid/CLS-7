"""
Timeliness Criticality Simulation Package
==========================================

Computational validation of the theoretical framework from:
Moran et al. (2024) "Timeliness criticality in complex systems"
arXiv:2309.15070v3

This package provides:
- Analytical solutions for critical parameters (theory module)
- Numba-accelerated Mean Field simulations (simulation module)
- Statistical analysis and hypothesis testing (analysis module)
- Parallel ensemble execution (ensemble module)

Quick Start
-----------
>>> from timeliness_simulation import theoretical_Bc, simulate_timeliness
>>> Bc = theoretical_Bc(K=5)  # Critical buffer for K=5
>>> history, delays = simulate_timeliness(N=1000, K=5, T=10000, B=3.5, seed=42)

Modules
-------
theory
    Analytical predictions from paper equations.
simulation
    Core simulation engine with Numba JIT.
analysis
    Statistical analysis, FSS, autocorrelation, avalanches.
ensemble
    Parallel ensemble execution framework.
"""

from .theory import (
    theoretical_Bc,
    theoretical_alpha_c,
    theoretical_alpha_above_Bc,
    verify_alpha_consistency,
    compute_theory_table
)

from .simulation import (
    simulate_timeliness,
    measure_velocity,
    measure_alpha,
    run_single_simulation
)

from .analysis import (
    estimate_Bc_constrained,
    estimate_Bc_free,
    bootstrap_Bc_estimate,
    test_slope_hypothesis,
    fss_paper_form,
    fss_simple_form,
    fit_fss,
    compute_autocorrelation,
    stretched_exponential,
    fit_autocorrelation,
    detect_avalanches,
    fit_power_law_tail
)

from .ensemble import (
    run_ensemble_sweep,
    run_multi_K_sweep,
    run_fss_sweep,
    SEED_BASE
)

from .meanfield_simulation import (
    meanfield_simulation,
    get_order_parameter
)

__version__ = "1.0.0"
__author__ = "Computational Validation Study"

__all__ = [
    # Theory
    'theoretical_Bc',
    'theoretical_alpha_c', 
    'theoretical_alpha_above_Bc',
    'verify_alpha_consistency',
    'compute_theory_table',
    # Simulation
    'simulate_timeliness',
    'measure_velocity',
    'measure_alpha',
    'run_single_simulation',
    'meanfield_simulation',
    'get_order_parameter',
    # Analysis
    'estimate_Bc_constrained',
    'estimate_Bc_free',
    'bootstrap_Bc_estimate',
    'test_slope_hypothesis',
    'fss_paper_form',
    'fss_simple_form',
    'fit_fss',
    'compute_autocorrelation',
    'stretched_exponential',
    'fit_autocorrelation',
    'detect_avalanches',
    'fit_power_law_tail',
    # Ensemble
    'run_ensemble_sweep',
    'run_multi_K_sweep',
    'run_fss_sweep',
    'SEED_BASE'
]
