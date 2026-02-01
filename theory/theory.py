"""
Theoretical predictions for timeliness criticality model.

This module implements analytical solutions from Moran et al. (2024)
"Timeliness criticality in complex systems" arXiv:2309.15070v3.

Key equations implemented:
- Eq. 5 / SI.23: Critical buffer Bc* via Lambert W function
- Eq. 7: Exponential tail exponent α
- Eq. SI.22: Critical exponent α_c*
"""

import numpy as np
from scipy.special import lambertw


def theoretical_Bc(K: int) -> float:
    """
    Compute the critical buffer size Bc* for connectivity K.
    
    Analytical solution from Eq. 5 / SI.23:
        Bc* = -W_{-1}(-1/(e*K))
    
    where W_{-1} is the -1 branch of the Lambert W function.
    
    Parameters
    ----------
    K : int
        Connectivity (number of random neighbors per node).
        
    Returns
    -------
    float
        Critical buffer size Bc*.
        
    Examples
    --------
    >>> theoretical_Bc(5)
    3.994314...
    """
    assert K >= 1, f"Connectivity K must be >= 1, got {K}"
    return -np.real(lambertw(-1.0 / (np.e * K), k=-1))


def theoretical_alpha_c(K: int) -> float:
    """
    Compute the critical tail exponent α_c* for connectivity K.
    
    From Eq. SI.22:
        α_c* = 1 - 1/Bc*
    
    Parameters
    ----------
    K : int
        Connectivity (number of random neighbors per node).
        
    Returns
    -------
    float
        Critical tail exponent α_c*.
        
    Examples
    --------
    >>> theoretical_alpha_c(5)
    0.749644...
    """
    Bc = theoretical_Bc(K)
    alpha_c = 1.0 - 1.0 / Bc
    assert 0 < alpha_c < 1, f"α_c* must be in (0,1), got {alpha_c}"
    return alpha_c


def theoretical_alpha_above_Bc(B: float, K: int) -> float:
    """
    Compute tail exponent α for buffer B > Bc* (stationary phase).
    
    From Eq. 7:
        α = 1 + W_0(-B*K*exp(-B)) / B
    
    where W_0 is the principal branch of Lambert W.
    For B <= Bc*, returns α_c* (the critical exponent).
    
    Parameters
    ----------
    B : float
        Buffer size.
    K : int
        Connectivity.
        
    Returns
    -------
    float
        Tail exponent α. Always <= 1.0 for valid parameters.
        
    Notes
    -----
    At B = Bc*, there's a square-root singularity:
        α - α_c* ~ (B - Bc*)^{1/2}
    """
    Bc = theoretical_Bc(K)
    if B <= Bc:
        return theoretical_alpha_c(K)
    
    arg = -B * K * np.exp(-B)
    w = np.real(lambertw(arg, k=0))
    alpha = 1.0 + w / B
    
    assert alpha <= 1.0, f"Invalid alpha={alpha} > 1.0 for B={B}, K={K}"
    return alpha


def verify_alpha_consistency(B: float, K: int, tol: float = 1e-10) -> bool:
    """
    Verify that α satisfies the self-consistency equation.
    
    The tail exponent must satisfy:
        1 - α = K * exp(-B * α)
    
    Parameters
    ----------
    B : float
        Buffer size.
    K : int
        Connectivity.
    tol : float, optional
        Tolerance for consistency check.
        
    Returns
    -------
    bool
        True if consistency equation holds within tolerance.
    """
    alpha = theoretical_alpha_above_Bc(B, K)
    lhs = 1 - alpha
    rhs = K * np.exp(-B * alpha)
    return abs(lhs - rhs) < tol


def compute_theory_table(K_values: list[int] = None) -> dict:
    """
    Compute theoretical predictions for multiple K values.
    
    Reproduces Table SI.1 from the paper.
    
    Parameters
    ----------
    K_values : list of int, optional
        Connectivity values to compute. Default: [2, 3, ..., 10].
        
    Returns
    -------
    dict
        Dictionary with keys 'K', 'Bc_theory', 'alpha_c_theory' containing arrays.
    """
    if K_values is None:
        K_values = list(range(2, 11))
    
    Bc_values = [theoretical_Bc(K) for K in K_values]
    alpha_c_values = [theoretical_alpha_c(K) for K in K_values]
    
    return {
        'K': np.array(K_values),
        'Bc_theory': np.array(Bc_values),
        'alpha_c_theory': np.array(alpha_c_values)
    }
