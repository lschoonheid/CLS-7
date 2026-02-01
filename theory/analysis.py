"""
Statistical analysis functions for timeliness criticality simulations.

Includes:
- Critical point estimation (constrained and free fitting)
- Finite-size scaling (FSS) analysis
- Hypothesis testing for slope = -1
- Autocorrelation and avalanche statistics
"""

import numpy as np
from scipy.stats import linregress, sem, norm
from scipy.optimize import curve_fit

from .theory import theoretical_Bc


# =============================================================================
# CRITICAL POINT ESTIMATION
# =============================================================================

def estimate_Bc_constrained(
    B_vals: np.ndarray, 
    v_mean: np.ndarray, 
    v_cutoff: float = 0.05
) -> tuple:
    """
    Estimate critical buffer Bc assuming theoretical slope = -1.
    
    From Eq. 6: v = Bc - B, so Bc = v + B for each point in moving phase.
    
    Parameters
    ----------
    B_vals : ndarray
        Array of buffer values.
    v_mean : ndarray
        Mean velocity at each B.
    v_cutoff : float, optional
        Minimum velocity to include in fit (to exclude stationary phase).
        
    Returns
    -------
    Bc_mean : float
        Mean Bc estimate.
    Bc_std : float
        Standard deviation of Bc estimates.
    r2 : float
        Placeholder (0.0 for constrained fit).
    """
    assert len(B_vals) == len(v_mean), "B_vals and v_mean must have same length"
    assert v_cutoff >= 0, f"v_cutoff must be non-negative, got {v_cutoff}"
    
    mask = v_mean > v_cutoff
    if np.sum(mask) < 3:
        return np.nan, np.nan, 0.0
    
    # If v = Bc - B, then Bc = v + B
    Bc_estimates = v_mean[mask] + B_vals[mask]
    return np.mean(Bc_estimates), np.std(Bc_estimates), 0.0


def estimate_Bc_free(
    B_vals: np.ndarray, 
    v_means: np.ndarray, 
    v_cutoff: float = 0.05
) -> tuple:
    """
    Estimate Bc using unconstrained linear regression.
    
    Fits v = slope * B + intercept, then Bc = -intercept / slope.
    
    Parameters
    ----------
    B_vals : ndarray
        Array of buffer values.
    v_means : ndarray
        Mean velocity at each B.
    v_cutoff : float, optional
        Minimum velocity to include.
        
    Returns
    -------
    Bc : float
        Estimated critical buffer.
    slope : float
        Fitted slope (theory predicts -1).
    slope_se : float
        Standard error of slope.
    r_squared : float
        Goodness of fit.
    """
    mask = v_means > v_cutoff
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan, 0.0
    
    B_fit = B_vals[mask]
    v_fit = v_means[mask]
    
    slope, intercept, r, p, se = linregress(B_fit, v_fit)
    
    # Bc is x-intercept: 0 = slope * Bc + intercept
    Bc = -intercept / slope if slope != 0 else np.nan
    
    return Bc, slope, se, r**2


def bootstrap_Bc_estimate(
    B_vals: np.ndarray, 
    v_matrix: np.ndarray, 
    n_bootstrap: int = 1000, 
    v_cutoff: float = 0.05
) -> tuple:
    """
    Bootstrap confidence interval for Bc estimate.
    
    Parameters
    ----------
    B_vals : ndarray
        Buffer values.
    v_matrix : ndarray of shape (M_trials, len(B_vals))
        Velocity matrix from ensemble.
    n_bootstrap : int, optional
        Number of bootstrap samples.
    v_cutoff : float, optional
        Velocity cutoff.
        
    Returns
    -------
    tuple of (median, low_2.5%, high_97.5%)
        Bootstrap confidence interval.
    """
    M = v_matrix.shape[0]
    Bc_samples = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(M, size=M, replace=True)
        v_boot = np.mean(v_matrix[indices, :], axis=0)
        
        Bc_est, _, _ = estimate_Bc_constrained(B_vals, v_boot, v_cutoff)
        if not np.isnan(Bc_est):
            Bc_samples.append(Bc_est)
            
    if not Bc_samples:
        return np.nan, np.nan, np.nan
        
    return tuple(np.percentile(Bc_samples, [50, 2.5, 97.5]))


def test_slope_hypothesis(
    B_vals: np.ndarray, 
    v_means: np.ndarray, 
    v_sem: np.ndarray, 
    expected_slope: float = -1.0, 
    v_cutoff: float = 0.05
) -> tuple:
    """
    Statistical test for H0: slope = -1 (paper's prediction).
    
    Uses weighted least squares with inverse variance weighting.
    
    Parameters
    ----------
    B_vals : ndarray
        Buffer values.
    v_means : ndarray
        Mean velocities.
    v_sem : ndarray
        Standard errors of mean velocities.
    expected_slope : float, optional
        Null hypothesis slope (default: -1.0).
    v_cutoff : float, optional
        Velocity cutoff.
        
    Returns
    -------
    slope : float
        Fitted slope.
    slope_se : float
        Standard error of slope.
    z_score : float
        Z-statistic for deviation from expected slope.
    p_value : float
        Two-tailed p-value.
    reject_null : bool
        True if null hypothesis rejected at α=0.05.
    """
    assert len(B_vals) == len(v_means) == len(v_sem), "Input arrays must have same length"
    assert v_cutoff >= 0, f"v_cutoff must be non-negative, got {v_cutoff}"
    
    mask = (v_means > v_cutoff) & (v_sem > 0)
    if np.sum(mask) < 4:
        return np.nan, np.nan, np.nan, np.nan, False

    B_fit = B_vals[mask]
    v_fit = v_means[mask]
    weights = 1.0 / (v_sem[mask]**2)
    
    # Weighted regression
    coeffs, cov = np.polyfit(B_fit, v_fit, 1, w=np.sqrt(weights), cov=True)
    slope = coeffs[0]
    slope_se = np.sqrt(cov[0, 0])
    
    # Z-test against expected slope
    z_score = (slope - expected_slope) / slope_se
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    
    reject_null = p_value < 0.05
    
    return slope, slope_se, z_score, p_value, reject_null


# =============================================================================
# FINITE-SIZE SCALING
# =============================================================================

def fss_paper_form(N: np.ndarray, Bc_inf: float, a: float, b: float) -> np.ndarray:
    """
    Paper's finite-size scaling formula (Eq. SI.29).
    
    Bc(N) = Bc* - 1/(a + b*ln(N))²
    
    This Brunet-Derrida correction predicts slow (ln N)^{-2} convergence.
    
    Parameters
    ----------
    N : ndarray
        System sizes.
    Bc_inf : float
        Thermodynamic limit Bc*.
    a, b : float
        Fitting parameters.
        
    Returns
    -------
    ndarray
        Bc(N) values.
    """
    return Bc_inf - 1.0 / (a + b * np.log(N))**2


def fss_simple_form(N: np.ndarray, Bc_inf: float, C: float) -> np.ndarray:
    """
    Simplified FSS formula (assumes a=0).
    
    Bc(N) = Bc* - C/(ln N)²
    
    Parameters
    ----------
    N : ndarray
        System sizes.
    Bc_inf : float
        Thermodynamic limit Bc*.
    C : float
        Amplitude parameter.
        
    Returns
    -------
    ndarray
        Bc(N) values.
    """
    return Bc_inf - C / (np.log(N))**2


def fit_fss(
    system_sizes: list, 
    Bc_estimates: np.ndarray, 
    Bc_errors: np.ndarray, 
    use_paper_form: bool = True, 
    K: int = 5
) -> tuple:
    """
    Fit finite-size scaling to extract Bc*(∞).
    
    Parameters
    ----------
    system_sizes : list of int
        System sizes N.
    Bc_estimates : ndarray
        Bc estimates for each N.
    Bc_errors : ndarray
        Standard errors of Bc estimates.
    use_paper_form : bool, optional
        If True, use full paper formula with (a, b). 
        If False, use simplified form.
    K : int, optional
        Connectivity for initial guess.
        
    Returns
    -------
    Bc_inf : float
        Extrapolated Bc*(∞).
    Bc_inf_err : float
        Standard error.
    fit_params : dict
        Fitted parameters.
    chi_squared : float
        Goodness of fit.
    """
    assert len(system_sizes) == len(Bc_estimates) == len(Bc_errors), \
        "All input arrays must have same length"
    assert K >= 1, f"K must be >= 1, got {K}"
    
    N_arr = np.array(system_sizes)
    Bc_arr = np.array(Bc_estimates)
    err_arr = np.array(Bc_errors)
    
    valid = ~np.isnan(Bc_arr) & ~np.isnan(err_arr) & (err_arr > 0)
    N_fit = N_arr[valid]
    Bc_fit = Bc_arr[valid]
    err_fit = err_arr[valid]
    
    if len(N_fit) < 3:
        return np.nan, np.nan, {}, np.nan
    
    Bc_theory = theoretical_Bc(K)
    
    if use_paper_form:
        try:
            popt, pcov = curve_fit(
                fss_paper_form, N_fit, Bc_fit,
                p0=[Bc_theory, 0.4, 0.15],
                sigma=err_fit,
                absolute_sigma=True,
                bounds=([Bc_theory - 0.5, 0.0, 0.0], [Bc_theory + 0.5, 2.0, 1.0]),
                maxfev=5000
            )
            perr = np.sqrt(np.diag(pcov))
            
            Bc_inf, a, b = popt
            Bc_inf_err = perr[0]
            
            residuals = Bc_fit - fss_paper_form(N_fit, *popt)
            chi_sq = np.sum((residuals / err_fit)**2)
            
            return Bc_inf, Bc_inf_err, {
                'a': a, 'b': b, 
                'a_err': perr[1], 'b_err': perr[2]
            }, chi_sq
        except Exception as e:
            print(f"Paper form fit failed: {e}")
            return np.nan, np.nan, {}, np.nan
    else:
        try:
            popt, pcov = curve_fit(
                fss_simple_form, N_fit, Bc_fit,
                p0=[Bc_theory, 5.0],
                sigma=err_fit,
                absolute_sigma=True,
                maxfev=5000
            )
            perr = np.sqrt(np.diag(pcov))
            
            Bc_inf, C = popt
            Bc_inf_err = perr[0]
            
            residuals = Bc_fit - fss_simple_form(N_fit, *popt)
            chi_sq = np.sum((residuals / err_fit)**2)
            
            return Bc_inf, Bc_inf_err, {'C': C, 'C_err': perr[1]}, chi_sq
        except Exception as e:
            print(f"Simple form fit failed: {e}")
            return np.nan, np.nan, {}, np.nan


# =============================================================================
# AUTOCORRELATION ANALYSIS
# =============================================================================

def compute_autocorrelation(history: np.ndarray, max_lag: int = None) -> np.ndarray:
    """
    Compute normalized autocorrelation function of delay history.
    
    C(lag) = <(x(t) - <x>)(x(t+lag) - <x>)> / Var(x)
    
    Parameters
    ----------
    history : ndarray
        Time series of mean delay.
    max_lag : int, optional
        Maximum lag to compute. Default: len(history)//4.
        
    Returns
    -------
    ndarray
        Autocorrelation function C(lag).
    """
    assert len(history) > 1, "history must have at least 2 points"
    
    x = history - np.mean(history)
    var = np.var(history)
    
    if var == 0:
        return np.ones(1)
    
    if max_lag is None:
        max_lag = len(history) // 4
    
    acf = np.zeros(max_lag)
    n = len(history)
    
    for lag in range(max_lag):
        if lag < n:
            acf[lag] = np.mean(x[:n-lag] * x[lag:]) / var
    
    return acf


def stretched_exponential(t: np.ndarray, t_ref: float, beta: float) -> np.ndarray:
    """
    Stretched exponential: exp(-(t/t_ref)^β).
    
    Parameters
    ----------
    t : ndarray
        Time lags.
    t_ref : float
        Characteristic decay time.
    beta : float
        Stretching exponent (β < 1 gives slower-than-exponential decay).
        
    Returns
    -------
    ndarray
        Stretched exponential values.
    """
    return np.exp(-(t / t_ref)**beta)


def fit_autocorrelation(acf: np.ndarray, max_fit_lag: int = None) -> tuple:
    """
    Fit stretched exponential to autocorrelation function.
    
    Paper finds β ≈ 0.82 for the mean delay ACF.
    
    Parameters
    ----------
    acf : ndarray
        Autocorrelation function.
    max_fit_lag : int, optional
        Maximum lag to include in fit.
        
    Returns
    -------
    t_ref : float
        Characteristic decay time (diverges as B → Bc+).
    beta : float
        Stretching exponent.
    """
    if max_fit_lag is None:
        above_threshold = np.where(acf > 0.05)[0]
        if len(above_threshold) > 10:
            max_fit_lag = above_threshold[-1] + 1
        else:
            max_fit_lag = min(len(acf), 1000)
    
    t_vals = np.arange(1, min(max_fit_lag, len(acf)))
    acf_vals = acf[1:min(max_fit_lag, len(acf))]
    
    valid = acf_vals > 0.01
    if np.sum(valid) < 5:
        return np.nan, np.nan
    
    t_fit = t_vals[valid]
    acf_fit = acf_vals[valid]
    
    try:
        popt, _ = curve_fit(
            stretched_exponential, t_fit, acf_fit,
            p0=[len(t_fit)/2, 0.8],
            bounds=([1, 0.1], [len(acf)*10, 2.0]),
            maxfev=5000
        )
        return popt[0], popt[1]
    except:
        return np.nan, np.nan


# =============================================================================
# AVALANCHE STATISTICS
# =============================================================================

def detect_avalanches(history: np.ndarray, threshold: float) -> tuple:
    """
    Detect avalanches where mean delay exceeds threshold.
    
    An avalanche is a continuous period where <τ> > threshold.
    
    Parameters
    ----------
    history : ndarray
        Mean delay time series.
    threshold : float
        Threshold value (typically set to B).
        
    Returns
    -------
    persistence_times : ndarray
        Duration of each avalanche.
    avalanche_sizes : ndarray
        Integrated excess (area above threshold) for each avalanche.
        
    Notes
    -----
    Paper predicts P(t_p) ~ t_p^{-3/2} (random walk return time).
    """
    assert len(history) > 0, "history must not be empty"
    assert threshold >= 0, f"threshold must be non-negative, got {threshold}"
    
    above = history > threshold
    
    persistence_times = []
    avalanche_sizes = []
    
    in_avalanche = False
    start_idx = 0
    
    for i, is_above in enumerate(above):
        if is_above and not in_avalanche:
            in_avalanche = True
            start_idx = i
        elif not is_above and in_avalanche:
            in_avalanche = False
            t_p = i - start_idx
            size = np.sum(history[start_idx:i] - threshold)
            
            if t_p > 0:
                persistence_times.append(t_p)
                avalanche_sizes.append(size)
    
    # Handle case where simulation ends in an avalanche
    if in_avalanche:
        t_p = len(history) - start_idx
        size = np.sum(history[start_idx:] - threshold)
        if t_p > 0:
            persistence_times.append(t_p)
            avalanche_sizes.append(size)
    
    return np.array(persistence_times), np.array(avalanche_sizes)


def fit_power_law_tail(
    data: np.ndarray, 
    x_min: float = None, 
    x_max: float = None
) -> tuple:
    """
    Fit power law P(x) ~ x^{-α} to tail of distribution.
    
    Uses log-binning and linear regression in log-log space.
    
    Parameters
    ----------
    data : ndarray
        Data array.
    x_min, x_max : float, optional
        Range for fitting. Default: 10th and 99th percentiles.
        
    Returns
    -------
    exponent : float
        Power law exponent α.
    exponent_se : float
        Standard error.
    r_squared : float
        Goodness of fit.
    """
    if len(data) < 50:
        return np.nan, np.nan, 0.0
    
    if x_min is None:
        x_min = np.percentile(data, 10)
    if x_max is None:
        x_max = np.percentile(data, 99)
    
    filtered = data[(data >= x_min) & (data <= x_max)]
    if len(filtered) < 30:
        return np.nan, np.nan, 0.0
    
    # Log-binned histogram
    log_bins = np.logspace(np.log10(x_min), np.log10(x_max), 30)
    hist, edges = np.histogram(filtered, bins=log_bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])  # Geometric mean
    
    valid = hist > 0
    if np.sum(valid) < 5:
        return np.nan, np.nan, 0.0
    
    log_x = np.log10(centers[valid])
    log_y = np.log10(hist[valid])
    
    slope, intercept, r, p, se = linregress(log_x, log_y)
    
    # Power law exponent is -slope (since P(x) ~ x^{-α})
    return -slope, se, r**2
