# For relative import to work, cd to the python_temp_criticality folder before running this script

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from joblib import Parallel, delayed
import pickle

import sys
import os

from tqdm import tqdm

try:
    sys.path.append(os.path.abspath(".."))
    from Timeliness_criticality import (
        synthetic_temporal_network_sparsity,  # pyright: ignore[reportAttributeAccessIssue]
    )
except ImportError as e:
    print("Error importing modules. Make sure to run this script from the 'python_temp_criticality' folder.")
    exit()


# Configuration
BASELINE_CACHE_FILE = "results.pkl"
SPARSITY_CACHE_FILE = "sparse_results.pkl"
LOAD_CACHED_BASELINE_RESULTS = True
LOAD_CACHED_SPARSITY_RESULTS = True
K = 5
N = 10000
T = 10000
T_START = 100
B_C = 3.674
BUFFERS = np.arange(0, 5, 0.05)
SPARSITIES = np.linspace(0, 1, 21)


def get_sparsities(
    buffers: npt.NDArray[np.float64],
    K: int,
    n: int,
    T: int,
    T_start: int,
    sparsities: npt.NDArray[np.float64],
) -> list:
    """
    Calculate metrics for synthetic temporal networks across different parameter combinations.
    This function runs parallel simulations of synthetic temporal networks with varying
    sparsity levels and buffer sizes, collecting results for analysis.
    Args:
        buffers (npt.NDArray[np.float64]): Array of buffer size values to test.
        K (int): Number of connections or degree parameter for the network.
        n (int): Number of nodes in the network.
        T (int): Total time steps for the simulation.
        T_start (int): Starting time step for measurements.
        sparsities (npt.NDArray[np.float64]): Array of sparsity values to test.
    Returns:
        list: A nested list of results where each element corresponds to a sparsity level,
            containing results for all buffer sizes at that sparsity level. The structure
            is: [[results for buffer_1, ..., buffer_n] at sparsity_1, ...].
    Notes:
        - Uses parallel processing with 10 workers to speed up computation.
        - Progress bars are displayed for both sparsity levels and buffer iterations.
        - Requires the `synthetic_temporal_network_sparsity` function to be defined.
    """

    sparse_results = []
    for sparsity in tqdm(sparsities, desc="Sparsity levels"):
        # Parallelize the simulation to make it faster
        sparse_results.append(
            Parallel(n_jobs=10)(
                delayed(synthetic_temporal_network_sparsity)(buffers[i], K, n, T, T_start, sparsity)
                for i in tqdm(range(len(buffers)), desc=f"Fetching sparsity results for sparsity={sparsity}")
            )
        )
    return sparse_results


def plot(
    sparse_results: list,
    buffers: npt.NDArray[np.float64],
    sparsities: npt.NDArray[np.float64],
    colormap=cm.Paired,  # pyright: ignore[reportAttributeAccessIssue]
):
    """
    Plot mean delay versus buffer size for different network sparsity levels.
    This function creates a scatter plot showing the relationship between buffer size (B)
    and mean delay (v) for various network sparsity values. It includes reference lines
    for the critical buffer size and critical mean delay.
    Parameters
    ----------
    sparse_results : list
        A nested list containing simulation results for different sparsity levels and buffers.
        Structure: sparse_results[sparsity_idx][buffer_idx][0] contains delay measurements.
    buffers : npt.NDArray[np.float64]
        Array of buffer sizes to plot on the x-axis.
    sparsities : npt.NDArray[np.float64]
        Array of sparsity values (between 0 and 1) used in the simulations.
    colormap : matplotlib colormap, optional
        Colormap to use for differentiating sparsity levels. Default is cm.Paired.
    Returns
    -------
    matplotlib display object
        The result of plt.show(), displaying the generated plot.
    Notes
    -----
    - Uses global variables B_C (critical buffer), N, K, T, and T_START for annotations
      and title.
    - The plot includes a vertical dashed line at the critical buffer size (B_C).
    - A horizontal dashed line indicates the critical mean delay.
    - Each sparsity level is represented by a different color according to the colormap.
    - The y-axis is limited to [0, 4] and x-axis spans the provided buffer range.
    """

    norm = mpl.colors.Normalize(vmin=0, vmax=len(sparsities), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=colormap)

    labels = [f"Sparsity = {round(100*(sparsities[i]))}%" for i in range(len(sparsities))]

    # Plot
    for i in range(len(buffers)):
        for j in range(len(sparsities)):
            plt.scatter(
                buffers[i],
                np.mean(sparse_results[j][i][0]),
                color=mapper.to_rgba(j),
                s=10,
                label=labels[j],
            )

    # Plot critical buffer line
    plt.axvline(x=B_C, color="k", linestyle="--", label=r"$B^{*}_c$")
    plt.text(B_C + 0.1, 2, f"$B^{{*}}_c = {B_C}$", rotation=0)

    # Plot critical mean delay line
    plt.axhline(y=B_C, color="r", linestyle="--", label=f"$v_c = {B_C}$")
    plt.text(buffers[0] + 0.1, B_C + 0.1, f"$v_{{B=0}} = {B_C}$", rotation=0)

    plt.ylabel(r"$v$")
    plt.xlabel(r"$B$")
    plt.legend(labels)
    plt.xlim([buffers[0], buffers[-1]])
    plt.ylim(bottom=0, top=4)
    plt.xticks(np.arange(buffers[0], buffers[-1] + 0.5, 0.5))
    plt.grid()
    plt.suptitle(r"$v$ versus $B$, Synthetic Temporal Networks")
    plt.title(f"$N$ = {N}, $K$ = {K}, $T$ = {T}, $T_{{transient}}$ = {T_START}")
    return plt.show()


if __name__ == "__main__":
    # Load cached sparsity results if they exist
    if LOAD_CACHED_SPARSITY_RESULTS:
        try:
            sparse_results = pickle.load(open(SPARSITY_CACHE_FILE, "rb"))
            print(f"Loaded existing sparse results from {SPARSITY_CACHE_FILE}")
        except FileNotFoundError:
            print("No existing sparse results found, running simulations...")
            sparse_results: list = get_sparsities(BUFFERS, K, N, T, T_START, SPARSITIES)
    else:
        sparse_results: list = get_sparsities(BUFFERS, K, N, T, T_START, SPARSITIES)
    pickle.dump(sparse_results, open(SPARSITY_CACHE_FILE, "wb"))

    plot(sparse_results, BUFFERS, SPARSITIES)
