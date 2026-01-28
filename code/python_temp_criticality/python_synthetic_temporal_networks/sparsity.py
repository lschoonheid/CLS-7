import numpy as np
import scipy
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle

import sys
import os

from tqdm import tqdm

sys.path.append(os.path.abspath(".."))
from Timeliness_criticality import (
    synthetic_temporal_network,  # pyright: ignore[reportAttributeAccessIssue]
    synthetic_temporal_network_sparsity,  # pyright: ignore[reportAttributeAccessIssue]
)

if __name__ == "__main__":
    B = 3
    K = 5
    n = 10000
    T = 10000
    T_start = 100

    mean_delay_propagation, mean_delays = synthetic_temporal_network(B, K, n, T, T_start)

    buffers = np.arange(0, 9, 0.05)

    # Parallelize the simulation to make it faster
    results: list = Parallel(n_jobs=10)(
        delayed(synthetic_temporal_network)(buffers[i], K, n, T, T_start)
        for i in tqdm(range(len(buffers)), desc="Fetching results")
    )  # pyright: ignore[reportAssignmentType]
    pickle.dump(results, open("results.pkl", "wb"))

    sparsities = [0.9, 0.5, 0.1, 0]

    sparse_results = []
    for sparsity in sparsities:
        sparse_results.append(
            Parallel(n_jobs=10)(
                delayed(synthetic_temporal_network_sparsity)(buffers[i], K, n, T, T_start, sparsity)
                for i in range(len(buffers))
            )
        )

    colors = ["blue", "red", "green", "orange"]

    # Plot
    for i in range(len(buffers)):
        plt.scatter(buffers[i], np.mean(results[i][0]), color=colors[3], s=10, label=f"Sparsity = {100*sparsities[3]}%")
        plt.scatter(
            buffers[i],
            np.mean(sparse_results[2][i][0]),
            color=colors[2],
            s=10,
            label=f"Sparsity = {100*sparsities[2]}%",
        )
        plt.scatter(
            buffers[i],
            np.mean(sparse_results[1][i][0]),
            color=colors[1],
            s=10,
            label=f"Sparsity = {100*sparsities[1]}%",
        )
        plt.scatter(
            buffers[i],
            np.mean(sparse_results[0][i][0]),
            color=colors[0],
            s=10,
            label=f"Sparsity = {100*sparsities[0]}%",
        )

    plt.ylabel(r"$v$")
    plt.xlabel(r"$B$")
    plt.legend(
        [
            f"Sparsity = {100*sparsities[3]}%",
            f"Sparsity = {100*sparsities[2]}%",
            f"Sparsity = {100*sparsities[1]}%",
            f"Sparsity = {100*sparsities[0]}%",
        ]
    )
    plt.title(r"Simple $v$ versus $B$ graph, with sparse STNs")
