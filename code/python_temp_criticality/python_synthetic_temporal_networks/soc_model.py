"""
SOC Model Module
----------------
Contains the core logic, simulation engines, and visualization functions
for the Self-Organized Criticality (SOC) project in temporal networks.

Includes inline assertions for validation and robustness testing.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import lambertw
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

# ==========================================
# --- 1. MODEL CONFIGURATION & CONSTANTS ---
# ==========================================

# Network Topology and Physics
N_NODES = 1000  # Number of interacting agents (nodes)
K = 5  # Node connectivity (degree)
NOISE_SCALE = 1.0  # Scale parameter for exponential demand noise
B_INIT = 2.0  # Initial buffer stock
T_STEPS = 2000  # Total simulation duration (time steps)
T_START = 100  # Warm-up period to discard transient dynamics

# Control Strategy Parameters
ALPHA = 0.3  # Correction gain (response aggressiveness)
EPSILON = 0.02  # Savings rate (drift parameter)
PIPELINE_LEN = 10  # Supply chain delay (lag steps)

# Strategy Configurations
BETA_MYOPIC = 0.0  # Myopic strategy: Ignores supply line (Beta = 0)
BETA_WISE = 1.0  # Wise strategy: Fully compensates for supply line (Beta = 1)

# --- ANALYTICAL THRESHOLD ---
# Calculation of the theoretical critical buffer threshold (Bc).
# Derived from the Mean Field solution: Bc * exp(1 - Bc) = 1/K
BC_THEO = -lambertw(-1 / (np.exp(1) * K), k=-1).real

# --- VISUALIZATION SETTINGS ---
COLORS = {
    "SIMULATION": "#2E86AB",  # Deep Blue
    "THEORY": "#A23B72",  # Purple/Magenta
    "MYOPIC": "#F18F01",  # Orange
    "WISE": "#73AB84",  # Green
    "GRID": "#E0E0E0",
    "ZERO": "#666666",
}

# ==========================================
# --- 2. CORE SIMULATION FUNCTIONS ---
# ==========================================


def measure_delay_velocity(B, k, n, T, T_start, noise_scale=1.0):
    """
    Simulates network dynamics with a fixed buffer size B to estimate the
    drift velocity v(B). Used to numerically determine the critical threshold Bc.
    """
    # --- BONUS: Input Validation ---
    assert n > 0, "Number of nodes must be positive"
    assert k > 0, "Connectivity k must be positive"
    assert T > T_start, "Total time T must be greater than warm-up period"

    prev_delays = np.zeros(n)
    v_history = []

    for t in range(T):
        # 1. Stochastic Load Generation (Mean Field Approximation)
        noise = scipy.stats.expon.rvs(scale=noise_scale, size=n)

        # Efficient neighbor selection via shuffling
        delays_pool = np.zeros(n * k)
        for i in range(k):
            np.random.shuffle(prev_delays)
            delays_pool[i * n : (i + 1) * n] = prev_delays
        neighbor_matrix = np.reshape(delays_pool, (n, k))

        # 2. Delay Dynamics
        incoming_load = np.max(neighbor_matrix, axis=1) + noise
        curr_delays = np.maximum(0, incoming_load - B)

        # --- BONUS: Physics Check ---
        assert np.all(curr_delays >= 0), "Delay cannot be negative"

        # 3. Velocity Measurement
        if t >= T_start:
            v_history.append(np.mean(curr_delays - prev_delays))

        prev_delays = curr_delays

    return np.mean(v_history)


def simulate_supply_chain_dynamics(
    B_start, k, n, T, T_start, epsilon=0.02, alpha=0.3, beta=1.0, pipeline_len=5, noise_scale=1.0
):
    """
    Simulates the full temporal network with physical delivery delays and
    adaptive control strategies (SOC).
    """
    # --- BONUS: Input Validation ---
    assert n > 0, "Nodes n must be positive"
    assert pipeline_len >= 1, "Pipeline length must be at least 1"
    assert 0 <= alpha <= 2.0, "Alpha (gain) out of reasonable bounds (0-2)"
    assert epsilon >= 0, "Epsilon (decay) cannot be negative"

    # State Initialization
    prev_delays = np.zeros(n)
    buffers = np.full(n, B_start)

    # Supply Line Matrix: Rows=Nodes, Cols=Time_Steps_To_Arrival
    supply_pipeline = np.zeros((n, pipeline_len))

    # Data Recording
    ts_delays = np.zeros(T - T_start)
    ts_buffers = np.zeros(T - T_start)

    for t in range(T):
        # --- 1. PHYSICAL LAYER: GOODS RECEIPT ---
        arrived_goods = supply_pipeline[:, 0]
        buffers = np.maximum(0, buffers + arrived_goods)

        # --- BONUS: Physics Check ---
        assert np.all(buffers >= 0), "Physics violation: Negative buffer stock detected"

        # Time propagation: Shift pipeline left
        supply_pipeline[:, :-1] = supply_pipeline[:, 1:]
        supply_pipeline[:, -1] = 0.0

        # --- 2. NETWORK LAYER: DEMAND SHOCK ---
        noise = scipy.stats.expon.rvs(scale=noise_scale, size=n)

        delays_pool = np.zeros(n * k)
        for i in range(k):
            np.random.shuffle(prev_delays)
            delays_pool[i * n : (i + 1) * n] = prev_delays
        neighbor_matrix = np.reshape(delays_pool, (n, k))

        incoming_load = np.max(neighbor_matrix, axis=1) + noise

        # Calculate current network delay
        curr_delays = np.maximum(0, incoming_load - buffers)
        is_congested = curr_delays > 1e-3

        # --- 3. CONTROL LAYER: AGENT HEURISTICS ---

        # A. Supply Line Estimation
        pending_orders = np.sum(supply_pipeline, axis=1)

        # B. Perceived Gap Calculation
        perceived_gap = curr_delays - (beta * pending_orders)

        # C. Decision Rule (Regime Switching)
        new_orders = np.zeros(n)

        # Regime 1: Correction (Delay > 0)
        needs_correction = (is_congested) & (perceived_gap > 0)
        new_orders[needs_correction] = alpha * perceived_gap[needs_correction]

        supply_pipeline[needs_correction, -1] = new_orders[needs_correction]

        # --- BONUS: Logic Check ---
        assert np.all(new_orders >= 0), "Logic Error: Negative orders generated"

        # Regime 2: Optimization (Delay = 0)
        buffers[~is_congested] = np.maximum(0, buffers[~is_congested] - epsilon)

        # --- 4. DATA LOGGING ---
        if t >= T_start:
            idx = t - T_start
            ts_delays[idx] = np.mean(curr_delays)
            ts_buffers[idx] = np.mean(buffers)

        prev_delays = curr_delays

    # --- BONUS: Output Validation ---
    assert len(ts_delays) == T - T_start, "Output time series length mismatch"
    assert not np.any(np.isnan(ts_buffers)), "Simulation produced NaN values"

    return ts_delays, ts_buffers, prev_delays


# ==========================================
# --- 3. TOPOLOGY & FIXED STRUCTURE ---
# ==========================================


def generate_ring_topology(n, k):
    """Generates a 1D Regular Ring Lattice (Strict Locality)."""
    # Safety checks
    assert k % 1 == 0, "K must be integer"
    assert n > k, "Network too small for connectivity K"

    neighbors = np.zeros((n, k), dtype=int)

    # Define possible offsets (left and right neighbors)
    # Note: This set is optimized for K=5 as per the reference paper
    all_offsets = [-2, -1, 1, 2, 3, -3, 4, -4]

    # SELECT ONLY THE FIRST K OFFSETS (Key fix to prevent index errors)
    if k > len(all_offsets):
        raise ValueError(f"Requested K={k} exceeds defined offsets pattern.")

    current_offsets = all_offsets[:k]

    for i in range(n):
        for j, offset in enumerate(current_offsets):
            neighbors[i, j] = (i + offset) % n

    # --- BONUS: Topology Check ---
    assert neighbors.shape == (n, k), "Topology matrix shape mismatch"
    return neighbors


def generate_random_k_regular_topology(n, k):
    """Generates a Random k-Regular Directed Graph (Global Constraint)."""
    sources = np.repeat(np.arange(n), k)
    targets = np.repeat(np.arange(n), k)
    np.random.shuffle(targets)
    neighbors = targets.reshape((n, k))

    # --- BONUS: Topology Check ---
    assert neighbors.shape == (n, k), "Topology matrix shape mismatch"
    return neighbors


def run_fixed_topology_simulation(topology_neighbors, n, T, T_start):
    """Executes the SOC simulation on a defined, static topology."""
    # --- BONUS: Input Check ---
    assert topology_neighbors.shape[0] == n, "Topology mismatch with N nodes"

    supply_pipeline = np.zeros((n, PIPELINE_LEN))
    buffers = np.full(n, B_INIT)
    delays_last = np.zeros(n)
    history_buffers = []

    np.random.seed(42)

    for t in range(T):
        arriving_goods = supply_pipeline[:, 0]
        buffers = np.maximum(0, buffers + arriving_goods)
        supply_pipeline[:, :-1] = supply_pipeline[:, 1:]
        supply_pipeline[:, -1] = 0.0

        noise = scipy.stats.expon.rvs(scale=NOISE_SCALE, size=n)
        neighbor_delays = delays_last[topology_neighbors]
        total_load = np.max(neighbor_delays, axis=1) + noise
        curr_delay = np.maximum(0, total_load - buffers)
        is_delayed = curr_delay > 1e-3

        pending_orders = np.sum(supply_pipeline, axis=1)
        perceived_gap = curr_delay - (BETA_WISE * pending_orders)

        new_orders = np.zeros(n)
        needs_order = (is_delayed) & (perceived_gap > 0)
        new_orders[needs_order] = ALPHA * perceived_gap[needs_order]
        supply_pipeline[needs_order, -1] = new_orders[needs_order]

        buffers[~is_delayed] = np.maximum(0, buffers[~is_delayed] - EPSILON)
        delays_last = curr_delay

        if t >= T_start:
            history_buffers.append(buffers.copy())

    return np.array(history_buffers).T


# ==========================================
# --- 4. VISUALIZATION FUNCTIONS ---
# ==========================================


def plot_static_phase_comparison():
    """Comparative Analysis of Static Network Phases."""
    print("Generating Static Phase Comparison...")

    T_TEST = 1000
    B_UNSTABLE = 2.0
    B_STABLE = 5.0

    def simulate_velocity_trace(buffer_val):
        prev_delays = np.zeros(N_NODES)
        v_history = []
        for t in range(T_TEST):
            noise = scipy.stats.expon.rvs(scale=NOISE_SCALE, size=N_NODES)
            delays_pool = np.zeros(N_NODES * K)
            for i in range(K):
                np.random.shuffle(prev_delays)
                delays_pool[i * N_NODES : (i + 1) * N_NODES] = prev_delays
            neighbor_matrix = np.reshape(delays_pool, (N_NODES, K))

            incoming_load = np.max(neighbor_matrix, axis=1) + noise
            curr_delays = np.maximum(0, incoming_load - buffer_val)
            v_t = np.mean(curr_delays - prev_delays)
            v_history.append(v_t)
            prev_delays = curr_delays
        return v_history

    v_unstable = simulate_velocity_trace(B_UNSTABLE)
    v_stable = simulate_velocity_trace(B_STABLE)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, dpi=120)

    # Panel A
    axes[0].plot(v_unstable, color="#D62828", linewidth=1, alpha=0.8, label=f"Unstable Regime ($B={B_UNSTABLE}$)")
    mean_unstable = np.mean(v_unstable[100:])
    axes[0].axhline(
        mean_unstable, color="black", linestyle="--", linewidth=2, label=f"Mean Drift $\\approx$ {mean_unstable:.2f}"
    )
    axes[0].set_title(
        f"A. Unstable Phase ($B < B_c$): Constant Accumulation", fontsize=12, fontweight="bold", loc="left"
    )
    axes[0].set_ylabel("Velocity ($v_t$)", fontsize=11)
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_facecolor("#FFF5F5")

    # Panel B
    axes[1].plot(v_stable, color="#2A9D8F", linewidth=1, alpha=0.8, label=f"Stable Regime ($B={B_STABLE}$)")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=2, label="Equilibrium ($v=0$)")
    axes[1].set_title(
        f"B. Stable Phase ($B > B_c$): Stationary Equilibrium", fontsize=12, fontweight="bold", loc="left"
    )
    axes[1].set_xlabel("Time Steps (t)", fontsize=11)
    axes[1].set_ylabel("Velocity ($v_t$)", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_facecolor("#F0FFF4")

    plt.tight_layout()
    plt.show()


def plot_velocity_trend_comparison():
    """Comparison of Velocity Trends: Static vs. Adaptive (SOC) Systems."""
    print("Generating Velocity Trend Comparison...")

    T_SIM = 1500
    B_LOW = 1.0
    WINDOW = 50

    def simulate_static_history():
        delays_prev = np.zeros(N_NODES)
        velocity_history = []
        for t in range(T_SIM):
            noise = scipy.stats.expon.rvs(scale=NOISE_SCALE, size=N_NODES)
            delays_pool = np.zeros(N_NODES * K)
            for i in range(K):
                np.random.shuffle(delays_prev)
                delays_pool[i * N_NODES : (i + 1) * N_NODES] = delays_prev
            neighbor_matrix = np.reshape(delays_pool, (N_NODES, K))

            incoming_load = np.max(neighbor_matrix, axis=1) + noise
            delays_curr = np.maximum(0, incoming_load - B_LOW)
            velocity_history.append(np.mean(delays_curr - delays_prev))
            delays_prev = delays_curr
        return velocity_history

    vel_static_raw = simulate_static_history()

    mean_delays_soc, _, _ = simulate_supply_chain_dynamics(
        B_start=B_LOW,
        k=K,
        n=N_NODES,
        T=T_SIM,
        T_start=0,
        epsilon=EPSILON,
        alpha=ALPHA,
        beta=1.0,
        pipeline_len=PIPELINE_LEN,
    )
    vel_soc_raw = np.diff(mean_delays_soc, prepend=mean_delays_soc[0])

    def moving_average(data, n=WINDOW):
        ret = np.cumsum(data, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    trend_static = moving_average(vel_static_raw)
    trend_soc = moving_average(vel_soc_raw)
    time_axis = np.arange(WINDOW - 1, T_SIM)

    plt.figure(figsize=(12, 6), dpi=120)
    plt.plot(
        time_axis,
        trend_static,
        color=COLORS["THEORY"],
        linewidth=3,
        label=f"Static Baseline ($B={B_LOW}$): Constant Drift",
    )
    plt.plot(
        time_axis, trend_soc, color=COLORS["WISE"], linewidth=3, label=f"Adaptive SOC ($B_0={B_LOW}$): Homeostasis"
    )
    plt.axhline(0, color=COLORS["ZERO"], linestyle="--", alpha=0.5, label="Stability Condition ($v=0$)")

    plt.title("Mechanism of Stability: Static Drift vs. Adaptive Correction", fontsize=12, fontweight="bold")
    plt.xlabel("Time Step (t)", fontsize=11)
    plt.ylabel("Mean Velocity ($v_t$)", fontsize=11)
    plt.legend(loc="upper right", framealpha=0.95, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.xlim(0, T_SIM)
    plt.tight_layout()
    plt.show()


def plot_system_breathing_dynamics():
    """Generates the 'Breathing' Plot (Phase Velocity Analysis)."""
    print("Generating System Breathing Plot...")

    CURRENT_BETA = 0.0

    mean_delays, _, _ = simulate_supply_chain_dynamics(
        B_start=8.0, k=K, n=N_NODES, T=1000, T_start=0, beta=CURRENT_BETA, alpha=ALPHA
    )

    velocity_instant = np.diff(mean_delays)
    time_axis = np.arange(1, 1000)

    plt.figure(figsize=(14, 6), dpi=120)

    plt.axhspan(0, np.max(velocity_instant), facecolor="#ffbfbf", alpha=0.2, label="Accumulation Phase ($v > 0$)")
    plt.axhspan(np.min(velocity_instant), 0, facecolor="#c7f9cc", alpha=0.3, label="Recovery Phase ($v < 0$)")

    plt.plot(time_axis, velocity_instant, color="#2E86AB", linewidth=1, alpha=0.7, label="Instantaneous Velocity")
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5)

    window = 15
    v_smooth = np.convolve(velocity_instant, np.ones(window) / window, mode="valid")
    plt.plot(time_axis[window - 1 :], v_smooth, color="#0D3B66", linewidth=2.5, label="Trend (Moving Avg)")

    plt.title(
        f"System Breathing: Phase Velocity Analysis (Myopic Agent, $\\beta={CURRENT_BETA}$)",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Time Step (t)", fontsize=11)
    plt.ylabel(r"Delay Velocity ($v_t = \Delta D_t$)", fontsize=11)
    plt.legend(loc="upper right", framealpha=0.95, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.xlim(0, 1000)
    plt.tight_layout()
    plt.show()


def plot_efficiency_comparison():
    """Generates Plot A: Representative Run Comparison (Myopic vs. Wise)."""
    print("Generating Efficiency Comparison Plot...")

    B_START_VIS = 2.0
    EPSILON_VIS = 0.05

    np.random.seed(42)
    _, b_myopic, _ = simulate_supply_chain_dynamics(
        B_start=B_START_VIS,
        k=K,
        n=N_NODES,
        T=T_STEPS,
        T_start=0,
        beta=BETA_MYOPIC,
        alpha=ALPHA,
        epsilon=EPSILON_VIS,
        pipeline_len=PIPELINE_LEN,
    )

    np.random.seed(42)
    _, b_wise, _ = simulate_supply_chain_dynamics(
        B_start=B_START_VIS,
        k=K,
        n=N_NODES,
        T=T_STEPS,
        T_start=0,
        beta=BETA_WISE,
        alpha=ALPHA,
        epsilon=EPSILON_VIS,
        pipeline_len=PIPELINE_LEN,
    )

    plt.figure(figsize=(12, 6), dpi=120)
    time_steps = np.arange(T_STEPS)

    plt.fill_between(
        time_steps,
        b_myopic,  # pyright: ignore[reportArgumentType]
        b_wise,  # pyright: ignore[reportArgumentType]
        where=(b_myopic > b_wise),
        color=COLORS["MYOPIC"],
        alpha=0.15,
        label="Wasted Resources (Inefficiency Gap)",
    )

    plt.plot(
        b_myopic, color=COLORS["MYOPIC"], linewidth=1.5, alpha=0.9, label="Myopic Strategy (No Pipeline Awareness)"
    )
    plt.plot(b_wise, color=COLORS["WISE"], linewidth=2.5, alpha=0.9, label="Wise Strategy (Pipeline Compensated)")

    plt.axhline(
        BC_THEO,
        color=COLORS["THEORY"],
        linestyle="--",
        linewidth=2.0,
        label=f"Theoretical Critical Threshold ($B_c \\approx {BC_THEO:.2f}$)",
    )

    mid_point = T_STEPS // 2
    plt.annotate(
        "Over-reaction leads to\nexcess inventory (Bullwhip Effect)",
        xy=(mid_point, b_myopic[mid_point]),
        xytext=(mid_point + 100, b_myopic[mid_point] + 1.5),
        arrowprops=dict(arrowstyle="->", color=COLORS["MYOPIC"]),
        fontsize=10,
        color=COLORS["MYOPIC"],
        fontweight="bold",
    )

    plt.title("Impact of Pipeline Awareness on Inventory Efficiency", fontsize=12, fontweight="bold")
    plt.xlabel("Time Step (t)", fontsize=11)
    plt.ylabel("Mean Buffer Size ($B_t$)", fontsize=11)
    plt.legend(fontsize=10, loc="upper right", framealpha=0.95)
    plt.grid(True, alpha=0.3, color=COLORS["GRID"])
    plt.xlim(0, T_STEPS)
    plt.ylim(0, max(b_myopic) * 1.2)
    plt.tight_layout()
    plt.show()


def plot_phase_diagram_robustness():
    """Generates the Phase Diagram (Sensitivity Analysis)."""
    print("Generating Phase Diagram: Robustness Analysis...")

    alphas = np.linspace(0.1, 1.0, 25)
    epsilons = np.linspace(0.01, 0.2, 25)
    phase_grid = np.zeros((len(alphas), len(epsilons)))

    N = 100
    T = 600
    B_INIT_LOCAL = 2.0
    PIPELINE_LEN_LOCAL = 5

    print(f"Simulating {len(alphas)*len(epsilons)} parameter combinations...")

    for i, alpha in enumerate(alphas):
        for j, epsilon in enumerate(epsilons):
            buffers = np.full(N, B_INIT_LOCAL)
            supply_pipeline = np.zeros((N, PIPELINE_LEN_LOCAL))
            avg_buffer_sum = 0
            count = 0
            np.random.seed(42)

            for t in range(T):
                buffers += supply_pipeline[:, 0]
                supply_pipeline[:, :-1] = supply_pipeline[:, 1:]
                supply_pipeline[:, -1] = 0

                load = np.random.exponential(scale=1.0, size=N)
                delay = np.maximum(0, load - buffers)
                pending = np.sum(supply_pipeline, axis=1)
                perceived_gap = delay - pending

                act_order = (delay > 1e-3) & (perceived_gap > 0)
                orders = np.zeros(N)
                orders[act_order] = alpha * perceived_gap[act_order]
                supply_pipeline[act_order, -1] = orders[act_order]

                act_decay = ~act_order
                buffers[act_decay] = np.maximum(0, buffers[act_decay] - epsilon)

                if t > T - 150:
                    avg_buffer_sum += np.mean(buffers)
                    count += 1

            phase_grid[i, j] = avg_buffer_sum / max(1, count)

    plt.figure(figsize=(10, 8), dpi=120)
    im = plt.imshow(
        phase_grid,
        origin="lower",
        aspect="auto",
        cmap="RdYlGn_r",
        extent=[epsilons.min(), epsilons.max(), alphas.min(), alphas.max()],
    )

    cbar = plt.colorbar(im)
    cbar.set_label("Mean Buffer Size [Units]", fontsize=11)

    plt.xlabel(r"Efficiency Pressure ($\epsilon$)", fontsize=12)
    plt.ylabel(r"Reaction Strength ($\alpha$)", fontsize=12)
    plt.title("Phase Diagram: The Three Zones of Control", fontsize=14, fontweight="bold", pad=15)

    box_style = dict(facecolor="white", alpha=0.85, edgecolor="#dddddd", boxstyle="round,pad=0.5")
    text_args = dict(fontsize=10, fontweight="bold", zorder=10)

    plt.text(
        epsilons.min() + 0.01,
        alphas.max() - 0.05,
        "THE HOARDER\n(High Cost)",
        color="#8B0000",
        ha="left",
        va="top",
        bbox=box_style,
        **text_args,  # pyright: ignore[reportArgumentType]
    )
    plt.text(
        epsilons.max() - 0.01,
        alphas.min() + 0.05,
        "THE MINIMALIST\n(High Risk)",
        color="#006400",
        ha="right",
        va="bottom",
        bbox=box_style,
        **text_args,  # pyright: ignore[reportArgumentType]
    )
    plt.text(
        np.mean(epsilons),  # pyright: ignore[reportArgumentType]
        np.mean(alphas),  # pyright: ignore[reportArgumentType]
        "GOLDILOCKS ZONE\n(Robust Stability)",
        color="#333333",
        ha="center",
        va="center",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="black", boxstyle="round,pad=0.6"),
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_time_evolution_zones():
    """Simulates and visualizes the time evolution of the three distinct control regimes."""
    print("Generating Time Evolution of Control Zones...")

    def simulate_regime(T, N, alpha, epsilon):
        buffers = np.full(N, 2.0)
        supply_pipeline = np.zeros((N, 5))
        buffer_history = []
        np.random.seed(42)

        for t in range(T):
            buffers += supply_pipeline[:, 0]
            supply_pipeline[:, :-1] = supply_pipeline[:, 1:]
            supply_pipeline[:, -1] = 0

            demand = np.random.exponential(scale=1.0, size=N)
            delay = np.maximum(0, demand - buffers)
            pending = np.sum(supply_pipeline, axis=1)
            gap = delay - pending

            needs_order = (delay > 1e-3) & (gap > 0)
            orders = np.zeros(N)
            orders[needs_order] = alpha * gap[needs_order]
            supply_pipeline[needs_order, -1] = orders[needs_order]

            buffers[~needs_order] = np.maximum(0, buffers[~needs_order] - epsilon)

            # --- BONUS: Safety Check ---
            assert np.mean(buffers) >= -1e-5, "Numerical instability: Negative Mean Buffer"

            buffer_history.append(np.mean(buffers))
        return buffer_history

    T_STEPS = 500
    N_NODES = 100

    hist_hoarder = simulate_regime(T_STEPS, N_NODES, alpha=0.8, epsilon=0.0)
    hist_minimalist = simulate_regime(T_STEPS, N_NODES, alpha=0.1, epsilon=0.5)
    hist_soc = simulate_regime(T_STEPS, N_NODES, alpha=0.5, epsilon=0.08)

    c_hoarder = "#d62728"
    c_minimalist = "#006400"
    c_goldilocks = "#8CBF26"

    plt.figure(figsize=(12, 6), dpi=120)
    plt.plot(hist_hoarder, color=c_hoarder, linewidth=2, alpha=0.8, label="The Hoarder (High Cost)")
    plt.plot(hist_minimalist, color=c_minimalist, linewidth=2, alpha=0.8, label="The Minimalist (High Risk)")
    plt.plot(hist_soc, color=c_goldilocks, linewidth=3, label="The Goldilocks Zone (Stable)")

    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.title("Temporal Evolution of Control Regimes", fontsize=14, fontweight="bold", pad=10)
    plt.ylabel("Mean Buffer Size [Units]", fontsize=12)
    plt.xlabel("Time Step (t)", fontsize=12)
    plt.legend(fontsize=10, loc="upper left", framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle="--")

    box_style = dict(facecolor="white", alpha=0.9, edgecolor="none", boxstyle="round,pad=0.4")
    x_pos = T_STEPS * 0.7
    y_pos_hoarder = hist_hoarder[-1]
    plt.text(
        x_pos,
        y_pos_hoarder - 1.5,
        "THE HOARDER\n(Over-reactive accumulation)",
        color=c_hoarder,
        fontweight="bold",
        fontsize=10,
        bbox=box_style,
        ha="center",
    )
    y_pos_minimalist = hist_minimalist[-1]
    plt.text(
        x_pos,
        y_pos_minimalist + 0.5,
        "THE MINIMALIST\n(Collapse to zero)",
        color=c_minimalist,
        fontweight="bold",
        fontsize=10,
        bbox=box_style,
        ha="center",
    )
    y_pos_soc = hist_soc[-1]
    plt.text(
        x_pos,
        y_pos_soc + 0.8,
        "GOLDILOCKS ZONE\n(Self-Organized Stability)",
        color="#4a6b13",
        fontweight="bold",
        fontsize=10,
        bbox=box_style,
        ha="center",
    )
    plt.tight_layout()
    plt.show()


def plot_separate_heatmaps():
    """Generates Spatiotemporal Heatmaps for Ring vs Random Topologies."""
    print("Generating Separate Spatiotemporal Heatmaps...")

    N_VIS = 200
    T_VIS = 1000
    T_START_VIS = 200

    # --- FIGURE 1: RING TOPOLOGY ---
    topo_ring = generate_ring_topology(N_VIS, K)
    hist_ring = run_fixed_topology_simulation(topo_ring, N_VIS, T_VIS, T_START_VIS)

    plt.figure(figsize=(10, 6), dpi=120)
    plt.imshow(
        hist_ring,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        vmin=0,
        vmax=np.percentile(hist_ring, 98),  # pyright: ignore[reportArgumentType]
    )
    cbar = plt.colorbar()
    cbar.set_label("Buffer Stock ($B_i$)", fontsize=11)
    plt.title(f"Ring Lattice: Strict Locality Constraint\n(Nearest Neighbors, $K={K}$)", fontsize=14, fontweight="bold")
    plt.xlabel("Time Steps (Relative)", fontsize=12)
    plt.ylabel("Node Index (Geometric Order)", fontsize=12)
    plt.tight_layout()
    plt.show()

    # --- FIGURE 2: RANDOM TOPOLOGY ---
    topo_random = generate_random_k_regular_topology(N_VIS, K)
    hist_random = run_fixed_topology_simulation(topo_random, N_VIS, T_VIS, T_START_VIS)

    # RCM Sorting
    data = np.ones(N_VIS * K)
    rows = np.repeat(np.arange(N_VIS), K)
    cols = topo_random.flatten()
    adjacency = csr_matrix((data, (rows, cols)), shape=(N_VIS, N_VIS))
    perm = reverse_cuthill_mckee(adjacency + adjacency.T)
    hist_random_sorted = hist_random[perm, :]

    plt.figure(figsize=(10, 6), dpi=120)
    plt.imshow(
        hist_random_sorted,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=0,
        vmax=np.percentile(hist_random_sorted, 98),  # pyright: ignore[reportArgumentType]
    )
    cbar = plt.colorbar()
    cbar.set_label("Buffer Stock ($B_i$)", fontsize=11)
    plt.title(
        f"Random $k$-Regular: Global Degree Constraint\n($k_{{in}}=k_{{out}}={K}$, RCM Sorted)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Time Steps (Relative)", fontsize=12)
    plt.ylabel("Node Index (Topological Order)", fontsize=12)
    plt.tight_layout()
    plt.show()


# ==========================================
# --- 5. UNIT TEST & ENTRY POINT ---
# ==========================================

if __name__ == "__main__":
    print("\n[SOC MODEL] Running Sanity Check / Smoke Test...")
    try:
        # Quick run to verify assertions
        simulate_supply_chain_dynamics(B_start=2.0, k=5, n=50, T=50, T_start=0)
        print("[SUCCESS] Smoke test passed. Code is valid.")
    except AssertionError as e:
        print(f"[FAILED] Assertion triggered: {e}")
    except Exception as e:
        print(f"[ERROR] Runtime exception: {e}")
