# This file contains the code for the simulations of the synthetic temporal network model.
# It contains functions to simulate regular STNs, and STNs with heterogeneity in K, and sparsity.
# Some parts of code may be difficult to grasp at first, since numerous NumPy optimizations are used.
# Author: Matthijs Romeijnders, 2023-2024.

import numpy as np
import scipy
import scipy.stats


def synthetic_temporal_network(B, k, n, T, T_start):
    """Simulate a synthetic temporal network.

    Args:
        B (float): The uniform buffer
        k (int): The number of connections per node
        n (int): The number of nodes
        T (int): The number of time steps
        T_start (int): The number of time steps before we start measuring delays, to decorrelate the data from the initialization.
        
    Returns:
        mean_delay_propagation (np.array): The mean delay propagation over time.
        mean_delays (np.array): The mean delay over time.
    """
    
    # Initialize the several delay arrays used for the simulation and the output.
    delays_last_iteration = np.zeros((n))
    delays_current_iteration = np.zeros((n))
    delays_selected = np.zeros((n*k))
    mean_delays = np.zeros((T-T_start))
    mean_delay_propagation = np.zeros((T-T_start))
    
    # We loop over the timerange, and simulate the network.
    for t in range(0, T):
        # Reset the delays_selected array.
        delays_selected = np.zeros((n*k))
        
        # Prepare exponential delays
        eps = scipy.stats.expon.rvs(size=(n))
        
        # Shuffle the delays from the last iteration k times.
        for i in range(k):
            np.random.shuffle(delays_last_iteration)
            # Reformat them each time into the selected delays array, this ensures exactly k connections per node.
            # While making sure that each node has exactly k connections.
            delays_selected[i*n:(i+1)*n] = delays_last_iteration
            
        # Reformat the delays_selected array into a matrix.
        delays_selected = np.reshape(delays_selected, (n, k))
        
        # Take the maximum delay of each node, and subtract the buffer.
        max_values = np.max(delays_selected, axis=1) - B
        
        # Add the exponential random variable to the delays.
        delays_current_iteration = np.where(max_values < 0, 0, max_values) + eps
        
        # If we are past the initialization phase, we can start measuring the delays.
        if t >= T_start:
            # Calculate the mean delay and the mean delay propagation.
            mean_delays[t-T_start] = np.mean(delays_current_iteration)
            mean_delay_propagation[t-T_start] = np.mean(delays_current_iteration - delays_last_iteration)
            
        # Update the delays_last_iteration array.
        delays_last_iteration = delays_current_iteration
        
    return mean_delay_propagation, mean_delays


def synthetic_temporal_network_heterogeneous_K(B, k, n, T, T_start, heterogeneity_range):
    """Simulate a synthetic temporal network, with heterogeneity in K.

    Args:
        B (float): The uniform buffer
        k (int): The number of connections per node
        n (int): The number of nodes
        T (int): The number of time steps
        T_start (int): The number of time steps before we start measuring delays, to decorrelate the data from the initialization.
        heterogeneity_range (int): The range around k that we allow for heterogeneity in k.
        
    Returns:
        mean_delay_propagation (np.array): The mean delay propagation over time.
        mean_delays (np.array): The mean delay over time.
    """
    if k - heterogeneity_range < 1:
        raise ValueError("Heterogeneity range cannot be larger than or equal to k.")
    
    # Initialize the several delay arrays used for the simulation and the output.
    delays_last_iteration = np.zeros((n))
    delays_current_iteration = np.zeros((n))
    delays_selected = np.zeros((n*k))
    mean_delays = np.zeros((T-T_start))
    mean_delay_propagation = np.zeros((T-T_start))
    
    k_probabilities = np.ones(len(np.arange(k-heterogeneity_range, k+heterogeneity_range+1))) / (2*heterogeneity_range+1)
    
    # We loop over the timerange, and simulate the network.
    for t in range(0, T):
        # Take a random sample from the uniform distribution.
        k_curr = np.random.choice(np.arange(k-heterogeneity_range, k+heterogeneity_range+1), 1, p=k_probabilities)[0]
        # Reset the delays_selected array.
        delays_selected = np.zeros((n*k_curr))
        
        # Prepare exponential delays
        eps = scipy.stats.expon.rvs(size=(n))
        
        # Shuffle the delays from the last iteration k times.
        for i in range(k_curr):
            np.random.shuffle(delays_last_iteration)
            # Reformat them each time into the selected delays array, this ensures exactly k connections per node.
            # While making sure that each node has exactly k connections.
            delays_selected[i*n:(i+1)*n] = delays_last_iteration
            
        # Reformat the delays_selected array into a matrix.
        delays_selected = np.reshape(delays_selected, (n, k_curr))
        
        # Take the maximum delay of each node, and subtract the buffer.
        max_values = np.max(delays_selected, axis=1) - B
        
        # Add the exponential random variable to the delays.
        delays_current_iteration = np.where(max_values < 0, 0, max_values) + eps
        
        # If we are past the initialization phase, we can start measuring the delays.
        if t >= T_start:
            # Calculate the mean delay and the mean delay propagation.
            mean_delays[t-T_start] = np.mean(delays_current_iteration)
            mean_delay_propagation[t-T_start] = np.mean(delays_current_iteration - delays_last_iteration)
            
        # Update the delays_last_iteration array.
        delays_last_iteration = delays_current_iteration
        
    return mean_delay_propagation, mean_delays
    
    
def synthetic_temporal_network_sparsity(B, k, n, T, T_start, sparsity):
    """Simulate a synthetic temporal network, with sparsity.

    Args:
        B (float): The uniform buffer
        k (int): The number of connections per node
        n (int): The number of nodes
        T (int): The number of time steps
        T_start (int): The number of time steps before we start measuring delays, to decorrelate the data from the initialization.
        sparsity (float): The sparsity of the STN, ranges from 0 to 1. 100% sparsity reflects a network where no nodes interact,
                            50% reflects an STN where half the agents are in events each time step, 0% will give a normal STN.
        
    Returns:
        mean_delay_propagation (np.array): The mean delay propagation over time.
        mean_delays (np.array): The mean delay over time.
    """
    
    # Initialize the several delay arrays used for the simulation and the output.
    delays_last_iteration = np.zeros((n))
    delays_current_iteration = np.zeros((n))
    delays_selected = np.zeros((n*k))
    mean_delays = np.zeros((T-T_start))
    mean_delay_propagation = np.zeros((T-T_start))
    
    # We loop over the timerange, and simulate the network.
    for t in range(0, T):
        if sparsity != 0:
            # The probability of a node being selected is 1-sparsity.
            n_selected = np.random.choice([0, 1], size=n, p=[sparsity, 1-sparsity])
            delays_last_iteration_orginal = delays_last_iteration.copy()
        else:
            n_selected = n
            
        # Reset the delays_selected array.
        delays_selected = np.zeros((n*k))
        
        # Prepare exponential delays
        eps = scipy.stats.expon.rvs(size=(n))
        
        # Shuffle the delays from the last iteration k times.
        for i in range(k):
            np.random.shuffle(delays_last_iteration)
            # Reformat them each time into the selected delays array, this ensures exactly k connections per node.
            # While making sure that each node has exactly k connections.
            delays_selected[i*n:(i+1)*n] = delays_last_iteration
            
        # Reformat the delays_selected array into a matrix.
        delays_selected = np.reshape(delays_selected, (n, k))
        
        # Take the maximum delay of each node, and subtract the buffer.
        max_values = np.max(delays_selected, axis=1) - B
        
        if sparsity != 0: 
            # Only propagate the delays of the nodes that are selected.
            delays_current_iteration = np.where(n_selected == 1, np.where(max_values < 0, 0, max_values) + eps, delays_last_iteration_orginal)
        else:
            delays_current_iteration = np.where(max_values < 0, 0, max_values) + eps
        
        # If we are past the initialization phase, we can start measuring the delays.
        if t >= T_start:
            # Calculate the mean delay and the mean delay propagation.
            mean_delays[t-T_start] = np.mean(delays_current_iteration)
            mean_delay_propagation[t-T_start] = np.mean(delays_current_iteration - delays_last_iteration)
            
        # Update the delays_last_iteration array.
        delays_last_iteration = delays_current_iteration
        
    return mean_delay_propagation, mean_delays



def sterman_stn_simulation(B, k, n, T, T_start, 
                           correction_speed=0.2, 
                           pipeline_awareness=1.0, 
                           effort_lag=3):
    """
    Simulates a Temporal Network using Sterman's local decision rules.
    
    Args:
        B (float): Structural buffer (passive absorption).
        k (int): Degree (connections per node).
        n (int): Number of nodes.
        T (int): Total time steps.
        T_start (int): Warm-up period to reach steady state.
        correction_speed (alpha): Sensitivity to backlog (0 to 1).
        pipeline_awareness (beta): Trust in the supply line (0=Miopia, 1=Rational).
        effort_lag (L): Delay between decision and impact.
        
    Returns:
        results (dict): Dictionary containing time-series of delays, effort, and waste.
    """
    
    # --- Initialization ---
    delays = np.zeros(n)
    
    # The Pipeline: Rows = nodes, Cols = time steps until arrival
    # effort_pipeline[:, 0] is the effort landing NOW.
    effort_pipeline = np.zeros((n, effort_lag))
    
    # Metrics Storage
    # We use (T - T_start) to store data after the system stabilizes
    history_size = T - T_start
    metrics = {
        "mean_delay": np.zeros(history_size),
        "total_effort_decided": np.zeros(history_size),
        "wasted_effort": np.zeros(history_size),
        "variability_index": np.zeros(history_size)
    }

    # Pre-generate noise for the entire simulation (Vectorized efficiency)
    all_eps = scipy.stats.expon.rvs(size=(T, n))

    # --- Main Simulation Loop ---
    for t in range(T):
        # 1. TEMPORAL COUPLING (The Shuffle)
        # We ensure exactly k connections in and out by shuffling the array k times
        delays_selected = np.zeros((n * k))
        for i in range(k):
            shuffled = np.random.permutation(delays)
            delays_selected[i*n : (i+1)*n] = shuffled
        
        # Reshape to (n, k) to find the worst-case input for each agent
        incoming_delays = np.reshape(delays_selected, (n, k))
        max_incoming = np.max(incoming_delays, axis=1) - B
        max_incoming = np.maximum(0, max_incoming) # Buffer absorption

        # 2. THE DECISION (Sterman's Rules)
        # Rule 1: Replacement (Base demand is max_incoming)
        # Rule 2: Backlog Correction (alpha * current_delay)
        # Rule 3: Pipeline awareness (beta * sum of efforts in transit)
        
        supply_line = np.sum(effort_pipeline, axis=1)
        
        # Decision Policy: How much extra effort to order?
        # desired_effort = Expected Demand + Correction for (Backlog - beta * SupplyLine)
        desired_extra_effort = correction_speed * (delays - pipeline_awareness * supply_line)
        
        # Total decision (constrained to be non-negative)
        # Note: Replacement is part of the flow; extra effort corrects the stock.
        total_decision = np.maximum(0, max_incoming + desired_extra_effort)

        # 3. THE EXECUTION (Lagged Impact)
        # Effort decided 'effort_lag' steps ago arrives now
        active_effort = effort_pipeline[:, 0]
        
        # Calculate resulting delay
        potential_delay = max_incoming + all_eps[t]
        
        # Logic for Waste: Effort exceeds what was needed to reach zero delay
        waste = np.maximum(0, active_effort - potential_delay)
        
        # Logic for New Delay: Cannot be negative
        new_delays = np.maximum(0, potential_delay - active_effort)

        # 4. MEMORY UPDATE (Pipeline Shift)
        # Shift the pipeline: efforts move one step closer to landing
        # We use roll to shift, then replace the last column with the new decision
        effort_pipeline = np.roll(effort_pipeline, -1, axis=1)
        effort_pipeline[:, -1] = total_decision

        # 5. DATA COLLECTION
        if t >= T_start:
            idx = t - T_start
            metrics["mean_delay"][idx] = np.mean(new_delays)
            metrics["total_effort_decided"][idx] = np.mean(total_decision)
            metrics["wasted_effort"][idx] = np.mean(waste)
            # Bullwhip proxy: Ratio of output variability to input noise
            metrics["variability_index"][idx] = np.std(new_delays) / np.std(all_eps[t])

        # State transition
        delays = new_delays

    return metrics
       