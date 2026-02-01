import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import pytest
import numpy as np
from soc_model import simulate_supply_chain_dynamics, generate_ring_topology

def test_simulation_reproducibility():
    """
    SCIENTIFIC VALIDITY: Ensures that setting a random seed produces 
    identical results (deterministic behavior).
    """
    params = {
        'B_start': 2.0, 'k': 2, 'n': 20, 'T': 50, 'T_start': 0, 'noise_scale': 1.0
    }
    
    # Run 1
    np.random.seed(42)
    d1, b1, _ = simulate_supply_chain_dynamics(**params)
    
    # Run 2 (Same Seed)
    np.random.seed(42)
    d2, b2, _ = simulate_supply_chain_dynamics(**params)
    
    # Run 3 (Different Seed)
    np.random.seed(123)
    d3, b3, _ = simulate_supply_chain_dynamics(**params)
    
    assert np.allclose(d1, d2), "Simulation not reproducible with same seed!"
    assert not np.allclose(d1, d3), "Different seeds produced identical results (unlikely)!"

def test_physics_constraints():
    """
    PHYSICS CHECK: Buffers and delays must never be negative.
    """
    delays, buffers, _ = simulate_supply_chain_dynamics(
        B_start=2.0, k=5, n=50, T=100, T_start=10, epsilon=0.1
    )
    
    assert np.min(buffers) >= 0, "Physics Violation: Negative buffer stock detected."
    assert np.min(delays) >= 0, "Physics Violation: Negative delay detected."

def test_mechanism_epsilon_decay():
    """
    LOGIC CHECK: With zero noise (no demand), buffers should decrease 
    over time due to epsilon (maintenance cost).
    """
    # Setup: Start with high buffer, no noise, strong decay
    delays, buffers, _ = simulate_supply_chain_dynamics(
        B_start=10.0, k=2, n=10, T=50, T_start=0,
        epsilon=0.1, noise_scale=0.0  # NO DOMANDA
    )
    
    initial_mean = buffers[0]
    final_mean = buffers[-1]
    
    assert final_mean < initial_mean, "Buffers did not decay despite epsilon pressure."

def test_ring_topology_logic():
    """
    TOPOLOGY CHECK: Verifies exact neighbor connections for a Ring Lattice.
    Logic: neighbors[i] should be predictable based on offsets.
    """
    N = 10
    K = 2 # Offsets dovrebbero essere [-2, -1] basato sul codice
    neighbors = generate_ring_topology(N, K)
    
    # Controllo Nodo 5
    # Vicini attesi: (5-2)=3, (5-1)=4
    expected_neighbors_node_5 = [3, 4]
    
    assert np.array_equal(sorted(neighbors[5]), sorted(expected_neighbors_node_5)), \
        f"Ring topology incorrect for Node 5. Got {neighbors[5]}"

if __name__ == "__main__":
    pytest.main()