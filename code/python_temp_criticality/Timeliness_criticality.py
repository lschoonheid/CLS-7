# Reference to original paper code
import sys
import os

sys.path.append(
    os.path.abspath("../../../timeliness_criticality_paper/python_temp_criticality/python_real_world_networks")
)
from DelayBufferNetwork import DelayBufferNetwork

sys.path.append(
    os.path.abspath("../../../timeliness_criticality_paper/python_temp_criticality/python_synthetic_temporal_networks")
)
from stn_simulations import (
    synthetic_temporal_network,
    synthetic_temporal_network_heterogeneous_K,
    synthetic_temporal_network_sparsity,
)
