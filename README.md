# Simulating Complex Systems Group 7 - Timeliness Criticality

This repository contains the code for the Simulating Complex Systems Group 7 project focused on Timeliness Criticality, based on the paper https://arxiv.org/abs/2309.15070 and using the respective code from [their repository](https://github.com/jose-moran/timeliness_criticality). 


For installation instructions of the python code refer to [INSTALL.md](INSTALL.md)

# Recreation
The analytical_validation notebook takes five hypotheses from the analytical mean field theory and tests them via simulation. Multiple trials are used as well as bootstrapping to ensure robust error estimates and accurate hypothesis tests. 

The details can be found as markdown descriptions in the notebook itself.

# Sparsity exploration
Different levels of sparsity in synthetic temporal networks (STNs) were evaluated to observe second order phase transition behavior. See [sparsity.py](code/python_temp_criticality/python_synthetic_temporal_networks/sparsity.py) for code to recreate the sparsity exploration. Make sure to cd to the [python_synthetic_temporal_networks](code/python_temp_criticality/python_synthetic_temporal_networks) folder before running this script.

# Real-world networks

Real-world temporal networks were explored as suggested by the original paper and heavily inspired by their code. See [real_world_networks.py](code/python_temp_criticality/python_real_world_networks/real_world_networks.py) for code to recreate the real-world networks exploration. Make sure to cd to the [python_real_world_networks](code/python_temp_criticality/python_real_world_networks) folder before running this script and to download the required datasets as mentioned in [INSTALL.md](INSTALL.md).

# Heavy-tailedness exploration
In the papaer is presented the possibility of modeling the noise with a heavier tailed distributuion. Check [meanfield_simulation.py](/theory/meanfield_simulation.py) for the code implementation. It can be used with any scipy.stats distribution. This module is for exploratory analysis only and should be used with a restricted number of simulations as it is not as efficient as the code present in [ensemble.py](/theory/ensemble.py).

# Self-organized Criticality (SOC) exploration

We implemented a Self-Organized Criticality (SOC) model applied to temporal networks, modifying the original logic to use purely local update rules.

The work done in the [SOC_CriticalityMatteo.ipynb](code/python_temp_criticality/python_synthetic_temporal_networks/SOC_CriticalityMatteo.ipynb) notebook includes:

1) Implementation of the model starting from a Mean Field approximation.

2) Statistical analysis of the system dynamics, specifically focusing on measuring delays and buffer sizes.

3) Final verification of the critical behavior on networks with a fixed spatial structure.