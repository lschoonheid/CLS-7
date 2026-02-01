# Simulating Complex Systems Group 7 - Timeliness Criticality

This repository contains the code for the Simulating Complex Systems Group 7 project focused on Timeliness Criticality, based on the paper https://arxiv.org/abs/2309.15070 and using parts of the respective code from [their repository](https://github.com/jose-moran/timeliness_criticality). 


For installation instructions of the python code refer to [INSTALL.md](INSTALL.md)

# Recreation
The theory folder contains a notebook and module files all written in python. The notebook runs simulations that test the analytical solutions of the mean-field thery through simulation. Hypothesis tests are used where suitable as well as ensembles for robustness. 

The notebook timeliness_criticality_simulation_logic contains details and explanations on the Hypothesis and the analytical theory being tested. 

# Sparsity exploration
Different levels of sparsity in synthetic temporal networks (STNs) were evaluated to observe second order phase transition behavior. See [sparsity.py](code/python_temp_criticality/python_synthetic_temporal_networks/sparsity.py) for code to recreate the sparsity exploration. Make sure to cd to the [python_synthetic_temporal_networks](code/python_temp_criticality/python_synthetic_temporal_networks) folder before running this script.

# Real-world networks

Real-world temporal networks were explored as suggested by the original paper and heavily inspired by their code. See [real_world_networks.py](code/python_temp_criticality/python_real_world_networks/real_world_networks.py) for code to recreate the real-world networks exploration. Make sure to cd to the [python_real_world_networks](code/python_temp_criticality/python_real_world_networks) folder before running this script and to download the required datasets as mentioned in [INSTALL.md](INSTALL.md).

# Heavy-tailedness exploration
In the papaer is presented the possibility of modeling the noise with a heavier tailed distributuion. Check [meanfield_simulation.py](/theory/meanfield_simulation.py) for the code implementation. It can be used with any scipy.stats distribution. This module is for exploratory analysis only and should be used with a restricted number of simulations as it is not as efficient as the code present in [ensemble.py](/theory/ensemble.py).

# Self-organized Criticality (SOC) exploration

We investigated the emergence of Self-Organized Criticality (SOC) in supply chain networks by implementing a model driven strictly by local update rules. The simulation module is found in [soc_model.py](code/python_temp_criticality/python_synthetic_temporal_networks/soc_model.py) and the analysis in [SOC_Analysis.ipynb](code/python_temp_criticality/python_synthetic_temporal_networks/SOC_Analysis.ipynb). This exploration covers:
1. The implementation of adaptive agents that self-tune buffer sizes to reach a "Goldilocks Zone" of stability.
2. A comparative analysis of **Reactive** (Pipeline-Agnostic) versus **Compensatory** (Pipeline-Aware) control strategies to quantify their impact on delay-induced oscillations (Bullwhip Effect).
3. A topological comparison between Random Regular Graphs and Ring Lattices to confirm robust critical behavior.
