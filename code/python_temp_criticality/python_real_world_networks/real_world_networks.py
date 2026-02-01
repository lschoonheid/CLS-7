# Download the Thiers13, Workplace15 dataset from sociopatterns.org


from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

import sys
import os

from tqdm import tqdm

sys.path.append(os.path.abspath(".."))

try:
    sys.path.append(os.path.abspath(".."))
    from Timeliness_criticality import DelayBufferNetwork  # pyright: ignore[reportAttributeAccessIssue]
except ImportError as e:
    print("Error importing modules. Make sure to run this script from the 'python_temp_criticality' folder.")
    exit()

DATA_FOLDER = "real_world_data/"
LOAD_CACHED_RESULTS = True

Result = Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


def get_thiers13(do_cache=LOAD_CACHED_RESULTS) -> pd.DataFrame:
    """
    Load and process the Thiers13 high school proximity network dataset.
    This function loads temporal network data from the HighSchool2013 proximity network dataset,
    preprocesses it by remapping node IDs and timestamps to sequential integers, and creates a
    DelayBufferNetwork object from the processed data.
    Args:
        do_cache (bool, optional): Whether to cache the processed network data to disk.
            Defaults to LOAD_CACHED_RESULTS.
    Returns:
        pd.DataFrame: A DelayBufferNetwork object containing the processed temporal network data
            with columns ['t', 'i', 'j', 'weight', 'event_id'].
    Notes:
        - If LOAD_CACHED_RESULTS is True, attempts to load cached data from "Thiers13.npy"
        - Node IDs (i, j) are remapped to sequential integers starting from 0
        - Timestamps (t) are remapped to sequential integers starting from 0
        - All edge weights are set to 0.5
        - Each event is assigned an event_id based on its timestamp
        - If do_cache is True, saves the processed network as "1_Thiers13" event arrays and dict
    Raises:
        FileNotFoundError: If the source CSV file is not found and no cached .npy file exists
    """

    if LOAD_CACHED_RESULTS:
        # Save to .npy
        try:
            event_data = np.load("Thiers13.npy", allow_pickle=True)
        except FileNotFoundError:
            event_data = pd.read_csv(
                DATA_FOLDER + "HighSchool2013_proximity_net.csv", delim_whitespace=True, header=None
            ).values
        np.save("Thiers13.npy", event_data)
    else:
        event_data = pd.read_csv(
            DATA_FOLDER + "HighSchool2013_proximity_net.csv", delim_whitespace=True, header=None
        ).values

    # Create a DataFrame for easier handling
    df = pd.DataFrame(event_data, columns=["t", "i", "j", "weight", "event_id"])
    df["event_id"] = df["t"].astype(int)  # Each event lasts 1 time step

    # Create a dictionary to map the old agent IDs to the new agent IDs
    agent_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(df["i"].unique()))}
    agent_id_mapj = {old_id: new_id for new_id, old_id in enumerate(sorted(df["j"].unique()))}
    t_map = {old_id: new_id for new_id, old_id in enumerate(sorted(df["t"].unique()))}
    weight_map = {old_id: 0.5 for _new_id, old_id in enumerate(sorted(df["weight"]))}

    # Map the old agent IDs to the new agent IDs in the DataFrame
    df["i"] = df["i"].map(agent_id_map)
    df["j"] = df["j"].map(agent_id_mapj)
    df["t"] = df["t"].map(t_map)
    df["weight"] = df["weight"].map(weight_map)

    dbn = DelayBufferNetwork(nettype="wd", from_df=df, uniform_time_range=True, dont_build_df=True, del_df=True)

    if do_cache:
        # save it in case you need it later
        dbn.save_event_arrays("1_Thiers13")
        dbn.save_event_dict("1_Thiers13")

    return dbn


def get_workplace15(do_cache=LOAD_CACHED_RESULTS) -> pd.DataFrame:
    """
    Load and process the Workplace15 temporal network dataset.

    This function reads the InVS workplace contact network data, processes it into a
    DelayBufferNetwork object, and optionally caches the results for faster future access.

    Args:
        do_cache (bool, optional): Whether to save processed network data to disk for
            future use. Defaults to LOAD_CACHED_RESULTS.

    Returns:
        pd.DataFrame: A DelayBufferNetwork object containing the processed temporal
            network with remapped node IDs and timestamps. Each contact event has:
            - i: source node (remapped to sequential integers)
            - j: target node (remapped to sequential integers)
            - t: timestamp (remapped to sequential integers)
            - weight: constant weight of 0.5
            - event_id: integer timestamp value

    Notes:
        - The function attempts to load cached .npy data if LOAD_CACHED_RESULTS is True
        - Original node IDs and timestamps are remapped to sequential integers starting from 0
        - If do_cache is True, the processed network is saved as "1_Workplace15"
        - The network is created with uniform_time_range=True and dont_build_df=True options
    """

    if LOAD_CACHED_RESULTS:
        # Save to .npy
        try:
            event_data = np.load("Workplace15.npy", allow_pickle=True)
        except FileNotFoundError:
            event_data = pd.read_csv(DATA_FOLDER + "tij_InVS.dat", delim_whitespace=True, header=None).values
        np.save("Workplace15.npy", event_data)
    else:
        event_data = pd.read_csv(DATA_FOLDER + "tij_InVS.dat", delim_whitespace=True, header=None).values

    # Create a DataFrame for easier handling
    df = pd.DataFrame(event_data, columns=["i", "j", "t"])
    df["weight"] = 0.5  # Assign a constant weight
    df["event_id"] = df["t"].astype(int)  # Each event lasts 1 time step

    # Create a dictionary to map the old agent IDs to the new agent IDs
    agent_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(df["i"].unique()))}
    agent_id_mapj = {old_id: new_id for new_id, old_id in enumerate(sorted(df["j"].unique()))}
    t_map = {old_id: new_id for new_id, old_id in enumerate(sorted(df["t"].unique()))}

    # Map the old agent IDs to the new agent IDs in the DataFrame
    df["i"] = df["i"].map(agent_id_map)
    df["j"] = df["j"].map(agent_id_mapj)
    df["t"] = df["t"].map(t_map)

    dbn = DelayBufferNetwork(nettype="wd", from_df=df, uniform_time_range=True, dont_build_df=True, del_df=True)

    if do_cache:
        # save it in case you need it later
        dbn.save_event_arrays("1_Workplace15")
        dbn.save_event_dict("1_Workplace15")

    return dbn


def sim(b, path: str) -> Result:
    """
    Simulate event propagation through a delay buffer network.
    This function creates a DelayBufferNetwork from a saved file, adds exponential delays
    and uniform event buffers, processes the delays without topology interaction, and 
    returns the resulting delay statistics.
    Args:
        b: Buffer size for the event buffer. Used to set uniform buffer capacity.
        path (str): File path to load the saved DelayBufferNetwork from.
    Returns:
        Result: A tuple containing:
            - event_delays: Array of current delays for each event
            - agent_delays: Mean delays per agent (averaged across axis 1)
    Note:
        - The network is instantiated inside the function to prevent memory issues 
          during parallelization
        - Uses exponential delay distribution with tau=1
        - Buffers are uniformly distributed
        - Both delays and buffers use event dictionary representation
        - Delay processing does not interact with network topology
    """

    # Build the DelayBufferNetwork inside the function to avoid memory issues with parallelization
    dbn = DelayBufferNetwork(load=True, path=path)
    dbn.add_delay(expon_distr_bool=True, tau=1, using_event_dict=True)
    dbn.add_event_buffer(buffer=b, uniform_buffer_bool=True, using_event_dict=True)
    dbn.process_delays_fast_arrays(interact_with_topology=False)
    event_delays = dbn.event_current_delay_array
    agent_delays = np.mean(dbn.agent_delays, axis=1)
    return event_delays, agent_delays


def plot(delays_prop: List[npt.NDArray[np.float64]], title: str):
    """
    Plot the mean proportion of delays as a function of buffer sizes.
    Parameters
    ----------
    delays_prop : List[npt.NDArray[np.float64]]
        A list of arrays containing delay proportions for each buffer size.
        Each array corresponds to one buffer size configuration.
    title : str
        The title/label to be displayed in the plot legend.
    Notes
    -----
    This function creates a scatter plot showing the relationship between buffer sizes
    and mean delay proportions. It assumes a global variable 'buffers' exists containing
    the buffer size values to plot on the x-axis.
    The plot displays:
    - X-axis: Buffer sizes (B)
    - Y-axis: Mean delay proportion (v)
    - Red scatter points for each buffer size
    - Grid for easier reading
    """
    


    for i, b in enumerate(buffers):
        plt.scatter(b, np.mean(delays_prop[i]), c="r")

    plt.legend([title])

    plt.xlabel(r"$B$")
    plt.ylabel(r"$v$")
    plt.xlim([buffers[0], buffers[-1]])
    plt.ylim(bottom=0)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Set sim settings
    buffers = np.arange(0, 5, 0.1)

    for name, _dbn in [("Thiers13", get_thiers13()), ("Workplace15", get_workplace15())]:
        # Execute simulations in parallel
        results: List[Result] = Parallel(n_jobs=4)(
            delayed(sim)(buffers[i], "1_" + name)
            for i in tqdm(range(len(buffers)), desc=f"Fetching results for {name}")
        )  # pyright: ignore[reportAssignmentType]

        # Reformat the results to give us $V$
        delays_prop: List[npt.NDArray[np.float64]] = []
        for i in range(len(results)):
            delays_prop.append(results[i][1][1:] - results[i][1][:-1])

        # Plot the results
        plot(delays_prop, name)
