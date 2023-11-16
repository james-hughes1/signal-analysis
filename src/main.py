import os
import numpy as np
from scipy.signal import periodogram
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt


def read_series_from_txt(filename: str) -> np.array:
    """Takes a .txt filename as input, and returns array whose
    elements are found from each line of the file.

    Args:
        filename (str): name of .txt data file to import, located in /data
        folder.

    Returns:
        np.array: converted 1D timeseries.
    """
    current_directory = os.getcwd()
    filepath = os.path.join(current_directory, "data", filename)
    file = open(filepath, "r")
    lines = file.readlines()
    file.close()
    timeseries = np.array([float(x) for x in lines])
    return timeseries


def estimate_wavelength(timeseries: np.array) -> int:
    """Uses the scipy.signal.periodogram to find strongest wavelength of
    timeseries, assuming unit time steps.

    Args:
        timeseries (np.array): timeseries to use.

    Returns:
        int: estimated wavelength
    """
    freq_arr, power_arr = periodogram(timeseries)
    max_freq_idx = np.argmax(power_arr)
    return round(1 / freq_arr[max_freq_idx])


def extract_periodicity(timeseries: np.array) -> np.array:
    """Uses estimate_wavelength to generate periodic features by optimising
    location of the period windows, via minimising the average pairwise L2
    dist of the periodic features.

    Args:
        timeseries (np.array): timeseries to use.

    Returns:
        np.array: 2D array whose rows are the exclusive periodic subsets of
        the timeseries.
    """
    wavelength = estimate_wavelength(timeseries)
    max_lag = len(timeseries) % wavelength
    num_periods = len(timeseries) // wavelength
    cost_lag = np.zeros(max_lag)
    # For each location of the period windows, find the average pairwise L2
    # dist of the subsets of the timeseries.
    for lag in range(max_lag):
        periods = timeseries[lag: num_periods*wavelength + lag].reshape(
            (num_periods, -1)
        )
        cost_lag[lag] = np.mean(pairwise_distances(periods))
    optimal_lag = np.argmin(cost_lag)
    extracted_periods = timeseries[
        optimal_lag: num_periods*wavelength + optimal_lag
    ].reshape((num_periods, -1))
    return extracted_periods


def plot_periodic_features(timeseries: np.array, filename: str):
    """Save plots of the periodic features of the given timeseries

    Args:
        timeseries (np.array): timeseries to use.
        filename (str): output file name, including extension (often .png).
    """
    periodic_features = extract_periodicity(timeseries)
    # Produce plot of original and of extracted periodic features.
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(timeseries)
    for i in range(len(periodic_features)):
        ax[1].plot(periodic_features[i])
    current_directory = os.getcwd()
    filepath = os.path.join(current_directory, "plots", filename)
    plt.savefig(filepath)


for i in range(1, 5):
    plot_periodic_features(
        read_series_from_txt("Period{}.txt".format(i)), "Plot{}.png".format(i)
    )
