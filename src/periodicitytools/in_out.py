"""!@file main.py
@brief Main code for analysing periodic features of time series.

@details This module contains tools for estimating the frequency of signals,
and extracting periodic features from time series data.
@author Created by J. Hughes on 016/11/2023
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from periodicitytools.analysis import extract_periodicity


def read_series_from_txt(filename: str) -> np.array:
    """!@brief Takes a .txt filename as input, and returns array whose
    elements are found from each line of the file.

    @param filename Name of .txt data file to import, located in /data
    folder.

    @return timeseries Converted 1D timeseries.
    """
    current_directory = os.getcwd()
    filepath = os.path.join(current_directory, "data", filename)
    file = open(filepath, "r")
    lines = file.readlines()
    file.close()
    timeseries = np.array([float(x) for x in lines])
    return timeseries


def plot_periodic_features(timeseries: np.array, filename: str):
    """!@brief Save plots of the periodic features of the given timeseries

    @param timeseries timeseries to use.
    @param filename output file name, including extension (often .png).
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
