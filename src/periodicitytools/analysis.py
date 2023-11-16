"""!@file main.py
@brief Main code for analysing periodic features of time series.

@details This module contains tools for estimating the frequency of signals,
and extracting periodic features from time series data.
@author Created by J. Hughes on 016/11/2023
"""


import numpy as np
from scipy.signal import periodogram
from sklearn.metrics import pairwise_distances


def estimate_wavelength(timeseries: np.array) -> int:
    """!@brief Uses the scipy.signal.periodogram to find strongest wavelength
    of timeseries, assuming unit time steps.

    @param timeseries Timeseries to use.

    @return wavelength Estimated wavelength.
    """
    freq_arr, power_arr = periodogram(timeseries)
    max_freq_idx = np.argmax(power_arr)
    return round(1 / freq_arr[max_freq_idx])


def extract_periodicity(timeseries: np.array) -> np.array:
    """!@brief Extract periodic features from time series.

    @details Uses estimate_wavelength to generate periodic features by
    optimising location of the period windows, via minimising the average
    pairwise L2 dist of the periodic features.

    @param timeseries timeseries to use.

    @return extracted_periods 2D array whose rows are the exclusive periodic
    subsets of the timeseries.
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
