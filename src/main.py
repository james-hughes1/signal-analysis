import os
import numpy as np
from scipy.signal import periodogram
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

def read_series_from_txt(filename):
    current_directory = os.getcwd()
    filepath = os.path.join(current_directory, "data", filename)
    file = open(filepath, "r")
    lines = file.readlines()
    file.close()
    timeseries = np.array([float(x) for x in lines])
    return timeseries

def estimate_wavelength(timeseries):
    freq_arr, power_arr = periodogram(timeseries)
    max_freq_idx = np.argmax(power_arr)
    return round(1 / freq_arr[max_freq_idx])

def extract_periodicity(timeseries):
    wavelength = estimate_wavelength(timeseries)
    max_lag = len(timeseries) % wavelength
    num_periods = len(timeseries) // wavelength
    cost_lag = np.zeros(max_lag)
    for lag in range(max_lag):
        periods = timeseries[lag : num_periods*wavelength + lag].reshape((num_periods, -1))
        cost_lag[lag] = np.mean(pairwise_distances(periods))
    optimal_lag = np.argmin(cost_lag)
    periodic_features = timeseries[optimal_lag : num_periods*wavelength + optimal_lag].reshape((num_periods, -1))

    return timeseries[optimal_lag : num_periods*wavelength + optimal_lag].reshape((num_periods, -1))

def plot_periodic_features(timeseries, filename):
    periodic_features = extract_periodicity(timeseries)
    fig, ax = plt.subplots(2, 1, figsize=(10,10))
    ax[0].plot(timeseries)
    for i in range(len(periodic_features)):
        ax[1].plot(periodic_features[i])
    current_directory = os.getcwd()
    filepath = os.path.join(current_directory, "plots", filename)
    plt.savefig(filepath)

for i in range(1,5):
    plot_periodic_features(read_series_from_txt("Period{}.txt".format(i)), "Plot{}.png".format(i))