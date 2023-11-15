import os
import numpy as np
from scipy.signal import periodogram

def read_series_from_txt(filename):
    current_directory = os.getcwd()
    filepath = os.path.join(current_directory, "src/data", filename)
    file = open(filepath, "r")
    lines = file.readlines()
    file.close()
    timeseries = np.array([float(x) for x in lines])
    return timeseries

def estimate_wavelength(timeseries):
    freq_arr, power_arr = periodogram(timeseries)
    max_freq_idx = np.argmax(power_arr)
    return 1 / freq_arr[max_freq_idx]

print(estimate_wavelength(read_series_from_txt("Period1.txt")))