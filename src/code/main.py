import os
import numpy as np

def read_series_from_txt(filename):
    current_directory = os.getcwd()
    filepath = os.path.join(current_directory, "src/data", filename)
    file = open(filepath, "r")
    lines = file.readlines()
    file.close()
    return np.array([float(x) for x in lines])

read_series_from_txt("Period1.txt")