from periodicitytools.in_out import read_series_from_txt
from periodicitytools.in_out import plot_periodic_features

for i in range(1, 5):
    plot_periodic_features(
        read_series_from_txt("Period{}.txt".format(i)), "Plot{}.png".format(i)
    )
