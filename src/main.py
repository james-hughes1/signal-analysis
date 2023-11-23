from periodicitytools.in_out import read_series_from_txt
from periodicitytools.in_out import plot_periodic_features
from line_profiler import LineProfiler

for i in range(1, 5):
    plot_periodic_features(
        read_series_from_txt("Period{}.txt".format(i)), "Plot{}.png".format(i)
    )


def signal_analysis_files(idx_list):
    for i in idx_list:
        plot_periodic_features(
            read_series_from_txt("Period{}.txt".format(i)),
            "Plot{}.png".format(i)
        )


lp = LineProfiler()
lp_wrapper = lp(signal_analysis_files)
lp_wrapper(list(range(1, 5)))
lp.print_stats()
