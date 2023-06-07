# First write bash that iterates through all results and runs base_plots.py on them
for i in `find ./results -name '*.pkl'` ; do python base_plots.py $i & done

# Next plot insets for various hand-chosen experimental settings.
# The inset_plot.py script takes arguments from the following parser:
# parser = argparse.ArgumentParser(description='Plot time series data.')
# parser.add_argument('filename', help='Path to pickle file containing time series data.')
# parser.add_argument('key1', help='First key for time series data extraction.')
# parser.add_argument('lr1', help='Learning rate associated with first key.', type=float)
# parser.add_argument('key2', help='Second key for time series data extraction.')
# parser.add_argument('lr2', help='Learning rate associated with second key.', type=float)
# parser.add_argument('window_length', help='Length of inset window.', default=60, type=int)
# parser.add_argument('window_start', help='Start of inset window.', default=None, type=int)
# parser.add_argument('window_loc', help='Location of inset window.', default='upper right', type=str)
# parser.add_argument('coverage_average_length', help='Length of moving average window for coverage.', default=50, type=int)
# parser.add_argument('coverage_inset', help='Boolean for whether to plot inset for coverage.', default=True, type=bool)
# parser.add_argument('set_inset', help='Boolean for whether to plot inset for sets.', default=True, type=bool)

# First set of inset plots compares ACI to Quantile tracker
python inset_plot.py --filename results/AMZN-better-forecaster.pkl --key1 ACI --lr1 0.1 --key2 Quantile --lr2 20 --window_length 400 --window_start 2300 --window_loc 'upper left' --coverage_average_length 50 &
python inset_plot.py --filename results/AMZN-better-forecaster.pkl --key1 ACI --lr1 0.005 --key2 Quantile --lr2 20 --window_length 400 --window_start 2300 --window_loc 'upper left' --coverage_average_length 50 &
python inset_plot.py --filename results/AMZN-better-forecaster.pkl --key1 ACI --lr1 0.1 --key2 Quantile --lr2 20 --window_length 400 --window_start 2300 --window_loc 'upper left' --coverage_average_length 50 --set_inset &
python inset_plot.py --filename results/AMZN-better-forecaster.pkl --key1 ACI --lr1 0.005 --key2 Quantile --lr2 20 --window_length 400 --window_start 2300 --window_loc 'upper left' --coverage_average_length 50 --set_inset &

# Second set of inset plots compares Quantile tracker to Quantile+Integrator
python inset_plot.py --filename results/GOOGL-better-forecaster.pkl --key1 Quantile --lr1 0.1 --key2 'Quantile+Integrator (log)' --lr2 0.1 --window_length 100 --window_start 2300 --window_loc 'upper left' --coverage_average_length 50 &
python inset_plot.py --filename results/GOOGL-better-forecaster.pkl --key1 Quantile --lr1 20 --key2 'Quantile+Integrator (log)' --lr2 5 --window_length 100 --window_start 2300 --window_loc 'upper left' --coverage_average_length 50 &
python inset_plot.py --filename results/GOOGL-better-forecaster.pkl --key1 Quantile --lr1 0.1 --key2 'Quantile+Integrator (log)' --lr2 0.1 --window_length 100 --window_start 2300 --window_loc 'upper left' --coverage_average_length 50 --set_inset &
python inset_plot.py --filename results/GOOGL-better-forecaster.pkl --key1 Quantile --lr1 20 --key2 'Quantile+Integrator (log)' --lr2 5 --window_length 100 --window_start 2300 --window_loc 'upper left' --coverage_average_length 50 --set_inset &

# Finally, compare Quantile+Integrator to Quantile+Integrator+Scorecaster
python inset_plot.py --filename results/elec2.pkl --key1 ACI --lr1 0.1 --key2 'Quantile+Integrator (log)+Scorecaster' --lr2 0.01 --window_length 100 --window_start 1500 --window_loc 'lower right' --coverage_average_length 50 --set_inset &
python inset_plot.py --filename results/COVID.pkl --key1 Quantile --lr1 0 --key2 'Quantile+Integrator (log)+Scorecaster' --lr2 1000 --window_length 15 --window_start 30 --window_loc 'upper right' --coverage_average_length 10 --set_inset &
