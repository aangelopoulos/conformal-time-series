for i in `find ./results -name '*.pkl'` ; do python base_plots.py $i & done
python inset_plot.py results/elec2.pkl ACI 0.005 Quantile+Integrator\ \(log\)+Scorecaster 0.01 100 1500 'lower right' 50 &
python inset_plot.py results/AMZN.pkl ACI 0.1 Quantile 20 400 1200 'lower right' 50 &
python inset_plot.py results/AMZN.pkl ACI 0.005 Quantile 20 400 1200 'lower right' 50 &
python inset_plot.py results/elec2.pkl Quantile 0.001 Quantile+Integrator\ \(log\) 0.001 100 500 'lower right' 50 &
python inset_plot.py results/COVID-national-cases-4wk.pkl Quantile+Integrator\ \(log\) 100000 Quantile+Integrator\ \(log\)+Scorecaster  100000 5 70 'upper right' 10
python inset_plot.py results/elec2.pkl Quantile 0.5 Quantile+Integrator\ \(log\) 0.5 100 500 'lower right' 50 &
