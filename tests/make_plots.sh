for i in `find ./results -name '*.pkl'` ; do python base_plots.py $i & done
python inset_plot.py results/elec2.pkl ACI 0.1 Quantile+Integrator\ \(log\)+Scorecaster 0.01 100 1500 &
python inset_plot.py results/AMZN.pkl ACI 0.1 Quantile 20 400 1200 &
python inset_plot.py results/AMZN.pkl ACI 0.01 Quantile 20 400 1200 &
python inset_plot.py results/elec2.pkl Quantile 0.001 Quantile+Integrator\ \(log\) 0.001 100 500 &
python inset_plot.py results/COVID-national-cases-4wk.pkl Quantile+Integrator\ \(log\) 500000 Quantile+Integrator\ \(log\)+Scorecaster  10000 5 70 &
python inset_plot.py results/elec2.pkl Quantile 0.5 Quantile+Integrator\ \(log\) 0.5 100 500 &
