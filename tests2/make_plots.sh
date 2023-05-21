for i in `find ./results -name '*.pkl'` ; do python base_plots.py $i & done
python inset_plot.py results/elec2-ar3.pkl aci 0.1 scorecaster 0.01 100 1500 &
python inset_plot.py results/AMZN-ar3.pkl aci 0.1 quantile 20 400 1200 &
python inset_plot.py results/AMZN-ar3.pkl aci 0.01 quantile 20 400 1200 &
python inset_plot.py results/elec2-ar3.pkl quantile 0.01 quantile_integrator_log 0.01 100 500 &
python inset_plot.py results/COVID-national-cases-4wk.pkl quantile_integrator_log 500000 quantile_integrator_log_scorecaster 10000 5 70 &
python inset_plot.py results/elec2-ar3.pkl quantile 0.5 quantile_integrator_log 0.5 100 500 &
