python inset_plot.py results/elec2-ar3.pkl aci 0.1 pid+ets 0.01 100 2500 &
python inset_plot.py results/AMZN-ar3.pkl aci 0.1 quantile 20 400 1200 &
python inset_plot.py results/AMZN-ar3.pkl aci 0.01 quantile 20 400 1200 &
python inset_plot.py results/elec2-ar3.pkl quantile 0.01 pi 0.01 100 500 &
python inset_plot.py results/COVID-national-cases-4wk.pkl pi 500000 pid+ets 10000 5 70 &
python inset_plot.py results/elec2-ar3.pkl quantile 0.5 pi 0.5 100 500 &
