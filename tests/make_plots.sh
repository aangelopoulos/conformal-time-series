for i in `find ./results -name '*.pkl'` ; do python base_plots.py $i & done
