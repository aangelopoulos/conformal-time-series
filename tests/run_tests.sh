export PYTORCH_ENABLE_MPS_FALLBACK=1
for i in `find ./configs -name '*.yaml'` ; do python base_test.py $i & done
#for i in `find ./configs -name '*.yaml'` ; do python base_test.py $i Quantile,'Quantile+Integrator (log)','Quantile+Integrator (log)+Scorecaster' & done # This kind of line will allow you to re-run only specific methods
