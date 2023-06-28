export PYTORCH_ENABLE_MPS_FALLBACK=1
for i in `find ./configs -name '*.yaml'` ; do python base_test.py $i & done
