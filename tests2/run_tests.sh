for i in `find ./configs -name '*.yaml'` ; do python base_test.py $i & done
