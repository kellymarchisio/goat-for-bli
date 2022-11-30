echo Getting GOAT code.
git clone https://github.com/neurodata/goat.git
echo GOAT code copied.
echo !!!! There is a bug in here that you need to manually correct !!!!
echo -e "\t Namely, after line 387 in third_party/goat/pkg/pkg/gmp/qap.py, you must add:"
echo -e "\t\t# Cannot assume partial_match is sorted"
echo -e "\t\tpartial_match = np.row_stack(sorted(partial_match, key=lambda x: x[0]))"
echo -e "\t  and the same after line 397 in third_party/goat/pkg/pkg/gmp/qapot.py"

git clone https://github.com/artetxem/vecmap.git
# Please pardon the hackiness... but it worked...
echo >> vecmap/embeddings.py
echo >> vecmap/embeddings.py
cat vecmap/cupy_utils.py >> vecmap/embeddings.py
sed -i 's/from cupy_utils import */## from cupy_utils import */g' vecmap/embeddings.py

# The below file is from Lilt's Alignment-Scripts package:
# https://github.com/lilt/alignment-scripts
wget https://raw.githubusercontent.com/lilt/alignment-scripts/master/scripts/combine_bidirectional_alignments.py

