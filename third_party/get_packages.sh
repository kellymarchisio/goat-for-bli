git clone https://github.com/microsoft/graspologic.git
echo > graspologic/graspologic/__init__.py

git clone https://github.com/artetxem/vecmap.git
# Please pardon the hackiness... but it worked...
echo >> vecmap/embeddings.py
echo >> vecmap/embeddings.py
cat vecmap/cupy_utils.py >> vecmap/embeddings.py
sed -i 's/from cupy_utils import */## from cupy_utils import */g' vecmap/embeddings.py

# The below file is from Lilt's Alignment-Scripts package:
# https://github.com/lilt/alignment-scripts
wget https://raw.githubusercontent.com/lilt/alignment-scripts/master/scripts/combine_bidirectional_alignments.py
