Euclidean vs. Graph-Based Framings for Bilingual Lexicon Induction
======================

This is an implementation of the experiments and combination system presented
in:
- Kelly Marchisio, Youngser Park, Ali Saad-Eldin,, Anton Alyakin Kevin Duh, Carey Priebe, and Philipp Koehn. 2021. **[An Analysis of Euclidean vs. Graph-Based Framing for Bilingual Lexicon Induction from Word Embedding Spaces](https://arxiv.org/abs/2109.12640)**. In *Findings of the Association for Computational Linguistics: EMNLP 2021*.

If you use this software for academic research, please cite the paper above.

Requirements
--------
- Python3
- CuPy
- Dask
- sklearn
- scipy
--------

Setup
-------
To download pretrained word embeddings, run `sh get_data.sh` from the embs/ folder.
To download MUSE dictionaries and create development sets, run `sh create_dicts.sh` from the dicts/ folder. 
To download Graspologic and Vecmap, run `sh get_packages.sh` from the third\_party/ folder.

Usage
-------
Note: All results are written to the exp/ directory.

To run the non-iterative experiments presented in Table 2 of the publication, run, for example:

	sh exps.sh single en de proc 100

for a single run of English-German using the "Procrustes" method (Euclidean view) and 100 seeds,
or the below for SGM (Graph-based view):

	sh exps.sh single en de sgm 100


To run iterative experiments presented in Table 4, one may run:

	sh exps.sh add-all en de proc 100
	sh exps.sh stoch-add en de proc 100
	sh exps.sh active-learn en de proc 100


For the combination system presented in Section 6, one may run the below for
English-German, starting with Iterative Procrustes and 100 seeds (Start:
IterProc from Table 5):

	sh combo-exps.sh en de proc 100

Output will be streamed to stdout. P@1 for -PullSGM for this example will be
seen at the end of program running. For -PullProc, one reads the P@1 for the
Forward direction of the last run of Iterative Procrustes. 
