Bilingual Lexicon Induction for Low-Resource Languages using Graph Matching via Optimal Transport
======================

This is an implementation of the experiments presented in:
- Kelly Marchisio, Ali Saad-Eldin, Kevin Duh, Carey Priebe, and Philipp Koehn. 2022. **[Bilingual Lexicon Induction for Low-Resource Languages using Graph Matching via Optimal Transport](https://arxiv.org/pdf/2210.14378.pdf)**. In *Association for Computational Linguistics: EMNLP 2022*.

If you use this software for academic research, please cite 2022 paper above.

This codebase is an extension of that used in the following paper, available at github.com/kellymarchisio/euc-v-graph-bli
- Kelly Marchisio, Youngser Park, Ali Saad-Eldin, Anton Alyakin, Kevin Duh, Carey Priebe, and Philipp Koehn. 2021. **[An Analysis of Euclidean vs. Graph-Based Framing for Bilingual Lexicon Induction from Word Embedding Spaces](https://aclanthology.org/2021.findings-emnlp.64)**. In Findings of the Association for Computational Linguistics: EMNLP 2021.



Requirements
--------
- Python3
- CuPy
- sklearn
- scipy
--------

Setup
-------
To download pretrained word embeddings, run `sh get_data.sh` from the embs/ folder.
To download MUSE dictionaries and create development sets, run `sh create_dicts.sh` from the dicts/ folder.
To download GOAT, run `sh get_packages.sh` from the third\_party/ folder.
Note: There is a small bug in the GOAT implementation that should be fixed before running this code (the published paper did fix the bug locally). The quick one-line fix is in `third_party/get_packages.sh`.

Usage
-------
Note: All results are written to the exp/ directory.

To run the main experiments presented in Table 1 and A3/4 of the publication, run, for example:

	sh exps.sh single en de goat 100
	sh exps.sh single en de sgm 100
	sh exps.sh single en de proc 100

for a run of English-German with 100 seeds using GOAT, SGM, or Procrustes.


To run iterative experiments presented in Table A5, one may run:

	sh exps.sh stoch-add en de proc 100


For the combination system, one may run the below for English-German starting with Iterative Procrustes and 100 seeds (-EG from Table 3, or GOAT -PG from Table A6).

	sh combo-exps.sh en de proc 100 barycenter goat

This command runs GOAT -PP from Table A6 (Start with GOAT, end with Iterative Procrustes):

	sh combo-exps.sh en de goat 100 barycenter goat

To use SGM instead of GOAT, you can run either of the below:

	sh combo-exps.sh en de proc 100 randomized sgm
	sh combo-exps.sh en de sgm 100 randomized sgm
