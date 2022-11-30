###############################################################################
#
# Seeded Graph Matching and Graph Matching via Optimal Transport demo over
#   word embeddings.
#
# Written by Kelly Marchisio, Oct 2020-2022.
#   Amended for public release, Oct 2021, Nov 2022.
#
# Parts of this code pull heavily from the Graspologic SGM Demo:
#   graspy.neurodata.io/tutorials/matching/sgm.html
#
###############################################################################

import numpy as np
import sys
from datetime import datetime
from functools import reduce

from third_party.goat.pkg.pkg.gmp import qapot, qap
from sklearn.utils import check_array, column_or_1d

###############################################################################
###############################################################################


def softsgm(A1, A2, A1_seeds=[], A2_seeds=[], iters=10, function='sgm',
        opts={}):
    '''Implements SoftSGM (Algorithm 3) from Fishkind et al. (2019).

    https://www.sciencedirect.com/science/article/pii/S0031320318303431
    Implementation inspiration:
        multiStart.r: http://www.cis.jhu.edu/~parky/D3M/VN_0.3.0.tar.gz
        and email from Ali Saad-Eldin on 7 April 2021.

    Options: If function == 'goat', runs Graph Matching via optimal transport
    (https://arxiv.org/abs/2111.05366) iteratively, like SoftSGM.
    '''
    P_dim = max(max(A1.shape), max(A2.shape))
    Ps = []
    Pi_trg_indices_prev = []
    for i in range(iters):
        Pi_trg_indices, _ = unshuffle(A1, A2, A1_seeds, A2_seeds, function,
                opts)
        Pi_trg_indices_prev = Pi_trg_indices
        Pi = np.eye(P_dim)[Pi_trg_indices, :] # From Ali.
        Ps.append(Pi)
    P_avg = reduce(lambda x,y: x + y, Ps) / iters
    return P_avg, Ps


def unshuffle(A1, A2, A1_seeds=[], A2_seeds=[], function='sgm', opts={}):
    # Opts should be a dictionary of options to pass to graph matching
    starttime = datetime.now()
    print('Unshuffle round start time:', starttime.strftime("%D %H:%M:%S"))
    sys.stdout.flush()


    A1_seeds = column_or_1d(A1_seeds)
    A2_seeds = column_or_1d(A2_seeds)
    partial_match = np.column_stack((A1_seeds, A2_seeds))
    opts.update(dict(partial_match=partial_match))
    if function == 'sgm':
        print('Running SGM')
        print('Options are:', opts)
        perm_inds = qap.quadratic_assignment(A1, A2, options=opts).col_ind
    elif function == 'goat':
        opts.update(dict(reg=500))
        print('Running GOAT')
        print('Options are:', opts)
        perm_inds = qapot.quadratic_assignment_ot(A1, A2, options=opts).col_ind
    # score = np.sum(A1 * A2[perm_inds][:, perm_inds])
    # print('Score:', score)
    A2_unshuffle = A2[np.ix_(perm_inds, perm_inds)]

    endtime = datetime.now()
    print('Unshuffle round end time:', endtime.strftime("%D %H:%M:%S"))
    if getattr((endtime-starttime), 'minutes', None):
        print('Total Approx Processing Time:', str((endtime-starttime).minutes))
    sys.stdout.flush()

    return perm_inds, A2_unshuffle

