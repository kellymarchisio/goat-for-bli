###############################################################################
#
# Seeded Graph Matching demo over word embeddings.
#
# Written by Kelly Marchisio, Oct 2020.
#   Amended for public release, Oct 2021.
#
# Parts of this code pull heavily from the Graspologic SGM Demo:
#   graspy.neurodata.io/tutorials/matching/sgm.html
#
###############################################################################

import numpy as np
import sys
from datetime import datetime
from functools import reduce
from third_party.graspologic.graspologic.match import gmp 

###############################################################################
###############################################################################


def softsgm(A1, A2, A1_seeds=[], A2_seeds=[], iters=10):
    '''Implements SoftSGM (Algorithm 3) from Fishkind et al. (2019).

    https://www.sciencedirect.com/science/article/pii/S0031320318303431
    Implementation inspiration:
        multiStart.r: http://www.cis.jhu.edu/~parky/D3M/VN_0.3.0.tar.gz
        and Ali Saad-Eldin on 7 April 2021.
    '''
    P_dim = max(max(A1.shape), max(A2.shape))
    Ps = []
    Pi_trg_indices_prev = []
    for i in range(iters):
        Pi_trg_indices, _ = unshuffle('rand', A1, A2, A1_seeds, A2_seeds)
        Pi_trg_indices_prev = Pi_trg_indices
        Pi = np.eye(P_dim)[Pi_trg_indices, :] # From Ali.
        Ps.append(Pi)
    P_avg = reduce(lambda x,y: x + y, Ps) / iters
    return P_avg, Ps


def unshuffle(init_method, A1, A2, A1_seeds=[], A2_seeds=[]):
    starttime = datetime.now()
    print('Unshuffle round start time:', starttime.strftime("%D %H:%M:%S"))
    sys.stdout.flush()

    sgm = gmp.GraphMatch(init=init_method).fit(A1, A2, A1_seeds, A2_seeds)
    perm_inds = sgm.perm_inds_
    A2_unshuffle = A2[np.ix_(perm_inds, perm_inds)]

    endtime = datetime.now()
    print('Unshuffle round end time:', endtime.strftime("%D %H:%M:%S"))
    if getattr((endtime-starttime), 'minutes', None):
        print('Total Approx Processing Time:', str((endtime-starttime).minutes))
    sys.stdout.flush()

    return perm_inds, A2_unshuffle


