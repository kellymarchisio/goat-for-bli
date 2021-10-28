###############################################################################
##
## Matrix utility operations
##
## Written by Kelly Marchisio for use on the CLSP grid, Oct 2020.
## Note: the `read` function is mostly copied from: 
##      https://github.com/artetxem/vecmap/blob/master/embeddings.py
## 
###############################################################################
import numpy as np
import random

def read(file, threshold=0, vocabulary=None, dtype='float'):
    # Note: This read function is mostly copied from:
    # https://github.com/artetxem/vecmap/blob/master/embeddings.py
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    all_words2ind = {}
    used_words = []
    m = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        all_words2ind[word] = i
        if vocabulary is None:
            used_words.append(word)
            m[i] = np.fromstring(vec, sep=' ', dtype=dtype)
        elif word in vocabulary:
            used_words.append(word)
            m.append(np.fromstring(vec, sep=' ', dtype=dtype))
    return (all_words2ind, np.array(m), used_words) if vocabulary is None else (
            used_words, np.array(m, dtype=dtype))


######
# Sources for keep_bottomk, keep_topk, etc functions:
#   https://stackoverflow.com/questions/31790819/scipy-sparse-csr-matrix-how-to-get-top-ten-values-and-indices
#   https://stackoverflow.com/questions/1623849/fastest-way-to-zero-out-low-values-in-array
######

def keep_bottomk(m, k):
    # Use keep_bottomk if best scores are the *lowest* - for instance, if
    # they're distances. Note: If there are ties for the cutoff value, it will
    # keep *all* numbers below the cutoff. So you could have more than k
    # results returned per row.
    m = np.copy(m)
    for i, row in enumerate(m):
        kth_idx = row.argsort()[:k][-1]
        kth_val = row[kth_idx]
        row[row > kth_val] = 0
    return m


def keep_topk(m, k, strict=True):
    # Keeps topk items per row in a matrix. Note: If there are ties for the
    # cutoff value, it will keep *all* numbers above the cutoff. So you could
    # have more than k results returned per row if strict=False.
    m = np.copy(m)
    for i, row in enumerate(m):
        kth_idx = row.argsort()[-k:][0]
        kth_val = row[kth_idx]
        row[row < kth_val] = 0
    if strict:
        for row in m:
            nonzero = list(np.flatnonzero(row))
            if len(nonzero) > k:
                idx_to_zero = random.sample(nonzero, len(nonzero) - 1)
                row[idx_to_zero] = 0
    return m


def keep_topk_over_minprob(m, k, min_prob=0, strict=True):
    m = np.copy(m)
    m_topk = keep_topk(m, k, strict)
    # Numpy indexing
    # https://stackoverflow.com/questions/28430904/set-numpy-array-elements-to-zero-if-they-are-above-a-specific-threshold
    m_topk[m_topk < min_prob] = 0
    return m_topk

