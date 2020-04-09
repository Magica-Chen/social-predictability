#!/usr/bin/env python
# -*- coding: utf-8 -*-

# entropy_functions.py
# (c) Zexun Chen, 2020-04-09
# sxtpy2010@gmail.com

import pandas as pd
import mpmath
import numpy as np
import collections

# As required by algorithm, N should be large, we set e as the threshold of N.
# if it is smaller than threshold, we will just print NA


def getPredictability(N, S, e=100):
    if (N >= e) & np.isfinite(S):
        f = lambda x: (((1 - x) / (N - 1)) ** (1 - x)) * x ** x - 2 ** (-S)
        root = mpmath.findroot(f, 1)
        return float(root.real)
    else:
        return np.nan


def shannon_entropy(seq):
    """Plain old Shannon entropy (in bits)."""
    C, n = collections.Counter(seq), float(len(seq))
    return -sum(c / n * np.log2(c / n) for c in list(C.values()))


# Since the LZ-entropy estimation only converge when the length is large,
# so we add one more arg for LZ-entropy function
def LZ_entropy(seq, lambdas=False, e=100):
    """Estimate the entropy rate of the symbols encoded in `seq`, a list of
    strings.

    Kontoyiannis, I., Algoet, P. H., Suhov, Y. M., & Wyner, A. J. (1998).
    Nonparametric entropy estimation for stationary processes and random
    fields, with applications to English text. IEEE Transactions on Information
    Theory, 44(3), 1319-1327.

    Bagrow, James P., Xipei Liu, and Lewis Mitchell. "Information flow reveals
    prediction limits in online social activity." Nature human behaviour 3.2
    (2019): 122-128.
    """
    N = len(seq)

    if N < e:
        return np.nan
    else:
        L = []
        for i, w in enumerate(seq):
            seen = True
            prevSeq = " %s " % " ".join(seq[0:i])
            c = i
            while seen and c < N:
                c += 1
                seen = (" %s " % " ".join(seq[i:c])) in prevSeq
            l = c - i
            L.append(l)

        if lambdas:
            return L
        return (1.0 * N / sum(L)) * np.log2(N)


# Since the LZ-cross_entropy estimation only converge when the length is large,
# so we add one more arg for this function
def LZ_cross_entropy(W1, W2, PTs, lambdas=False, e=100):
    """Find the cross entropy H_cross(W2|W1), how many bits we would need to
    encode the data in W2 using the information in W1. W1 and W2 are lists of
    strings, PTs is a list of integers with the same length as W2 denoting the
    relative time ordering of W1 vs. W2. These integers tell us the position
    PTs[x] = i in W1 such that all symbols in W1[:i] occurred before the x-th
    word in W2.

    Bagrow, James P., Xipei Liu, and Lewis Mitchell. "Information flow reveals
    prediction limits in online social activity." Nature human behaviour 3.2
    (2019): 122-128.
    """

    lenW1 = len(W1)
    lenW2 = len(W2)

    if lenW1 < e | lenW2 < e:
        return np.nan
    else:
        L = []
        for j, (wj, i) in enumerate(zip(W2, PTs)):
            seen = True
            prevW1 = " %s " % " ".join(W1[:i])
            c = j
            while seen and c < lenW2:
                c += 1
                seen = (" %s " % " ".join(W2[j:c]) in prevW1)
            l = c - j
            L.append(l)

        if lambdas:
            return L

        if sum(L) == lenW2:
            return np.nan
        else:
            return (1.0 * lenW2 / sum(L)) * np.log2(lenW1)


def first_name_finder(s):
    if '-' in s:
        id = s.index('-')
        return s[:id]
    else:
        return 'unknown'