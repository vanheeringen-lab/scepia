# Copyright (c) 2020 Simon van Heeringen <simon.vanheeringen@gmail.com>
#
# This module is free software. You can redistribute it and/or modify it under
# the terms of the MIT License, see the file LICENSE included with this
# distribution.
from numba import njit
import numpy as np


@njit
def mean1(a):
    n = len(a)
    b = np.empty(n)
    for i in range(n):
        b[i] = a[i].mean()
    return b


@njit
def std1(a):
    n = len(a)
    b = np.empty(n)
    for i in range(n):
        b[i] = a[i].std()
    return b


@njit
def fast_corr(a, b):
    """ Correlation """
    n, k = a.shape
    m, k = b.shape

    mu_a = mean1(a)
    mu_b = mean1(b)
    sig_a = std1(a)
    sig_b = std1(b)

    out = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            out[i, j] = (a[i] - mu_a[i]) @ (b[j] - mu_b[j]) / k / sig_a[i] / sig_b[j]

    return out
