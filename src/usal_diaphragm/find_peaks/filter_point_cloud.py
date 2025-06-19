# -*- coding: utf-8 -*-
"""
Created on Sat Feb 05 16:25:47 2022

@author: ptresadern
"""
import numpy as np


def filter_isolated_points_from(ii, jj, kk,
                                threshold=3.0,
                                n_neighbours=2):
    """Filter out points that have no nearby neighbours"""
    iii, jjj, kkk = [], [], []
    t_sqrd = threshold**2

    ijk_mat = np.array([ii, jj, kk]).transpose()

    n = len(ijk_mat)

    # Compute the distance between every pair of points
    d_sqrd_mat = np.zeros((n, n))
    for i, pi in enumerate(ijk_mat):
        d_sqrd_mat[i, i] = t_sqrd * 2

        for j0, pj in enumerate(ijk_mat[i + 1:]):
            j = j0 + i + 1
            dp = pi - pj
            d_sqrd_mat[j, i] = d_sqrd_mat[i, j] = np.inner(dp, dp)

    # Keep only the ones that are <threshold distance from
    # at least n_neighbours.
    for i in range(n):
        z = d_sqrd_mat[i]
        if sum(z < t_sqrd) >= n_neighbours:
            iii.append(ijk_mat[i, 0])
            jjj.append(ijk_mat[i, 1])
            kkk.append(ijk_mat[i, 2])

    return iii, jjj, kkk
