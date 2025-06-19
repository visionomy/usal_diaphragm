# -*- coding: utf-8 -*-
"""
Created on Sat Feb 05 16:25:47 2022

@author: ptresadern
"""
import numpy as np
from scipy import linalg

# Random Sample Consensus (RanSaC) is a relatively old method
# but was popular in its day because it works pretty well.
# It was designed to be robust in the presence of outliers that can
# have a disproportionate effect on the estimate.

# The idea is to take a sample of points that is *just* enough to
# estimate the parameters of the model. (Only two points are
# necessary to estimate the parameters of a line, for example.)
# You take a sample, estimate the model parameters and then see how many
# other points are roughly in agreement (e.g. is their distance from the
# fitted model below some threshold).

# Do that a fixed number of times (n_test here)
# and keep the parameters with the largest support (consensus).
# Any samples that include an outlier will result in parameters that
# are well wide of the mark, that therefore have little support and
# that are duly thrown away.

# In this implementation, I return all of the candidates, sorted from
# best to worst.


def create_fitter(fitter_type, args_obj):
    """-"""
    if fitter_type == "plane":
        fitter = RansacPlane(args_obj)
    elif fitter_type == "parabola":
        fitter = RansacParabola(args_obj)
    elif fitter_type == "sphere":
        fitter = RansacSphere(args_obj)

    return fitter


class _RansacFitterBase:

    """
    Base class for fitter that uses RanSaC to estimate a surface
    """

    def __init__(self, args_obj):
        """-"""
        # Number of surfaces to test
        self._n_test = args_obj.n_surfaces_test

        if args_obj.randseed > 0:
            seed = args_obj.randseed
        else:
            seed = None

        self._rng = np.random.default_rng(seed)

    def fit_to(self, ii, jj, kk,
               dist_threshold=25.0):
        """-"""
        ijk = np.array([ii, jj, kk], dtype=np.float64).transpose()
        n = len(ijk)

        # Normalize points for numerical stability
        mn = ijk.mean(axis=0)
        sd = ijk.std(axis=0)
        ijk -= mn
        ijk /= sd

        a_mat = self._a_mat_from(ijk)

        tuples_list = []
        inliers = []
        for _ in range(self._n_test):
            # Keep looking for parameters until a valid guess is found
            # (usually on the first attempt).
            x_vec = None
            while x_vec is None or x_vec[0] > -0.5:
                inds = np.argsort(self._rng.random(n))
                x_vec = self._fit_surface_to(ijk[inds])

            # Predict the points on the parameterised surface
            pred_ii_normalized = np.matmul(a_mat, x_vec)
            pred_ii_natural = (pred_ii_normalized * sd[0]) + mn[0]

            # Compute the error
            e_vec = abs(ii - pred_ii_natural)

            # Assign a score based on the number of observed points
            # within a given distance from the fitted surface.
            inliers = e_vec < dist_threshold
            tuples_list.append((x_vec, sum(inliers), inliers))

        # Sort the estimates from best to worst
        tuples_list.sort(
            key=lambda tup: tup[1],
            reverse=True,
        )

        best_x_vecs = np.array([v[0] for v in tuples_list])
        scores_list = [v[1] for v in tuples_list]
        inliers_list = [v[2] for v in tuples_list]

        return (
            best_x_vecs, scores_list, inliers_list
        )

    def surface_from(self, ii, jj, kk, params_vec):
        """Return a set of surface points for given parameters"""
        ijk = np.array([ii, jj, kk], dtype=np.float64).transpose()

        mn = ijk.mean(axis=0)
        sd = ijk.std(axis=0)
        ijk -= mn
        ijk /= sd

        a_mat = self._a_mat_from(ijk)

        ijk[:, 0] = np.matmul(a_mat, params_vec)

        ijk *= sd
        ijk += mn

        return ijk[:, 0], ijk[:, 1], ijk[:, 2]


class RansacPlane(_RansacFitterBase):

    """Fitter that approximates points with a plane"""

    def _fit_surface_to(self, rand_ijk):
        """-"""
        rand_ijk = rand_ijk[:3]

        b_vec = rand_ijk[:, 0]
        a_mat = self._a_mat_from(rand_ijk)
        try:
            x_vec = np.linalg.solve(a_mat, b_vec)
        except np.linalg.LinAlgError:
            x_vec = None

        if x_vec is not None and not np.allclose(np.matmul(a_mat, x_vec), b_vec):
            x_vec = None

        return x_vec

    def _a_mat_from(self, ijk_mat):
        """-"""
        return np.array([
            ijk_mat[:, 1],
            ijk_mat[:, 2],

            np.ones(len(ijk_mat)),
        ]).transpose()


class RansacParabola(_RansacFitterBase):

    """Fitter that approximates points with a parabola"""

    def _fit_surface_to(self, rand_ijk):
        """-"""
        rand_ijk = rand_ijk[:6]

        b_vec = rand_ijk[:, 0]
        a2_mat = self._a_mat_from(rand_ijk)

        try:
            x_vec = np.linalg.solve(a2_mat, b_vec)
        except np.linalg.LinAlgError:
            x_vec = None

        if x_vec is not None and not np.allclose(np.matmul(a2_mat, x_vec), b_vec):
            x_vec = None

        return x_vec

    def _a_mat_from(self, ijk_mat):
        """-"""
        return np.array([
            ijk_mat[:, 1]**2,
            ijk_mat[:, 2]**2,

            2 * ijk_mat[:, 1] * ijk_mat[:, 2],
            2 * ijk_mat[:, 1],
            2 * ijk_mat[:, 2],

            np.ones(len(ijk_mat)),
        ]).transpose()


class RansacSphere(_RansacFitterBase):

    """-"""

    def fit_to(self, ii, jj, kk,
               dist_threshold=25.0):
        """Estimate centre and radius of a sphere from a point cloud"""
        # If you take a pair of points from the surface of a sphere,
        # draw a line from one to the other, and compute the plane
        # that bisects the line in 3D, the centre of the sphere lies
        # somewhere in that plane.
        # Do that for lots of points and the intersection of all the
        # planes should give you the actual centre, from which you can
        # compute its radius.
        del dist_threshold 

        ii = list(ii)
        jj = list(jj)
        kk = list(kk)
        xyz = np.array([ii, jj, kk]).transpose()

        n, d = xyz.shape
        assert d == 3

        k = self._n_test
        
        nrml_mat = np.zeros((k, 3))
        nrml_p_vec = np.zeros(k)
        for i in range(k):
            p1, p2 = self.__pair_of_points_from(xyz)

            nrml_dir = p2 - p1
            nrml = nrml_dir / np.sqrt(np.dot(nrml_dir, nrml_dir))
            nrml_mat[i] = nrml

            midp = (p1 + p2) / 2.0
            nrml_p_vec[i] = np.dot(nrml, midp)

        # This is the estimated centre
        c_est = np.matmul(linalg.pinv(nrml_mat), nrml_p_vec)

        xyz -= c_est
        r_est_vec = np.sqrt((xyz**2).sum(axis=1))
        r_est = r_est_vec.sum() / n

        return c_est, r_est

    @staticmethod
    def __pair_of_points_from(xyz):
        """-"""
        n = xyz.shape[0]

        same_point = True
        while same_point:
            p1 = xyz[np.random.randint(n)]
            p2 = xyz[np.random.randint(n)]
            same_point = (p1 == p2).all()

        return p1, p2
