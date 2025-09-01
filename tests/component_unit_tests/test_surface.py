import mock

import numpy as np

from usal_diaphragm.fit_surface.fit_surface import RansacSphere


def test_fitting():
    """-"""
    r = 10
    n = 300

    c = r * np.random.randn(3)
    t, p = np.random.rand(2, n) * 2 * np.pi

    xyz = r * np.array([
        np.sin(t) * np.cos(p),
        np.sin(t) * np.sin(p),
        np.cos(t),
    ]).transpose()

    xyz += c

    args = mock.Mock()
    args.n_surfaces_test = 200
    args.randseed = 12345

    c_est, r_est = RansacSphere(args).fit_to(xyz[:,0], xyz[:,1], xyz[:,2])

    assert np.allclose(c, c_est)
    assert np.allclose(r, r_est)

    print("Test passed")

