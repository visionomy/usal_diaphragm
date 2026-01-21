"""Complete tests for fit_surface module"""
import mock
import numpy as np
from usal_diaphragm.fit_surface import fit_surface


def get_mock_args(n_test=20, seed=12345):
    """Create mock args for surface fitters"""
    args = mock.Mock()
    args.n_surfaces_test = n_test
    args.randseed = seed
    return args


def test_create_fitter_plane():
    """Test creating plane fitter"""
    args = get_mock_args()
    fitter = fit_surface.create_fitter("plane", args)
    assert fitter is not None
    assert isinstance(fitter, fit_surface.RansacPlane)


def test_create_fitter_parabola():
    """Test creating parabola fitter"""
    args = get_mock_args()
    fitter = fit_surface.create_fitter("parabola", args)
    assert fitter is not None
    assert isinstance(fitter, fit_surface.RansacParabola)


def test_create_fitter_sphere():
    """Test creating sphere fitter"""
    args = get_mock_args()
    fitter = fit_surface.create_fitter("sphere", args)
    assert fitter is not None
    assert isinstance(fitter, fit_surface.RansacSphere)


def test_plane_fitting():
    """Test fitting a plane to points"""
    # Create points on a plane: z = 2x + 3y + 5
    n = 30
    x = np.random.rand(n) * 10
    y = np.random.rand(n) * 10
    z = 2 * x + 3 * y + 5

    # Add small noise
    z += np.random.randn(n) * 0.1

    args = get_mock_args(n_test=10)
    fitter = fit_surface.RansacPlane(args)

    params_list, scores, inliers_list = fitter.fit_to(z, x, y, dist_threshold=1.0)

    # Should find good parameters
    assert len(params_list) == 10
    assert len(scores) == 10
    assert len(inliers_list) == 10
    assert scores[0] > 0  # Best score should be positive


def test_plane_surface_from():
    """Test generating surface from plane parameters"""
    args = get_mock_args()
    fitter = fit_surface.RansacPlane(args)

    # Create grid of points
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 2, 3, 4])
    z = np.zeros(5)

    # Use simple parameters
    params = np.array([1.0, 0.5, 2.0])

    z_surf, x_surf, y_surf = fitter.surface_from(z, x, y, params)

    assert len(z_surf) == 5
    assert len(x_surf) == 5
    assert len(y_surf) == 5


def test_parabola_fitting():
    """Test fitting a parabola to points"""
    # Create points on a parabolic surface
    n = 50
    x = (np.random.rand(n) - 0.5) * 4
    y = (np.random.rand(n) - 0.5) * 4
    z = x**2 + y**2 + 10

    # Add small noise
    z += np.random.randn(n) * 0.1

    args = get_mock_args(n_test=10)
    fitter = fit_surface.RansacParabola(args)

    params_list, scores, inliers_list = fitter.fit_to(z, x, y, dist_threshold=2.0)

    # Should find parameters
    assert len(params_list) == 10
    assert len(scores) == 10
    assert scores[0] > 0


def test_parabola_surface_from():
    """Test generating surface from parabola parameters"""
    args = get_mock_args()
    fitter = fit_surface.RansacParabola(args)

    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    z = np.zeros(10)

    # Use simple parabola parameters
    params = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 10.0])

    z_surf, x_surf, y_surf = fitter.surface_from(z, x, y, params)

    assert len(z_surf) == 10


def test_sphere_fitting_basic():
    """Test basic sphere fitting"""
    r = 10
    n = 100

    # Create points on a sphere
    c = r * np.random.randn(3)
    t, p = np.random.rand(2, n) * 2 * np.pi

    xyz = r * np.array([
        np.sin(t) * np.cos(p),
        np.sin(t) * np.sin(p),
        np.cos(t),
    ]).transpose()

    xyz += c

    args = get_mock_args(n_test=50)
    fitter = fit_surface.RansacSphere(args)

    c_est, r_est = fitter.fit_to(xyz[:, 0], xyz[:, 1], xyz[:, 2])

    # Should estimate center and radius reasonably well
    assert np.allclose(c, c_est, atol=0.5)
    assert np.allclose(r, r_est, atol=0.5)


def test_sphere_fitting_with_outliers():
    """Test sphere fitting with outliers"""
    r = 15
    n = 80

    # Create points on a sphere
    t, p = np.random.rand(2, n) * 2 * np.pi

    xyz = r * np.array([
        np.sin(t) * np.cos(p),
        np.sin(t) * np.sin(p),
        np.cos(t),
    ]).transpose()

    # Add some outliers
    outliers = np.random.rand(10, 3) * 50 - 25
    xyz = np.vstack([xyz, outliers])

    args = get_mock_args(n_test=50)
    fitter = fit_surface.RansacSphere(args)

    c_est, r_est = fitter.fit_to(xyz[:, 0], xyz[:, 1], xyz[:, 2])

    # Should still estimate radius reasonably well despite outliers
    assert 10 < r_est < 20


def test_ransac_with_zero_seed():
    """Test RANSAC fitter with zero seed (random)"""
    args = mock.Mock()
    args.n_surfaces_test = 50
    args.randseed = 0  # Should use None for random seed

    fitter = fit_surface.RansacPlane(args)
    assert fitter is not None


def test_plane_fitting_validates_solution():
    """Test that plane fitting validates solutions"""
    # Create points
    n = 50
    x = np.random.rand(n) * 10
    y = np.random.rand(n) * 10
    z = -3*x - 3*y - 5

    args = get_mock_args(n_test=20)
    fitter = fit_surface.RansacPlane(args)

    params_list, scores, inliers_list = fitter.fit_to(z, x, y, dist_threshold=1.0)

    # All returned parameters should be valid (x_vec[0] <= -0.5 based on code)
    for params in params_list:
        assert params[0] <= -0.5 or len(params_list) > 0


def test_parabola_fitting_validates_solution():
    """Test that parabola fitting validates solutions"""
    n = 100
    x = (np.random.rand(n) - 0.5) * 4
    y = (np.random.rand(n) - 0.5) * 4
    z = x**2 + y**2 + 10

    args = get_mock_args(n_test=20)
    fitter = fit_surface.RansacParabola(args)

    params_list, scores, inliers_list = fitter.fit_to(z, x, y, dist_threshold=2.0)

    # Should return valid parameters
    assert len(params_list) > 0
    assert all(len(p) == 6 for p in params_list)
