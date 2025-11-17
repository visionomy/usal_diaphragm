"""Tests for filter_point_cloud module"""
import numpy as np
from usal_diaphragm.find_peaks import filter_point_cloud


def test_filter_isolated_points_basic():
    """Test filtering of isolated points"""
    # Create a cluster of 3 nearby points
    ii = np.array([10.0, 10.1, 10.2, 50.0])
    jj = np.array([20.0, 20.1, 20.2, 60.0])
    kk = np.array([30.0, 30.1, 30.2, 70.0])

    # The isolated point (50, 60, 70) should be filtered out
    iii, jjj, kkk = filter_point_cloud.filter_isolated_points_from(
        ii, jj, kk, threshold=3.0, n_neighbours=2
    )

    # Should only have the cluster of 3 points
    assert len(iii) == 3
    assert len(jjj) == 3
    assert len(kkk) == 3


def test_filter_no_filtering():
    """Test case where no filtering occurs"""
    # All points are close together
    ii = np.array([10.0, 10.5, 11.0])
    jj = np.array([20.0, 20.5, 21.0])
    kk = np.array([30.0, 30.5, 31.0])

    iii, jjj, kkk = filter_point_cloud.filter_isolated_points_from(
        ii, jj, kk, threshold=3.0, n_neighbours=2
    )

    # All points should remain
    assert len(iii) == 3
    assert len(jjj) == 3
    assert len(kkk) == 3


def test_filter_all_isolated():
    """Test case where all points are isolated"""
    # All points far apart
    ii = np.array([0.0, 100.0, 200.0])
    jj = np.array([0.0, 100.0, 200.0])
    kk = np.array([0.0, 100.0, 200.0])

    iii, jjj, kkk = filter_point_cloud.filter_isolated_points_from(
        ii, jj, kk, threshold=3.0, n_neighbours=2
    )

    # No points should remain
    assert len(iii) == 0
    assert len(jjj) == 0
    assert len(kkk) == 0


def test_filter_custom_threshold():
    """Test with custom threshold"""
    ii = np.array([0.0, 1.0, 2.0, 10.0])
    jj = np.array([0.0, 0.0, 0.0, 0.0])
    kk = np.array([0.0, 0.0, 0.0, 0.0])

    # With threshold 5.0, the point at 10.0 should be filtered
    iii, jjj, kkk = filter_point_cloud.filter_isolated_points_from(
        ii, jj, kk, threshold=5.0, n_neighbours=2
    )

    # Should keep the first 3 points
    assert len(iii) == 3


def test_filter_single_neighbour():
    """Test with n_neighbours=1"""
    # Two pairs of points
    ii = np.array([0.0, 1.0, 100.0, 101.0, 200.0])
    jj = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    kk = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    iii, jjj, kkk = filter_point_cloud.filter_isolated_points_from(
        ii, jj, kk, threshold=5.0, n_neighbours=1
    )

    # Should keep 4 points (two pairs)
    assert len(iii) == 4
