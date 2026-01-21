"""
Module for filtering isolated points from point clouds.
"""
import numpy as np

# Default filtering parameters
DEFAULT_DISTANCE_THRESHOLD = 3.0
DEFAULT_MIN_NEIGHBOURS = 2


def filter_isolated_points_from(ii, jj, kk, threshold=DEFAULT_DISTANCE_THRESHOLD, n_neighbours=DEFAULT_MIN_NEIGHBOURS):
    """
    Filter out points that have too few nearby neighbours.

    This function removes isolated points from a point cloud by checking
    each point's distance to all other points and keeping only those that
    have at least n_neighbours within the specified threshold distance.

    Args:
        ii: Array of i-coordinates
        jj: Array of j-coordinates
        kk: Array of k-coordinates
        threshold: Maximum distance for points to be considered neighbours (default: 3.0)
        n_neighbours: Minimum number of neighbours required to keep a point (default: 2)

    Returns:
        Tuple of (iii, jjj, kkk) containing the filtered point coordinates
    """
    filtered_ii = []
    filtered_jj = []
    filtered_kk = []

    threshold_squared = threshold**2

    ijk_matrix = np.array([ii, jj, kk]).transpose()
    n_points = len(ijk_matrix)

    # Compute pairwise squared distances
    distance_squared_matrix = np.zeros((n_points, n_points))
    for i, point_i in enumerate(ijk_matrix):
        # Set diagonal to large value so point doesn't count as its own neighbour
        distance_squared_matrix[i, i] = threshold_squared * 2

        for j0, point_j in enumerate(ijk_matrix[i + 1:]):
            j = j0 + i + 1
            diff = point_i - point_j
            dist_sq = np.inner(diff, diff)
            distance_squared_matrix[j, i] = distance_squared_matrix[i, j] = dist_sq

    # Keep only points with sufficient neighbours
    for i in range(n_points):
        distances = distance_squared_matrix[i]
        num_close_neighbours = sum(distances < threshold_squared)

        if num_close_neighbours >= n_neighbours:
            filtered_ii.append(ijk_matrix[i, 0])
            filtered_jj.append(ijk_matrix[i, 1])
            filtered_kk.append(ijk_matrix[i, 2])

    return np.array(filtered_ii), np.array(filtered_jj), np.array(filtered_kk)
