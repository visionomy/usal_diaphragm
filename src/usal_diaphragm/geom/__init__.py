"""Geometry utility functions for line-circle intersections."""
import numpy as np


def line_circle_intersection(uv1, uv2, cu, cv, r):
    """
    Calculate intersection point between a line and a circle.

    Args:
        uv1: First point on the line (u, v)
        uv2: Second point on the line (u, v)
        cu: Circle center u-coordinate
        cv: Circle center v-coordinate
        r: Circle radius

    Returns:
        Tuple of (u, v) coordinates of the intersection point
    """
    # Calculate line parameters: v = m*u + c
    slope = (uv2[1] - uv1[1]) / (uv2[0] - uv1[0])
    intercept = uv1[1] - (slope * uv1[0])

    # Quadratic equation coefficients for line-circle intersection
    coeff_a = (1 + slope**2)
    coeff_b = 2*slope*intercept - 2*slope*cv - 2*cu
    coeff_c = intercept**2 + cu**2 + cv**2 - 2*intercept*cv - r**2

    # Try first solution
    discriminant = coeff_b**2 - 4*coeff_a*coeff_c
    u = (-coeff_b + np.sqrt(discriminant)) / (2*coeff_a)

    # If first solution is outside line segment, try second solution
    if not min(uv1[0], uv2[0]) < u < max(uv1[0], uv2[0]):
        u = (-coeff_b - np.sqrt(discriminant)) / (2*coeff_a)

    v = slope*u + intercept

    return u, v

