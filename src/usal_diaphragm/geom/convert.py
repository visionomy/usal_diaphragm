"""
Module to convert between polar and Cartesian coordinate systems for ultrasound data.
"""
import numpy as np

# Constants for coordinate system reference angles
PHI_REFERENCE = np.pi / 2.0
THETA_REFERENCE = np.pi / 2.0
POLAR_COORD_TOLERANCE = 1e-3


def to_cartesian(ii, jj, kk, vol_props):
    """
    Convert probe's polar i,j,k coordinates to Cartesian x,y,z coordinates.

    Args:
        ii: Radial indices (depth from probe)
        jj: Theta angle indices
        kk: Phi angle indices
        vol_props: Dictionary containing volume properties (dimensions, angles, resolution)

    Returns:
        Tuple of (xx, yy, zz) arrays in Cartesian coordinates
    """
    nj = vol_props["dim_j"]
    nk = vol_props["dim_k"]
    assert len(vol_props["theta_angles"]) == nj
    assert len(vol_props["phi_angles"]) == nk

    radial_spacing_mm = vol_props["rad_res"]
    radial_spacing_start_mm = vol_props["offset1"] * vol_props["rad_res"]
    bmode_radius = -vol_props["offset2"] * vol_props["rad_res"]

    xx = []
    yy = []
    zz = []

    for i, j, k in zip(ii, jj, kk):
        phi = vol_props["phi_angles"][int(k)] - PHI_REFERENCE
        theta = vol_props["theta_angles"][int(j)] - THETA_REFERENCE

        # Rotate about theta from the origin
        # Rotate about phi from a point displaced by bmode_radius from the origin
        radius = radial_spacing_start_mm + i * radial_spacing_mm
        x = radius * np.sin(theta)
        q = radius * np.cos(theta) - bmode_radius
        y = q * np.sin(-phi)
        z = q * np.cos(-phi) + bmode_radius

        xx.append(x)
        yy.append(y)
        zz.append(z)

    return np.array(xx), np.array(yy), np.array(zz)


def to_polar(xx, yy, zz, vol_props, imin=0, interp="nearest"):
    """
    Convert Cartesian x,y,z coordinates to probe's polar i,j,k coordinates.

    Args:
        xx: X-coordinates in Cartesian space
        yy: Y-coordinates in Cartesian space
        zz: Z-coordinates in Cartesian space
        vol_props: Dictionary containing volume properties
        imin: Minimum radial index to consider (default: 0)
        interp: Interpolation method - currently only "nearest" supported

    Returns:
        Tuple of (ii, jj, kk) arrays in polar coordinates
    """
    ni = vol_props["dim_i"]
    nj = vol_props["dim_j"]
    nk = vol_props["dim_k"]
    assert len(vol_props["theta_angles"]) == nj
    assert len(vol_props["phi_angles"]) == nk

    radial_spacing_mm = vol_props["rad_res"]
    radial_spacing_start_mm = vol_props["offset1"] * vol_props["rad_res"]
    bmode_radius = -vol_props["offset2"] * vol_props["rad_res"]

    theta_angles = np.array(vol_props["theta_angles"])
    phi_angles = np.array(vol_props["phi_angles"])

    ii = []
    jj = []
    kk = []

    for x, y, z in zip(xx, yy, zz):
        # Convert from Cartesian to polar
        q = np.sqrt(y**2 + (z - bmode_radius)**2)
        radius = np.sqrt(x**2 + (q + bmode_radius)**2)

        i = int((radius - radial_spacing_start_mm) / radial_spacing_mm)
        theta = np.arccos((q + bmode_radius) / radius)
        phi = -np.arcsin(y / q)

        if interp == "nearest":
            # Adjust theta based on x sign
            if x > 0:
                theta = THETA_REFERENCE + theta
            else:
                theta = THETA_REFERENCE - theta

            # Find nearest theta index with tolerance
            if theta_angles[0] - POLAR_COORD_TOLERANCE <= theta <= theta_angles[-1] + POLAR_COORD_TOLERANCE:
                j = np.argmin(abs(theta_angles - theta))
            else:
                j = -1

            phi = phi + PHI_REFERENCE

            # phi_angles are ordered high to low
            if phi_angles[-1] - POLAR_COORD_TOLERANCE <= phi <= phi_angles[0] + POLAR_COORD_TOLERANCE:
                k = np.argmin(abs(phi_angles - phi))
            else:
                k = -1

        if imin <= i < ni and 0 <= j < nj and 0 <= k < nk:
            ii.append(i)
            jj.append(j)
            kk.append(k)
        else:
            ii.append(-1)
            jj.append(-1)
            kk.append(-1)

    return np.array(ii), np.array(jj), np.array(kk)
