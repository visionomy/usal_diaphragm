"""
Module to convert from rectangular to cartesian coordinates.
"""
import numpy as np


def to_cartesian(ii, jj, kk, vol_props):
    """Convert probe's i,j,k values to cartesian x,y,z ones."""
    nj = vol_props["dim_j"]
    nk = vol_props["dim_k"]
    assert len(vol_props["theta_angles"]) == nj
    assert len(vol_props["phi_angles"]) == nk

    radialSpacingMm = vol_props["rad_res"]
    radialSpacingStartMm = vol_props["offset1"] * vol_props["rad_res"]
    bModeRadius = -vol_props["offset2"] * vol_props["rad_res"]

    xx = []
    yy = []
    zz = []

    phi_ref = np.pi / 2.0
    theta_ref = np.pi / 2.0
    for i, j, k in zip(ii, jj, kk):
        phi = vol_props["phi_angles"][int(k)] - phi_ref
        theta = vol_props["theta_angles"][int(j)] - theta_ref

        # rotate about theta from the origin
        # rotate about phi from a point displaced by bModeRadius from the origin

        r = radialSpacingStartMm + i * radialSpacingMm
        x = r * np.sin(theta)
        q = r * np.cos(theta) - bModeRadius
        y = q * np.sin(-phi)
        z = q * np.cos(-phi) + bModeRadius

        xx.append(x)
        yy.append(y)
        zz.append(z)

    return np.array(xx), np.array(yy), np.array(zz)


def to_polar(xx, yy, zz, vol_props,
             imin=0,
             interp="nearest"):
    """Convert probe's i,j,k values to cartesian x,y,z ones."""
    ni = vol_props["dim_i"]
    nj = vol_props["dim_j"]
    nk = vol_props["dim_k"]
    assert len(vol_props["theta_angles"]) == nj
    assert len(vol_props["phi_angles"]) == nk

    radialSpacingMm = vol_props["rad_res"]
    radialSpacingStartMm = vol_props["offset1"] * vol_props["rad_res"]
    bModeRadius = -vol_props["offset2"] * vol_props["rad_res"]

    theta_angles = np.array(vol_props["theta_angles"])
    phi_angles = np.array(vol_props["phi_angles"])

    ii = []
    jj = []
    kk = []

    phi_ref = np.pi / 2.0
    theta_ref = np.pi / 2.0

    delta = 1e-3
    for x, y, z in zip(xx, yy, zz):
        # r = radialSpacingStartMm + i * radialSpacingMm
        # x = r * np.sin(theta)
        # q = r * np.cos(theta) - bModeRadius
        # y = q * np.sin(-phi)
        # z = q * np.cos(-phi) + bModeRadius

        q = np.sqrt(y**2 + (z-bModeRadius)**2)
        r = np.sqrt(x**2 + (q+bModeRadius)**2)

        i = int((r - radialSpacingStartMm) / radialSpacingMm)
        theta = np.arccos((q+bModeRadius) / r)
        phi = -np.arcsin(y / q)

        if interp == "nearest":
            if x > 0:
                theta = theta_ref + theta
            else:
                theta = theta_ref - theta

            if theta_angles[0] - delta <= theta <= theta_angles[-1] + delta:
                j = np.argmin(abs(theta_angles - theta))
            else:
                j = -1

            phi = phi + phi_ref

            # phi_angles are ordered high to low
            if phi_angles[-1] - delta <= phi <= phi_angles[0] + delta:
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
