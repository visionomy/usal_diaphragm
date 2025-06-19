import numpy as np


def line_circle_intersection(uv1, uv2, cu, cv, r):
    """-"""
    m = (uv2[1] - uv1[1]) / (uv2[0] - uv1[0])
    c = uv1[1] - (m *  uv1[0])

    aa = (1 + m**2)
    bb = 2*m*c - 2*m*cv - 2*cu
    cc = c**2 + cu**2 + cv**2 -2*c*cv - r**2

    u = (-bb + np.sqrt(bb**2 - 4*aa*cc)) / (2*aa)
    if not min(uv1[0], uv2[0]) < u < max(uv1[0], uv2[0]):
        u = (-bb - np.sqrt(bb**2 - 4*aa*cc)) / (2*aa)

    v = m*u + c
    
    return u, v

