"""
Class for displaying a 3D scatter plot of points
"""
import os 

import numpy as np
from matplotlib import pyplot as plt

from usal_diaphragm.visualize import view_base
from usal_diaphragm.geom import convert


class VolumeSlice(view_base.ViewBase):

    def _create_axes(self):
        """Set up the axes"""
        return [
            self._fig.add_subplot()
        ]

    def _plot_frame(self, fi, ax=None):
        """-"""
        if ax is None:
            ax = self._axes_list[0]

        ax.cla()
        ax.set_title("Central slice with peaks")

        v = self._masked_volume(fi)

        # dim(v) = [nk, nj, ni, nchannels=3]
        mid_k = int(len(v) / 2)

        # volumes are indexed (k,j,i) but peaks are stored as (i,j,k)
        ax.imshow(
            v[mid_k, :, :, :],
            aspect="auto",
            origin="lower"
        )
        ax.set_xlabel("i")
        ax.set_ylabel("j")

        p = self._frames_dict.get("peaks")
        if p:
            ii, jj, kk = p[fi]
            ii = ii[kk == mid_k]
            jj = jj[kk == mid_k]
            ax.scatter(ii - self._ztrim, jj, marker=".")

        ax.axis("off")
        ax.axis("equal")

    def _movie_subdir(self):
        """-"""
        return "points_plot_movie"


class CartesianVolumeSlice(VolumeSlice):

    def __init__(self, fig, filename, frames_dict, vol_props, args):
        """-"""
        super().__init__(fig, filename, frames_dict, vol_props, args)

        self._yoffset = float(args.yoffset) / 1000.0  # m
        self._dy_thresh = args.dy_thresh / 1000.0  # m

        self._flip_x = args.flip_x
        self._flip_y = args.flip_y
        self._direction_line_points = args.direction_points

        self._ii = None
        self._jj = None 
        self._kk = None 
        self._nu = None
        self._nv = None

        self._nuv = 300
        self._zmin = self._zmax = self._xmax = None

        self._compute_ijk_inds()

    def _plot_frame(self, fi, ax=None):
        """-"""
        if ax is None:
            ax = self._axes_list[0]

        ax.cla()
        ax.set_title(f"Slice y={self._yoffset*1000}mm with peaks")

        img = np.ones((self._nv, self._nu, 3))

        vol = self._masked_volume(fi)

        # volumes are indexed (k,j,i) but peaks are stored as (i,j,k)
        for v in range(self._nv):
            for u in range(self._nu):
                if self._kk[v,u] > 0 and self._jj[v,u] > 0 and self._ii[v,u] > 0:
                    img[v,u,:] = vol[self._kk[v,u], self._jj[v,u], self._ii[v,u],:]

        if self._flip_x:
            img = img[:,::-1,:]
        if self._flip_y:
            img = img[::-1,:,:]

        ax.imshow(
            img,
            aspect="auto",
            origin="lower"
        )

        if self._frames_dict.get("peaks"):
            ii, jj, kk = self._frames_dict.get("peaks")[fi]
            xx, yy, zz = convert.to_cartesian(ii, jj, kk, self._vol_props)
            inds = abs(yy - self._yoffset) < self._dy_thresh
            uu, vv = self._xyz_to_uv(xx[inds], yy[inds], zz[inds])
            ax.scatter(uu, vv, marker=".")
        else:
            xx, yy, zz = None, None, None

        if self._frames_dict.get("sphere_params") is not None:
            self._plot_arc(fi, ax)
        elif self._frames_dict.get("surface"):
            ii, jj, kk = self._frames_dict.get("surface")[fi]
            xx, yy, zz = convert.to_cartesian(ii, jj, kk, self._vol_props)
            inds = abs(yy - self._yoffset) < self._dy_thresh
            uu, vv = self._xyz_to_uv(xx[inds], yy[inds], zz[inds])
            ax.scatter(uu, vv, marker=".")

        self._plot_excursion(fi, ax)

        ax.axis("off")
        ax.axis("equal")
        ax.set_xlim([0, self._nu])
        ax.set_ylim([0, self._nv])

        if len(self._direction_line_points) < 2:
            pts = plt.ginput(timeout=0.001)
            if pts != []:
                px, py, pz = self._uv_to_xyz(pts[0][0], pts[0][1])
                self._direction_line_points.append((px[0]*1000.0, pz[0]*1000.0))
                print(self._direction_line_points[-1])
                ax.scatter(pts[0][0], pts[0][1], marker=".")

    def _plot_excursion(self, fi, ax):
        if len(self._direction_line_points) > 0:
            xz1 = np.array(self._direction_line_points[0]) / 1000.0  # mm->m
            uv1 = self._xyz_to_uv(xz1[0], self._yoffset, xz1[1])
            ax.scatter(uv1[0], uv1[1], marker=".")

        if len(self._direction_line_points) > 1:
            xz2 = np.array(self._direction_line_points[1]) / 1000.0  # mm->m
            uv2 = self._xyz_to_uv(xz2[0], self._yoffset, xz2[1])
            ax.scatter(uv2[0], uv2[1], marker=".")
    
            ax.plot([uv1[0], uv2[0]], [uv1[1], uv2[1]], "--")

        if self._frames_dict.get("excursion") is not None:
            _, xint, zint, _ = self._frames_dict.get("excursion")[fi]
            uint, vint = self._xyz_to_uv(xint, self._yoffset, zint)
            ax.scatter(uint, vint, marker="*")

    def _plot_arc(self, fi, ax):
        cx, cy, cz, r = self._frames_dict.get("sphere_params")[fi][:4]
        theta = np.linspace(0, 2*np.pi, 1000)
        rprime = np.sqrt(r**2 - (self._yoffset - cy)**2)
        uu, vv = self._xyz_to_uv(
            cx + rprime*np.sin(theta),
            self._yoffset,
            cz + rprime*np.cos(theta),
        )
        ax.plot(uu, vv, ":")
        
    def _compute_ijk_inds(self):
        """-"""
        _, _, self._zmax = convert.to_cartesian(
            [self._vol_props["dim_i"]-1], [self._vol_props["dim_j"] // 2], [self._vol_props["dim_k"] // 2],
            self._vol_props
        )
        _, _, self._zmin = convert.to_cartesian(
            [0], [self._vol_props["dim_j"]-1], [self._vol_props["dim_k"]-1],
            self._vol_props
        )
        self._xmax, _, _ = convert.to_cartesian(
            [self._vol_props["dim_i"]-1], [self._vol_props["dim_j"] - 1], [self._vol_props["dim_k"] // 2],
            self._vol_props
        )

        zrange = self._zmax-self._zmin
        xrange = 2*self._xmax
        if zrange < xrange:
            self._nv = self._nuv
            self._nu = int(xrange/zrange * self._nuv)
        else:
            self._nu = self._nuv
            self._nv = int(zrange/xrange * self._nuv)

        xx, zz = np.meshgrid(
            np.linspace(-self._xmax, self._xmax, self._nu), 
            np.linspace(self._zmin, self._zmax, self._nv)
        )
        yy = self._yoffset * np.ones(zz.shape)

        ii, jj, kk = convert.to_polar(
            xx.flatten(), yy.flatten(), zz.flatten(),
            self._vol_props,
            imin=0,
        )

        self._ii = ii.reshape(zz.shape)
        self._jj = jj.reshape(zz.shape)
        self._kk = kk.reshape(zz.shape)

    def _xyz_to_uv(self, xx, yy, zz):
        """-"""
        del yy

        zrange = float(self._zmax-self._zmin)
        xrange = float(2*self._xmax)
        if zrange < xrange:
            self._nv = self._nuv
            self._nu = int(xrange/zrange * self._nuv)
        else:
            self._nu = self._nuv
            self._nv = int(zrange/xrange * self._nuv)

        uu = ((xx - 0.0) / xrange) * self._nu + (self._nu / 2)
        vv = ((zz - float(self._zmin)) / zrange) * self._nv

        return uu, vv

    def _uv_to_xyz(self, uu, vv):
        """-"""
        zrange = self._zmax-self._zmin
        xrange = 2*self._xmax
        if zrange < xrange:
            self._nv = self._nuv
            self._nu = int(xrange/zrange * self._nuv)
        else:
            self._nu = self._nuv
            self._nv = int(zrange/xrange * self._nuv)

        xx = ((uu - (self._nu / 2)) / self._nu) * xrange
        yy = self._yoffset
        zz = ((vv / self._nv) * zrange) + self._zmin

        return xx, yy, zz

