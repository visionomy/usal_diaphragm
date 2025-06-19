import os
import tempfile
import yaml 

import numpy as np 
import pickle

from usal_diaphragm.app import _app_args

from usal_diaphragm.vol.read_4d_vol import read_4d_vol_from
from usal_diaphragm.fit_surface import fit_surface
from usal_diaphragm.find_peaks import peak_finders
from usal_diaphragm.geom import convert, line_circle_intersection


class DiaphragmApp:

    def __init__(self, description, app_args):
        """-"""
        self._description = description
        self._temp_dir = None

        self._args = self._get_args_from(app_args[1:])
            
        if self._args.dist_weight < 0:
            raise ValueError("dist_weight argument should not be negative.")
 
        if self._args.dev and self._args.randseed <= 0:
            self._args.randseed = 3141592

        if self._args.randseed > 0:
            np.random.seed(self._args.randseed)

        self._vol_props = None
        self._frames_dict = {}

    def main(self):
        """-"""
        if self._args.config == "new":
            msg = f"Created new config file at {self._args.input}"
            print(msg)
            res = []
        else:
            res = [
                self._process_file(
                    os.path.join(self._args.input_root, self._args.input_vol), 
                )
            ]

            return res

    # Nonexported methods

    def _process_file(self, full_filename):
        """-"""
        print("Opening " + full_filename)
        self._set_temp_dir_from(full_filename)

        return self._process_source(full_filename)
    
    def _process_source(self, full_filename):
        """-"""
        raise NotImplementedError

    def _crunch_the_numbers(self, full_filename):
        """-"""
        self._load_and_preprocess(full_filename)
        self._process_peaks()

        if self._args.surface != "none" and self._frames_dict["peaks"] != []:
            all_inliers_lists, best_candidates_path = self._process_polar_surface()

            best_polar_params_path = self._frames_dict["best_params"]
            surface_frames = [
                self._rect_fitter.surface_from(*point_clouds_t, best_polar_params_path[t])
                for t, point_clouds_t in enumerate(self._frames_dict["peaks"])
            ]
            self._frames_dict.update({
                "surface": surface_frames,
            })

            if self._args.sphere:   
                self._process_sphere_surface(surface_frames, best_candidates_path)

                # Compute the points on the surface for every frame in the
                # sequence.
                self._get_polar_surface_points(all_inliers_lists, best_candidates_path)

                self._project_surface_centre_path()

                # Don't try to combine this line with the one above
                self._frames_dict.update({
                    "rms_error": self._rms_error(),
                })
                
                if len(self._args.direction_points) == 2:
                    self._frames_dict.update({
                        "excursion": self._excursion(),
                    })

        return self._frames_dict

    def _project_surface_centre_path(self):
        """Reduce the 3D trajectory of the 'centre' of the surface
        to a 1D trajectory via SVD.
        
        The 'centre' is defined as the point on the surface closest
        to the mean of all surface points.
        """
        surface_frames = self._frames_dict["surface"]
        surface_centre_path = self._frames_dict["surface_ref"]

        surface_centre_path = np.array(surface_centre_path)
        surface_centre_path[:,0], surface_centre_path[:,1], surface_centre_path[:,2] = convert.to_cartesian(
            surface_centre_path[:,0],
            surface_centre_path[:,1],
            surface_centre_path[:,2],
            self._vol_props,
        )

        surface_ref_mean = surface_centre_path.mean(axis=0)
        surface_centre_path -= surface_ref_mean

        # Add dummy values for averages that we're about to compute
        sphere_params = self._frames_dict["sphere_params"]
        sphere_params = np.hstack([
            sphere_params, 
            -1*np.ones((len(sphere_params), 2))
        ])

        # Get average centre and radius over the sequence
        self._update_averages_in_sphere_params(
            sphere_params, self._vol_props, surface_frames
        )

        u, s, v = np.linalg.svd(surface_centre_path, full_matrices=False)
        # Extra column exported as "surf_disp"
        sphere_params = np.hstack([sphere_params, np.zeros((len(sphere_params), 1))])
        sphere_params[:,-1] = u[:,0] * s[0]
                
        surface_centre_path = np.outer(u[:,0] * s[0], v[0,:])
        surface_centre_path += surface_ref_mean

        self._frames_dict.update({
            "sphere_params": sphere_params,
            "surface_ref": np.array(surface_centre_path),
        })

    def _get_polar_surface_points(self, all_inliers_lists, best_candidates_path):
        """-"""
        best_polar_params_path = self._frames_dict["best_params"]

        surface_frames = []
        surface_centre_path = []
        for t, (ii, jj, kk) in enumerate(self._frames_dict["peaks"]):
            cand = best_candidates_path[t]
            p_vec = best_polar_params_path[t]
            inliers = all_inliers_lists[t][cand]

            sii, sjj, skk = self._rect_fitter.surface_from(ii, jj, kk, p_vec)
            surface_frames.append(
                (sii[inliers], sjj[inliers], skk[inliers])
            )

            surface_pts = np.array(surface_frames[-1]).transpose()
            surface_mean = surface_pts.mean(axis=0)
            diff = surface_pts - surface_mean
            dist = np.sqrt((diff**2).sum(axis=1))
            surface_ref = surface_pts[np.argmin(dist)]
            surface_centre_path.append(surface_ref)

            self._frames_dict["peaks"][t] = tuple(
                self._frames_dict["peaks"][t][d][inliers]
                for d in range(3)
            )
            
        # Update the single structure
        self._frames_dict.update({
            "surface": surface_frames,
            "surface_ref": np.array(surface_centre_path),
        })

    def _process_sphere_surface(self, surface_frames, best_candidates_path):
        """-"""
        polar_params_list = self._frames_dict["params"][0]
        best_polar_params_path = self._frames_dict["best_params"]

        all_sphere_params = self._get_sphere_params(
            self._vol_props, 
            [[bps_t] for bps_t in best_polar_params_path], 
            self._frames_dict["peaks"],
        )
        sphere_params = np.array([
            all_sphere_params[t][0]
            for t, _ in enumerate(best_candidates_path)
        ])

        # Find frames that look as though the parameters have been
        # estimated incorrectly.
        if self._args.replace_outliers:
            outliers, estimated_sd = self._find_outliers_in(sphere_params)

            # Look for better parameters in the frames where we
            # think they are estimated badly.
            outliers, estimated_sd = self._replace_outliers_in(
                surface_frames, polar_params_list, sphere_params,
                self._frames_dict["peaks"], outliers, estimated_sd, self._vol_props
            )
    
            self._frames_dict["outliers"] = outliers

        self._frames_dict["sphere_params"] = sphere_params

    def _process_polar_surface(self):
        """-"""
        self._rect_fitter = fit_surface.create_fitter(self._args.surface, self._args)

        polar_params_list, all_scores_vecs, all_inliers_lists = \
                self._get_surface_params(self._frames_dict["peaks"])

        # Choose a single set of parameters for every frame
        best_candidates_path = self._find_param_path_from(
            polar_params_list, all_scores_vecs,
            self._args.n_surfaces_keep,
            self._args.dist_weight,
        )

        best_polar_params_path = np.array([
            polar_params_list[t][candidate]
            for t, candidate in enumerate(best_candidates_path)
        ])
        
        # Update the single structure
        self._frames_dict.update({
            "params": (polar_params_list, self._args.n_surfaces_keep),
            "best_params": best_polar_params_path,
        })

        return all_inliers_lists, best_candidates_path

    def _process_peaks(self):
        if self._args.peaks != "none":
            point_clouds = self._get_peaks(self._frames_dict["raw"])
        else:
            self._args.surface = "none"
            point_clouds = []

        # Put everything into a dictionary for convenience.
        self._frames_dict.update({
            "peaks": point_clouds,
        })
        
    def _load_and_preprocess(self, full_filename):
        """-"""
        res = read_4d_vol_from(full_filename)
        if res:
            vol_arrays_list, self._vol_props = res
        else:
            return

        # Trim the first tstart frames from the beginning
        vol_arrays_list = vol_arrays_list[self._args.tstart:]

        # Trim to just the first n_frames_max frames
        vol_arrays_list = vol_arrays_list[:self._args.n_frames_max]
        self._frames_dict.update({
            "raw": vol_arrays_list,
        })

    def _rms_error(self):
        rms_error = []

        y_offset = self._args.slice_offsets[0]  # mm
        y_offset /= 1000.0  # m

        y_thresh = self._args.dy_thresh # mm
        y_thresh /= 1000.0  # m

        for fi in range(len(self._frames_dict.get("sphere_params"))):
            cx, cy, cz, r = self._frames_dict.get("sphere_params")[fi][:4]
            ii, jj, kk = self._frames_dict.get("peaks")[fi]
            xx, yy, zz = convert.to_cartesian(ii, jj, kk, self._vol_props)

            inds = abs(yy - y_offset) < y_thresh

            dx = xx[inds] - cx
            dy = yy[inds] - cy
            dz = zz[inds] - cz
            r_delta = np.sqrt(dx**2 + dy**2 + dz**2)
            r_scale = r / r_delta
            xx2 = (dx * r_scale) + cx
            yy2 = (dy * r_scale) + cy
            zz2 = (dz * r_scale) + cz

            if len(dx) > 0:
                squared_errors = (xx[inds]-xx2)**2 + (yy[inds]-yy2)**2 + (zz[inds]-zz2)**2
                mean_sqrd_error = np.mean(squared_errors)
                rms_error.append(np.sqrt(mean_sqrd_error)*1000.0)  # mm
            else:
                rms_error.append(-1)

        return rms_error

    def _excursion(self):
        """-"""
        excursion = []

        y_offset = self._args.slice_offsets[0]
        xz1, xz2 = np.array(self._args.direction_points, dtype=np.float64)

        y_offset /= 1000.0
        xz1 /= 1000.0
        xz2 /= 1000.0

        vec1 = np.array(xz2) - np.array(xz1)
        mag1 = np.sqrt(np.dot(vec1, vec1))
        vec1 /= mag1

        for fi in range(len(self._frames_dict.get("sphere_params"))):
            cx, cy, cz, r = self._frames_dict.get("sphere_params")[fi][:4]
            rprime = np.sqrt(r**2 - (y_offset - cy)**2)
            
            xzint = line_circle_intersection(xz1, xz2, cx, cz, rprime)

            vec2 = np.array(xzint) - xz1
            mag2 = np.sqrt(np.dot(vec2, vec2))
            cos_theta = np.dot(vec1, vec2) / mag2

            d = mag2*cos_theta
            excursion.append([d, *xzint])

        exc_array= np.array(excursion)
        excursion_m = abs(max(exc_array[:,0]) - min(exc_array[:,0]))
        print(f"Excursion range = {excursion_m*1000.0:.3g}mm")

        mean_excursion = sum([row[0] for row in excursion])/len(excursion)
        for row in excursion:
            row.append((row[0] - mean_excursion)*1000.0)
    
        return excursion
    
    def _get_peaks(self, vol_arrays_list):
        """-"""
        peaks_file = os.path.join(self._temp_dir, "peaks.bin")
        if self._args.dev and os.path.exists(peaks_file):
            print("Loading peaks from " + peaks_file)
            with open(peaks_file, "rb") as fid:
                point_clouds = pickle.load(fid)
        else:
            point_clouds = self._point_clouds_from(vol_arrays_list)
            if self._args.dev:
                with open(peaks_file, "wb") as fid:
                    pickle.dump(point_clouds, fid)

        return point_clouds

    def _point_clouds_from(self, vol_arrays_list):
        """-"""
        point_clouds = []

        peak_finder = peak_finders.create(self._args.peaks, self._args)
        for i, volume in enumerate(vol_arrays_list):
            print(
                "Finding peaks in frame {:03d} of {:03d}".format(
                    i + 1, len(vol_arrays_list)
                ),
                end="\r",
            )
            point_clouds.append(peak_finder.find_peaks_in(volume))

        print("")

        return point_clouds

    def _get_surface_params(self, point_clouds):
        """-"""
        surface_params_file = os.path.join(self._temp_dir, "surface_params.bin")
        if self._args.dev and os.path.exists(surface_params_file):
            print("Loading surfaces from " + surface_params_file)
            with open(surface_params_file, "rb") as fid:
                surface_tuple = pickle.load(fid)
        else:
            surface_tuple = self._surface_params_from(point_clouds)
            if self._args.dev:
                with open(surface_params_file, "wb") as fid:
                    pickle.dump(surface_tuple, fid)

        polar_params_list, all_scores_vecs, all_inliers_lists = surface_tuple

        return polar_params_list, all_scores_vecs, all_inliers_lists

    def _surface_params_from(self, point_clouds):
        """-"""
        polar_params_list = []
        all_scores_vecs = []
        all_inliers_lists = []

        for i, (ii, jj, kk) in enumerate(point_clouds):
            print(
                "Finding surfaces in frame {:03d} of {:03d}".format(
                    i + 1, len(point_clouds)
                ),
                end="\r",
            )

            best_n_params, best_n_scores, inliers_list_i = self._rect_fitter.fit_to(
                ii, jj, kk, 
                self._args.dist_threshold,
            )

            polar_params_list.append(best_n_params)
            all_scores_vecs.append(best_n_scores)
            all_inliers_lists.append(inliers_list_i)

        print("")

        return polar_params_list, all_scores_vecs, all_inliers_lists

    def _get_sphere_params(self, vol_props, polar_params_list, point_clouds):
        """-"""
        # sphere_params_file = os.path.join(self._temp_dir, "sphere_params.bin")
        # if self._args.dev and os.path.exists(sphere_params_file):
        #     print("Loading spheres from " + sphere_params_file)
        #     with open(sphere_params_file, "rb") as fid:
        #         sphere_params = pickle.load(fid)
        # else:
        sphere_params = self._sphere_params_from(
            vol_props, polar_params_list, point_clouds
        )
        # if self._args.dev:
        #     with open(sphere_params_file, "wb") as fid:
        #         pickle.dump(sphere_params, fid)

        return np.array(sphere_params)

    def _sphere_params_from(self, vol_props, polar_params_list, point_clouds):
        """-"""
        sphere_params = []

        for t, params_vecs_t in enumerate(polar_params_list):
            print(
                "Fitting sphere in frame {:03d} of {:03d}".format(
                    t + 1, len(polar_params_list)
                ),
                end="\r",
            )

            sphere_params_t = []
            for i, params_vec_ti in enumerate(params_vecs_t):
                surfii, surfjj, surfkk = self._rect_fitter.surface_from(
                    *point_clouds[t], params_vec_ti 
                )
                cart_ii, cart_jj, cart_kk = convert.to_cartesian(
                    surfii, surfjj, surfkk, 
                    vol_props
                )

                c, r = self._cart_fitter.fit_to(cart_ii, cart_jj, cart_kk)
                sphere_params_t.append([c[0], c[1], c[2], r])

            sphere_params.append(sphere_params_t)

        print("")

        return sphere_params

    def _replace_outliers_in(
            self, surface_frames, polar_params_list, sphere_params,
            point_clouds, outlier_indicators, estimated_sd, vol_props):
        """-"""
        # Repeat until either no outliers are left or we run out of
        # proposed surfaces.

        k = 1
        while sum(outlier_indicators) > 0 and k < self._args.n_surfaces_keep:
            print(str(sum(outlier_indicators)) + " outlier frames remaining")

            outlier_frames = [
                i
                for i, is_outlier in enumerate(outlier_indicators)
                if is_outlier
            ]

            for t in outlier_frames:
                ii, jj, kk = point_clouds[t]

                # Instead of choosing the "best" set of parameters
                # (which have been deemed to be an outlier), try the next
                # best set, and then the next best, and so on.
                surfii, surfjj, surfkk = self._rect_fitter.surface_from(
                    ii, jj, kk, polar_params_list[t][k]
                )

                # Replace the original surface points with this new
                # estimate.
                surface_frames[t] = (surfii, surfjj, surfkk)

                # surfii = surfii + self._args.ztrim
                cart_ii, cart_jj, cart_kk = convert.to_cartesian(
                    surfii, surfjj, surfkk, 
                    vol_props
                )

                # Replace the old sphere parameters with this new
                # estimate
                sf = fit_surface.create_fitter("sphere", self._args)
                c, r = sf.fit_to(cart_ii, cart_jj, cart_kk)
                sphere_params[t][:4] = [c[0], c[1], c[2], r]

            # Recompute averages
            self._update_averages_in_sphere_params(
                sphere_params, vol_props, surface_frames
            )

            # Recompute statistics for the new points and identify
            # any remaining outliers.
            outlier_indicators, estimated_sd = self._find_outliers_in(sphere_params)

            k += 1

        return outlier_indicators, estimated_sd

    def _update_averages_in_sphere_params(self, sphere_params, vol_props, surface_frames):
        """Add average centre and average radius to the sphere parameters"""
        mat = sphere_params
        mat_mean = mat.sum(axis=0) / len(mat)

        # Average/fixed centre of the sphere over the sequence
        c_mean = mat_mean[:3]
        c_diff = mat[:, :3] - c_mean

        # Distance of the i'th centre from the mean at every frame.
        # (Note: this will always be positive and therefore effectively a 
        # full-wave rectified version of the displacement.)
        c_rad = np.sqrt((c_diff**2).sum(axis=1))

        for i, (surfii, surfjj, surfkk) in enumerate(surface_frames):
            # Peaks in cartesian space
            cart_ii, cart_jj, cart_kk = convert.to_cartesian(
                surfii, surfjj, surfkk, 
                vol_props
            )
            surface_xyz = np.array([cart_ii, cart_jj, cart_kk]).transpose()
            surface_diff = surface_xyz - c_mean

            # Average distance to the fixed for every point in this frame
            surface_rad_t = np.sqrt((surface_diff**2).sum() / len(surface_diff))

            # Exported as "r_norm" and "c_norm"
            sphere_params[i][4:6] = [surface_rad_t, c_rad[i]]

        return 
        
    @staticmethod
    def _find_outliers_in(sphere_params):
        """-"""
        # Subtract the median radius from the estimated radius over the
        # whole sequence.
        # Median is more robust to outliers, which is what we want to detect.
        c_rad = np.array(sphere_params)[:, -1]
        c_rad -= np.median(c_rad)

        # Random number generator
        rng = np.random.default_rng(12345)

        inds = list(range(len(c_rad)))
        sd_list = []
        for _ in range(100):
            # Choose five frames at random from the sequence
            inds = rng.permutation(inds)
            selected_c_rad = c_rad[inds[:5]]

            # Compute the standard deviation of the radius for those
            # five frames.
            sd_list.append(np.std(selected_c_rad))

        # Take the median of the s.d. for the 100 random samples.
        # The thinking is that outliers will give an inflated s.d.
        # but as long as at least half of our random samples contain
        # no outliers then the median s.d. should be about right.
        estimated_sd = np.median(sd_list)

        # Define an outlier as any frame where the estimated displacement
        # of the sphere's centre from the fixed centre is more than three
        # standard deviations.
        outliers = (c_rad > 3 * estimated_sd)

        print(
            str(sum(outliers)) + " outliers found from"
            " estimated s.d. of " + str(1000.0 * estimated_sd) + " mm"
        )

        return outliers, estimated_sd
    
    @staticmethod
    def _find_param_path_from(polar_params_list, scores_vecs, k, dist_weight):
        """Return a reasonable time series from a list of candidates"""
        # Every frame comes with a number of candidate surface parameters
        # Each set of parameters has a score
        # This function returns a weighted average from the set of
        # candidates so that every frame has one set of parameters only.

        # params_path = []
        # for params_i, scores_i in zip(polar_params_list, scores_vecs):
        #     params_i = params_i[:k]
        #     scores_i = scores_i[:k]

        #     nz = [s > 0 for s in scores_i]
        #     if any(nz):
        #         # Use the mean of the parameters for whic
        #         mean_p = sum([
        #             weight * params
        #             for weight, params in zip(scores_i, params_i)
        #         ])
        #         mean_p /= sum(scores_i)
        #     else:
        #         mean_p = params_i[0]

        #     params_path.append(mean_p)

        # return params_path
    
        nt = len(scores_vecs)
        best_candidates_path = [None] * nt

        best_score = -1
        best_t = None
        best_candidate = None
        # best_params = None
        for t, scores_t in enumerate(scores_vecs):
            if max(scores_t[:k]) > best_score:
                best_t = t
                best_candidate = np.argmax(scores_t[:k])
                best_score = scores_t[best_candidate]

        best_candidates_path[best_t] = best_candidate

        def _update(best_candidates_path, t, offset):
            prev_best_candidate = best_candidates_path[t-offset]
            prev_best_params = polar_params_list[t-offset][prev_best_candidate]

            dist_from_current = polar_params_list[t][:k] - prev_best_params
            dist_from_current = np.sqrt((dist_from_current**2).sum(axis=1))

            # A negative dist_weight penalises trajectories where consecutive parameter
            # vectors are far apart.
            overall_scores = scores_vecs[t][:k] - dist_weight*dist_from_current
            best_candidates_path[t] = np.argmax(overall_scores)

        # forwards
        for t in range(best_t+1, nt):
            _update(best_candidates_path, t, 1)

        # backwards
        for t in reversed(range(0, best_t)):
            _update(best_candidates_path, t, -1)

        return best_candidates_path

    def _set_temp_dir_from(self, full_filename):
        """-"""
        if self._args.dev:
            # Create a temporary folder for intermediate values to speed
            # up development
            tempdir = os.path.join(
                tempfile.gettempdir(), 
                "diaphragm_temp",
                os.path.relpath(full_filename, self._args.input_root) + ".tmp",
            )
            os.makedirs(tempdir, exist_ok=True)
        else:
            tempdir = "."

        self._temp_dir = tempdir
        print(f"Created temp dir {os.path.abspath(tempdir)}")

    def _get_args_from(self, app_args):
        """-"""
        args = self._arg_parser().parse_args(app_args)

        if args.config == "new":
            self._write_new_config(args)
        elif args.input_vol != "":
            if args.input != "":
                msg = "Cannot supply two input files"
                raise IOError(msg)
            else:
                if not args.input_vol.endswith(".vol"):
                    msg = "input_vol filename must end with '.vol'"
                    raise IOError(msg)
        else:
            if args.input != "":
                if args.input.endswith(".yaml"):
                    args.config = args.input
                elif args.input.endswith(".vol"):
                    args.input_vol = args.input
            else:
                msg = "Must supply either an input .vol or .yaml file"
                raise IOError(msg)

        if args.config != "" and args.config != "new":
            with open(os.path.join(args.input_root, args.config), "rt") as fp:
                loaded_config_dict = yaml.safe_load(fp)

            app_args_as_str = " ".join(app_args)
            
            for k, v in loaded_config_dict.items():
                if k in args and f"--{k}" not in app_args_as_str:
                    args.__setattr__(k, v)

            if args.input_root == "":
                args.input_root = os.path.dirname(os.path.abspath(args.config))

        return args

    def _write_new_config(self, args):
        with open(args.input, "wt") as fp:
            output_dict = dict(args.__dict__)
            for k in ["config", "input"]:
                del output_dict[k]
            yaml.safe_dump(output_dict, fp)

    def _arg_parser(self):
        """Define all the arguments that the user can pass to the program"""
        parser = _app_args.new_parser(
            "Script to process a 4D volume and output"
            " graphical feedback."
        )

        _app_args.add_input_args_to(parser)
        _app_args.add_trim_args_to(parser)
        _app_args.add_peaks_args_to(parser)
        _app_args.add_surface_args_to(parser)
        _app_args.add_plot_args_to(parser)

        return parser
