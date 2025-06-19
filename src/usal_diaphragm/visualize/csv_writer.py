import csv

import numpy as np


class CsvWriter:
    
    def __init__(self, frames_dict, full_source, args):
        """-"""
        self._frames_dict = frames_dict 
        self._filename = full_source

        self._surface = args.surface
        self.csv_params = None

        self._suffix = args.csv_suffix
        
    def save_csv_files(self):
        """-"""
        if self._surface != "none":
            self._save_surface_points_to_csv()
            self._save_params_to_csv()
        else:
            print(
                "No surface parameters to save because"
                " args.surface == 'none'"
            )

        return

    def _save_surface_points_to_csv(self):
        """Save x,y,z values for surface points in every frame of the sequence"""
        csv_data = [
            ["Frame", "i", "j", "k"]
        ]
        for fi, (ii, jj, kk) in enumerate(self._frames_dict["surface"]):
            for ii_i, jj_i, kk_i in zip(ii, jj, kk):
                csv_data.append([fi, ii_i, jj_i, kk_i])

        output_csv_file = f"{self._filename}_surface_points{self._suffix}.csv"
        with open(output_csv_file, "w", newline="\n") as fid:
            wtr = csv.writer(fid)
            wtr.writerows(csv_data)

        return

    def _save_params_to_csv(self):
        """Save surface parameters for every frame."""
        # Column names
        csv_data = [
            ["Frame", "a", "b", "c"]
        ]

        pp = self._frames_dict["best_params"]
        for fi, (a, b, c) in enumerate(pp):
            csv_data.append([fi, a, b, c])

        if "sphere_params" in self._frames_dict:
            csv_data[0].extend(["cx", "cy", "cz", "r", "r_norm", "c_norm", "surf_disp", "Rel. r_norm (mm)"])
            sp = np.array(self._frames_dict["sphere_params"]) * 1000.0  # m -> mm

            # Add relative r_norm.
            surface_rad = sp[:, 4]
            sp = np.hstack([sp, (surface_rad-surface_rad.mean()).reshape((-1,1))])

            for fi, vals in enumerate(sp):
                csv_data[fi+1].extend(vals)

        if "excursion" in self._frames_dict:
            csv_data[0].extend(["excursion (m)", "x_int (m)", "z_int (m)", "Rel. excursion (mm)"])
            sp = np.array(self._frames_dict["excursion"])
            for fi, vals in enumerate(sp):
                csv_data[fi+1].extend(vals)

        if "rms_error" in self._frames_dict:
            csv_data[0].extend(["RMS error (mm)"])
            sp = np.array(self._frames_dict["rms_error"])
            for fi, val in enumerate(sp):
                csv_data[fi+1].append(val)

        output_csv_file = f"{self._filename}_surface_params{self._suffix}.csv"
        with open(output_csv_file, "w", newline="\n") as fid:
            wtr = csv.writer(fid)
            wtr.writerows(csv_data)

        print("Surface parameters saved to " + output_csv_file)

        self.csv_params = csv_data
