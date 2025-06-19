'''
Created on 15 Feb 2022

@author: ptresadern
'''
import os

from matplotlib import pyplot as pp
# from matplotlib.animation import FuncAnimation

import numpy as np
import skimage

IMG_FOLDER = r"F:\Visionomy\001 University of Salford\001 Diaphragm motion\Image files\Image formats\AVI files\IMG_20211216_11_4_frames\subset"


def main():
    """-"""
    all_files = os.listdir(IMG_FOLDER)
    png_files = [
        f for f in all_files
        if f.endswith(".png")
    ]
    images_list = [
        pp.imread(os.path.join(IMG_FOLDER, f))[110:360, 765:950, :]
        for f in png_files
    ]
    # images_list = [images_list[0]]
    n_img = len(images_list)

    m = int(np.sqrt(n_img))
    n = int(n_img / m) + 1

    fig = pp.figure()

    for i, img in enumerate(images_list):
        pp.subplot(m, n, i + 1)

        greyscale_img = img.sum(axis=2) / 3
        # filtered_img = skimage.filters.sobel_h(greyscale_img)
        # filtered_img = skimage.filters.roberts(greyscale_img)
        # filtered_img = skimage.filters.prewitt_h(greyscale_img)
        filtered_img = skimage.filters.difference_of_gaussians(
            greyscale_img,
            low_sigma=[1, 0],
            high_sigma=[2, 0],
        )

        # _find_crosshair_in(images_list[0])
        # _animate_frames_in(images_list)

        peaks = np.array(
            _find_highest_vertical_peaks_in(
                filtered_img,
                n_per_col=1,
            )
        )
        peak_map = _peak_map_from(filtered_img, peaks)

        hspace, angles, dists = skimage.transform.hough_line(
            peak_map,
            theta=np.linspace(-3 * np.pi / 4, -np.pi / 4, 181)
        )
        good_peaks = skimage.transform.hough_line_peaks(hspace, angles, dists)

        # hspace, angles, dists = skimage.transform.hough_line(
        #     peak_map,
        #     theta=np.linspace(-np.pi / 4, np.pi / 4)
        # )
        # bad_peaks = skimage.transform.hough_line_peaks(hspace, angles, dists)
        # bad_peaks = (
        #     bad_peaks[0][:2],
        #     bad_peaks[1][:2],
        #     bad_peaks[2][:2],
        # )
        bad_peaks = (
            [0, 0],
            [-0.208, 0.208],
            [118, 65],
        )

        img_sz = img.shape
        _plot_lines_from(img_sz, good_peaks, color="y")
        # _plot_lines_from(img_sz, bad_peaks, color="r")

        peaks = _filter_peaks(peaks, good_peaks, bad_peaks)
        if len(peaks) > 0:
            pp.plot(peaks[:, 1], peaks[:, 0], 'c.')

        pp.imshow(greyscale_img, cmap="gray")
        # pp.imshow(filtered_img, cmap="gray")

        pp.title("Frame {:d}".format(i))
        pp.axis("off")

    pp.show()

    return


def _find_crosshair_in(img):
    """-"""
    cross_hairs = []
    for ri, row in enumerate(img):
        for ci, pixel in enumerate(row):
            r, g, _ = pixel
            if r > g:
                cross_hairs.append([ri, ci])

    cross_hairs = np.array(cross_hairs).sum(axis=0) / len(cross_hairs)

    return cross_hairs


# def _animate_frames_in(images_list):
#     """-"""
#     # def animate(i):
#     #     """-"""
#     #     img = images_list[i]
#     #     img = img[110:360, 765:950, :]
#     #     pp.imshow(img)
#
#     anim = FuncAnimation(
#         fig, animate,
#         interval=100, frames=len(t) - 1
#     )
#
#     pp.draw()
#     pp.show()


def _find_highest_vertical_peaks_in(img,
                                    n_per_col=1,
                                    abs_threshold=0,
                                    diff_threshold=0):
    """-"""
    peaks = []
    for c in range(0, img.shape[1]):
        max_so_far = -1.0
        rmax, cmax = -1, -1

        peaks_i = []
        for r in range(1, img.shape[0] - 1):
            is_max = (
                img[r, c] > abs_threshold and
                img[r, c] - img[r - 1, c] > diff_threshold and
                img[r, c] - img[r + 1, c] > diff_threshold
            )
            if is_max:
                peaks_i.append([img[r, c], r, c])

        if len(peaks_i) > 0:
            peaks_i.sort(reverse=True)
            peaks.extend([
                [pk[1], pk[2]] for pk in
                peaks_i[:n_per_col]
            ])

        # if img[r, c - 1] != 0 and img[r, c + 1] != 0:
        #     peaks.append([rmax, cmax])

    return peaks

    # peaks1 = skimage.feature.peak_local_max(img, min_distance=1)
    #
    # peaks2 = skimage.feature.peak_local_max(img[:, 1:], min_distance=1)
    # peaks2[:, 1] += 1
    #
    # return list(peaks1) + list(peaks2)


def _peak_map_from(img, peaks_list):
    """-"""
    peak_map = np.zeros(img.shape)
    for r, c in peaks_list:
        peak_map[r, c] = 1.0

    return peak_map


def _plot_lines_from(img_sz, hough_peaks, color="b"):
    """-"""
    ax = pp.gca()
    for _, angle, dist in zip(*hough_peaks):
        x0, y0 = dist * np.array([np.cos(angle), np.sin(angle)])
        ax.axline((x0, y0), slope=np.tan(angle + np.pi / 2), color=color)

    return


def _filter_peaks(peaks_in, good_peaks, bad_peaks,
                  dist_threshold=5):
    """-"""
    peaks_out = []
    for peak_i in peaks_in:
        keep_this_peak = (
            _get_min_dist_from(peak_i, good_peaks) <= dist_threshold and
            _get_min_dist_from(peak_i, bad_peaks) > 2 * dist_threshold and
            True
        )
        if keep_this_peak:
            peaks_out.append(peak_i)

    return np.array(peaks_out)


def _get_min_dist_from(peak_i, hough_peaks):
    """-"""
    min_perp_dist = np.inf
    for _, angle, dist in zip(*hough_peaks):
        # angle += np.pi / 2
        dvec = np.array([np.cos(angle), np.sin(angle)])
        a = np.array(dist * dvec)
        b = np.array([peak_i[1], peak_i[0]])
        ab = b - a

        angle += np.pi / 2
        dvec = np.array([np.cos(angle), np.sin(angle)])
        proj = np.inner(dvec, ab) * dvec
        diff = ab - proj
        perp_dist = np.sqrt(np.inner(diff, diff))

        if perp_dist < min_perp_dist:
            min_perp_dist = perp_dist

    return min_perp_dist


if __name__ == "__main__":
    main()
