import os
import pickle
import numpy as np
import tifffile as tif
from scipy import ndimage
import matplotlib.pyplot as plt
from progressbar import progressbar
from circles.circle_math import get_line_profile, circle_dist_to_obj, circle_dist_overlap
from circles.data_handling import load_updated_pickle
from circles.misc import assert_lists, remove_nones


def display(image, color='Reds'):
    """Display image (or stack)"""

    if len(image.shape) == 3:
        nimgs, _, _ = image.shape
        fig, ax = plt.subplots(1, nimgs, figsize=(15, 15))
        for i in range(nimgs):
            ax[i].imshow(image[i], cmap=color)
    else:
        fig, ax = plt.subplots(1, figsize=(7.5, 7.5))
        ax.imshow(image, cmap=color)


def thresh_img(img, path='', sigma=1, thresh_factor=.5):
    """Threshold input image """

    smoothed_img = ndimage.gaussian_filter(img, sigma, output=float)
    smoothed_max = np.max(smoothed_img)

    if smoothed_max > 10:
        thresholded_img = smoothed_img > smoothed_max * thresh_factor
    else:
        thresholded_img = np.zeros(smoothed_img.shape)

    fig, ax = plt.subplots(1, figsize=(7.5, 7.5))
    ax.imshow(thresholded_img, cmap='Greens')

    if path != '':
        plt.savefig(path)

    return thresholded_img


def obtain_data_figs(dst, src, lw):
    """Obtain the figure data for the circles"""

    img_list = os.listdir(dst)

    for num, name in zip(progressbar(range(len(img_list))), img_list):
        img_path = f'{dst}/{name}'

        centers, rads = load_updated_pickle(img_path)
        if centers:
            centers, rads = assert_lists(centers, rads)
            centers, rads = remove_nones(centers, rads)

            pex = tif.imread(f'{src}/{name}.tiff')[0]
            atg = tif.imread(f'{src}/{name}.tiff')[1]

            lp_pex = get_line_profile(pex, centers, rads, lw, simg=1, path=img_path)
            lp_atg = get_line_profile(atg, centers, rads, lw + 1, simg=1, col='Greens', path=img_path)
            pickle.dump([lp_pex, lp_atg], open(f'{img_path}/line_profiles.p', "wb"))

            bin_atg = thresh_img(atg, path=f'{img_path}/atg30_binary.png')

            for i in range(len(lp_pex)):
                distance_list = []
                overlap_list = []
                circle1 = [centers[i][0], centers[i][1], rads[i]]
                if np.sum(bin_atg) == 0:
                    obj_dist = np.zeros(len(lp_pex[0]))
                else:
                    obj_dist = circle_dist_to_obj(pex, bin_atg, circle1, disp_imgs=1, path=img_path, circle_num=i + 1)
                distance_list.append(obj_dist)
                lsum = np.mean(lp_pex[i], axis=1)
                asum = np.mean(lp_atg[i], axis=1)

                for j in range(len(lp_pex)):
                    if j != i:
                        circle2 = [centers[j][0], centers[j][1], rads[j]]
                        dlist, olist = circle_dist_overlap(pex, circle1, circle2, lw, disp_imgs=1, path=img_path,
                                                           circle1_num=i + 1, circle2_num=j + 1)
                        distance_list.append(dlist)
                        overlap_list.append(olist)
                pickle.dump([distance_list, overlap_list], open(f'{img_path}/circle{i + 1}_dist_overlap.p', "wb"))
                fig, ax = plt.subplots(1)
                ax.plot((lsum - np.min(lsum)) / (np.max(lsum) - np.min(lsum)))
                ax.plot((asum - np.min(asum)) / (np.max(asum) - np.min(asum)))
            plt.close('all')
