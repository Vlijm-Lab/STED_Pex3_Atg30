import os
import pickle
import numpy as np
import pandas as pd
import tifffile as tif
import matplotlib.pyplot as plt
from scipy import ndimage


def tiff_list(path):
    """Returns list with all tiff files in given path"""

    file_list = os.listdir(path)
    name_list = [file for file in file_list if file.split('.')[-1] == 'tiff']

    return name_list


def prevent_overwrite(file_path):
    """Manually remove, to avoid overwriting data"""

    if os.path.isfile(file_path):
        print(f'{file_path} already exists. Remove first to start')
        exit()


def create_folder(path):
    """Create folder if not existing"""

    if not os.path.isdir(path):
        os.mkdir(path)


def folder_structure(path):
    """Create the folders for storage"""

    create_folder(path)
    create_folder(f'{path}/line_figs')
    create_folder(f'{path}/line_figs_n')


def init_dataframe():
    columns = ["File name",
               "Circle number",
               "#Circles in image",
               "#Data points",
               "Circumference (nm)",
               "R (raw)",
               "p (raw)",
               "R (ma)",
               "p (ma)",
               "mean Pex3",
               "mean ATG30",
               "inside mean",
               "outside mean",
               "inside percentage",
               "outside percentage"]

    df = pd.DataFrame(columns=columns)

    return df


def add_empty_row(df):
    """Empty row with corresponding index"""

    idx = len(df)
    df.loc[idx] = len(df.columns) * [np.nan]

    return df, idx


def load_image(im_name, thresh, show_images=0):
    """Load images"""

    im = tif.imread(f'{im_name}')
    pex = im[0]

    simg = ndimage.gaussian_filter(im[1], 1, output=float)
    bin_img = (ndimage.gaussian_filter(im[1], 1, output=float) > thresh)

    ar = np.where(bin_img)
    if show_images:
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))

        ax[0, 0].imshow(pex, cmap='Reds')
        ax[0, 1].imshow(im[1], cmap='Reds')
        ax[1, 0].imshow(simg, cmap='Reds')
        ax[1, 1].imshow(bin_img, cmap='Reds')

    return pex, bin_img, ar


def load_updated_pickle(path):
    """Load the circle centers and radii"""

    original_path = f'{path}/circle_data.p'
    updated_path = f'{path}/circle_data_manual_update.p'

    if os.path.isfile(updated_path):
        centers, rads = pickle.load(open(updated_path, "rb"))
    else:
        centers, rads = pickle.load(open(original_path, "rb"))

    return centers, rads


def get_image_lists(path):
    """Obtain lists of the sted and conf images"""

    image_list = os.listdir(path)
    sted_names = []
    conf_names = []
    for image_name in image_list:
        if len(image_name) == 9 and image_name[-4:] == 'tiff':
            sted_names.append(f'{path}/{image_name}')
        elif len(image_name) == 14 and image_name[-4:] == 'tiff':
            conf_names.append(f'{path}/{image_name}')

    sted_names.sort()
    conf_names.sort()

    for sted_path, conf_path in zip(sted_names, conf_names):
        if not sted_path[:-5] == conf_path[:-10]:
            print(f'{sted_path[:-5]} != {conf_path[:-10]} -- make sure the input data is correct.')
            break

    return sted_names, conf_names


def store_equal_sized(src, dst, pixel_ratio):
    """Combine stack of STED and confocal imageo of unequal dimensions"""

    create_folder(dst)
    sted_names, conf_names = get_image_lists(src)

    for sted_path, conf_path in zip(sted_names, conf_names):
        sted = tif.imread(sted_path)[0]
        conf = tif.imread(conf_path)[1]

        conf_large = np.repeat(np.repeat(conf, pixel_ratio, axis=1), pixel_ratio, axis=0)
        stacked_image = np.stack((sted, conf_large))
        tif.imwrite(f'{dst}/{sted_path.split("/")[-1]}', stacked_image)
