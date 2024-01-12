import os
import pickle
import cv2 as cv
import numpy as np
import tifffile as tif
from circles import prevent_overwrite

resize_factor = 5
path = 'data/combined'
centers_pickle = 'data/centers.p'


def print_coords(all_coords):
    """Print the clicked coords for each image"""

    for num, coords in enumerate(all_coords):
        print(f'Image num {num} has {len(coords)} circles with coordinates:')
        for x, y in coords:
            print(f'({x, y})')
        print(f'                ')


def mouse_event(event, x, y, flags, param):
    """Draw circles"""

    global img_resized, coord_list, img, h, w, name

    if event == cv.EVENT_LBUTTONDOWN:
        img_resized = cv.circle(img_resized, (x, y), 5, color=(255, 255, 255), thickness=3)
        cv.imshow(name, img_resized)
        coord_list.append([x // resize_factor, y // resize_factor])
        print(x, y)

    if event == cv.EVENT_RBUTTONDOWN:
        coord_list = []
        img_resized = cv.resize(img, (h * resize_factor, w * resize_factor))
        cv.imshow(name, img_resized)


prevent_overwrite(centers_pickle)
comp_list = []
for i, file_name in enumerate(os.listdir(path)):
    name = file_name[:-5]
    file_path = f'{path}/{file_name}'

    img = tif.imread(file_path)[0]
    h, w = img.shape
    img = img.reshape((h, w, 1))
    img = np.array(img * 255 / np.max(img), dtype=np.uint8)
    img = cv.applyColorMap(img, cv.COLORMAP_HOT)
    img_resized = cv.resize(img, (h * resize_factor, w * resize_factor))

    coord_list = []
    cv.imshow(name, img_resized)
    cv.setMouseCallback(name, mouse_event)
    cv.waitKey(0)
    cv.destroyAllWindows()
    comp_list.append(coord_list)

pickle.dump(comp_list, open(centers_pickle, "wb"))
print_coords(comp_list)
