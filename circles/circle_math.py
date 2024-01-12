import math
import numpy as np
import scipy
import pickle
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import circle_fit
from progressbar import progressbar
from skimage import measure
from circles.data_handling import create_folder, tiff_list, load_image


def rotate(origin, point, angle):
    """Returns rotated point"""

    rotated_x = math.cos(angle) * (point[0] - origin[0]) - math.sin(angle) * (point[1] - origin[1])
    rotated_y = math.sin(angle) * (point[0] - origin[0]) + math.cos(angle) * (point[1] - origin[1])
    rotated_point = [origin[0] + rotated_x, origin[1] + rotated_y]

    return rotated_point


def rotate_circle(circle, degrees):
    """Obtain coordinates of circle point"""

    x, y, r = circle

    rotation = -math.radians(degrees)
    circle_point = rotate([x, y], [x - r, y], rotation)

    return circle_point


def rotate_points(circles, main_circle, num_points, path):
    """Rotate points in main circle"""

    circle = circles[main_circle]
    rest_circles = circles.copy()
    rest_circles.pop(main_circle)

    in_list = list(np.zeros(num_points))

    fig, ax = plt.subplots(1, figsize=(10, 10))
    for i in range(num_points):
        degrees = i / num_points * 360
        circle_point = rotate_circle(circle, degrees)
        if check_if_inside(circle_point, rest_circles):
            in_list[i] = True
            color = 'tab:green'
        else:
            color = 'tab:red'

        ax.scatter(circle_point[0], -circle_point[1], s=30, color=color, alpha=.5)

    plt.savefig(f'{path}/test{main_circle:02d}.png')

    return in_list


def get_intersections(circle1, circle2):
    """Obtain intersecting coordinates of two circles"""

    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    center1 = np.array([x1, y1])
    center2 = np.array([x2, y2])

    dist = np.linalg.norm(center1 - center2)

    if (dist > r1 + r2) or (dist < np.abs(r1 - r2)) or (dist == 0 and r1 == r2):
        return []
    else:
        c1 = (r1 ** 2 - r2 ** 2 + dist ** 2) / (2 * dist)
        c2 = math.sqrt(r1 ** 2 - c1 ** 2)

        point = [x1 + c1 * (x2 - x1) / dist, y1 + c1 * (y2 - y1) / dist]
        intersect1 = [point[0] + c2 * (y2 - y1) / dist, point[1] - c2 * (x2 - x1) / dist]
        intersect2 = [point[0] - c2 * (y2 - y1) / dist, point[1] + c2 * (x2 - x1) / dist]

        return intersect1, intersect2


def inside_outside_line_profile(circle, circle_num, num_points, path):
    """Compute line profile which is inside"""

    circles_min = circle.copy()
    circles_min.pop(circle_num)
    points = []

    for circle_second in circles_min:
        points_cur = get_intersections(circle[circle_num], circle_second)
        if points_cur:
            for point in points_cur:
                points.append(point)

    line_profile_in = rotate_points(circle, circle_num, num_points, path)

    return line_profile_in


def circle_dist_to_obj(img, bin_img, circle, disp_imgs=True, path='', circle_num=1):
    """
    Calculate shortest distance for each part of the circle to the objectc
    """

    x, y, r = circle

    if disp_imgs:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap='Reds')
        ax.imshow(bin_img, cmap='gray_r', alpha=0.5)
        circ = Circle((x, y), r, color='r', fill=False, lw=1, alpha=1)
        ax.add_patch(circ)

    distance_list = []
    obj_coords = np.where(bin_img)
    circum = 2 * math.pi * r * 23
    steps = int(circum / 10)

    x_best, y_best = 0, 0
    for i in range(0, steps):
        shortest_d = 1000
        xi, yi = rotate([x, y], [x - r, y], -math.radians(i / steps * 360))

        for xo, yo in zip(obj_coords[1], obj_coords[0]):
            d = np.linalg.norm([xi - xo, yi - yo])
            if d < shortest_d:
                shortest_d = d
                x_best = xo
                y_best = yo

        distance_list.append(shortest_d)
        if disp_imgs:
            ax.plot([xi, x_best], [yi, y_best], 'b', alpha=0.1)

    if disp_imgs:
        plt.savefig(f'{path}/circle{circle_num}_to_obj.png')

    return distance_list


def calculate_overlap(circle1, circle2, degrees, linewidth, accuracy=100):
    """
    Calculate overlap for line with linewidth of point circle1 with circle2.
    The accuracy is determined by the number of steps taken on the line.
    """

    # init coordinates and radii of circles
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    # init counter
    counter = 0

    # loop through num. accuracy steps
    for i in range(accuracy):

        # calculate point on line between inner and outer ring of circle1
        linepoint = (i * 2 * linewidth) / accuracy - linewidth
        xi1, yi1 = rotate([x1, y1], [x1 - r1 + linepoint, y1], -math.radians(degrees))

        # calculate distance to center of circle2
        dist = np.linalg.norm([x2 - xi1, y2 - yi1])

        # boundaries inner and outer circle to center of circle2
        p1 = r2 + linewidth
        p2 = r2 - linewidth

        # count if dist falls between p1 and p2 bounds
        if (p1 > dist > p2) or (p1 < dist < p2):
            counter += 1

    return counter / accuracy


def shortest_dist_circles(circle1, circle2, degrees, linewidth):
    """
    Calculate the shortest distance to any point at circle2 from point at circumference of circle1 at given rotation
    in degrees.

    The line width determines the width of both line profiles of both circles
    """

    # init x-coordinate, y-coordinate and radius of both circles
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    # coordinates of point at circumference of circle1 at degrees
    xi, yi = rotate([x1, y1], [x1 - r1, y1], -math.radians(degrees))
    d = np.linalg.norm([x2 - xi, y2 - yi])
    p = (d - r2) / d

    # calculate overlap
    overlap = calculate_overlap(circle1, circle2, degrees, linewidth)

    # calculate coordinates
    xti = (xi + (x2 - xi) * p)
    yti = (yi + (y2 - yi) * p)
    coords = np.array([[xi, xti], [yi, yti]])
    distance = np.linalg.norm([xi - xti, yi - yti])

    return coords, distance, overlap


def circle_dist_overlap(img, circle1, circle2, linewidth, disp_imgs=0, path='', circle1_num=1, circle2_num=2):
    """
    Calculate the shortest distances from all points of circle1 to circle2. If there is overlap, calculate the
    percantage overlap.

    The linewidth determines the width of both line profiles of both circles
    """

    # Iniatialize image and circles (including linewidths)
    if disp_imgs:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap='Reds')

        colors = ['red', 'blue']

        # for each circle, display centre, circumference, and inner and outer linewidth
        for [x, y, r], color in zip([circle1, circle2], colors):
            ax.plot(x, y, 'k.')
            ax.plot([x, x + r], [y, y], 'k')

            circ = Circle((x, y), r, color='black', fill=False)
            ax.add_patch(circ)

    # init lists
    distance_list = []
    overlap_list = []

    x1, y1, r1 = circle1

    circum = 2 * math.pi * r1 * 23
    steps = int(circum / 10)
    # calculate from all rotations of circle 1
    for i in range(0, steps):

        # calculate distances and overlaps
        coordinates, distance, overlap = shortest_dist_circles(circle1, circle2, i / steps * 360, linewidth)

        distance_list.append(distance)
        overlap_list.append(overlap)

        # Draw distance lines:
        if disp_imgs:
            if len(coordinates) == 1:
                xb, yb = rotate([x1, y1], [x1 - r1 - linewidth, y1], -math.radians(i))
                xu, yu = rotate([x1, y1], [x1 - r1 + linewidth, y1], -math.radians(i))

                ax.plot([xb, xu], [yb, yu], 'r--', alpha=0.3)
                ax.plot(coordinates[0][0], coordinates[0][1], 'r.')
            else:
                ax.plot(coordinates[0], coordinates[1], 'b', alpha=0.1)
    if disp_imgs:
        plt.savefig(f'{path}/circle{circle1_num}_to_circle{circle2_num}_dist.png')

    return distance_list, overlap_list


def estimate_circles(image, centers_list, lw=2, show_image=0):
    """Circle estimation"""

    smoothed_image = ndimage.gaussian_filter(image, 1.0, output=float) * (image > 20)
    ho, wo = smoothed_image.shape

    rads = []
    il_lists = []

    if show_image:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap='Reds')

    for x, y in centers_list:

        pim = np.zeros((2 * ho, 2 * wo))
        ys = int(ho - y)
        xs = int(wo - x)
        pim[ys:ys + ho, xs:xs + wo] = smoothed_image

        ilist = []
        hl, wl = pim.shape

        for i in range(360):

            rimg = ndimage.rotate(np.copy(pim), i, reshape=False)

            fatline = rimg[ho - lw:ho + lw + 1, :]

            smoothed_hor = np.sum(fatline, axis=0)

            peaks = scipy.signal.find_peaks(smoothed_hor)
            if len(peaks[0]) > 1:
                inner_idx = peaks[0][0]
                outer_idx = peaks[0][-1]

                max_outer_peak = 0
                max_inner_peak = 0
                for p in peaks[0]:
                    m = wo
                    if p > m:
                        if smoothed_hor[p] > max_outer_peak:
                            max_outer_peak = smoothed_hor[p]
                            outer_idx = p
                    else:
                        if smoothed_hor[p] > max_inner_peak:
                            max_inner_peak = smoothed_hor[p]
                            inner_idx = p

                ilist.append((inner_idx, outer_idx))

        il = []
        ol = []
        ol2 = []
        dif = []
        for idx in ilist:
            il.append(idx[0])
            ol.append(idx[1])
            ol2.append(wl - idx[1])
            dif.append(idx[1] - idx[0])

        il_lists.append(np.asarray(il) * ((dif < np.median(dif) * 1.1) * (dif > np.median(dif) * 0.9)))
        rad = np.mean(np.asarray(dif)[(dif < np.median(dif) * 1.2) * (dif > np.median(dif) * 0.8)])
        rad = rad / 2
        rads.append(rad)

        if show_image:
            circ = Circle((x, y), rad - lw, color='black', fill=False, alpha=0.5)
            ax.add_patch(circ)
            circ = Circle((x, y), rad + lw, color='black', fill=False, alpha=0.5)
            ax.add_patch(circ)
            ax.plot(x, y, 'k.')

    return il_lists, rads


def improve_circles(image, radii, coordinate_list, path, lw):
    """Improve the found circles"""

    smoothed_image = ndimage.gaussian_filter(image, 1.0, output=float) * (image > 20)
    ho, wo = smoothed_image.shape

    tlists = []
    updated_coordinates = []
    updated_radii = []

    for (x, y), num in zip(coordinate_list, range(len(coordinate_list))):

        pim = np.zeros((2 * ho, 2 * wo))
        ys = int(ho - y)
        xs = int(wo - x)
        pim[ys:ys + ho, xs:xs + wo] = smoothed_image

        tlist = []

        for i in range(360):

            val = radii[num][i]
            if val != 0:
                tlist.append(rotate([ho, wo], [val, ho], math.radians(i)))

        tlists.append(np.array(tlist) - np.array([xs, ys]))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='Reds')

    for i in range(len(tlists)):

        a1, a2, a3, a4 = circle_fit.hyper_fit(tlists[i])
        if (not np.isnan(a1)) and (not np.isnan(a2)) and (not np.isnan(a3)):
            circ = Circle((a1, a2), a3 - lw, color='black', fill=False, alpha=0.5)
            ax.add_patch(circ)
            circ = Circle((a1, a2), a3 + lw, color='black', fill=False, alpha=0.5)
            ax.add_patch(circ)
            ax.plot(a1, a2, 'k.')

            updated_coordinates.append([a1, a2])
            updated_radii.append(a3)

    plt.savefig(f'{path}/circles.png')

    lp = show_ring_profile(image, updated_coordinates, updated_radii)
    return updated_coordinates, updated_radii, lp


def outer_circles(centers, rads, linewidth):
    """Returns outer circles with added linewidth"""

    circle_list = []

    for center, radius, in zip(centers, rads):
        if center:
            circle_list.append([center[0], center[1], radius + linewidth])

    return circle_list


def show_ring_profile(image, clist, rads, lw=2):
    """Display ring profile"""

    lp = []
    fat_lp = []
    cnt = 0

    for x, y in clist:
        cur_lp = []
        cur_fat_lp = []
        rad = rads[cnt]

        h, w = image.shape
        pim = np.zeros((2 * h, 2 * w))
        ys = int(h - y)
        xs = int(w - x)
        pim[ys:ys + h, xs:xs + w] = image

        h, w = pim.shape

        for i in range(360):
            rimg = ndimage.rotate(np.copy(pim), -i, reshape=False)

            part = rimg[int(h / 2), int(w / 2 - rad) - lw:int(w / 2 - rad) + lw + 1]
            cur_lp.append(part)

            fat_part = rimg[int(h / 2), int(w / 2 - rad) - 15:int(w / 2 - rad) + 15 + 1]
            cur_fat_lp.append(fat_part)

        cnt += 1
        lp.append(cur_lp)
        fat_lp.append(cur_fat_lp)

    for i in range(len(lp)):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(np.array(lp[i]).T, cmap='Reds')
        fig, ax = plt.subplots(figsize=(12, 12))
        ad = np.array(fat_lp[i]).T
        ax.imshow(ad, cmap='Reds')
        hi, wi = ad.shape
        ax.plot([0, wi], [hi // 2 - lw, hi // 2 - lw], color='Black', alpha=0.8)
        ax.plot([0, wi], [hi // 2 + lw, hi // 2 + lw], color='Black', alpha=0.8)
        ax.set_xlim(0, wi)

    return lp


def estimate_small_circles(img, path, x, y, threshold_ratio=.3):
    """Calculate circles which are too small for normal fit"""

    binary_img = img > threshold_ratio * np.max(img)
    labels = measure.label(binary_img)
    reg_props = measure.regionprops(labels)

    mdist = np.inf
    for r in reg_props:
        if r.area > 10:
            yc, xc = r.centroid
            cdist = np.linalg.norm([x - xc, y - yc])
            if cdist < mdist:
                mdist = cdist
                yu, xu = yc, xc
                ru = r.major_axis_length / 2

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(img, cmap='Reds')

    ax.plot(xu, yu, 'k.')
    circ = Circle((xu, yu), ru, color='Black', fill=False)
    ax.add_patch(circ)
    circ = Circle((xu, yu), ru - 4, color='Black', fill=False)
    ax.add_patch(circ)
    plt.savefig(f'{path}/circles.png')

    return [xu, yu], ru


def fit_circles(analysis_path, data_path, centers_path, lw, small=False):
    """"Circle fit for all input data in data path"""

    create_folder(analysis_path)
    name_list = tiff_list(data_path)
    comp_list = pickle.load(open(centers_path, "rb"))

    for num, name in zip(progressbar(range(len(name_list))), name_list):
        file_name = f'{analysis_path}/{name.split(".")[0]}'
        create_folder(file_name)

        pex, bin_img, ar = load_image(f'{data_path}/{name}', 5, show_images=1)
        centers = comp_list[num]

        if not small:
            il_lists, rads = estimate_circles(pex, centers)
            clist_u, rads_u, lp = improve_circles(pex, il_lists, centers, file_name, lw)
        else:
            x, y = centers[0]
            clist_u, rads_u = estimate_small_circles(pex, file_name, x, y)

        pickle.dump([clist_u, rads_u], open(f'{file_name}/circle_data.p', "wb"))
        plt.close('all')


def get_line_profile(image, clist, rads, linewidth, simg=0, col='Reds', path=''):
    """Obtain line profile of circle in given image"""

    lp = []
    fat_lp = []
    cnt = 0

    for ring_num, [x, y] in enumerate(clist):
        cur_lp = []
        cur_fat_lp = []
        rad = rads[cnt]

        h, w = image.shape
        pim = np.zeros((2 * h, 2 * w))
        ys = int(h - y)
        xs = int(w - x)
        pim[ys:ys + h, xs:xs + w] = image

        h, w = pim.shape

        circum = 2 * math.pi * rad * 23
        # print(circum)
        steps = int(circum / 10)

        for i in range(steps):
            # print(-(i/steps*360))
            rimg = ndimage.rotate(np.copy(pim), -(i / steps * 360), reshape=False)

            part = rimg[int(h / 2), int(w / 2 - rad) - linewidth:int(w / 2 - rad) + linewidth]
            cur_lp.append(part)

            fat_part = rimg[int(h / 2), int(w / 2 - rad) - 15:int(w / 2 - rad) + 15]
            cur_fat_lp.append(fat_part)

        cnt += 1
        lp.append(cur_lp)
        fat_lp.append(cur_fat_lp)

        if simg:
            fig, ax = plt.subplots(1, figsize=(7.5, 7.5))
            ax.imshow(image, cmap=col)
            circ = Circle((x, y), rad - linewidth, color='black', fill=False, alpha=0.5)
            ax.add_patch(circ)
            circ = Circle((x, y), rad + linewidth, color='black', fill=False, alpha=0.5)
            ax.add_patch(circ)
            ax.plot(x, y, 'k.')
            if col == 'Greens':
                plt.savefig(f'{path}/atg30_ring{ring_num + 1}.png')
            for i in range(len(lp)):
                fig, ax = plt.subplots(figsize=(12, 12))
                ax.imshow(np.array(lp[i]).T, cmap=col)
                fig, ax = plt.subplots(figsize=(12, 12))
                ad = np.array(fat_lp[i]).T
                ax.imshow(ad, cmap=col)
                hi, wi = ad.shape
                ax.plot([0, wi], [hi // 2 - linewidth, hi // 2 - linewidth], color='Yellow', alpha=0.8)
                ax.plot([0, wi], [hi // 2 + linewidth, hi // 2 + linewidth], color='Yellow', alpha=0.8)
                ax.set_xlim(0, wi)

    return lp


def in_circle(point, circle):
    """Check if point is in given circle"""

    x_point, y_point = point
    x_circle, y_circle, r_circle = circle
    inside = (x_point - x_circle) ** 2 + (y_point - y_circle) ** 2 <= r_circle ** 2

    return inside


def check_if_inside(circle, rest_circles):
    """Check if """
    inside = False
    for compare_circle in rest_circles:
        if in_circle(circle, compare_circle):
            inside = True
            break

    return inside
