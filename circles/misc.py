import numpy as np


def moving_average(x, kernel_size):
    """Returns moving average"""

    moving_avg = np.convolve(x, np.ones(kernel_size), 'same') / kernel_size
    return moving_avg


def unit_to_list(var):
    """If var is not a list, convert to list"""

    if type(var) != list:
        var = [var]

    return var


def assert_lists(list1, list2):
    """Assert list of lists"""

    if type(list1[0]) == np.float64:
        list1 = [list1]
        list2 = [list2]

    return list1, list2


def remove_nones(list1, list2):
    """Remove empty entries in list"""

    list1 = [x for x in list1 if x]
    list2 = [x for x in list2 if x]

    return list1, list2
