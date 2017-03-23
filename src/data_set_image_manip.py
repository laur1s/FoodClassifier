import numpy as np


# set of functions for processing images directly from training and testing datasets

def substract_mean(x_train, x_test):
    """Returns imageset with substracted mean image of the dateset
    this method should make different classes more distinguishable be remove values that
    repeats in most of the images.
    """
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image
    return x_train, x_test, mean_image


