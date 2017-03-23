import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
import random


def load_images(directory):
    images = [plt.imread(os.path.join(directory, file))
              for file in os.listdir(directory)
              if '.JPEG' in file]
    return images


def display_examples(images):
    indexes = random.sample(range(len(images)), 4)
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(images[indexes[0]])
    axarr[0, 1].imshow(images[indexes[1]])
    axarr[1, 0].imshow(images[indexes[2]])
    axarr[1, 1].imshow(images[indexes[3]])


def resize_img(images):
    resized = []
    for image in images:
        resized.append(imresize(image, [50, 50], 'bilinear'))
    return resized
