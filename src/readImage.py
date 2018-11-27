#!/usr/bin/env python
# Created by lanrete at 11/17/18

import os
from PIL import Image

import numpy as np
import scipy.cluster


def read_image(image_path):
    """
    The functions to read raw image and return a numpy array out of that.
    The array would be a 3d-array with the same size of the image itself.
    Each element in the 2d-array will be a tuple, which holds the RGB information of that pixel
    Parameters
    ----------
    image_path

    Returns
    -------

    """
    if not os.path.exists(image_path):
        raise ValueError(f'The path does\'nt exist. Please check again. {os.path.abspath(image_path)}')
    image = Image.open(image_path)
    image = reduce_size(image, 0.25)
    image_array = np.asarray(image)
    shape = image_array.shape
    # This will `unstack` the original array into a linear fashion.
    # scipy.product(shape[:2]) get the total number of points in the images
    # reshape(x, 3) reshape the current 3d-array from h*w*3 => (h*w)*3
    image_array = image_array.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
    return image_array


def reduce_size(image, multiplier):
    multiplier = min(0.5, multiplier)
    h, w = image.size[0], image.size[1]
    new_h, new_w = int(h * multiplier), int(w * multiplier)
    return image.resize((new_h, new_w), Image.ANTIALIAS)


def extract_tones(image_array, num_tones):
    # TODO Sort the colors by the degree of dominance
    codes, dist = scipy.cluster.vq.kmeans(image_array, num_tones)
    vecs, dist = scipy.cluster.vq.vq(image_array, codes)
    counts, bins = scipy.histogram(vecs, len(codes))

    print(codes)

    pass


def main():
    image_path = '../data/city.jpg'
    image_array = read_image(image_path)
    print(image_array.shape)
    print(image_array[0])
    extract_tones(image_array, 5)
    pass


if __name__ == '__main__':
    main()
