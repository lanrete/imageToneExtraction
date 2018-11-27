#!/usr/bin/env python
# Created by lanrete at 11/27/18


import os
from PIL import Image, ImageDraw

import numpy as np
import scipy.cluster


def read_file(image_path):
    """
    Create object from local path
    Parameters
    ----------
    image_path: str
        Path where the images sit

    Returns
    -------
    ret: ImageExtractor
        A basic object created from the local path given
    """
    if not os.path.exists(image_path):
        raise ValueError('Image path {} not exist.'.format(image_path))
    img = ImageExtractor()
    img.raw_image = Image.open(image_path)
    img.initialized = True
    return img


class ImageExtractor(object):
    def __init__(self):
        self.image = None
        self.raw_image = None
        self.image_array = None
        self.tones = None
        self.initialized = False
        self.tone_image = None
        pass

    def reduce_size(self, max_width):
        h, w = self.raw_image.size[0], self.raw_image.size[1]
        if w <= max_width:
            self.image = self.raw_image
            return self
        new_w = int(max_width)
        new_h = int(new_w / w * h)
        self.image = self.raw_image.resize((new_h, new_w), Image.ANTIALIAS)
        return self

    def unstack_pixel(self):
        image_array = np.asarray(self.image)
        shape = image_array.shape
        # This will `unstack` the original array into a linear fashion.
        # scipy.product(shape[:2]) get the total number of points in the images
        # reshape(x, 3) reshape the current 3d-array from h*w*3 => (h*w)*3
        self.image_array = image_array.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
        return self

    def extract_tones(self, num):
        image_array = self.image_array
        codes, dist = scipy.cluster.vq.kmeans(image_array, num)
        # vecs, dist = scipy.cluster.vq.vq(image_array, codes)
        # counts, bins = scipy.histogram(vecs, len(codes))
        codes = [[int(_) for _ in code] for code in codes]
        self.tones = codes
        return self

    def print_tones(self):
        for ind, each_tone in enumerate(self.tones, 1):
            print(f'Tone {ind}: {each_tone}')
        return

    def combine_tones(self):
        image_width, image_height = self.image.size[0], self.image.size[1]

        width = round(image_width / len(self.tones))
        height = width

        tone_image = Image.new('RGB', (image_width, height + image_height))
        tone_image.paste(self.image, (0, 0))

        for ind, each_tone in enumerate(self.tones):
            tone_tuple = (each_tone[0], each_tone[1], each_tone[2])
            temp_image = Image.new('RGB', (width, height), tone_tuple)
            tone_image.paste(temp_image, (width * ind, image_height))
        self.tone_image = tone_image
        return self

    def add_borders(self):
        image_width, image_height = self.image.size[0], self.image.size[1]
        width = round(image_width / len(self.tones))
        height = width

        border_width = 10
        border_color = self.__get_board_color()
        draw = ImageDraw.Draw(self.tone_image)
        draw.line(xy=((0, image_height), (image_width - 1, image_height)),
                  fill=border_color,
                  width=border_width)
        for ind, _ in enumerate(self.tones[:-1], 1):
            draw.line(xy=((width * ind, image_height), (width * ind, image_height + height)),
                      fill=border_color,
                      width=int(width / 20))
        return self

    def __get_board_color(self):
        return 'white'


def main():
    pass


if __name__ == '__main__':
    main()
