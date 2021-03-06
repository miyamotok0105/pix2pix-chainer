# -*- coding: utf-8 -*-
import numpy as np
from scipy.misc import imread, imresize, imsave

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = imread(filepath)
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
    img = imresize(img, (256, 256))
    # img = np.transpose(img, (2, 0, 1))
    return img

def save_img(img, filename):
    img = deprocess_img(img)
    img = img.numpy()
    img *= 255.0
    img = img.clip(0, 255)
    img = np.transpose(img, (1, 2, 0))
    img = imresize(img, (250, 200, 3))
    img = img.astype(np.uint8)
    imsave(filename, img)
    print("Image saved as {}".format(filename))

def preprocess_img(img):
    # [0,255] image to [0,1]
    min = img.min()
    max = img.max()
    img.add_(-min).mul_(1.0 / (max - min))
    # RGB to BGR
    # [0,1] to [-1,1]
    img = img.mul_(2).add_(-1)
    # check that input is in expected range
    assert img.max() <= 1, 'badly scaled inputs'
    assert img.min() >= -1, "badly scaled inputs"
    return img

def deprocess_img(img):
    # BGR to RGB
    # [-1,1] to [0,1]
    img = img.add_(1).div_(2)
    return img
