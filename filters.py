import math
import numpy as np
from scipy import ndimage


def to_grayscale(img):
    if img.ndim == 2:
        return img.astype(float)
    r = img[:, :, 0].astype(float)
    g = img[:, :, 1].astype(float)
    b = img[:, :, 2].astype(float)
    return 0.299 * r + 0.587 * g + 0.114 * b


def add_noise(img, std=20.0):
    if std < 0:
        raise ValueError("std must be >= 0")
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img + noise, 0, 255)


def gaussian_blur(img, sigma=1.2):
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    return ndimage.gaussian_filter(img.astype(float), sigma=sigma)


def sobel_edges(img):
    gx = ndimage.sobel(img.astype(float), axis=1)
    gy = ndimage.sobel(img.astype(float), axis=0)
    grad = np.hypot(gx, gy)
    if np.max(grad) == 0:
        return grad
    return grad / np.max(grad) * 255.0


def adaptive_threshold(img, window_size=25, c=10.0):
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be odd and >= 3")
    local_mean = ndimage.uniform_filter(img.astype(float), size=window_size)
    return np.where(img < local_mean - c, 0.0, 255.0)

def median_blur(img, size=3):
    if size < 3 or size % 2 == 0:
        raise ValueError("size must be odd and >= 3")
    return ndimage.median_filter(img.astype(float), size=size)


def add_salt_pepper(img, prob=0.05):
    out = img.astype(float).copy()
    rnd = np.random.rand(*img.shape)
    out[rnd < prob / 2] = 0.0
    out[rnd > 1 - prob / 2] = 255.0
    return out