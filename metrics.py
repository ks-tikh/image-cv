import math
import numpy as np


def mse(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("shapes must match")
    diff = img1.astype(float) - img2.astype(float)
    return float(np.mean(diff ** 2))


def psnr(img1, img2):
    err = mse(img1, img2)
    if err == 0:
        return float("inf")
    return 10.0 * math.log10((255.0 ** 2) / err)


def contrast(img):
    return float(np.std(img.astype(float)))
