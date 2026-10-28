import numpy as np
import pytest
from PIL import Image

from filters import adaptive_threshold, add_noise, gaussian_blur, sobel_edges, to_grayscale
from metrics import contrast, mse, psnr


def test_grayscale_shape():
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    rgb[:, :, 0] = 100
    gray = to_grayscale(rgb)
    assert gray.shape == (20, 30)
    assert gray[0, 0] == pytest.approx(29.9)


def test_mse_same_image_is_zero():
    img = np.ones((16, 16)) * 120
    assert mse(img, img) == 0.0


def test_mse_different_shapes_fails():
    a = np.zeros((10, 10))
    b = np.zeros((10, 12))
    with pytest.raises(ValueError):
        mse(a, b)


@pytest.mark.parametrize("bad_sigma", [-0.1, -5.0])
def test_negative_sigma_fails(bad_sigma):
    img = np.ones((10, 10))
    with pytest.raises(ValueError):
        gaussian_blur(img, sigma=bad_sigma)


@pytest.mark.parametrize("bad_window", [0, 2, 4, -3])
def test_bad_window_size_fails(bad_window):
    img = np.ones((20, 20))
    with pytest.raises(ValueError):
        adaptive_threshold(img, window_size=bad_window)


def test_sobel_flat_image_zero():
    flat = np.full((20, 20), 50.0)
    assert np.max(sobel_edges(flat)) == pytest.approx(0.0)


@pytest.mark.parametrize("noise_std", [20.0, 30.0, 40.0])
def test_gaussian_filter_improves_psnr(noise_std):
    np.random.seed(0)
    raw = np.array(Image.open("input.jpg"))
    gray = to_grayscale(raw)
    noisy = add_noise(gray, std=noise_std)
    denoised = gaussian_blur(noisy, sigma=1.0)

    assert psnr(gray, denoised) > psnr(gray, noisy)
    assert mse(gray, denoised) < mse(gray, noisy)


def test_binary_output_values():
    raw = np.array(Image.open("input.jpg"))
    gray = to_grayscale(raw)
    binary = adaptive_threshold(gray)
    unique = set(np.unique(binary))
    assert unique.issubset({0.0, 255.0})
    assert contrast(binary) > 0
