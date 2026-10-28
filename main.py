import os
import numpy as np
from PIL import Image

from filters import adaptive_threshold, add_noise, gaussian_blur, sobel_edges, to_grayscale, add_salt_pepper, median_blur
from metrics import contrast, mse, psnr


def save_img(arr, filename):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(filename)


def main():
    os.makedirs("results", exist_ok=True)

    np.random.seed(42)
    raw = np.array(Image.open("input.jpg"))
    gray = to_grayscale(raw)

    noisy = add_noise(gray, std=20.0)
    denoised = gaussian_blur(noisy, sigma=1.2)
    edges = sobel_edges(denoised)
    binary = adaptive_threshold(denoised, window_size=25, c=10.0)

    save_img(noisy, "results/noisy.png")
    save_img(denoised, "results/denoised.png")
    save_img(edges, "results/edges.png")
    save_img(binary, "results/binary.png")

    print(f"MSE noisy:    {mse(gray, noisy):.2f}")
    print(f"MSE denoised: {mse(gray, denoised):.2f}")
    print(f"PSNR noisy:    {psnr(gray, noisy):.2f} dB")
    print(f"PSNR denoised: {psnr(gray, denoised):.2f} dB")
    print(f"Contrast before: {contrast(gray):.2f}, binary: {contrast(binary):.2f}")

    sp_noisy = add_salt_pepper(gray, prob=0.05)
    sp_gauss = gaussian_blur(sp_noisy, sigma=1.2)
    sp_median = median_blur(sp_noisy, size=3)
    save_img(sp_noisy, "results/salt_pepper.png")
    save_img(sp_median, "results/median_denoised.png")
    print(f"Salt&Pepper PSNR -> Gauss: {psnr(gray, sp_gauss):.2f} dB, Median: {psnr(gray, sp_median):.2f} dB")


if __name__ == "__main__":
    main()