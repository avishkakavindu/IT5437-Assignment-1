"""
(3) Write your own function to equalize the histogram of an image. Apply this function to the
runway image.
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

"""
Manual histogram equalization
Steps:
  1 - Calculate the histogram of the image (count of pixel intensities)
  2 - Compute the cumulative distribution function (CDF) from the histogram
  3 - Normalize the CDF to the range [0, 255]
  4 - Map the original pixel values to the new equalized values using the normalized CDF

"""


def equalize_histogram(im):
    # convert to LAB color space
    lab = cv.cvtColor(im, cv.COLOR_BGR2LAB)

    # Split channels
    L, a, b = cv.split(lab)

    # compute histogram
    hist = np.zeros(256, dtype=int)
    for i in range(256):
        hist[i] = np.sum(L == i)

    cdf = hist.cumsum()  # cumulative distribution function

    # normalize
    cdf_min = cdf[np.nonzero(cdf)][0]  # first non-zero
    total_pixels = L.size
    cdf_normalized = np.round((cdf - cdf_min) / (total_pixels - cdf_min) * 255).astype(
        np.uint8
    )

    # map original pixel values to equalized values
    L_equalized = cdf_normalized[L]

    # compute histogram AFTER
    hist_eq = np.zeros(256, dtype=int)
    for i in range(256):
        hist_eq[i] = np.sum(L_equalized == i)

    # merge channels and convert back to BGR
    lab_equalized = cv.merge((L_equalized, a, b))
    img_equalized = cv.cvtColor(lab_equalized, cv.COLOR_LAB2BGR)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].plot(hist, color="black")
    axes[0, 1].set_title("Histogram Before Equalization")
    axes[0, 1].set_xlabel("Pixel Intensity")
    axes[0, 1].set_ylabel("Pixel Count")
    axes[0, 1].set_xlim([0, 255])

    axes[1, 0].imshow(cv.cvtColor(img_equalized, cv.COLOR_BGR2RGB))
    axes[1, 0].set_title("Equalized Image")
    axes[1, 0].axis("off")

    axes[1, 1].plot(hist_eq, color="black")
    axes[1, 1].set_title("Histogram After Equalization")
    axes[1, 1].set_xlabel("Pixel Intensity")
    axes[1, 1].set_ylabel("Pixel Count")
    axes[1, 1].set_xlim([0, 255])

    plt.suptitle("Histogram Equalization", fontsize=14)
    plt.tight_layout()
    plt.show()
    return img_equalized


im = cv.imread("./data/runway.png")
equalize_histogram(im)

""" 
Here, we worked on the L channel (luminance) of the image.
We first extracted the L channel, counted the occurrences of each intensity,
and then used the cumulative distribution function (CDF) to redistribute the
intensities across the full range of 0–255, as shown in the histograms.
This process ensures that very bright areas become slightly darker and
very dark areas become slightly lighter, effectively enhancing contrast
and improving visibility in both dark and bright regions of the image.
"""
