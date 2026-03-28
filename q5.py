"""
5) Gaussian filtering:
(a) Using NumPy, compute a normalized 5×5 Gaussian kernel for σ=2.
(b) Visualize a 51×51 computed Gaussian kernel as a 3D surface plot, where the kernel
coefficients represent the height.
(c) Apply Gaussian smoothing to a given grayscale image using the manually computed
Gaussian kernel.
(d) Do the same using OpenCV’s built-in cv.GaussianBlur() function.
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

size = 5
sigma = 2

# coordinate grid
ax = np.arange(-(size // 2), (size // 2 + 1))  # array with values from -2 to 2
print(ax)
xx, yy = np.meshgrid(ax, ax)


# Gausian kernel
kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
print("Raw kernel: ", kernel)

# normzalize the kernel
kernel = kernel / np.sum(kernel)
print("Normalized kernel: ", kernel)

# ------------------------------------------

# 51x51 kernel
size = 51
sigma = 10  # random value

ax = np.arange(-(size // 2), size // 2 + 1)
xx, yy = np.meshgrid(ax, ax)

kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
kernel /= np.sum(kernel)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(xx, yy, kernel)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Kernel Value")

plt.show()


im = cv.imread("./data/a1images/emma.jpg", cv.IMREAD_GRAYSCALE)

# apply gausian kernel
# get height and width of the image
height, width = im.shape

# Adding extra pixels around the border of the image to handle the edges during convolution
pad = size // 2
padded = np.pad(im, pad, mode="constant")  # pad the image with zeros(default value)

output = np.zeros_like(im)


t_start = time.perf_counter()
for i in range(height):
    for j in range(width):
        region = padded[i : i + size, j : j + size]
        output[i, j] = np.sum(region * kernel)
t_manual = time.perf_counter() - t_start
print(f"Manual Gaussian blur time:  {t_manual:.4f}s")

plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(im, cmap="gray")
plt.axis("off")

# Blurred image
plt.subplot(1, 2, 2)
plt.title("Gaussian Blurred Image")
plt.imshow(output, cmap="gray")
plt.axis("off")

plt.show()

# ----------------
# Gaussian blur using OpenCV
t_start = time.perf_counter()
output_cv = cv.GaussianBlur(im, (51, 51), sigma)
t_cv = time.perf_counter() - t_start
print(f"OpenCV GaussianBlur time:   {t_cv:.4f}s")
print(f"Speedup (manual/opencv):    {t_manual / t_cv:.1f}x")

plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(im, cmap="gray")
plt.axis("off")

# Manual convolution
plt.subplot(1, 3, 2)
plt.title("Manual Gaussian Blur")
plt.imshow(output, cmap="gray")
plt.axis("off")

# OpenCV result
plt.subplot(1, 3, 3)
plt.title("OpenCV GaussianBlur")
plt.imshow(output_cv, cmap="gray")
plt.axis("off")

plt.show()

""" 
Conclusion:
Gaussian kernel gives higher weight to nearby pixels (3D visualization) so it won't change 
overall brightness of the image. Larger the kernel size more the smoothing effect.

  - since the custom blurring adds padding the image by adding 0 values pixels(black pixel)
    so when kernel calculate the values around the edges of the image it pushes twards zero which is why
    edges are darker in custom gaussian blurring.
  - we  can fix this by using 'reflect' mode in np.pad() which will reflect the image across the border instead of adding zeros.
  - And that mode is the one used by OpenCV's GaussianBlur() function, which is why it produces a more natural-looking blur without darkening the edges.
  
  - And OpenCv's GaussianBlur() is much faster than the manual implementation by 344X times, because it is optimized
  and likely uses multi-threading and SIMD instructions, while the manual version is a simple nested loop implementation that is not optimized for performance.

"""
