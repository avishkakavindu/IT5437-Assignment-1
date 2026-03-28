"""
(6) Derivative of Gaussian:
  (a) Consider the two–dimensional Gaussian function
  Show that its first–order partial derivatives are given by
  (b) Using NumPy, compute normalized 5×5 kernels corresponding to the derivatives of a
  Gaussian for σ=2 in the x- and y-directions.
  (c) Visualize a 51×51 derivative-of-Gaussian kernel (for either the x or y direction) as a
  3D surface plot, where the kernel coefficients represent the height.
  (d) Apply the computed derivative-of-Gaussian kernels to a given grayscale image to obtain
  the image gradients in the horizontal and vertical directions.
  (e) Using OpenCV, compute the image gradients by applying cv.Sobel(). Compare the
  results with those obtained above and comment on any observed differences.
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# TODO Partial Derivation need to be added

size = 5
sigma = 2

ax = np.arange(-(size // 2), size // 2 + 1)
xx, yy = np.meshgrid(ax, ax)

kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
kernel = kernel / np.sum(kernel)  # normalize Gaussian

kernel_x = -(xx / sigma**2) * kernel
kernel_y = -(yy / sigma**2) * kernel

# normalize
kernel_x = kernel_x / np.sum(np.abs(kernel_x))  # detects vertical edges
kernel_y = kernel_y / np.sum(np.abs(kernel_y))  # detects horizontal edges

# ----------------------------

size = 51
sigma = 10  # standard deviation

# Coordinate grid
ax = np.arange(-(size // 2), size // 2 + 1)
xx, yy = np.meshgrid(ax, ax)

# Gaussian
kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
kernel /= np.sum(kernel)  # normalize Gaussian

# Derivative kernels
kernel_x = -(xx / sigma**2) * kernel
kernel_y = -(yy / sigma**2) * kernel

fig = plt.figure(figsize=(8, 6))
ax3d = fig.add_subplot(111, projection="3d")

ax3d.plot_surface(xx, yy, kernel_x, cmap="viridis")

ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Kernel Value")
ax3d.set_title("Derivative-of-Gaussian Kernel (X-direction)")

plt.show()
""" 
Derivative of Gaussian = Gaussian × linear term
It is used to detect edges:
kernel_x → vertical edges (intensity changes along x)
kernel_y → horizontal edges (intensity changes along y)
Smoothing still happens because of the Gaussian part
"""
# -------------------------------

im = cv.imread("./data/a1images/emma.jpg", cv.IMREAD_GRAYSCALE)

# image height and width
height, width = im.shape

size = 51
pad = size // 2

# Pad image to handle edges
padded = np.pad(im, pad, mode="reflect")  # reflect is better than zero padding

# Initialize outputs
Ix = np.zeros_like(im, dtype=float)
Iy = np.zeros_like(im, dtype=float)

# Manual convolution
for i in range(height):
    for j in range(width):
        region = padded[i : i + size, j : j + size]
        Ix[i, j] = np.sum(region * kernel_x)
        Iy[i, j] = np.sum(region * kernel_y)

gradient_magnitude = np.sqrt(Ix**2 + Iy**2)

# Normalize for display
gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude) * 255).astype(
    np.uint8
)


plt.figure(figsize=(15, 5))

# Horizontal gradient
plt.subplot(1, 3, 1)
plt.title("Horizontal Gradient (Ix - kernel_x)")
plt.imshow(Ix, cmap="gray")
plt.axis("off")

# Vertical gradient
plt.subplot(1, 3, 2)
plt.title("Vertical Gradient (Iy - kernel_y)")
plt.imshow(Iy, cmap="gray")
plt.axis("off")

# Gradient magnitude
plt.subplot(1, 3, 3)
plt.title("Gradient Magnitude")
plt.imshow(gradient_magnitude, cmap="gray")
plt.axis("off")

plt.show()

""" 
Derivative-of-Gaussian = Gaussian × linear term
Convolving with DoG kernel = smoothing + derivative → robust edge detection
This is essentially how Sobel or Canny operators work internally
"""
# ------------------------------------------------------

# compute horizontal and vertical gradients using Sobel
Ix_sobel = cv.Sobel(im, cv.CV_64F, 1, 0, ksize=5)  # dx=1, dy=0
Iy_sobel = cv.Sobel(im, cv.CV_64F, 0, 1, ksize=5)  # dx=0, dy=1

# Gradient magnitude
gradient_sobel = np.sqrt(Ix_sobel**2 + Iy_sobel**2)
gradient_sobel = (gradient_sobel / np.max(gradient_sobel) * 255).astype(np.uint8)

plt.figure(figsize=(15, 5))

#  DoG gradients
plt.subplot(1, 3, 1)
plt.title("Gradient Magnitude (DoG)")
plt.imshow(gradient_magnitude, cmap="gray")
plt.axis("off")

# Sobel gradients
plt.subplot(1, 3, 2)
plt.title("Gradient Magnitude (Sobel)")
plt.imshow(gradient_sobel, cmap="gray")
plt.axis("off")

# Difference
plt.subplot(1, 3, 3)
plt.title("Difference (DoG - Sobel)")
plt.imshow(np.abs(gradient_magnitude - gradient_sobel), cmap="hot")
plt.axis("off")

plt.show()

""" 
Using OpenCV's Sobel is faster and convenient, 
but the derivative-of-Gaussian method provides more 
control over smoothing and can produce cleaner gradients in noisy images.
"""
