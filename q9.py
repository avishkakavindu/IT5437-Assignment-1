"""
(9) Carry out image sharpening on am image of your choice
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

im = cv.imread("./data/a1images/emma.jpg")


# Sharpening kernel of 3x3
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# Apply filter
sharpened = cv.filter2D(im, -1, kernel)

# Convert sharpened image to RGB
sharpened_rgb = cv.cvtColor(sharpened, cv.COLOR_BGR2RGB)

# Plot original and sharpened images
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# Sharpened image
plt.subplot(1, 2, 2)
plt.imshow(sharpened_rgb)
plt.title("Sharpened Image")
plt.axis("off")

plt.tight_layout()
plt.show()
