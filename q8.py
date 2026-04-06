"""
(8) Fig. 4 is an image corrupted with salt and pepper noise.
(a) Apply Gaussian smoothing.
(b) Apply median filtering.
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


im = cv.imread("./data/emma.png")

# Apply Gaussian sian smoothing
blurred = cv.GaussianBlur(im, (5, 5), 0)

""" 
How it Works
  1 - Window Selection: A square "window" or kernel (e.g., 3x3 or 5x5) slides over the image.
  2 - Sorting: All pixel values within that window are sorted in numerical order.
  3 - Median Selection: The middle value (the median) is selected from the sorted list.
  4 - Replacement: The center pixel of the window is replaced by this median value
  so white and black pixels (salt and pepper noise) are pushed to the edges of the sorted list
  and do not affect the median value as much as they would affect the mean in Gaussian smoothing.
"""
# Apply median filtering
median = cv.medianBlur(im, 5)

# plot
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(im)
ax[0].set_title("Original Image")
ax[1].imshow(blurred)
ax[1].set_title("Gaussian Smoothing")
ax[2].imshow(median)
ax[2].set_title("Median Filtering")
plt.show()

""" 
Conclusion:
- Gaussian smoothing is effective for reducing Gaussian noise but may blur edges and details in the image.
- But for salt paper noise, median filtering is more effective as it can preserve edges while removing the noise, making it a better choice for this type of corruption.
  - Median blur does that by replacing each pixel with the median value of the neighboring pixels(process above mentioned),
  which helps to eliminate the outliers (salt and pepper noise) while preserving the edges.
"""
