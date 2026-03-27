"""
4. Fig. 3 is a photo of a woman standing in front of an open window2. Convert this to grayscale.
(a) Use Otsu thesholding to obtain the binary mask for the foreground comprising the
woman and the room. Report the resulting threshold value.
(b) Carry out histogram equalization only for the foreground region. What are the hidden
features that are revealed in the resulting image?
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

im = cv.imread("./data/a1images/shadow.png")

# convert to gray scale
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
ret, binary_mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
""" 
Otsu's method automatically finds the best threshold value by analyzing the image's histogram. 
It tries every possible threshold (0–255) and picks the one that best separates the pixels
into two groups - dark pixels (woman + room) and bright pixels (window) - by minimizing
how spread out the values are within each group.                                                           

In this image, most pixels are very dark (the room interior), with a small cluster of
bright pixels (the window). Even though the two groups are unequal in size, Otsu still
finds the optimal split point between them.

  `ret` — the threshold value Otsu chose
  `binary_mask` — the resulting binary image where:
      pixels above the threshold (bright window) → 255 (white)
      pixels below the threshold (dark foreground) → 0 (black)
"""

# 'ret' is the threshold value chosen by Otsu
print("Otsu's threshold value:", ret)


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Grayscale Image")
axes[0].axis("off")

axes[1].hist(gray.ravel(), bins=256, range=(0, 256), color="gray")
axes[1].axvline(x=ret, color="red", linewidth=2, label=f"Otsu threshold = {int(ret)}")
axes[1].set_title("Grayscale Histogram")
axes[1].set_xlabel("Pixel Intensity")
axes[1].set_ylabel("Frequency")
axes[1].legend()

axes[2].imshow(binary_mask, cmap="gray")
axes[2].set_title("Binary Mask (Otsu)")
axes[2].axis("off")

plt.tight_layout()
plt.show()


""" 
Only the foreground has darker pixels, earlier we got the mask for the foreground, 
so we can apply histogram equalization only to the foreground pixels.
This can be done using the similar method we used for 
question 3
"""

# create boolean foreground mask with boolean values
# since the binary mask has 255 for background and 0 for foreground
# we need it as boolean mask with True for 0 and False for 255
foreground_mask = binary_mask == 0

# extract foreground pixels
foreground_pixels = gray[foreground_mask]

# compute histogram for foreground
hist = np.zeros(256, dtype=int)
for i in range(256):
    hist[i] = np.sum(foreground_pixels == i)

cdf = hist.cumsum()  # cumulative distribution function

# normalize
cdf_min = cdf[np.nonzero(cdf)][0]  # first non-zero
total_pixels = foreground_pixels.size  # get only foreground pixels count
cdf_normalized = np.round((cdf - cdf_min) / (total_pixels - cdf_min) * 255).astype(
    np.uint8
)

# map original pixel values to equalized values
result = gray.copy()
result[foreground_mask] = cdf_normalized[gray[foreground_mask]]

# compute histograms for before/after comparison
hist_before = np.zeros(256, dtype=int)
hist_after = np.zeros(256, dtype=int)
for i in range(256):
    hist_before[i] = np.sum(foreground_pixels == i)
    hist_after[i] = np.sum(result[foreground_mask] == i)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].imshow(gray, cmap="gray")
axes[0, 0].set_title("Original Grayscale")
axes[0, 0].axis("off")

axes[0, 1].plot(hist_before, color="black")
axes[0, 1].set_title("Foreground Histogram Before Equalization")
axes[0, 1].set_xlabel("Pixel Intensity")
axes[0, 1].set_ylabel("Pixel Count")
axes[0, 1].set_xlim([0, 255])

axes[1, 0].imshow(result, cmap="gray")
axes[1, 0].set_title("Foreground Equalized Image")
axes[1, 0].axis("off")

axes[1, 1].plot(hist_after, color="black")
axes[1, 1].set_title("Foreground Histogram After Equalization")
axes[1, 1].set_xlabel("Pixel Intensity")
axes[1, 1].set_ylabel("Pixel Count")
axes[1, 1].set_xlim([0, 255])

plt.suptitle("Foreground Histogram Equalization", fontsize=14)
plt.tight_layout()
plt.show()

fig2, axes2 = plt.subplots(1, 2, figsize=(18, 10))
axes2[0].imshow(gray, cmap='gray')
axes2[0].set_title("Original Grayscale", fontsize=14)
axes2[0].axis('off')
axes2[1].imshow(result, cmap='gray')
axes2[1].set_title("Foreground Equalized", fontsize=14)
axes2[1].axis('off')
plt.tight_layout()
plt.show()
