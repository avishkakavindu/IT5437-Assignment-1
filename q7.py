"""
7) Write a program to zoom images by a given factor s∈(0,10]. You must use a function to
zoom the image, which can handle
(a) nearest-neighbor, and
(b) bilinear interpolation.
I have included several images, large originals, and their zoomed-out versions. Test you
algorithm by computing the normalized sum of squared difference (SSD) when you scale-up
the given small images to match the size of the large original images. The SSD should be
small when comparing with the original images.
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


original_img = cv.imread("./data/a1images/a1q8images/im01.png")
small_image = cv.imread("./data/a1images/a1q8images/im01small.png")

# computer the scale factor
scale_factor = original_img.shape[0] / small_image.shape[0]
print(f"Scale factor: {scale_factor}")
new_size = (original_img.shape[1], original_img.shape[0])


def zoom_img(img, scale_factor, method="nearest"):
    h, w, c = img.shape

    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    zoomed = np.zeros((new_h, new_w, c), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            # get original coordinates
            x = i / scale_factor
            y = j / scale_factor

            if method == "nearest":
                xi = int(round(x))
                yj = int(round(y))

                xi = min(max(xi, 0), h - 1)
                yj = min(max(yj, 0), w - 1)

                zoomed[i, j] = img[xi, yj]
            elif method == "bilinear":
                x1 = int(np.floor(x))
                y1 = int(np.floor(y))
                x2 = min(x1 + 1, h - 1)
                y2 = min(y1 + 1, w - 1)

                dx = x - x1
                dy = y - y1

                top = (1 - dy) * img[x1, y1] + dy * img[x1, y2]
                bottom = (1 - dy) * img[x2, y1] + dy * img[x2, y2]

                pixel = (1 - dx) * top + dx * bottom
                zoomed[i, j] = pixel

    return zoomed


def normalized_ssd(img1, img2):
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    return np.sum(diff**2) / img1.size


# Apply zoom
zoom_nn = zoom_img(small_image, scale_factor, method="nearest")
zoom_bl = zoom_img(small_image, scale_factor, method="bilinear")

# Resize to match exactly (safety step)
zoom_nn = cv.resize(zoom_nn, (original_img.shape[1], original_img.shape[0]))
zoom_bl = cv.resize(zoom_bl, (original_img.shape[1], original_img.shape[0]))

# Compute SSD
ssd_nn = normalized_ssd(original_img, zoom_nn)
ssd_bl = normalized_ssd(original_img, zoom_bl)

print("Normalized SSD (Nearest):", ssd_nn)
print("Normalized SSD (Bilinear):", ssd_bl)


original_rgb = cv.cvtColor(original_img, cv.COLOR_BGR2RGB)
small_rgb = cv.cvtColor(small_image, cv.COLOR_BGR2RGB)
zoom_nn_rgb = cv.cvtColor(zoom_nn, cv.COLOR_BGR2RGB)
zoom_bl_rgb = cv.cvtColor(zoom_bl, cv.COLOR_BGR2RGB)

diff_nn = np.abs(original_img.astype(np.float32) - zoom_nn.astype(np.float32))
diff_bl = np.abs(original_img.astype(np.float32) - zoom_bl.astype(np.float32))

diff_nn = diff_nn.astype(np.uint8)
diff_bl = diff_bl.astype(np.uint8)

diff_nn_rgb = cv.cvtColor(diff_nn, cv.COLOR_BGR2RGB)
diff_bl_rgb = cv.cvtColor(diff_bl, cv.COLOR_BGR2RGB)


fig1, axes1 = plt.subplots(1, 2, figsize=(12, 6))
axes1[0].imshow(original_rgb)
axes1[0].set_title("Original Image")
axes1[0].axis("off")
axes1[1].imshow(zoom_nn_rgb)
axes1[1].set_title(f"Nearest Neighbor (SSD={ssd_nn:.2f})")
axes1[1].axis("off")
fig1.tight_layout()

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
axes2[0].imshow(original_rgb)
axes2[0].set_title("Original Image")
axes2[0].axis("off")
axes2[1].imshow(zoom_bl_rgb)
axes2[1].set_title(f"Bilinear (SSD={ssd_bl:.2f})")
axes2[1].axis("off")
fig2.tight_layout()

plt.show()

""" 
Conclusion:
Visually we can see the nearest neighbor zoomed image has pixalated image and th e
bileanear option provided more smooth an quality image compared to the nearest neigbour one.
  - thisis because nearest neighbor simply copies the value of the nearest pixel, 
  which can lead to blocky artifacts, especially when the scale factor is large.
  
The normalized SSD for bilinear interpolation is lower than that of nearest neighbor interpolation,
indicating that bilinear interpolation provides a more accurate reconstruction of the original image. 
However, the SSD is not negligible due to the loss of information during the downsampling 
process, which cannot be fully recovered during upscaling.
"""
