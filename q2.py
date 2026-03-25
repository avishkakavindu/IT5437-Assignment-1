"""
Consider the image shown in Fig. 21.
(a) Apply gamma correction to the L plane in the L*&b*color space and state the γ value.
(b) Show the histograms of the original and corrected images
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

""" 
The CIELAB color space (often written as L*a*b*) separates an image into three channels:
  - L* → Lightness (brightness)
  - a* → Green-Red color axis
  - b* → Blue-Yellow color axis
  
  So L plane is the Lightness channel, which represents the brightness of the image.
  Steps:
    1 - convert image to L*a*b* color space
    2 - extract L channel
    3 - apply gamma correction to L channel
    4 - recombine channels and convert back to RGB 
"""


def gamma_on_L_plane(im, gamma):
    # convert BRG to LAB
    lab = cv.cvtColor(im, cv.COLOR_BGR2LAB)

    # split channels
    L, a, b = cv.split(lab)

    # normalize L channel to [0, 1]
    normalized_L = L / 255.0

    # apply gamma correction to L channel
    l_gamma_corrected = np.uint8(np.power(normalized_L, gamma) * 255)

    # merge channels back
    lab_corrected = cv.merge((l_gamma_corrected, a, b))

    # plot hist
    plot_hist(L, l_gamma_corrected)

    # Show results
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    plt.title("Original RGB")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(im)
    plt.title("Original BGR")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(cv.cvtColor(lab, cv.COLOR_LAB2RGB))
    plt.title("Original LAB")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(cv.cvtColor(lab_corrected, cv.COLOR_LAB2RGB))
    plt.title(f"Gamma on L = {gamma}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_hist(original, corrected):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(original.flatten(), bins=256, range=[0, 256])
    plt.title("Original L Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(corrected.flatten(), bins=256, range=[0, 256])
    plt.title(f"Gamma Corrected L Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


im = cv.imread("./data/a1images/highlights_and_shadows.jpg")
gamma_on_L_plane(im, gamma=0.8)

""" 
Conclusion:
  - We selected the L channel (luminance/lightness) instead of the RGB channels
  because adjusting RGB directly would distort the colors of the image.
  By applying gamma correction to the L channel only, we can modify brightness
  while preserving color information. A gamma value of 0.8 provides a balanced output,
  enhancing brightness moderately so that the stone textures remain clearly visible.
  Reducing gamma further towards 0 brightens the image more strongly, pushing pixel
  values closer to 255 and potentially overexposing the image.
"""
