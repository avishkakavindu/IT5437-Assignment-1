import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

""" 
Intensity Transformation

- technique where we change the brightness values (intensity) of pixels in an image.
    - Each pixel has a value (e.g., 0 = black, 255 = white)
    - Intensity transformation modifies those values to improve the image
    
    s=T(r)

      r = original pixel intensity
      s = new pixel intensity
      T = transformation function
"""

im = cv.imread("./data/runway.png", cv.IMREAD_GRAYSCALE)
img_rgb = cv.cvtColor(im, cv.COLOR_BGR2RGB)


"""
Gamma Correction

- technique used to adjust the brightness of an image in a non-linear way.

  s = c ⋅ r**γ

    r = input pixel value (normalized, usually 0–1)
    s = output pixel value
    γ (gamma) = controls brightness
    c = constant (often 1)
 
"""


def get_gamma_corrected(im, gamma=0.5):
    normalized = im / 255.0

    # apply gamma correction
    gamma_corrected = np.power(normalized, gamma)  # s = c * r^γ, where c=1

    # Scale back to [0, 255]
    gamma_corrected = np.uint8(gamma_corrected * 255)

    # Plot side by side
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(gamma_corrected)
    plt.title(f"Gamma = {gamma}")
    plt.axis("off")

    plt.show()


# Gamma correction with γ=0.5
get_gamma_corrected(img_rgb)

# Gamma correction with γ=2.0
get_gamma_corrected(img_rgb, gamma=2.0)


"""
Conclusion: 
  - Since we normalize the image before the gamma correction pixel values are between 0-1
      0 = black, 1 = white
  - When pixel values are closer to zero(black) the exponentiation with γ < 1 will increase the pixel values, making the image brighter.
      when r = 0.5 and gamma = 0.5, s = 0.5^0.5 = 0.707 (brighter)
  - When pixel values are closer to one(white) the exponentiation with γ > 1 will decrease the pixel values, making the image darker.
      when r = 0.5 and gamma = 2.0, s = 0.5^2.0 = 0.25 (darker)
"""
