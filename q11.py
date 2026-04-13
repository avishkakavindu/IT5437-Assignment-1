"""
11) Relationship Between Spatial Filtering and Frequency Response3:
Consider a 2D image f(x,y) and a filter h(x,y), where the filtered image is
g(x, y) = f(x,y)∗h(x,y).
(a) Using the Convolution Theorem, explain how spatial domain filtering corresponds to
an operation in the frequency domain.
(b) For each of the following filters, describe their qualitative effect in the frequency domain
and relate it to their spatial behavior:
• Averaging (box) filter
• Gaussian filter
• Laplacian filter
(c) Explain why Gaussian filtering avoids ringing artifacts compared to an ideal low-pass
filter.
(d) An image is corrupted by high-frequency noise. Justify which of the above filters is
most suitable for noise reduction, using both spatial and frequency domain arguments
"""

import numpy as np
import cv2 as cv
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

f = cv.imread("./data/a1images/emma.jpg", cv.IMREAD_GRAYSCALE)
f = f.astype(np.float32)

# Simple 5x5 averaging filter
h = np.ones((5, 5), dtype=np.float32) / 25

# Spatial domain convolution
g_spatial = convolve2d(f, h, mode="same", boundary="wrap")

# Frequency domain multiplication
rows, cols = f.shape

# Pad to image size
H_padded = np.zeros((rows, cols), dtype=np.float32)

kh, kw = h.shape
center_r, center_c = kh // 2, kw // 2

# Place kernel centered at (0,0) for FFT
H_padded[:kh, :kw] = h
H_padded = np.roll(H_padded, -center_r, axis=0)
H_padded = np.roll(H_padded, -center_c, axis=1)

# FFT
F = np.fft.fft2(f)
H = np.fft.fft2(H_padded)

# Multiply in frequency domain
G = F * H

# Inverse FFT
g_freq = np.fft.ifft2(G)
g_freq = np.real(g_freq)

difference = np.abs(g_spatial - g_freq)

print("Max difference:", np.max(difference))


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Spatial Convolution")
plt.imshow(g_spatial, cmap="gray")

plt.subplot(1, 3, 2)
plt.title("Frequency Domain")
plt.imshow(g_freq, cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Difference")
plt.imshow(difference, cmap="gray")

plt.show()

""" 
Conclussion for (a):
  Using the Convolution Theorem, spatial convolution of an image
  with a filter corresponds to multiplication of their Fourier 
  transforms in the frequency domain. This allows filtering to 
  be performed efficiently by transforming to the frequency domain,
  multiplying, and applying the inverse transform.
"""

"""
Conclussion for (b) 
1. Averaging (Box) Filter
  Frequency Domain:
    Acts as a low-pass filter
    Passes low frequencies (smooth regions)
    Attenuates high frequencies (edges, noise)
    Frequency response has ripples (sinc-like shape) → not smooth
  Spatial Domain:
    Replaces each pixel with the average of neighbors
    Produces blurring/smoothing
    Reduces noise but also blurs edges

Because the frequency response is not smooth → may introduce artifacts (ringing)

2. Gaussian Filter
  Frequency Domain:
    Also a low-pass filter
    Has a smooth, bell-shaped response
    No sharp cutoff, no ripples
  Spatial Domain:
    Performs weighted averaging (center pixels matter more)
    Produces smooth blur
    Preserves structure better than box filter

Smooth frequency response → no ringing artifacts
Most natural and stable smoothing filter

3. Laplacian Filter
  Frequency Domain:
    Acts as a high-pass filter
    Suppresses low frequencies
    Strongly amplifies high frequencies
  Spatial Domain:
    Computes second derivative
    Highlights edges and fine details
    Enhances intensity changes

Since noise is high-frequency → it also amplifies noise
"""

""" 
Conclussion for (c)
  Ringing artifacts come from sharp cutoffs in the frequency domain.
  
  Ideal low pass filter, frequency domain: has a perfect  abrupt cutoff. SO  this sudden transition
  cause oscilations when transformed back. 
  
  But in Gaussian filter, frequency domain: has smooth gradual decaying. SO no sudden transition → no oscillations → no ringing artifacts.
  
  Fundamental principle:
    Sharp changes in frequency → oscillations in space (ringing)
    Smooth changes in frequency → smooth response in space
"""

""" 
Conclussion (D):

  Gaussian filtering is suitable because, high frequency noise image has high frequency components. So we need a filter
  to supress high frequencies while preserving low frequencies. So, Averaging cause ripples in frequency response, uneven suppression 
  and can introduce artifacts. 
  Gaussian filter will gradually attenuate high frequencies without introducing ripples, so it will effectively reduce noise while preserving image details.
  
  Laplacian filter is high pass filter so not suitable.
"""
