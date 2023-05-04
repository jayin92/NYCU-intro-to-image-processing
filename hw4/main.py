import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("test1.tif", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("test2.tif", cv2.IMREAD_GRAYSCALE)

# Get the Spectrum using Fourier Transform
f1 = np.fft.fft2(img1)
f2 = np.fft.fft2(img2)

fshift1 = np.fft.fftshift(f1)
fshift2 = np.fft.fftshift(f2)

magnitude_spectrum1 = 20*np.log(np.abs(fshift1))
magnitude_spectrum2 = 20*np.log(np.abs(fshift2))

plt.figure(figsize=(15, 7))
plt.subplot(241), plt.imshow(img1, cmap='gray')
plt.title('Input Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(242), plt.imshow(magnitude_spectrum1, cmap='gray')
plt.title('Magnitude Spectrum 1'), plt.xticks([]), plt.yticks([])

plt.subplot(245), plt.imshow(img2, cmap='gray')
plt.title('Input Image 2'), plt.xticks([]), plt.yticks([])
plt.subplot(246), plt.imshow(magnitude_spectrum2, cmap='gray')
plt.title('Magnitude Spectrum 2'), plt.xticks([]), plt.yticks([])


width = 25

h, w = img1.shape

mask1 = np.ones((img1.shape[0], img1.shape[1]), np.uint8) * 255.0

mask1 = cv2.line(mask1, (w//2, 0), (w//2, h//2-width), 0, 5)
mask1 = cv2.line(mask1, (w//2, h//2+width), (w//2, h-1), 0, 5)
mask1 = cv2.GaussianBlur(mask1, (21, 21), 0)


fshift1 = fshift1 * mask1 / 255
magnitude_spectrum1 = magnitude_spectrum1 * mask1 / 255

f_ishift1 = np.fft.ifftshift(fshift1)
img_back1 = np.fft.ifft2(f_ishift1)
img_back1 = np.abs(img_back1)
img_back1 = np.clip(img_back1, 0, 255)

plt.subplot(243), plt.imshow(magnitude_spectrum1, cmap='gray')
plt.title('Filtered Magnitude Spectrum 1'), plt.xticks([]), plt.yticks([])
plt.subplot(244), plt.imshow(img_back1, cmap='gray')
plt.title('Filtered Image 1'), plt.xticks([]), plt.yticks([])

cords = [
    (4, 55),
    (44, 55),
    (84, 55),
    (164, 55),
    (208, 55),
    (243, 55),
    (0, 112),
    (38, 112),
    (82, 112),
    (160, 112),
    (203, 112),
    (240, 112)
]

radius = 15


mask2 = np.ones((img2.shape[0], img2.shape[1]), np.uint8) * 255.0

for x, y in cords:
    mask2 = cv2.circle(mask2, (y, x), radius, 0, -1)

mask2 = cv2.GaussianBlur(mask2, (31, 31), 0)

fshift2 = fshift2 * mask2 / 255
magnitude_spectrum2 = magnitude_spectrum2 * mask2 / 255

# apply mask and inverse DFT
f_ishift2 = np.fft.ifftshift(fshift2)
img_back2 = np.fft.ifft2(f_ishift2)
img_back2 = np.abs(img_back2)
img_back2 = np.clip(img_back2, 0, 255)


plt.subplot(247), plt.imshow(magnitude_spectrum2, cmap='gray')
plt.title('Filtered Magnitude Spectrum 2'), plt.xticks([]), plt.yticks([])
plt.subplot(248), plt.imshow(img_back2, cmap='gray')
plt.title('Filtered Image 2'), plt.xticks([]), plt.yticks([])

plt.savefig('report/result.png', bbox_inches='tight', dpi=300)
plt.show()
