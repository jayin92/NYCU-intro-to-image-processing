import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read image
img = cv2.imread("Q1.jpg", cv2.IMREAD_GRAYSCALE)

# Draw histogram
plt.hist(img.ravel(), 256, [0, 256])

h, w = img.shape

total = h * w

cnt = np.zeros(256, dtype=np.int32)

for i in range(h):
    for j in range(w):
        cnt[img[i, j]] += 1

prob = np.zeros(256, dtype=np.float32)

for i in range(256):
    prob[i] = cnt[i] / total

cdf = np.zeros(256, dtype=np.float32)

cdf[0] = prob[0]

for i in range(1, 256):
    cdf[i] = cdf[i - 1] + prob[i]

cdf = cdf * 255


for i in range(h):
    for j in range(w):
        img[i, j] = round(cdf[img[i, j]])

plt.hist(img.ravel(), 256, [0, 256])

plt.legend(('original', 'equalized'), loc='upper left')
plt.xlabel('Intensity')
plt.ylabel('Number of pixels')

# save plt in compact way
plt.savefig('hist_equal_plt.png', dpi=300, bbox_inches='tight')

cv2.imwrite("Q1_equal.jpg", img)
