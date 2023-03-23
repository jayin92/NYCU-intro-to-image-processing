import cv2
import numpy as np

# Read image
img = cv2.imread("Q1.jpg", cv2.IMREAD_GRAYSCALE)

ans = cv2.equalizeHist(img)

cv2.imwrite("Q1_equal1.jpg", ans)

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

cdf = cdf.astype(np.uint8)

for i in range(h):
    for j in range(w):
        img[i, j] = cdf[img[i, j]]

cv2.imwrite("Q1_equal.jpg", img)
