import cv2
import numpy as np
from tqdm import tqdm

K = 1
size = 5
sigma = 25

def guassian(x, y):
    return K * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

def clamp(x, a, b):
    return max(a, min(x, b))

# Read image
img = cv2.imread("Q3.jpg", cv2.IMREAD_GRAYSCALE)

comp = cv2.GaussianBlur(img, (size, size), sigma)

# padding
h, w = img.shape
res = np.zeros((h, w), dtype=np.float32)


for i in tqdm(range(h)):
    for j in range(w):
        total = 0
        for k in range(-size // 2 + 1, size // 2 + 1):
            for l in range(-size // 2 + 1, size // 2 + 1):
                x = clamp(i + k, 0, h - 1)
                y = clamp(j + l, 0, w - 1)
                res[i, j] += img[x, y] * guassian(k, l)
                total += guassian(k, l)
        res[i, j] /= total

res = res.astype(np.uint8)

cv2.imwrite("Q3_gaussian.jpg", res)