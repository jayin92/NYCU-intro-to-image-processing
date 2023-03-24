import cv2
import numpy as np

# Read image
img = cv2.imread('test.jpg')

hh, ww = img.shape[:2] # 360 * 600

h = hh // 3
w = ww // 3

# Exchange Position
tmp = img[0:h, 0:w].copy()
img[0:h, 0:w] = img[0:h, 2*w:3*w]
img[0:h, 2*w:3*w] = tmp

# Gray Scale
for i in range(2*h, 3*h):
    for j in range(0, w):
        res = int(img[i, j, 0]) + int(img[i, j, 1]) + int(img[i, j, 2])
        img[i, j] = res // 3

# Gray Scale and intensity resolution from 256 to 4
for i in range(2*h, 3*h):
    for j in range(2*w, 3*w):
        res = int(img[i, j, 0]) + int(img[i, j, 1]) + int(img[i, j, 2])
        img[i, j] = res // 3
        img[i, j] = img[i, j] // 64 * 64

# Red Color Filter
for i in range(h, 2*h):
    for j in range(0, w):
        b, g, r = img[i, j]
        b = int(b)
        g = int(g)
        r = int(r)
        res = b + g + r
        if not (r > 150 and r * 0.6 > g and r * 0.6 > b):
            img[i, j] = res // 3

# Yellow Color Filter
for i in range(h, 2*h):
    for j in range(2*w, 3*w):
        b, g, r = img[i, j]
        b = int(b)
        g = int(g)
        r = int(r)
        res = b + g + r
        if not ((g + r) * 0.3 > b and abs(g - r) < 50):
            img[i, j] = res // 3

# Channel Operation
for i in range(2*h, 3*h):
    for j in range(w, 2*w):
        img[i, j, 1] = min(255, int(img[i, j, 1]) * 2)

# Bilinaer Interpolation
tar = img[0:h, w:2*w].copy()
res = np.zeros((2*h, 2*w, 3), np.uint8)

for i in range(0, 2*h):
    for j in range(0, 2*w):
        x = i / 2
        y = j / 2
        x1 = i // 2
        y1 = j // 2
        x2 = x1 + 1
        y2 = y1 + 1
        x1 = min(x1, h - 1)
        y1 = min(y1, w - 1)
        x2 = min(x2, h - 1)
        y2 = min(y2, w - 1)
        x = x - x1
        y = y - y1
        top = (1 - x) * tar[x1, y1] + x * tar[x2, y1]
        bottom = (1 - x) * tar[x1, y2] + x * tar[x2, y2]
        res[i, j] = (1 - y) * top + y * bottom

img[0:h, w:2*w] = res[0:h, 0:w]

# Bicubic Interpolation
tar = img[h:2*h, w:2*w].copy().astype(np.float32)
res = np.zeros((2*h, 2*w, 3), np.int32)

for i in range(0, 2*h):
    for j in range(0, 2*w):
        x1 = i // 2
        y1 = j // 2
        tmp = []
        for k in range(-1, 3):
            x = x1 + k
            if x < 0:
                x = 0
            elif x >= h:
                x = h - 1
            p = []
            for l in range(-1, 3):
                y = y1 + l
                if y < 0:
                    y = 0
                elif y >= w:
                    y = w - 1
                p.append(tar[x, y])
            
            yy = j / 2 - y1
            tmp.append((-0.5*p[0]+1.5*p[1]-1.5*p[2]+0.5*p[3])*yy**3 + (p[0]-2.5*p[1]+2*p[2]-0.5*p[3])*yy**2 + (-0.5*p[0]+0.5*p[2])*yy + p[1])
        
        xx = i / 2 - x1
        res[i, j] = (-0.5*tmp[0]+1.5*tmp[1]-1.5*tmp[2]+0.5*tmp[3])*xx**3 + (tmp[0]-2.5*tmp[1]+2*tmp[2]-0.5*tmp[3])*xx**2 + (-0.5*tmp[0]+0.5*tmp[2])*xx + tmp[1]

res = np.clip(res, 0, 255)
img[h:2*h, w:2*w] = res[0:h, 0:w]

cv2.imwrite('result.jpg', img)
cv2.imshow('image', img)
cv2.waitKey(0)
