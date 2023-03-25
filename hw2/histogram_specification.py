import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image

tar = cv2.imread("Q1.jpg", cv2.IMREAD_GRAYSCALE)
ref = cv2.imread("Q2.jpg", cv2.IMREAD_GRAYSCALE)

tar_h, tar_w = tar.shape
ref_h, ref_w = ref.shape

tar_total = tar_h * tar_w
ref_total = ref_h * ref_w

tar_cnt = np.zeros(256, dtype=np.int32)
ref_cnt = np.zeros(256, dtype=np.int32)

plt.hist(tar.ravel(), 256, [0, 256])


for i in range(tar_h):
    for j in range(tar_w):
        tar_cnt[tar[i, j]] += 1

for i in range(ref_h):
    for j in range(ref_w):
        ref_cnt[ref[i, j]] += 1



tar_prob = np.zeros(256, dtype=np.float32)
ref_prob = np.zeros(256, dtype=np.float32)

for i in range(256):
    tar_prob[i] = tar_cnt[i] / tar_total
    ref_prob[i] = ref_cnt[i] / ref_total

tar_cdf = np.zeros(256, dtype=np.float32)
ref_cdf = np.zeros(256, dtype=np.float32)

tar_cdf[0] = tar_prob[0]
ref_cdf[0] = ref_prob[0]

for i in range(1, 256):
    tar_cdf[i] = tar_cdf[i - 1] + tar_prob[i]
    ref_cdf[i] = ref_cdf[i - 1] + ref_prob[i]

tar_map = np.zeros(256, dtype=np.uint8)

for i in range(256):
    min_diff = 1.0
    for j in range(256):
        diff = abs(tar_cdf[i] - ref_cdf[j])
        if diff < min_diff:
            min_diff = diff
            tar_map[i] = j


for i in range(tar_h):
    for j in range(tar_w):
        tar[i, j] = tar_map[tar[i, j]]

plt.hist(tar.ravel(), 256, [0, 256])
plt.hist(ref.ravel(), 256, [0, 256])


plt.legend(('before matching', 'after matching', 'target'), loc='upper left')
plt.xlabel('Intensity')
plt.ylabel('Number of pixels')

# save plt in compact way
plt.savefig('hist_match_plt.png', dpi=300, bbox_inches='tight')
plt.show()


cv2.imwrite("Q1_spec.jpg", tar)