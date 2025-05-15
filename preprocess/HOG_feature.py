import os
import cv2
import numpy as np
from numpy import linalg as LA

IMG = 'D:\BTL_CSDLDPT\Moutain_Search\images\gsun_0a94d45397d3bd4aeb4b20d1d2d91375.jpg'

# Chưa giảm chiều, có thế tối ưu thêm
def hog(img_gray, cell_size=8, block_size=2, bins=9):
    img = img_gray
    h, w = img.shape  # 256,256

    # gradient
    xkernel = np.array([[-1, 0, 1]])
    ykernel = np.array([[-1], [0], [1]])
    dx = cv2.filter2D(img, cv2.CV_32F, xkernel)
    dy = cv2.filter2D(img, cv2.CV_32F, ykernel)

    # histogram
    magnitude = np.sqrt(np.square(dx) + np.square(dy))
    orientation = np.arctan(np.divide(dy, dx + 0.00001))  # radian
    orientation = np.degrees(orientation)  # -90 -> 90
    orientation += 90  # 0 -> 180

    num_cell_x = w // cell_size  # 32
    num_cell_y = h // cell_size  # 32
    hist_tensor = np.zeros([num_cell_y, num_cell_x, bins])  # 32 x 32 x 9
    for cx in range(num_cell_x):
        for cy in range(num_cell_y):
            ori = orientation[cy * cell_size:cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]
            mag = magnitude[cy * cell_size:cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]
            hist, _ = np.histogram(ori, bins=bins, range=(0, 180), weights=mag)  # 1-D vector, 9 elements
            hist_tensor[cy, cx, :] = hist
        pass
    pass

    # normalization
    redundant_cell = block_size - 1
    feature_tensor = np.zeros(
        [num_cell_y - redundant_cell, num_cell_x - redundant_cell, block_size * block_size * bins])
    for bx in range(num_cell_x - redundant_cell):  # 7
        for by in range(num_cell_y - redundant_cell):  # 15
            by_from = by
            by_to = by + block_size
            bx_from = bx
            bx_to = bx + block_size
            v = hist_tensor[by_from:by_to, bx_from:bx_to, :].flatten()  # to 1-D array (vector)
            feature_tensor[by, bx, :] = v / LA.norm(v, 2)
            # avoid NaN:
            if np.isnan(feature_tensor[by, bx, :]).any():  # avoid NaN (zero division)
                feature_tensor[by, bx, :] = v

    return feature_tensor.flatten()  # 34596 features


def main(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(src=img, dsize=(256, 256))
    f = hog(img)
    print('Extracted feature vector of %s. Shape:' % img_path)
    print('Feature size:', f.shape)
    print('Features (HOG):', f)
    pass


if __name__ == "__main__":
    print('Start running HOG on image @ %s' % IMG)
    main(IMG)
