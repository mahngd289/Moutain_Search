import cv2
import numpy as np


def extract_eoh_from_canny(img, low_thresh, high_thresh, sobel_size=3, eoh_bins=9):
    """
    - Dùng OpenCV Canny để phát hiện cạnh
    - Tính Gx, Gy để lấy hướng cạnh (angle)
    - Tính EOH từ các điểm là cạnh (canny_mask > 0)
    """

    # 1. Làm mịn
    blurred = img

    # 2. Dùng Canny từ OpenCV
    canny_mask = cv2.Canny(blurred, threshold1=low_thresh, threshold2=high_thresh, apertureSize=sobel_size)

    # 3. Tính Gx, Gy bằng Sobel để lấy hướng cạnh
    Gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=sobel_size)
    Gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=sobel_size)

    angle = np.arctan2(Gy, Gx) * 180 / np.pi
    angle[angle < 0] += 180  # Đưa về [0, 180)

    # 4. Trích góc tại vị trí là biên thực sự
    edge_angles = angle[canny_mask > 0]
    hist, _ = np.histogram(edge_angles, bins=eoh_bins, range=(0, 180))
    hist_normalized = hist / np.sum(hist) if np.sum(hist) > 0 else hist

    return canny_mask, hist_normalized


def scale_to_0_255(img):
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val == 0:
        return np.zeros_like(img, dtype=np.uint8)
    new_img = (img - min_val) / (max_val - min_val) * 255
    return new_img.astype(np.uint8)


def my_canny_full_with_eoh(img, min_val, max_val, sobel_size=3, is_L2_gradient=False, eoh_bins=9):
    """
    Hàm thực hiện thuật toán Canny + trả về đặc trưng Edge Orientation Histogram (EOH)
    """

    # 1. Làm mịn ảnh
    smooth_img = cv2.GaussianBlur(img, (5, 5), sigmaX=0.6, sigmaY=0.6)

    # 2. Tính gradient
    Gx = cv2.Sobel(smooth_img, cv2.CV_64F, 1, 0, ksize=sobel_size)
    Gy = cv2.Sobel(smooth_img, cv2.CV_64F, 0, 1, ksize=sobel_size)

    if is_L2_gradient:
        edge_gradient = np.sqrt(Gx**2 + Gy**2)
    else:
        edge_gradient = np.abs(Gx) + np.abs(Gy)

    angle = np.arctan2(Gy, Gx) * 180 / np.pi
    angle[angle < 0] += 180  # Đưa về dải [0, 180)

    # 3. Làm tròn hướng cạnh về 0, 45, 90, 135 độ
    rounded_angle = angle.copy()
    rounded_angle[(angle <= 22.5) | (angle > 157.5)] = 0
    rounded_angle[(angle > 22.5) & (angle <= 67.5)] = 45
    rounded_angle[(angle > 67.5) & (angle <= 112.5)] = 90
    rounded_angle[(angle > 112.5) & (angle <= 157.5)] = 135

    # 4. Non-maximum suppression
    keep_mask = np.zeros_like(img, dtype=np.uint8)
    for y in range(1, img.shape[0] - 1):
        for x in range(1, img.shape[1] - 1):
            curr_angle = rounded_angle[y, x] # Lấy hướng cạnh đã làm tròn
            curr_grad = edge_gradient[y, x] # Lấy độ mạnh grad của cạnh tại điểm hiện tại

            if curr_angle == 0:
                neighbors = [edge_gradient[y, x - 1], edge_gradient[y, x + 1]]
            elif curr_angle == 45:
                neighbors = [edge_gradient[y - 1, x + 1], edge_gradient[y + 1, x - 1]]
            elif curr_angle == 90:
                neighbors = [edge_gradient[y - 1, x], edge_gradient[y + 1, x]]
            elif curr_angle == 135:
                neighbors = [edge_gradient[y - 1, x - 1], edge_gradient[y + 1, x + 1]]
            else:
                continue

            if curr_grad >= max(neighbors):
                keep_mask[y, x] = 255
            else:
                edge_gradient[y, x] = 0

    # 5. Hysteresis thresholding
    canny_mask = np.zeros_like(img, dtype=np.uint8)
    canny_mask[(keep_mask > 0) & (edge_gradient >= min_val)] = 255

    # 6. Tính EOH (chỉ tại các pixel biên thực sự)
    edge_angles = angle[canny_mask > 0]
    hist, _ = np.histogram(edge_angles, bins=eoh_bins, range=(0, 180))
    hist_normalized = hist / np.sum(hist) if np.sum(hist) > 0 else hist

    return scale_to_0_255(canny_mask), hist_normalized


gray = cv2.imread('E:\OnSchool\Moutain_Search\images\gsun_01a0b4a162e02e6d3913b3a620540297.jpg', cv2.IMREAD_GRAYSCALE)
gray = cv2.resize(gray, (256, 256))
# Gọi hàm Canny + EOH
canny_img, eoh_feature = my_canny_full_with_eoh(gray, min_val=100, max_val=150)
canny_result, eoh_feature_noCustom = extract_eoh_from_canny(gray, low_thresh=100, high_thresh=150)
# Hiển thị ảnh Canny
cv2.imshow("Canny", canny_img)
cv2.imshow("OpenCV Canny", canny_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In vector đặc trưng EOH
print("EOH descriptor:", eoh_feature)
print("EOH descriptor:", eoh_feature_noCustom)