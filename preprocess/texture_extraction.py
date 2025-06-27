import math

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_texture_features(image, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
    """Trích xuất đặc trưng kết cấu sử dụng GLCM từ scikit-image"""
    # Chuyển sang ảnh xám và giảm số mức xám để tính toán nhanh hơn
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = (gray / 16).astype(np.uint8)  # Giảm xuống 16 mức xám

    # Chuyển góc từ radian sang độ cho scikit-image
    angles_deg = [a * 180 / np.pi for a in angles]

    # Tính toán GLCM
    glcm = graycomatrix(gray, distances, angles_deg, levels=16, symmetric=True, normed=True)

    # Tính các thuộc tính từ GLCM
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))

    return np.array([contrast, dissimilarity, homogeneity, energy, correlation])

def quantize_image(image, levels=16):
    """Giảm số mức xám của ảnh về số levels mong muốn."""
    return np.floor(image / (256 / levels)).astype(np.uint8)

def get_offset_from_radian(angle_rad, distance):
    """Chuyển hướng từ radian sang offset (dy, dx)."""
    dy = int(round(-math.sin(angle_rad) * distance))
    dx = int(round(math.cos(angle_rad) * distance))
    return dy, dx

def compute_glcm_custom(image, distance=1, angle_rad=0.0, levels=16):
    """
    Tính GLCM theo hướng angle (radian) và độ lệch distance.

    Các tham số:
        image: Ảnh grayscale đầu vào.
        distance: Độ lệch pixel.
        angle_rad: Góc tính bằng radian.
        levels: Số mức xám.

    Returns:
        GLCM: Ma trận (levels x levels).
    """
    image = quantize_image(image, levels)
    glcm = np.zeros((levels, levels), dtype=np.uint32)

    dy, dx = get_offset_from_radian(angle_rad, distance)
    rows, cols = image.shape

    for y in range(rows):
        for x in range(cols):
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                i = image[y, x]
                j = image[ny, nx]
                glcm[i, j] += 1

    return glcm

#Xây dựng các hàm đặc trưng của GLCM
def normalize_glcm(glcm):
    """Chuẩn hóa GLCM thành xác suất (probability matrix)."""
    glcm_sum = glcm.sum()
    return glcm / glcm_sum if glcm_sum != 0 else glcm

def compute_contrast(glcm):
    """Tính Contrast: tổng trọng số bình phương của hiệu mức xám."""
    levels = glcm.shape[0]
    i, j = np.indices((levels, levels))
    return np.sum(glcm * (i - j) ** 2)

def compute_dissimilarity(glcm):
    """Tính Dissimilarity: tổng trọng số của khoảng cách tuyệt đối."""
    levels = glcm.shape[0]
    i, j = np.indices((levels, levels))
    return np.sum(glcm * np.abs(i - j))

def compute_homogeneity(glcm):
    """Tính Homogeneity: giá trị càng lớn khi các giá trị nằm gần đường chéo chính."""
    levels = glcm.shape[0]
    i, j = np.indices((levels, levels))
    return np.sum(glcm / (1.0 + (i - j) ** 2))

def compute_asm(glcm):
    """Tính ASM (Angular Second Moment) - tổng bình phương ma trận GLCM."""
    return np.sum(glcm ** 2)

def compute_energy(glcm):
    """Tính Energy (căn bậc hai của ASM)."""
    return np.sqrt(compute_asm(glcm))

def compute_correlation(glcm):
    """Tính Correlation: đo mức độ tuyến tính giữa cặp mức xám."""
    levels = glcm.shape[0]
    i, j = np.indices((levels, levels))
    mean_i = np.sum(i * glcm)
    mean_j = np.sum(j * glcm)
    std_i = np.sqrt(np.sum((i - mean_i) ** 2 * glcm))
    std_j = np.sqrt(np.sum((j - mean_j) ** 2 * glcm))

    if std_i == 0 or std_j == 0:
        return 0  # tránh chia cho 0

    return np.sum((i - mean_i) * (j - mean_j) * glcm) / (std_i * std_j)

def extract_glcm_features_from_manual(image, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
    """
    Nhận GLCM (đã tính thủ công) và trả về các đặc trưng texture.
    GLCM sẽ được chuẩn hóa nội bộ.
    """
    glcm = compute_glcm_custom(image)
    glcm_norm = normalize_glcm(glcm)

    contrast = compute_contrast(glcm_norm)
    dissimilarity = compute_dissimilarity(glcm_norm)
    homogeneity = compute_homogeneity(glcm_norm)
    energy = compute_energy(glcm_norm)
    correlation = compute_correlation(glcm_norm)

    return np.array([contrast, dissimilarity, homogeneity, energy, correlation])