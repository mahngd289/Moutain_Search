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