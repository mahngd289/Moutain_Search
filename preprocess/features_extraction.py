import cv2
import numpy as np
from sklearn.preprocessing import normalize
from skimage.feature import graycomatrix, graycoprops


def extract_color_histogram(image, bins=16):
    """Trích xuất histogram màu từ ảnh"""
    # Chuyển sang không gian màu HSV (phù hợp hơn cho phân tích ảnh phong cảnh)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Tính histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins], [0, 180, 0, 256, 0, 256])
    # Chuẩn hóa histogram
    hist = cv2.normalize(hist, hist).flatten()
    return hist


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


def extract_edge_features(image, ksize=3):
    """Trích xuất đặc trưng cạnh"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Phát hiện cạnh sử dụng Canny
    edges = cv2.Canny(gray, 100, 200)
    # Tính histogram hướng cạnh
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    mag, ang = cv2.cartToPolar(gx, gy)

    # Histogram của góc cạnh
    bins = 36  # Chia 360 độ thành 36 bins
    hist = np.zeros(bins)
    edge_count = 0
    
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i, j] > 0:
                edge_count += 1
                bin_idx = int(ang[i, j] * 180 / np.pi * bins / 360)
                hist[bin_idx % bins] += 1
    
    # Handle case when no edges are detected
    if edge_count == 0:
        # Return uniform distribution instead of zeros
        return np.ones(bins) / bins
    
    # Manual normalization to avoid division by zero issues
    sum_hist = np.sum(hist)
    if sum_hist > 0:
        hist = hist / sum_hist
    else:
        hist = np.ones(bins) / bins
        
    return hist


def extract_all_features(image_path):
    """Trích xuất tất cả đặc trưng từ ảnh"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")

    # Resize ảnh nếu cần
    image = cv2.resize(image, (224, 224))

    # Trích xuất các đặc trưng
    color_hist = extract_color_histogram(image)
    texture_feats = extract_texture_features(image)
    edge_feats = extract_edge_features(image)

    # Ghép tất cả đặc trưng lại
    all_features = np.concatenate([color_hist, texture_feats, edge_feats])

    return all_features