#Xây dựng GLCM
import numpy as np
import cv2
import math
from skimage.feature import graycomatrix, graycoprops

#Chuyển ảnh sang Grayscale và giảm mức xám còn 16
def quantize_image(image, levels=16): #Lấy
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.floor(image / (256 / levels)).astype(np.uint8)


#Lấy hướng dy, dx tính từ đầu góc trên bên trái shape
def get_offset(angle_rad, distance): #Lấy
    """Trả về offset (dy, dx) theo chuẩn skimage (row tăng là xuống, col tăng là phải)."""
    dy = int(round(math.sin(angle_rad) * distance))   # Không đảo dấu
    dx = int(round(math.cos(angle_rad) * distance))
    return dy, dx

#Tính GLCM có shape dạng: [levels, levels, num_distances, num_angles] 4 chiều
def compute_glcm_custom(image, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=16, symmetric=True, normed=True): #Lấy
    """
    Tính GLCM theo hướng angle (radian) và độ lệch distance.

    Các tham số:
        image: Ảnh grayscale đầu vào.
        distance: Độ lệch pixel.
        angle_rad: Góc tính bằng radian.
        levels: Số mức xám.

    Returns:
        GLCM: Ma trận (levels x levels x 4).
    """
    image = quantize_image(image, levels)
    rows, cols = image.shape

    glcm = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.uint32)

    for d_idx, distance in enumerate(distances):
        for a_idx, angle in enumerate(angles):
            dy, dx = get_offset(angle, distance)

            for y in range(rows):
                for x in range(cols):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx < cols:
                        i = image[y, x]
                        j = image[ny, nx]
                        glcm[i, j, d_idx, a_idx] += 1
                        if symmetric:
                            glcm[j, i, d_idx, a_idx] += 1

    if normed:
        glcm = glcm.astype(np.float64)
        sums = glcm.sum(axis=(0,1), keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            glcm = np.divide(glcm, sums, where=sums != 0)

    return glcm

# --- Ví dụ sử dụng ---
# Load ảnh bình thường
#img = cv2.imread("C:\\Users\LEGION PC\Downloads\data\\anhdauvao3.jpeg")

# Tham số đầu vào
distances = [1]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# GLCM thủ công
#glcm_manual = compute_glcm_custom(img)

# GLCM skimage
#glcm_lib = graycomatrix(quantize_image(img, 16), distances, angles, levels=16, symmetric=True, normed=True)

# Kiểm tra giống nhau
#print(np.allclose(glcm_manual, glcm_lib))

#Có thể print ra glcm_manual và glcm_lib để xét sự giống nhau
#Hàm xây đặc trưng
def graycoprops_custom(glcm, prop): #Lấy
    """
    Tính đặc trưng từ GLCM 4D giống skimage.feature.graycoprops.

    Parameters:
        glcm: ndarray (levels, levels, num_distances, num_angles), đã chuẩn hoá.
        prop: tên đặc trưng: 'contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation'

    Returns:
        ndarray (num_distances, num_angles) chứa đặc trưng ứng với mỗi (d, a). Với num_d = 1 và num a = 4 sẽ có 4 tập kết quả
    """
    levels = glcm.shape[0]
    I, J = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')

    I = I[:, :, np.newaxis, np.newaxis]  # shape (levels, levels, 1, 1)
    J = J[:, :, np.newaxis, np.newaxis]

    if prop == 'contrast':
        return np.sum(((I - J)**2) * glcm, axis=(0, 1))

    elif prop == 'dissimilarity':
        return np.sum(np.abs(I - J) * glcm, axis=(0, 1))

    elif prop == 'homogeneity':
        return np.sum(glcm / (1. + (I - J)**2), axis=(0, 1))

    elif prop == 'ASM':
        return np.sum(glcm**2, axis=(0, 1))

    elif prop == 'energy':
        asm = np.sum(glcm**2, axis=(0, 1))
        return np.sqrt(asm)

    elif prop == 'correlation':
        eps = 1e-10
        mean_i = np.sum(I * glcm, axis=(0, 1))
        mean_j = np.sum(J * glcm, axis=(0, 1))

        std_i = np.sqrt(np.sum((I - mean_i)**2 * glcm, axis=(0, 1)))
        std_j = np.sqrt(np.sum((J - mean_j)**2 * glcm, axis=(0, 1)))

        numerator = np.sum((I - mean_i) * (J - mean_j) * glcm, axis=(0, 1))
        return numerator / (std_i * std_j + eps)

    else:
        raise ValueError(f"Unsupported property '{prop}'. Choose from: contrast, dissimilarity, homogeneity, ASM, energy, correlation.")

def features_extraction_mean(glcm): #Lấy
    contrast = graycoprops_custom(glcm, 'contrast')         # shape (num_d, num_a)
    dissimilarity = graycoprops_custom(glcm, 'dissimilarity')
    homogeneity = graycoprops_custom(glcm, 'homogeneity')
    energy = graycoprops_custom(glcm, 'energy')
    correlation = graycoprops_custom(glcm, 'correlation')
    # Return trung bình các đặc trưng
    features = np.array([contrast.mean(), dissimilarity.mean(), homogeneity.mean(), energy.mean(), correlation.mean()])
    return features

def features_extraction_mean_lib(glcm):
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))

    return np.array([contrast, dissimilarity, homogeneity, energy, correlation])

#features = features_extraction_mean(glcm_manual)
#print(features)
#features_lib = features_extraction_mean_lib(glcm_lib)
#print(features_lib)