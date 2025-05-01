import cv2
import numpy as np

# TODO: conda install scikit-image==0.24.0
from skimage import color, feature
from sklearn.decomposition import PCA
from scipy.signal import find_peaks



# TODO : Tối ưu lại phương pháp tìm số cụm cho K-means (Xét xem việc tính toán có mang lại nhiều ý nghĩa không). Có thể in ra để xem số cụm, sử dụng VGG16 để so sánh kết quả.
def find_optimal_k(image, max_k=10):
    """Tìm số cụm màu tối ưu với phương pháp đa tiêu chí"""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pixels = lab_image.reshape(-1, 3).astype(np.float32)

    # Lấy mẫu cho hiệu suất tốt hơn
    if pixels.shape[0] > 20000:
        indices = np.random.choice(pixels.shape[0], 20000, replace=False)
        pixels = pixels[indices]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # 1. Phân tích độ phức tạp màu sắc (cải tiến)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    # Tính trên từng kênh màu riêng biệt để có đánh giá tốt hơn
    h_unique = np.unique(h.flatten())
    s_unique = np.unique(s.flatten())
    v_unique = np.unique(v.flatten())

    # Đếm số màu sắc thực sự khác biệt dựa trên tất cả các kênh
    h_weight = 2.0  # Hue quan trọng hơn trong cảm nhận màu sắc
    s_weight = 1.0
    v_weight = 0.5  # Value ít quan trọng nhất

    # Công thức cân bằng: trọng số cho mức độ quan trọng của từng kênh
    weighted_diversity = (len(h_unique) * h_weight +
                          len(s_unique) * s_weight +
                          len(v_unique) * v_weight) / (h_weight + s_weight + v_weight)

    color_diversity = min(int(weighted_diversity // 25), max_k)

    # 2. Phân tích mức độ phân đoạn của ảnh
    # Phát hiện cạnh để xác định độ phức tạp của cảnh
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
    segmentation_estimate = min(int(edge_density * 15) + 3, max_k)

    # 3. Tính inertia và áp dụng phương pháp "knee detection"
    inertias = []
    for k in range(1, max_k + 1):
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        # Tính inertia nhanh hơn với vectorization
        cluster_distances = np.zeros(len(pixels))
        for j in range(k):
            mask = (labels.flatten() == j)
            if np.any(mask):
                cluster_distances[mask] = np.sum((pixels[mask] - centers[j]) ** 2, axis=1)

        inertias.append(np.sum(cluster_distances))

    # Chuẩn hóa inertias
    inertias = np.array(inertias)
    normalized_inertias = inertias / inertias[0]

    # Tìm "knee point" - điểm có sự thay đổi lớn nhất trong gradient
    gradients = np.diff(normalized_inertias)
    acceleration = np.diff(gradients)

    # Tìm điểm có gia tốc âm lớn nhất (thay đổi lớn nhất trong gradient)
    k_knee = np.argmax(-acceleration) + 2  # +2 vì diff hai lần và chỉ số bắt đầu từ 1

    # 4. Kết hợp các tiêu chí
    # Nếu các phương pháp cho kết quả gần nhau, điều đó cho thấy k đó thực sự tốt
    methods = [color_diversity, segmentation_estimate, k_knee]

    # Lấy giá trị trung vị để tránh outlier
    k_optimal = int(np.median(methods))

    # Áp dụng heuristic cho ảnh núi:
    # - Ảnh núi thường có ít nhất 4 màu chính (bầu trời, núi, thực vật, đất/đá)
    k_optimal = max(4, k_optimal)
    k_optimal = min(k_optimal, max_k)

    return k_optimal


# TODO : Kiểm tra lại số chiều của thuộc tính này trong enhanced_color_vector (có thể bị thiếu)
def extract_dominant_colors(image, k=None):
    """Trích xuất k màu chủ đạo từ ảnh với k thích ứng"""
    # Nếu k không được chỉ định, tìm k tối ưu
    if k is None:
        k = find_optimal_k(image)

    # Chuyển sang không gian màu Lab cho nhận thức màu tốt hơn
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Reshape thành danh sách pixel
    pixels = lab_image.reshape(-1, 3).astype(np.float32)

    # Định nghĩa điều kiện dừng
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # K-Means clustering
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Tính số lượng pixel cho mỗi cụm
    counts = np.bincount(labels.flatten(), minlength=k)

    # Sắp xếp các cụm theo số lượng pixel giảm dần
    sorted_indices = np.argsort(counts)[::-1]

    # Tạo vector đặc trưng: center + tỷ lệ
    result = []
    for i in sorted_indices:
        # Chuyển về không gian HSV để lưu trữ - phù hợp với màu cảnh quan
        bgr_color = cv2.cvtColor(np.uint8([[centers[i]]]), cv2.COLOR_LAB2BGR)[0][0]
        hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]

        # Thêm giá trị H, S, V
        result.extend(hsv_color.tolist())

        # Thêm tỷ lệ
        result.append(counts[i] / len(labels))

    # Đảm bảo chiều cố định bằng cách padding nếu cần
    while len(result) < 20:  # Đảm bảo 5 màu * 4 giá trị
        result.extend([0, 0, 0, 0])

    return np.array(result[:20])  # Lấy chính xác 20 giá trị đầu tiên


def extract_color_regions(image, regions=5):
    """Trích xuất histogram màu theo vùng"""
    height, width = image.shape[:2]
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Chia ảnh thành các vùng dọc (từ trên xuống dưới)
    region_height = height // regions
    region_features = []

    # Trọng số cho các vùng (vùng giữa quan trọng hơn cho ảnh núi)
    weights = [0.15, 0.2, 0.3, 0.2, 0.15]  # Cho 5 vùng

    for i in range(regions):
        # Xác định vùng
        y_start = i * region_height
        y_end = (i + 1) * region_height if i < regions - 1 else height

        region = hsv_image[y_start:y_end, :]

        # Tính histogram với số bin ít hơn
        hist = cv2.calcHist([region], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten() * weights[i]

        region_features.extend(hist)

    return np.array(region_features)


def detect_horizon(image):
    """Phát hiện đường chân trời trong ảnh đồi núi"""
    # Chuyển đổi hình ảnh sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng bộ lọc Canny để phát hiện cạnh
    edges = cv2.Canny(gray, 50, 150)

    # Áp dụng phép biến đổi Hough để phát hiện đường thẳng
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    height, width = image.shape[:2]
    horizon_features = np.zeros(5)  # Vector đặc trưng mô tả đường chân trời

    if lines is not None:
        # Tính toán histogram của các đường theo góc
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:  # Tránh chia cho 0
                angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                angles.append(angle)

        if angles:
            # Tạo histogram của các góc
            hist, bins = np.histogram(angles, bins=18, range=(-90, 90))

            # Tìm góc phổ biến nhất
            dominant_angle_idx = np.argmax(hist)
            dominant_angle = (bins[dominant_angle_idx] + bins[dominant_angle_idx + 1]) / 2

            # Giả định các đường có góc gần với góc phổ biến là đường chân trời
            horizon_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:
                    angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                    if abs(angle - dominant_angle) < 10:
                        horizon_lines.append(line[0])

            if horizon_lines:
                # Tính toán vị trí trung bình của các đường chân trời
                y_positions = []
                for x1, y1, x2, y2 in horizon_lines:
                    y_positions.append((y1 + y2) / 2)

                # Đặc trưng 1: Vị trí tương đối của đường chân trời (từ 0 đến 1)
                horizon_position = np.median(y_positions) / height
                horizon_features[0] = horizon_position

                # Đặc trưng 2: Độ cong của đường chân trời
                std_y = np.std(y_positions) / height
                horizon_features[1] = std_y

                # Đặc trưng 3: Góc của đường chân trời (chuẩn hóa)
                horizon_features[2] = (dominant_angle + 90) / 180

                # Đặc trưng 4: Số lượng đường chân trời được phát hiện (chuẩn hóa)
                horizon_features[3] = min(len(horizon_lines) / 20, 1.0)

                # Đặc trưng 5: Độ phức tạp của đường chân trời
                complexity = len(set(tuple(line) for line in horizon_lines)) / len(
                    horizon_lines) if horizon_lines else 0
                horizon_features[4] = complexity

    # Nếu không tìm thấy đường chân trời, giá trị mặc định
    if np.sum(horizon_features) == 0:
        # Giá trị mặc định: vị trí ở giữa ảnh, không có độ cong, không có góc
        horizon_features[0] = 0.5  # Vị trí mặc định ở giữa

    return horizon_features


def extract_seasonal_features(image):
    """Trích xuất đặc trưng phản ánh mùa"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Định nghĩa dải màu theo mùa
    # Mùa hè: xanh lá (H: ~60-120)
    # Mùa thu: vàng/cam/đỏ (H: ~0-60) với độ bão hòa cao
    # Mùa đông: trắng (S thấp, V cao) hoặc xanh biển (H: ~120-180)
    # Mùa xuân: xanh lá non (H: ~60-100) với độ bão hòa cao

    # Mặt nạ cho từng dải màu
    summer_mask = ((h >= 60) & (h <= 120) & (s >= 50))
    autumn_mask = ((h <= 30) | (h >= 330)) & (s >= 100)
    winter_mask = ((s <= 50) & (v >= 150)) | ((h >= 120) & (h <= 180))
    spring_mask = ((h >= 60) & (h <= 100) & (s >= 100) & (v >= 100))

    # Tính tỷ lệ pixel thuộc về mỗi mùa
    summer_ratio = np.sum(summer_mask) / (h.shape[0] * h.shape[1] + 1e-10)
    autumn_ratio = np.sum(autumn_mask) / (h.shape[0] * h.shape[1] + 1e-10)
    winter_ratio = np.sum(winter_mask) / (h.shape[0] * h.shape[1] + 1e-10)
    spring_ratio = np.sum(spring_mask) / (h.shape[0] * h.shape[1] + 1e-10)

    # Thêm vài đặc trưng thống kê khác
    # Nhiệt độ màu (màu ấm vs màu lạnh)
    warm_mask = (h <= 60) | (h >= 330)
    cool_mask = (h > 60) & (h < 330)
    warm_ratio = np.sum(warm_mask) / (h.shape[0] * h.shape[1] + 1e-10)
    cool_ratio = np.sum(cool_mask) / (h.shape[0] * h.shape[1] + 1e-10)

    # Độ đa dạng màu (phân tán của histogram màu)
    h_diversity = np.std(cv2.calcHist([h], [0], None, [36], [0, 180])) / 50

    return np.array([summer_ratio, autumn_ratio, winter_ratio, spring_ratio,
                     warm_ratio, cool_ratio, h_diversity])


def extract_color_contrast(image):
    """Trích xuất đặc trưng tương phản màu"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Tính các đặc trưng thống kê
    h_mean, h_std = np.mean(h), np.std(h)
    s_mean, s_std = np.mean(s), np.std(s)
    v_mean, v_std = np.mean(v), np.std(v)

    # Tính toán các đặc trưng tương phản
    h_contrast = h_std / (h_mean + 1e-10)
    s_contrast = s_std / (s_mean + 1e-10)
    v_contrast = v_std / (v_mean + 1e-10)

    return np.array([h_mean, h_std, h_contrast, s_mean, s_std, s_contrast, v_mean, v_std, v_contrast])


# TODO : Hiện tại số chiều là 361 có thể khá lớn nên giảm số chiều bằng PCA (hàm để giảm số chiều này chưa được triển khai). Nếu triển khai nên dùng 1 model PCS được huấn luyện trước để giảm chi phí tính toán
def reduce_dimensions(features, n_components=100):
    """Giảm chiều dữ liệu sử dụng PCA"""
    # Reshape để PCA có thể hoạt động với vector một chiều
    original_shape = features.shape
    features_reshaped = features.reshape(1, -1)

    # Áp dụng PCA
    pca = PCA(n_components=min(n_components, features_reshaped.shape[1]))
    reduced_features = pca.fit_transform(features_reshaped)[0]

    return reduced_features



# TODO : Chưa triển khai giảm số chiều (để giảm số chiều nên chạy hàm update_features để đảm bảo đã có dữ liệu trong bảng mountain_images)
def improved_color_features(image, reduce_dim=False, n_components=100):
    """Tích hợp tất cả các đặc trưng màu cải tiến"""
    # 1. Màu chủ đạo (với k thích ứng)
    dominant_colors = extract_dominant_colors(image)  # ~20 chiều

    # 2. Histogram màu theo vùng (5 vùng, mỗi vùng có 8x8 bins)
    regional_hists = extract_color_regions(image, regions=5)  # 5*8*8 = 320 chiều

    # 3. Đặc trưng tương phản màu (9 đặc trưng)
    contrast_features = extract_color_contrast(image)  # 9 chiều

    # 4. Đặc trưng đường chân trời (5 đặc trưng)
    horizon_features = detect_horizon(image)  # 5 chiều

    # 5. Đặc trưng theo mùa (7 đặc trưng)
    seasonal_features = extract_seasonal_features(image)  # 7 chiều

    # Ghép tất cả đặc trưng
    combined_features = np.concatenate([
        dominant_colors,  # ~20 chiều
        regional_hists,  # 320 chiều
        contrast_features,  # 9 chiều
        horizon_features,  # 5 chiều
        seasonal_features  # 7 chiều
    ])  # Tổng ~361 chiều

    # Chuẩn hóa
    norm = np.linalg.norm(combined_features)
    if norm > 0:
        combined_features = combined_features / norm

    # Áp dụng giảm chiều nếu được yêu cầu
    if reduce_dim:
        combined_features = reduce_dimensions(combined_features, n_components)

    return combined_features