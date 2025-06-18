import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
Xử lý tốt các ảnh núi đa dạng (mùa đông ít màu, mùa thu nhiều màu)
Không phân cụm quá chi tiết cho ảnh đơn giản hoặc quá thô cho ảnh phức tạp
'''
def elbow_optimal_k(image, max_k=5):

    '''LAB được thiết kế để gần với cách con người nhận thức màu sắc
    L = độ sáng, a = kênh màu xanh lá-đỏ, b = kênh màu xanh dương-vàng'''
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    '''Biến đổi ảnh 2D thành ma trận, mỗi hàng là một pixel với 3 giá trị (L,a,b)'''
    pixels = lab_image.reshape(-1, 3).astype(np.float32)

    '''Lấy mẫu tối đa 20k pixel và không trùng lặp'''
    if pixels.shape[0] > 20000:
        indices = np.random.choice(pixels.shape[0], 20000, replace=False)
        pixels = pixels[indices]

    '''Định nghĩa tiêu chí dừng cho thuật toán K-means,  Với ảnh núi có nhiều gradient màu và chi tiết, cần một điểm cân bằng giữa phân cụm chi tiết và tốc độ xử lý'''
    '''cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER kết hợp hai điều kiện dừng qua phép toán bitwise OR'''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

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
    elbow_point = np.argmax(-acceleration) + 2  # +2 vì diff hai lần và index bắt đầu từ 1

    # Giới hạn k trong khoảng hợp lý
    k_optimal = max(2, min(elbow_point, max_k))

    # Áp dụng heuristic cho ảnh núi:
    # - Ảnh núi thường có ít nhất 3-4 màu chính
    k_optimal = max(3, k_optimal)

    return k_optimal


def extract_dominant_colors(image, k=None):

    # Nếu k không được chỉ định, tìm k tối ưu
    if k is None:
        k = elbow_optimal_k(image)

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

    # Sắp xếp các cụm theo số lượng pixel giảm dần - Màu với tỷ lệ cao luôn được giữ lại, màu ít hơn có thể bị loại
    sorted_indices = np.argsort(counts)[::-1]

    '''Mô hình cảnh quan núi: Hầu hết ảnh núi có khoảng 5 thành phần màu chính:  
        Bầu trời (xanh/xám/trắng)
        Đỉnh núi (nâu/xám/tuyết)
        Thực vật (xanh lá/vàng)
        Mặt nước (nếu có) hoặc đất đá
        Bóng râm/vùng tối'''
    result = []

    # Tìm màu chủ đạo của mỗi cụm
    for i in sorted_indices:
        # Chuyển về không gian HSV để lưu trữ - phù hợp với màu cảnh quan
        bgr_color = cv2.cvtColor(np.uint8([[centers[i]]]), cv2.COLOR_LAB2BGR)[0][0]
        hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]

        # 3 giá trị HSV (không gian màu phù hợp với cảnh tự nhiên)
        result.extend(hsv_color.tolist())

        # 1 giá trị tỷ lệ (% diện tích của màu đó trong ảnh)
        result.append(counts[i] / len(labels))

    '''Nắm bắt đặc tính riêng: Mỗi ảnh núi được mô tả bằng số màu tối ưu với đặc điểm riêng
    Ưu tiên màu quan trọng: Các màu chiếm tỷ lệ nhỏ không làm nhiễu kết quả
    Tăng độ chính xác: Ảnh đơn giản (mùa đông) được biểu diễn với ít cụm màu, ảnh phức tạp (mùa thu) có nhiều cụm hơn
    Hai ảnh núi với K khác nhau vẫn có thể so sánh trực tiếp thông qua vector 20 chiều này, vì màu chủ đạo quan trọng nhất luôn được giữ lại và sắp xếp theo thứ tự giảm dần về tỷ lệ xuất hiện.'''
    # Đảm bảo chiều cố định bằng cách padding nếu cần
    while len(result) < 20:  # Đảm bảo 5 màu * 4 giá trị
        result.extend([0, 0, 0, 0])

    return np.array(result[:20])  # Lấy chính xác 20 giá trị đầu tiên


"""Trích xuất histogram màu theo vùng"""
def extract_color_regions(image, regions=5, visualize=False):
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
        # Nếu là vùng cuối cùng, lấy toàn bộ chiều cao còn lại để đảm bảo không bỏ sót pixel nào
        y_end = (i + 1) * region_height if i < regions - 1 else height

        # Cắt một khung hình chữ nhật từ ảnh HSV từ tọa độ y_start đến y_end và lấy tất cả các cột (toàn bộ chiều rộng).
        region = hsv_image[y_start:y_end, :]

        '''
        [region]: Vùng ảnh đang xét (một phần của ảnh HSV)
        [0, 1]: Chỉ định kênh màu được sử dụng - kênh 0 (Hue - màu sắc) và kênh 1 (Saturation - độ bão hòa). Không sử dụng kênh Value (độ sáng).
        None: Không sử dụng mặt nạ, tức là tất cả các pixel trong vùng đều được tính
        [8, 8]: Số lượng bins (thùng) cho mỗi kênh - 8 bins cho Hue và 8 bins cho Saturation, tạo thành lưới 8x8 (64 giá trị)
        [0, 180, 0, 256]: Khoảng giá trị của mỗi kênh - Hue từ 0-180 (trong OpenCV) và Saturation từ 0-256
        '''
        '''
        Nắm bắt được mối tương quan giữa màu sắc và độ bão hòa
        Phân biệt được, ví dụ, màu xanh nhạt (xanh ít bão hòa) với màu xanh đậm (xanh nhiều bão hòa)
        Cung cấp thông tin chi tiết hơn về phân phối màu sắc, quan trọng khi phân tích ảnh phong cảnh
        '''
        hist = cv2.calcHist([region], [0, 1], None, [8, 8], [0, 180, 0, 256])
        orig_hist = hist.copy()
        hist = cv2.normalize(hist, hist).flatten() * weights[i]

        if visualize:
            plt.subplot(2, regions, i + 1)
            # Hiển thị vùng ảnh
            region_bgr = cv2.cvtColor(region, cv2.COLOR_HSV2BGR)
            plt.imshow(cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB))
            plt.title(f'Vùng {i + 1}')
            plt.axis('off')

            plt.subplot(2, regions, i + 1 + regions)
            # Reshape và hiển thị histogram 2D
            hist_display = cv2.normalize(orig_hist, None, 0, 255, cv2.NORM_MINMAX)
            plt.imshow(hist_display, interpolation='nearest')
            plt.title(f'Histogram vùng {i + 1}')
            plt.colorbar()
            plt.xlabel('Saturation')
            plt.ylabel('Hue')

        region_features.extend(hist)

    if visualize:
        plt.tight_layout()
        plt.show()

    return np.array(region_features)


def extract_seasonal_features(image):
    """Trích xuất đặc trưng phản ánh mùa"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Định nghĩa dải màu theo mùa
    # Mùa hè: xanh lá (H: ~60-120)
    # Mùa thu: vàng/cam/đỏ (H: ~0-60) với độ bão hòa cao
    # Mùa đông: trắng (S thấp, V cao) hoặc xanh biển (H: ~120-180)
    # Mùa xuân: xanh lá non (H: ~60-100) với độ bão hòa cao

    '''
    Thành phần Hue (h): Giá trị 60-120 trong không gian HSV tương ứng với dải màu xanh lá cây. Trong OpenCV, Hue nằm trong khoảng 0-180, với:  
    60: xanh lá non
    90: xanh lá đậm
    120: xanh lá pha xanh dương
    Thành phần Saturation (s): Yêu cầu độ bão hòa từ 50 trở lên đảm bảo rằng chỉ các màu xanh lá có độ sống động đủ cao mới được tính, loại bỏ các màu xám nhạt.
    '''
    summer_mask = ((h >= 60) & (h <= 120) & (s >= 50))

    '''
    Thành phần Hue (h):  
    (h <= 30) | (h >= 330): Mặt nạ này chọn dải màu đỏ-cam-vàng, đặc trưng của lá mùa thu
    Trong OpenCV (thang 0-180): ≤ 30 bao gồm đỏ, cam và vàng; ≥ 330 không tồn tại vì Hue chỉ từ 0-180, đây có thể là lỗi logic
    Thành phần Saturation (s):  
    (s >= 100): Yêu cầu độ bão hòa cao, đảm bảo chỉ chọn các màu sống động, rực rỡ đặc trưng của lá mùa thu
    Loại bỏ các màu xám, nâu nhạt không đặc trưng cho mùa thu
    Ý nghĩa: Xác định tỷ lệ các điểm ảnh có màu đỏ/cam/vàng sống động, đặc trưng cho cảnh núi vào mùa thu khi lá cây chuyển màu
    '''
    autumn_mask = ((h <= 30) | (h >= 330)) & (s >= 100)

    '''
    Điều kiện 1 - Các màu trắng/xám nhạt:  
    (s <= 50) & (v >= 150): Chọn các pixel có độ bão hòa thấp (gần với màu xám/trắng) và độ sáng cao
    Đại diện cho tuyết, mây, sương giá đặc trưng trong ảnh núi mùa đông
    Điều kiện 2 - Các màu lạnh:  
    (h >= 120) & (h <= 180): Bao gồm dải màu từ xanh lục-xanh dương đến tím nhạt
    Đại diện cho bóng tối lạnh và bầu trời mùa đông
    Ý nghĩa: Xác định tỷ lệ các điểm ảnh có màu đặc trưng của mùa đông - hoặc là màu sáng không màu (tuyết) hoặc các màu lạnh (xanh dương)
    '''
    winter_mask = ((s <= 50) & (v >= 150)) | ((h >= 120) & (h <= 180))

    '''
    Thành phần Hue (h):  
    (h >= 60) & (h <= 100): Chọn dải màu từ xanh vàng đến xanh lá cây non
    Đặc trưng cho màu lá non mùa xuân, khác với xanh đậm của mùa hè
    Thành phần Saturation (s) và Value (v):  
    (s >= 100): Yêu cầu độ bão hòa cao, đảm bảo màu tươi sáng
    (v >= 100): Yêu cầu độ sáng vừa phải trở lên, loại bỏ các màu xanh tối
    Ý nghĩa: Xác định tỷ lệ các điểm ảnh có màu xanh lá non, tươi và sáng - đặc trưng của thảm thực vật mới mọc trong mùa xuân
    '''
    spring_mask = ((h >= 60) & (h <= 100) & (s >= 100) & (v >= 100))

    # Tính tỷ lệ pixel thuộc về mỗi mùa
    summer_ratio = np.sum(summer_mask) / (h.shape[0] * h.shape[1] + 1e-10)
    autumn_ratio = np.sum(autumn_mask) / (h.shape[0] * h.shape[1] + 1e-10)
    winter_ratio = np.sum(winter_mask) / (h.shape[0] * h.shape[1] + 1e-10)
    spring_ratio = np.sum(spring_mask) / (h.shape[0] * h.shape[1] + 1e-10)

    return np.array([summer_ratio, autumn_ratio, winter_ratio, spring_ratio])


def extract_color_contrast(image):
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

'''
Hai cảnh núi có thể có độ tương phản tương tự nhưng một cảnh đa dạng hơn về màu sắc
Cảnh núi C: Núi tuyết trắng dưới bầu trời xanh  
Có tương phản cao giữa vùng trắng và xanh
Chỉ có 2-3 màu chính
Cảnh núi D: Núi mùa thu với nhiều sắc lá  
Cũng có tương phản giữa các vùng màu
Có nhiều màu (đỏ, cam, vàng, xanh lá, xanh dương...)
→ extract_color_contrast sẽ cho giá trị tương tự (vì cùng có tương phản) → extract_weighted_diversity sẽ cho giá trị cao hơn nhiều cho cảnh D (nhiều màu hơn)
'''
def extract_weighted_diversity(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv_image)

    '''Tính các giá trị màu sắc duy nhất trên từng kênh'''
    h_unique = np.unique(h.flatten())
    s_unique = np.unique(s.flatten())
    v_unique = np.unique(v.flatten())

    h_weight = 2.0  # Hue quan trọng hơn trong cảm nhận màu sắc
    s_weight = 1.0
    v_weight = 0.5  # Value ít quan trọng nhất

    weighted_diversity = (len(h_unique) * h_weight +
                          len(s_unique) * s_weight +
                          len(v_unique) * v_weight) / (h_weight + s_weight + v_weight)

    max_possible = (180 * h_weight + 256 * s_weight + 256 * v_weight) / (h_weight + s_weight + v_weight)
    return weighted_diversity / max_possible


def improved_color_features(image):

    # 1. Màu chủ đạo (~20 chiều)
    dominant_colors = extract_dominant_colors(image)

    # 2. Histogram màu theo vùng (5 vùng, mỗi vùng có 8x8 bins)
    regional_hists = extract_color_regions(image, regions=5)

    # 3. Đặc trưng tương phản màu (9 đặc trưng)
    contrast_features = extract_color_contrast(image)

    # 4. Đặc trưng theo mùa (4 đặc trưng)
    seasonal_features = extract_seasonal_features(image)

    # 5. Weighted diversity feature (1 chiều)
    diversity_feature = np.array([extract_weighted_diversity(image)])

    dominant_colors_norm = dominant_colors / (np.linalg.norm(dominant_colors) + 1e-10)
    regional_hists_norm = regional_hists / (np.linalg.norm(regional_hists) + 1e-10)
    contrast_features_norm = contrast_features / (np.linalg.norm(contrast_features) + 1e-10)
    seasonal_features_norm = seasonal_features / (np.linalg.norm(seasonal_features) + 1e-10)

    # Ghép tất cả đặc trưng
    combined_features = np.concatenate([
        dominant_colors_norm * 1.5,
        regional_hists_norm,
        contrast_features_norm * 1.2,
        seasonal_features_norm * 1.8,
        diversity_feature * 1.5
    ])  # Tổng ~354 chiều

    return combined_features