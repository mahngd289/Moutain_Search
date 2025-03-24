import numpy as np


def cosine_similarity(vec1, vec2):
    """Tính toán độ tương tự cosine giữa 2 vector"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0

    return dot_product / (norm_vec1 * norm_vec2)


def euclidean_distance(vec1, vec2):
    """Tính khoảng cách Euclidean giữa 2 vector"""
    return np.linalg.norm(vec1 - vec2)


def get_similar_images(query_features, feature_type="vgg16", metric="cosine", top_k=3):
    """
    Tìm ảnh tương tự dựa trên đặc trưng và độ đo

    Args:
        query_features: Vector đặc trưng của ảnh truy vấn
        feature_type: Loại đặc trưng cần so sánh
        metric: Phương pháp đo độ tương đồng ('cosine' hoặc 'euclidean')
        top_k: Số lượng kết quả trả về

    Returns:
        List các ID và độ tương đồng của các ảnh giống nhất
    """
    # Lấy tất cả đặc trưng từ database
    cur.execute(
        "SELECT image_id, feature_value FROM image_features WHERE feature_type = %s",
        (feature_type,)
    )
    results = cur.fetchall()

    similarities = []

    for image_id, feature_json in results:
        db_features = np.array(json.loads(feature_json))

        if metric == "cosine":
            similarity = cosine_similarity(query_features, db_features)
            # Với cosine similarity, giá trị càng lớn càng tương đồng
            similarities.append((image_id, similarity))
        else:  # euclidean
            distance = euclidean_distance(query_features, db_features)
            # Với euclidean distance, giá trị càng nhỏ càng tương đồng
            # Đổi dấu để sắp xếp giảm dần như cosine
            similarities.append((image_id, -distance))

    # Sắp xếp theo độ tương đồng giảm dần
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Trả về top_k kết quả
    return similarities[:top_k]


def search_similar_images(query_image_path, feature_type="vgg16", top_k=3):
    """
    Hàm tìm kiếm ảnh tương tự từ một ảnh đầu vào

    Args:
        query_image_path: Đường dẫn đến ảnh truy vấn
        feature_type: Loại đặc trưng sử dụng
        top_k: Số lượng ảnh tương tự trả về

    Returns:
        List các đường dẫn đến ảnh tương tự nhất
    """
    # Trích xuất đặc trưng từ ảnh truy vấn
    if feature_type == "vgg16":
        query_features = extract_features_vgg16(query_image_path)
    elif feature_type == "color_histogram":
        query_features = extract_color_histogram(cv2.imread(query_image_path))
    # Thêm các loại đặc trưng khác nếu cần

    # Tìm các ảnh tương tự
    similar_images = get_similar_images(query_features, feature_type, top_k=top_k)

    # Lấy đường dẫn đến các ảnh tương tự
    image_paths = []
    for image_id, similarity in similar_images:
        cur.execute("SELECT image_path FROM mountain_images WHERE id = %s", (image_id,))
        path = cur.fetchone()[0]
        image_paths.append((path, similarity))

    return image_paths