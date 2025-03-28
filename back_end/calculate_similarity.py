import numpy as np
import json
import cv2
from helper.db_connection import get_db_connection
from preprocess.deep_learning_extraction import extract_features_vgg16
from preprocess.features_extraction import extract_color_histogram


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
    """
    # Lấy kết nối
    conn = get_db_connection()
    cur = conn.cursor()

    # Lấy tất cả đặc trưng từ database
    cur.execute(
        "SELECT image_id, feature_value FROM image_features WHERE feature_type = %s",
        (feature_type,)
    )
    results = cur.fetchall()

    similarities = []

    for image_id, feature_json in results:
        # db_features = np.array(json.loads(feature_json))

        # Kiểm tra kiểu dữ liệu của feature_json
        if isinstance(feature_json, str):
            # Nếu là chuỗi JSON, chuyển thành đối tượng Python
            db_features = np.array(json.loads(feature_json))
        else:
            # Nếu đã là list Python, chuyển trực tiếp thành mảng NumPy
            db_features = np.array(feature_json)

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
    """
    # Lấy kết nối
    conn = get_db_connection()
    cur = conn.cursor()

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