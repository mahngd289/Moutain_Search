import numpy as np
import cv2
from helper.db_connection import get_db_connection
from preprocess.deep_learning_extraction import extract_features_vgg16
from preprocess.texture_extraction import extract_texture_features
from preprocess.texture_extraction import extract_glcm_features_from_manual
from preprocess.color_extraction import improved_color_features
from preprocess.eoh_extraction import my_canny_full_with_eoh
from Moutain_Search.preprocess.texture_custom import *

def parse_vector(vec):
    """Convert any vector format to proper numpy array"""
    if vec is None:
        return None

    try:
        # If already a numpy array, just ensure it's float64
        if isinstance(vec, np.ndarray):
            return vec.astype(np.float64)

        # If it's a string representation like "[0.123, 0.456, ...]"
        if isinstance(vec, str):
            if vec.startswith('[') and vec.endswith(']'):
                vec = vec[1:-1]
            # Split by comma and convert each value to float
            return np.array([float(x.strip()) for x in vec.split(',') if x.strip()], dtype=np.float64)

        # For list or other iterables
        return np.array(vec, dtype=np.float64)

    except Exception as e:
        print(f"Error parsing vector: {str(e)[:100]}")
        return np.array([], dtype=np.float64)


def cosine_similarity(vec1, vec2):
    """Tính toán độ tương tự cosine giữa 2 vector"""
    # Parse vectors to ensure they're numeric arrays
    vec1 = parse_vector(vec1)
    vec2 = parse_vector(vec2)

    # Check if vectors are valid
    if vec1 is None or vec2 is None or len(vec1) == 0 or len(vec2) == 0:
        return 0

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0

    return dot_product / (norm_vec1 * norm_vec2)


def euclidean_distance(vec1, vec2):
    """Tính khoảng cách Euclidean giữa 2 vector"""
    # Parse vectors to ensure they're numeric arrays
    vec1 = parse_vector(vec1)
    vec2 = parse_vector(vec2)

    # Check if vectors are valid
    if vec1 is None or vec2 is None or len(vec1) == 0 or len(vec2) == 0:
        return float('inf')  # Return infinite distance for invalid vectors

    vec1 = np.nan_to_num(vec1)
    vec2 = np.nan_to_num(vec2)

    return np.linalg.norm(vec1 - vec2)


def get_similar_images_vgg16(query_features, metric="cosine", top_k=3):
    """Tìm ảnh tương tự dựa trên đặc trưng VGG16"""
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        if metric == "cosine":
            cur.execute("""
                SELECT image_id, 1 - (vgg16_vector <=> %s::vector) as similarity
                FROM image_features_unified
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_features.tolist(), top_k))
        else:  # euclidean
            cur.execute("""
                SELECT image_id, -(vgg16_vector <-> %s::vector) as similarity
                FROM image_features_unified
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_features.tolist(), top_k))

        results = cur.fetchall()
    except Exception as e:
        print(f"Lỗi khi truy vấn pgvector: {e}")

        # Phương án dự phòng nếu pgvector không hoạt động
        cur.execute("SELECT image_id, vgg16_vector FROM image_features_unified")
        db_results = cur.fetchall()
        similarities = []

        for image_id, db_features in db_results:
            if db_features is None:
                continue

            try:
                # Explicit conversion to float array
                db_features = np.array(db_features, dtype=np.float64)
                query_features_arr = np.array(query_features, dtype=np.float64)

                if metric == "cosine":
                    similarity = cosine_similarity(query_features_arr, db_features)
                else:  # euclidean
                    similarity = -euclidean_distance(query_features_arr, db_features)

                similarities.append((image_id, similarity))
            except Exception as e:
                print(f"Lỗi khi xử lý vector: {e}")
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = [(image_id, similarity) for image_id, similarity in similarities if not np.isnan(similarity)]
        results = similarities[:top_k]

    # Lấy đường dẫn ảnh và giá trị tương tự
    image_results = []
    for image_id, similarity in results:
        cur.execute("SELECT image_path FROM mountain_images WHERE id = %s", (image_id,))
        result = cur.fetchone()
        if result:
            image_results.append((result[0], similarity))

    conn.close()
    return image_results


def get_weighted_similar_images(query_features_dict, weights, metric="cosine", top_k=3):
    """
    Tìm ảnh tương tự dựa trên các đặc trưng có trọng số
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Lấy tất cả ảnh từ DB
    cur.execute("""
        SELECT i.id, i.image_path, f.color_vector, f.texture_vector, f.edge_vector, f.vgg16_vector
        FROM mountain_images i
        JOIN image_features_unified f ON i.id = f.image_id
    """)

    all_images = cur.fetchall()
    weighted_similarities = []

    for record in all_images:
        try:
            image_id, image_path, color_vec, texture_vec, edge_vec, vgg16_vec = record

            # Tính độ tương tự cho từng loại đặc trưng
            similarities = {}

            # Đặc trưng màu
            if weights["color"] > 0 and color_vec is not None and "color" in query_features_dict:
                # Sử dụng hàm parse_vector
                color_vec_array = parse_vector(color_vec)
                query_color = parse_vector(query_features_dict["color"])

                if color_vec_array is not None and query_color is not None:
                    if metric == "cosine":
                        similarities["color"] = cosine_similarity(query_color, color_vec_array)
                    else:
                        similarities["color"] = -euclidean_distance(query_color, color_vec_array)
                else:
                    similarities["color"] = 0
            else:
                similarities["color"] = 0

            # Đặc trưng kết cấu
            if weights["texture"] > 0 and texture_vec is not None and "texture" in query_features_dict:
                # Sử dụng hàm parse_vector
                texture_vec_array = parse_vector(texture_vec)
                query_texture = parse_vector(query_features_dict["texture"])

                if texture_vec_array is not None and query_texture is not None:
                    if metric == "cosine":
                        similarities["texture"] = cosine_similarity(query_texture, texture_vec_array)
                    else:
                        similarities["texture"] = -euclidean_distance(query_texture, texture_vec_array)
                else:
                    similarities["texture"] = 0
            else:
                similarities["texture"] = 0

            # Đặc trưng cạnh
            if weights["edge"] > 0 and edge_vec is not None and "edge" in query_features_dict:
                # Sử dụng hàm parse_vector
                edge_vec_array = parse_vector(edge_vec)
                query_edge = parse_vector(query_features_dict["edge"])

                if edge_vec_array is not None and query_edge is not None:
                    if metric == "cosine":
                        similarities["edge"] = cosine_similarity(query_edge, edge_vec_array)
                    else:
                        similarities["edge"] = -euclidean_distance(query_edge, edge_vec_array)
                else:
                    similarities["edge"] = 0
            else:
                similarities["edge"] = 0

            # # Đặc trưng VGG16
            # if "vgg16" in weights and weights["vgg16"] > 0 and vgg16_vec is not None and "vgg16" in query_features_dict:
            #     # Sử dụng hàm parse_vector
            #     vgg16_vec_array = parse_vector(vgg16_vec)
            #     query_vgg16 = parse_vector(query_features_dict["vgg16"])
            #
            #     if vgg16_vec_array is not None and query_vgg16 is not None:
            #         if metric == "cosine":
            #             similarities["vgg16"] = cosine_similarity(query_vgg16, vgg16_vec_array)
            #         else:
            #             similarities["vgg16"] = -euclidean_distance(query_vgg16, vgg16_vec_array)
            #     else:
            #         similarities["vgg16"] = 0
            # else:
            #     similarities["vgg16"] = 0

            # Tính tổng có trọng số
            total_weight = sum(weights.values())
            if total_weight > 0:
                weighted_similarity = (
                    weights["color"] * similarities["color"] +
                    weights["texture"] * similarities["texture"] +
                    weights["edge"] * similarities["edge"]
                ) / total_weight
            else:
                weighted_similarity = 0

            weighted_similarities.append((image_path, weighted_similarity))
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {record[1] if len(record) > 1 else 'unknown'}: {e}")
            continue

    # Sắp xếp theo độ tương tự giảm dần
    weighted_similarities.sort(key=lambda x: x[1], reverse=True)

    # Lọc giá trị NaN
    weighted_similarities = [(path, sim) for path, sim in weighted_similarities if not np.isnan(sim)]

    conn.close()
    return weighted_similarities[:top_k]


def search_similar_images_weighted(query_image_path, weights, metric="cosine", top_k=3):
    """
    Tìm kiếm ảnh tương tự sử dụng trọng số của các đặc trưng

    query_image_path: Đường dẫn đến ảnh truy vấn
    weights: Dict chứa trọng số cho từng loại đặc trưng
    metric: Loại độ đo ("cosine" hoặc "euclidean")
    top_k: Số lượng kết quả trả về
    """
    # Đọc và tiền xử lý ảnh
    image = cv2.imread(query_image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ {query_image_path}")

    image = cv2.resize(image, (224, 224))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Trích xuất tất cả đặc trưng cần thiết
    query_features = {}

    if weights["color"] > 0:
        query_features["color"] = improved_color_features(image)

    if weights["texture"] > 0:
        # query_features["texture"] = extract_texture_features(image)
        # query_features["texture"] = extract_glcm_features_from_manual(image)
        glcm_manual = compute_glcm_custom(image)
        query_features["texture"] = features_extraction_mean(glcm_manual)
    if weights["edge"] > 0:
        _, query_features["edge"] = my_canny_full_with_eoh(gray, min_val=100, max_val=150, eoh_bins=8)

    # if "vgg16" in weights and weights["vgg16"] > 0:
    #     query_features["vgg16"] = extract_features_vgg16(query_image_path)

    # Tìm kiếm ảnh tương tự với các đặc trưng có trọng số
    return get_weighted_similar_images(query_features, weights, metric, top_k)


def search_similar_images_vgg16(query_image_path, metric="cosine", top_k=3):
    """Tìm kiếm ảnh tương tự chỉ sử dụng đặc trưng VGG16"""
    query_features = extract_features_vgg16(query_image_path)
    return get_similar_images_vgg16(query_features, metric, top_k)
