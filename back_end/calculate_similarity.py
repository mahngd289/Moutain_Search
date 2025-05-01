import numpy as np
import cv2
from helper.db_connection import get_db_connection
from preprocess.deep_learning_extraction import extract_features_vgg16
from preprocess.features_extraction import extract_color_histogram, extract_texture_features, extract_edge_features
from preprocess.enhanced_color_features import improved_color_features


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
    # Handle potential NaN values
    vec1 = np.nan_to_num(vec1)
    vec2 = np.nan_to_num(vec2)
    return np.linalg.norm(vec1 - vec2)


def get_similar_images_unified(query_features, feature_type="vgg16", metric="cosine", top_k=3):
    """
    Tìm ảnh tương tự dựa trên đặc trưng và độ đo sử dụng bảng unified
    """
    # Lấy kết nối
    conn = get_db_connection()
    cur = conn.cursor()

    # Xác định cột vector cần sử dụng
    vector_column = f"{feature_type}_vector"
    
    try:
        if metric == "cosine":
            # Thử sử dụng cosine_similarity của pgvector
            cur.execute(f"""
                SELECT image_id, 1 - ({vector_column} <=> %s::vector) as similarity
                FROM image_features_unified
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_features.tolist(), top_k))
            results = cur.fetchall()
        else:  # euclidean
            # Thử sử dụng euclidean_distance của pgvector
            cur.execute(f"""
                SELECT image_id, -({vector_column} <-> %s::vector) as similarity
                FROM image_features_unified
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_features.tolist(), top_k))
            results = cur.fetchall()
    except Exception as e:
        print(f"Không thể sử dụng pgvector: {e}")
        # Fallback nếu không có pgvector
        cur.execute(f"SELECT image_id, {vector_column} FROM image_features_unified")
        db_results = cur.fetchall()
        similarities = []
        
        for image_id, db_features in db_results:
            if db_features is None:
                continue
                
            # Chuyển thành numpy array nếu cần
            if not isinstance(db_features, np.ndarray):
                db_features = np.array(db_features)
            
            if metric == "cosine":
                similarity = cosine_similarity(query_features, db_features)
            else:  # euclidean
                similarity = -euclidean_distance(query_features, db_features)
            
            similarities.append((image_id, similarity))
        
        # Sắp xếp theo độ tương đồng giảm dần
        similarities.sort(key=lambda x: x[1], reverse=True)
        # Xử lý giá trị NaN
        similarities = [(image_id, similarity) for image_id, similarity in similarities if not np.isnan(similarity)]
        
        results = similarities[:top_k]

    return results


def search_similar_images(query_image_path, feature_type="vgg16", top_k=3):
    """
    Hàm tìm kiếm ảnh tương tự từ một ảnh đầu vào
    """
    # Lấy kết nối
    conn = get_db_connection()
    cur = conn.cursor()

    # Đọc ảnh truy vấn
    image = cv2.imread(query_image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ {query_image_path}")
    
    # Resize ảnh để phù hợp với mô hình
    image = cv2.resize(image, (224, 224))
    
    # Trích xuất đặc trưng từ ảnh truy vấn dựa trên loại được chọn
    if feature_type == "vgg16":
        query_features = extract_features_vgg16(query_image_path)
    elif feature_type == "color":
        query_features = extract_color_histogram(image, bins=16)
    elif feature_type == "enhanced_color":
        query_features = improved_color_features(image)
    elif feature_type == "texture":
        query_features = extract_texture_features(image)
    elif feature_type == "edge":
        query_features = extract_edge_features(image)
    else:
        raise ValueError(f"Loại đặc trưng không hợp lệ: {feature_type}")
    
    # Tìm các ảnh tương tự
    similar_images = get_similar_images_unified(query_features, feature_type, top_k=top_k)
    
    # Lấy đường dẫn đến các ảnh tương tự
    image_paths = []
    for image_id, similarity in similar_images:
        cur.execute("SELECT image_path FROM mountain_images WHERE id = %s", (image_id,))
        result = cur.fetchone()
        if result:
            path = result[0]
            image_paths.append((path, similarity))
    
    return image_paths