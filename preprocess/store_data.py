import psycopg2
import os
import json
import numpy as np
import cv2
from PIL import Image
import io


from preprocess.deep_learning_extraction import extract_features_vgg16
from preprocess.features_extraction import extract_color_histogram


try:
    # Kết nối đến PostgreSQL
    conn = psycopg2.connect(
        dbname="moutains_db",
        user="admin_user",
        password="123456",
        host="localhost"
    )
    cur = conn.cursor()
    print("Kết nối thành công!")
except Exception as e:
    print(f"Lỗi kết nối: {e}")


# Hàm lưu ảnh vào database
def save_image_to_db(image_path, store_binary=True):
    """
    Lưu ảnh vào cơ sở dữ liệu

    Args:
        image_path: Đường dẫn đến file ảnh
        store_binary: True để lưu ảnh dưới dạng BYTEA, False chỉ lưu đường dẫn
    """
    filename = os.path.basename(image_path)

    if store_binary:
        # Đọc ảnh thành binary data
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # Chèn vào database
        cur.execute(
            "INSERT INTO mountain_images (file_name, image_data, image_path) VALUES (%s, %s, %s) RETURNING id",
            (filename, psycopg2.Binary(image_data), image_path)
        )
    else:
        # Chỉ lưu đường dẫn
        cur.execute(
            "INSERT INTO mountain_images (file_name, image_path) VALUES (%s, %s) RETURNING id",
            (filename, image_path)
        )

    image_id = cur.fetchone()[0]
    conn.commit()
    return image_id


# Hàm lưu đặc trưng vào database
def save_features_to_db(image_id, features, feature_type="vgg16"):
    """
    Lưu vector đặc trưng vào cơ sở dữ liệu

    Args:
        image_id: ID của ảnh trong bảng mountain_images
        features: Vector đặc trưng
        feature_type: Loại đặc trưng (vgg16, color_hist, ...)
    """
    # Chuyển numpy array thành dạng có thể lưu trong DB
    features_json = json.dumps(features.tolist())

    # Chèn vào database
    cur.execute(
        "INSERT INTO image_features (image_id, feature_type, feature_value) VALUES (%s, %s, %s)",
        (image_id, feature_type, features_json)
    )
    conn.commit()


# Hàm để xử lý toàn bộ thư mục ảnh
def process_image_directory(directory):
    """Xử lý toàn bộ thư mục ảnh và lưu vào database"""
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory, filename)

            # Lưu ảnh vào database
            image_id = save_image_to_db(image_path)

            # Trích xuất đặc trưng
            vgg_features = extract_features_vgg16(image_path)
            color_features = extract_color_histogram(cv2.imread(image_path))

            # Lưu đặc trưng vào database
            save_features_to_db(image_id, vgg_features, "vgg16")
            save_features_to_db(image_id, color_features, "color_histogram")

            print(f"Đã xử lý ảnh {filename}")