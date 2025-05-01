import psycopg2
import os
import numpy as np
import cv2
from PIL import Image
import argparse
import skimage

from preprocess.deep_learning_extraction import extract_features_vgg16
from preprocess.features_extraction import extract_color_histogram, extract_texture_features, extract_edge_features
from preprocess.enhanced_color_features import improved_color_features


def get_db_connection():
    """Tạo và trả về kết nối đến cơ sở dữ liệu"""
    try:
        # Kết nối đến PostgreSQL
        conn = psycopg2.connect(
            dbname="moutains_db",
            user="admin_user",
            password="123456",
            host="localhost"
        )

        # Tạo bảng nếu chưa tồn tại
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS mountain_images (
            id SERIAL PRIMARY KEY,
            file_name VARCHAR(255) NOT NULL,
            image_path VARCHAR(255) NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS image_features_unified (
            image_id INTEGER PRIMARY KEY REFERENCES mountain_images(id),
            color_vector vector(4096),
            texture_vector vector(5),
            edge_vector vector(36),
            vgg16_vector vector(512),
            enhanced_color_vector vector(361)
        )
        """)

        conn.commit()
        print("Kết nối thành công!")
        return conn
    except Exception as e:
        print(f"Lỗi kết nối: {e}")
        return None


# Khởi tạo kết nối
conn = get_db_connection()
cur = conn.cursor()


# Hàm lưu ảnh vào database
def save_image_to_db(image_path, store_binary=False):
    """
    Lưu ảnh vào cơ sở dữ liệu

    Args:
        image_path: Đường dẫn đến file ảnh
        store_binary: mặc định False, chỉ lưu đường dẫn
    """
    filename = os.path.basename(image_path)

    # Kiểm tra xem ảnh đã tồn tại trong database chưa
    cur.execute("SELECT id FROM mountain_images WHERE file_name = %s", (filename,))
    result = cur.fetchone()

    if result:
        # Ảnh đã tồn tại, trả về id
        return result[0]
    else:
        # Ảnh chưa tồn tại, thêm mới
        cur.execute(
            "INSERT INTO mountain_images (file_name, image_path) VALUES (%s, %s) RETURNING id",
            (filename, image_path)
        )
        image_id = cur.fetchone()[0]
        conn.commit()
        return image_id


# Hàm lưu các đặc trưng vào bảng hợp nhất
def save_features_unified(image_id, color_hist, enhanced_color, texture_feats, edge_feats, vgg_feats):
    """
    Lưu tất cả đặc trưng của một ảnh vào một bản ghi duy nhất trong bảng image_features_unified

    Args:
        image_id: ID của ảnh
        color_hist: Đặc trưng histogram màu
        enhanced_color: Đặc trưng màu cải tiến
        texture_feats: Đặc trưng kết cấu
        edge_feats: Đặc trưng cạnh
        vgg_feats: Đặc trưng VGG16
    """

    # Kiểm tra xem bản ghi đã tồn tại chưa
    cur.execute("SELECT 1 FROM image_features_unified WHERE image_id = %s", (image_id,))
    if cur.fetchone():
        # Cập nhật nếu đã tồn tại
        cur.execute("""
            UPDATE image_features_unified SET
                color_vector = %s::vector,
                enhanced_color_vector = %s::vector,
                texture_vector = %s::vector,
                edge_vector = %s::vector,
                vgg16_vector = %s::vector
            WHERE image_id = %s
        """, (
            color_hist.tolist(), enhanced_color.tolist(), texture_feats.tolist(),
            edge_feats.tolist(), vgg_feats.tolist(), image_id
        ))
    else:
        # Chèn mới nếu chưa tồn tại
        cur.execute("""
            INSERT INTO image_features_unified (
                image_id, color_vector, enhanced_color_vector, texture_vector, edge_vector, vgg16_vector
            ) VALUES (%s, %s::vector, %s::vector, %s::vector, %s::vector, %s::vector)
        """, (
            image_id,
            color_hist.tolist(), enhanced_color.tolist(),
            texture_feats.tolist(), edge_feats.tolist(), vgg_feats.tolist()
        ))

    conn.commit()


# Hàm trích xuất và cập nhật tất cả đặc trưng của ảnh
def extract_and_save_features(image_id, image_path):
    """
    Trích xuất và lưu/cập nhật tất cả đặc trưng của một ảnh

    Args:
        image_id: ID của ảnh
        image_path: Đường dẫn đến file ảnh
    """
    try:
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"Không thể đọc ảnh {image_path}")
            return False

        # Resize ảnh nếu cần
        image = cv2.resize(image, (224, 224))

        # Trích xuất các đặc trưng
        color_hist = extract_color_histogram(image, bins=16)  # Histogram màu gốc
        enhanced_color = improved_color_features(image)  # Đặc trưng màu cải tiến
        texture_feats = extract_texture_features(image)
        edge_feats = extract_edge_features(image)
        vgg_feats = extract_features_vgg16(image_path)

        # Lưu tất cả đặc trưng trong một bản ghi
        save_features_unified(image_id, color_hist, enhanced_color, texture_feats, edge_feats, vgg_feats)

        return True
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_path}: {e}")
        return False


# Hàm để xử lý toàn bộ thư mục ảnh
def process_image_directory(directory):
    """Xử lý toàn bộ thư mục ảnh và lưu vào database"""
    total_processed = 0
    total_failed = 0

    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory, filename)
            print(f"Đang xử lý: {filename}")

            # Lưu ảnh vào database (chỉ lưu đường dẫn)
            image_id = save_image_to_db(image_path)

            # Trích xuất và lưu đặc trưng
            if extract_and_save_features(image_id, image_path):
                total_processed += 1
                print(f"✅ Đã xử lý thành công: {filename}")
            else:
                total_failed += 1
                print(f"❌ Xử lý thất bại: {filename}")

    print(f"Tổng kết: Thành công {total_processed}, Thất bại {total_failed}")


# Hàm cập nhật đặc trưng cho tất cả ảnh trong database
def update_all_features():
    """Cập nhật tất cả đặc trưng cho các ảnh có trong database"""
    # Lấy tất cả ảnh từ database
    cur.execute("SELECT id, image_path FROM mountain_images")
    images = cur.fetchall()

    total = len(images)
    success = 0
    failed = 0

    for i, (image_id, image_path) in enumerate(images):
        print(f"Đang cập nhật {i + 1}/{total}: {os.path.basename(image_path)}")

        if extract_and_save_features(image_id, image_path):
            success += 1
        else:
            failed += 1

    print(f"Đã hoàn thành cập nhật: {success} thành công, {failed} thất bại")


if __name__ == "__main__":

    # Gọi hàm này trước khi xử lý ảnh
    # initialize_pca_model()

    # Xử lý từ đầu
    process_image_directory("E:\\OnSchool\\Moutain_Search\\images")

    # Cập nhật đặc trưng
    # update_all_features()
