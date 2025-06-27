import psycopg2
import os
import cv2

from preprocess.deep_learning_extraction import extract_features_vgg16
from preprocess.texture_extraction import extract_texture_features
from preprocess.texture_extraction import extract_glcm_features_from_manual
from preprocess.color_extraction import improved_color_features
from preprocess.eoh_extraction import my_canny_full_with_eoh


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

        # Đảm bảo autocommit là False để kiểm soát transaction
        conn.autocommit = False

        # Tạo bảng nếu chưa tồn tại
        cur = conn.cursor()

        # Tạo bảng mountain_images
        cur.execute("""
        CREATE TABLE IF NOT EXISTS mountain_images (
            id SERIAL PRIMARY KEY,
            file_name VARCHAR(255) NOT NULL,
            image_path VARCHAR(255) NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Tạo extension vector và bảng lưu đặc trưng
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Reset bảng để lưu trữ lại từ đầu - chỉ lưu 4 đặc trưng
        cur.execute("""
        DROP TABLE IF EXISTS image_features_unified
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS image_features_unified (
            image_id INTEGER PRIMARY KEY REFERENCES mountain_images(id),
            color_vector vector(354),
            texture_vector vector(5),
            edge_vector vector(8),
            vgg16_vector vector(512)
        )
        """)

        conn.commit()
        print("Kết nối và tạo bảng thành công!")
        return conn
    except Exception as e:
        print(f"Lỗi kết nối hoặc tạo bảng: {e}")
        if 'conn' in locals() and conn is not None:
            conn.rollback()
        return None


def save_image_to_db(conn, image_path):
    """Lưu ảnh vào cơ sở dữ liệu và trả về ID"""
    cur = conn.cursor()
    filename = os.path.basename(image_path)

    try:
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
    except Exception as e:
        conn.rollback()
        print(f"Lỗi khi lưu ảnh vào DB: {e}")
        return None


def save_features_unified(conn, image_id, color_feats, texture_feats, edge_feats, vgg_feats):
    """Lưu 4 đặc trưng của một ảnh vào bảng image_features_unified"""
    if image_id is None:
        return False

    cur = conn.cursor()
    try:
        # Kiểm tra xem bản ghi đã tồn tại chưa
        cur.execute("SELECT 1 FROM image_features_unified WHERE image_id = %s", (image_id,))
        if cur.fetchone():
            # Cập nhật nếu đã tồn tại
            cur.execute("""
                UPDATE image_features_unified SET
                    color_vector = %s::vector,
                    texture_vector = %s::vector,
                    edge_vector = %s::vector,
                    vgg16_vector = %s::vector
                WHERE image_id = %s
            """, (
                color_feats.tolist(), texture_feats.tolist(),
                edge_feats.tolist(), vgg_feats.tolist(), image_id
            ))
        else:
            # Chèn mới nếu chưa tồn tại
            cur.execute("""
                INSERT INTO image_features_unified (
                    image_id, color_vector, texture_vector, edge_vector, vgg16_vector
                ) VALUES (%s, %s::vector, %s::vector, %s::vector, %s::vector)
            """, (
                image_id,
                color_feats.tolist(), texture_feats.tolist(),
                edge_feats.tolist(), vgg_feats.tolist()
            ))

        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Lỗi khi lưu đặc trưng: {e}")
        return False


def extract_and_save_features(conn, image_id, image_path):
    """Trích xuất và lưu 4 đặc trưng của một ảnh"""
    try:
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"Không thể đọc ảnh {image_path}")
            return False

        # Resize ảnh
        image = cv2.resize(image, (224, 224))

        # Chuyển sang ảnh xám cho trích xuất biên
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Trích xuất các đặc trưng
        color_feats = improved_color_features(image)  # Đặc trưng màu
        texture_feats = extract_glcm_features_from_manual(image)  # Đặc trưng kết cấu
        _, edge_feats = my_canny_full_with_eoh(gray, min_val=100, max_val=150, eoh_bins=8)  # Đặc trưng cạnh (EOH)
        vgg_feats = extract_features_vgg16(image_path)  # Đặc trưng deep learning

        # Lưu 4 đặc trưng
        return save_features_unified(conn, image_id, color_feats, texture_feats, edge_feats, vgg_feats)
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_path}: {e}")
        return False


def process_image_directory(directory):
    """Xử lý toàn bộ thư mục ảnh và lưu vào database"""
    conn = get_db_connection()
    if conn is None:
        print("Không thể kết nối tới cơ sở dữ liệu!")
        return

    total_processed = 0
    total_failed = 0

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory, filename)
            print(f"Đang xử lý: {filename}")

            # Lưu ảnh vào database
            image_id = save_image_to_db(conn, image_path)

            if image_id is not None:
                # Trích xuất và lưu đặc trưng
                if extract_and_save_features(conn, image_id, image_path):
                    total_processed += 1
                    print(f"✅ Đã xử lý thành công: {filename}")
                else:
                    total_failed += 1
                    print(f"❌ Xử lý thất bại: {filename}")
            else:
                total_failed += 1
                print(f"❌ Không thể lưu thông tin ảnh vào database: {filename}")

    conn.close()
    print(f"Tổng kết: Thành công {total_processed}, Thất bại {total_failed}")


if __name__ == "__main__":
    # Xử lý từ đầu tất cả ảnh trong thư mục
    process_image_directory("E:\\OnSchool\\Moutain_Search\\images")