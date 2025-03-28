from preprocess.store_data import process_image_directory
import os


def main():
    # Đường dẫn tới thư mục chứa ảnh núi
    image_directory = os.path.join(os.path.dirname(__file__), "images")

    # Xử lý và lưu tất cả ảnh vào database
    process_image_directory(image_directory)

    print(f"Đã hoàn thành việc lưu ảnh từ thư mục {image_directory} vào database")


if __name__ == "__main__":
    main()