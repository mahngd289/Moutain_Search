import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import preprocess.color_extraction as ce


def visualize_weighted_diversity():
    # Đường dẫn đến thư mục chứa ảnh (thay thế bằng đường dẫn thực tế của bạn)
    image_folder = "E:/OnSchool/Moutain_Search/images"

    # Các định dạng ảnh hỗ trợ
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

    # Lưu trữ kết quả
    diversity_values = []
    image_paths = []

    print(f"Đang đọc ảnh từ thư mục: {image_folder}")

    # Duyệt qua tất cả file trong thư mục
    for file_name in os.listdir(image_folder):
        # Kiểm tra nếu là file ảnh
        if any(file_name.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(image_folder, file_name)

            try:
                # Đọc ảnh
                image = cv2.imread(image_path)
                if image is None:
                    continue

                # Tính weighted_diversity
                diversity = ce.extract_weighted_diversity(image)

                # Lưu kết quả
                diversity_values.append(diversity)
                image_paths.append(image_path)

                print(f"Ảnh: {file_name}, Weighted Diversity: {diversity:.4f}")

            except Exception as e:
                print(f"Lỗi xử lý ảnh {file_name}: {e}")

    # Hiển thị phân phối weighted_diversity
    plt.figure(figsize=(12, 8))

    # Biểu đồ phân phối
    plt.subplot(2, 1, 1)
    plt.hist(diversity_values, bins=30, color='skyblue', edgecolor='black')
    plt.title('Phân phối Weighted Diversity của các ảnh')
    plt.xlabel('Weighted Diversity')
    plt.ylabel('Số lượng ảnh')

    # Hiển thị một số ảnh mẫu với weighted_diversity khác nhau
    n_samples = min(5, len(image_paths))
    if n_samples > 0:
        # Lấy các mẫu phân bố đều trên dải giá trị
        indices = np.linspace(0, len(diversity_values) - 1, n_samples, dtype=int)
        sorted_indices = np.argsort(diversity_values)
        sample_indices = sorted_indices[indices]

        plt.subplot(2, 1, 2)
        for i, idx in enumerate(sample_indices):
            # Đọc và hiển thị ảnh mẫu
            img = cv2.imread(image_paths[idx])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Hiển thị ảnh thu nhỏ với giá trị diversity
            plt.subplot(2, n_samples, n_samples + i + 1)
            plt.imshow(img_rgb)
            plt.title(f"WD: {diversity_values[idx]:.3f}")
            plt.axis('off')

    # Hiển thị thống kê
    print("\nThống kê Weighted Diversity:")
    print(f"Min: {min(diversity_values):.4f}")
    print(f"Max: {max(diversity_values):.4f}")
    print(f"Mean: {np.mean(diversity_values):.4f}")
    print(f"Median: {np.median(diversity_values):.4f}")
    print(f"Std: {np.std(diversity_values):.4f}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "/images/gsun_0a89000e8e72db9b91e1d6ae905195db.jpg"
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    # enhanced_color_features.elbow_optimal_k(image)
    # enhanced_color_features.extract_color_regions(image, 5, True)
    features = ce.improved_color_features(image)
    print(features.shape)
    print(features)


