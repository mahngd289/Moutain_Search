import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def visualize_dominant_colors(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    # Xác định số lượng màu tối ưu
    k = ce.elbow_optimal_k(image)

    # Chuyển sang không gian màu Lab
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pixels = lab_image.reshape(-1, 3).astype(np.float32)

    # K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Tính số lượng pixel cho mỗi cụm
    counts = np.bincount(labels.flatten(), minlength=k)

    # Sắp xếp các cụm theo số lượng pixel giảm dần
    sorted_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_indices]
    percentages = sorted_counts / np.sum(counts) * 100

    # Chuẩn bị hình ảnh kết quả
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Hiển thị ảnh gốc
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Ảnh gốc')
    ax1.axis('off')

    # Tạo bảng màu
    height = 50
    width = int(image.shape[1] * 0.8)
    color_chart = np.zeros((height * k, width, 3), dtype=np.uint8)

    for i, idx in enumerate(sorted_indices):
        # Chuyển từ Lab sang BGR
        bgr_color = cv2.cvtColor(np.uint8([[centers[idx]]]), cv2.COLOR_LAB2BGR)[0][0]
        # Vẽ màu và phần trăm
        color_chart[i * height:(i + 1) * height, :] = bgr_color
        ax2.text(width + 10, i * height + height // 2, f"{percentages[i]:.1f}%",
                 va='center', ha='left', fontsize=10)

    # Hiển thị bảng màu
    ax2.imshow(cv2.cvtColor(color_chart, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Màu chủ đạo (k={k})')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_seasonal_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    # Trích xuất đặc trưng mùa
    seasonal = ce.extract_seasonal_features(image)

    # Hiển thị ảnh và biểu đồ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Ảnh gốc
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Ảnh gốc')
    ax1.axis('off')

    # Biểu đồ cột phân bố mùa
    seasons = ['Hè', 'Thu', 'Đông', 'Xuân']
    colors = ['#4CAF50', '#FF9800', '#2196F3', '#8BC34A']

    ax2.bar(seasons, seasonal, color=colors)
    ax2.set_ylim(0, max(seasonal) * 1.2)  # Tạo khoảng trống trên cùng
    ax2.set_title('Phân bố đặc trưng mùa')

    # Thêm giá trị phần trăm
    for i, v in enumerate(seasonal):
        ax2.text(i, v + 0.01, f"{v * 100:.1f}%", ha='center')

    plt.tight_layout()
    plt.show()

    return seasonal


def visualize_image_regions(image_path):
    # Đọc và chuẩn bị ảnh
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    height, width = image.shape[:2]

    # Số vùng phân tích
    regions = 5
    region_height = height // regions

    # Chuyển sang không gian màu HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tạo layout 2 hàng x 5 cột: hàng trên cho vùng, hàng dưới cho histogram
    fig, axes = plt.subplots(2, regions, figsize=(15, 6))

    for i in range(regions):
        # Xác định vùng
        y_start = i * region_height
        y_end = (i + 1) * region_height if i < regions - 1 else height
        region = hsv_image[y_start:y_end, :]

        # Tính histogram 2D (Hue x Saturation)
        hist = cv2.calcHist([region], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX)

        # Hiển thị vùng ảnh (hàng 0)
        region_bgr = cv2.cvtColor(region, cv2.COLOR_HSV2BGR)
        axes[0, i].imshow(cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f'Vùng {i + 1}', fontweight='bold')
        axes[0, i].axis('off')

        # Hiển thị histogram 2D (hàng 1)
        im = axes[1, i].imshow(hist, interpolation='nearest', cmap='viridis')
        axes[1, i].set_title(f'Histogram {i + 1}')

        # Điều chỉnh kích thước nhãn
        axes[1, i].tick_params(axis='both', labelsize=8)

        # Thêm colorbar nhỏ gọn cho mỗi histogram
        divider = make_axes_locatable(axes[1, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # Điều chỉnh layout
    plt.tight_layout()
    plt.suptitle('Phân tích phân bố màu sắc theo vùng',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.9, wspace=0.3)
    plt.show()


if __name__ == "__main__":
    image_path = "E:\OnSchool\Moutain_Search\images\gsun_0dfb573bdb7f650160561747a2de2add.jpg"
    # image = cv2.imread(image_path)
    # image = cv2.resize(image, (224, 224))

    visualize_dominant_colors(image_path)
    visualize_seasonal_features(image_path)
    visualize_image_regions(image_path)




