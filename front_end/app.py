import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import sys
import os

# Thêm đường dẫn gốc vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import hàm tìm kiếm từ backend
from back_end.calculate_similarity import search_similar_images_weighted, search_similar_images_vgg16


class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống tìm kiếm ảnh đồi núi")
        self.root.geometry("800x700")  # Kích thước cửa sổ rộng hơn

        # Ảnh truy vấn hiện tại
        self.query_image_path = None

        # Tạo giao diện
        self.create_widgets()

    def create_widgets(self):
        # Frame chứa ảnh truy vấn
        self.query_frame = tk.LabelFrame(self.root, text="Ảnh truy vấn")
        self.query_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.query_label = tk.Label(self.query_frame)
        self.query_label.pack(padx=10, pady=10)

        # Frame chọn ảnh
        self.select_frame = tk.Frame(self.root)
        self.select_frame.pack(fill="x", padx=10, pady=5)

        self.select_button = tk.Button(self.select_frame, text="Chọn ảnh", command=self.select_image, width=15)
        self.select_button.pack(side=tk.LEFT, padx=10)

        # Frame chứa lựa chọn kiểu tìm kiếm
        self.options_frame = tk.LabelFrame(self.root, text="Lựa chọn kiểu tìm kiếm")
        self.options_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Biến để lưu lựa chọn
        self.search_type = tk.IntVar(value=1)  # Mặc định là lựa chọn 1

        # Tạo radio buttons
        self.radio1 = tk.Radiobutton(
            self.options_frame,
            text="Lựa chọn 1: Tìm kiếm theo trọng số đặc trưng",
            variable=self.search_type,
            value=1,
            command=self.toggle_search_options
        )
        self.radio1.grid(row=0, column=0, sticky="w", padx=10, pady=5)

        self.radio2 = tk.Radiobutton(
            self.options_frame,
            text="Lựa chọn 2: Tìm kiếm theo VGG16",
            variable=self.search_type,
            value=2,
            command=self.toggle_search_options
        )
        self.radio2.grid(row=0, column=1, sticky="w", padx=10, pady=5)

        # Frame chứa các tùy chọn cho Lựa chọn 1
        self.weights_frame = tk.LabelFrame(self.options_frame, text="Trọng số cho các đặc trưng")
        self.weights_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)

        # Tạo các slider trọng số
        self.weight_vars = {
            "color": tk.DoubleVar(value=1.0),
            "texture": tk.DoubleVar(value=1.0),
            "edge": tk.DoubleVar(value=1.0)
        }

        # Slider cho trọng số màu sắc
        tk.Label(self.weights_frame, text="Màu sắc:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        tk.Scale(
            self.weights_frame,
            from_=0.0, to=2.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.weight_vars["color"],
            length=200
        ).grid(row=0, column=1, sticky="ew", padx=10, pady=5)

        # Slider cho trọng số kết cấu
        tk.Label(self.weights_frame, text="Kết cấu:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        tk.Scale(
            self.weights_frame,
            from_=0.0, to=2.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.weight_vars["texture"],
            length=200
        ).grid(row=1, column=1, sticky="ew", padx=10, pady=5)

        # Slider cho trọng số cạnh
        tk.Label(self.weights_frame, text="Cạnh:").grid(row=0, column=2, sticky="w", padx=10, pady=5)
        tk.Scale(
            self.weights_frame,
            from_=0.0, to=2.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.weight_vars["edge"],
            length=200
        ).grid(row=0, column=3, sticky="ew", padx=10, pady=5)

        # Slider cho trọng số VGG16
        # tk.Label(self.weights_frame, text="VGG16:").grid(row=1, column=2, sticky="w", padx=10, pady=5)
        # tk.Scale(
        #     self.weights_frame,
        #     from_=0.0, to=2.0,
        #     resolution=0.1,
        #     orient=tk.HORIZONTAL,
        #     variable=self.weight_vars["vgg16"],
        #     length=200
        # ).grid(row=1, column=3, sticky="ew", padx=10, pady=5)

        # Frame cho lựa chọn độ đo
        self.metric_frame = tk.Frame(self.options_frame)
        self.metric_frame.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        tk.Label(self.metric_frame, text="Độ đo:").pack(side=tk.LEFT, padx=5)

        # Lựa chọn độ đo
        self.metric_var = tk.StringVar(value="cosine")
        metric_combo = ttk.Combobox(
            self.metric_frame,
            textvariable=self.metric_var,
            values=["cosine", "euclidean"],
            state="readonly",
            width=15
        )
        metric_combo.pack(side=tk.LEFT, padx=5)

        # Frame chứa các tùy chọn cho Lựa chọn 2
        self.vgg16_frame = tk.LabelFrame(self.options_frame, text="Tìm kiếm với VGG16")
        self.vgg16_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)

        # Lựa chọn độ đo cho VGG16
        tk.Label(self.vgg16_frame, text="Độ đo VGG16:").pack(side=tk.LEFT, padx=5)

        self.vgg16_metric_var = tk.StringVar(value="cosine")
        vgg16_metric_combo = ttk.Combobox(
            self.vgg16_frame,
            textvariable=self.vgg16_metric_var,
            values=["cosine", "euclidean"],
            state="readonly",
            width=15
        )
        vgg16_metric_combo.pack(side=tk.LEFT, padx=5)

        # Ẩn frame VGG16 ban đầu (vì mặc định chọn lựa chọn 1)
        self.vgg16_frame.grid_remove()

        # Nút tìm kiếm
        self.search_button = tk.Button(self.root, text="Tìm kiếm", command=self.search_images, width=15, height=2)
        self.search_button.pack(pady=10)

        # Frame chứa kết quả
        self.results_frame = tk.LabelFrame(self.root, text="Kết quả tìm kiếm")
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Tạo 3 label cho 3 ảnh kết quả
        self.result_frames = []
        self.result_labels = []
        self.similarity_labels = []

        for i in range(3):
            frame = tk.Frame(self.results_frame)
            frame.pack(side=tk.LEFT, padx=10, pady=10, expand=True)
            self.result_frames.append(frame)

            label = tk.Label(frame)
            label.pack(padx=5, pady=5)
            self.result_labels.append(label)

            sim_label = tk.Label(frame, text="")
            sim_label.pack(padx=5)
            self.similarity_labels.append(sim_label)

    def toggle_search_options(self):
        """Hiển thị/ẩn các tùy chọn dựa trên lựa chọn tìm kiếm"""
        if self.search_type.get() == 1:  # Lựa chọn 1
            self.weights_frame.grid()
            self.metric_frame.grid()
            self.vgg16_frame.grid_remove()

        else:  # Lựa chọn 2
            self.weights_frame.grid_remove()
            self.metric_frame.grid_remove()
            self.vgg16_frame.grid()

    def select_image(self):
        """Chọn ảnh truy vấn"""
        # Mở hộp thoại chọn file
        self.query_image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")])

        if self.query_image_path:
            # Hiển thị ảnh đã chọn
            img = Image.open(self.query_image_path)
            img = img.resize((200, 200))  # Resize để hiển thị
            img_tk = ImageTk.PhotoImage(img)

            self.query_label.configure(image=img_tk)
            self.query_label.image = img_tk  # Giữ tham chiếu

    def search_images(self):
        """Thực hiện tìm kiếm ảnh"""
        if not self.query_image_path:
            tk.messagebox.showinfo("Thông báo", "Vui lòng chọn ảnh truy vấn trước!")
            return

        try:
            # Xác định chế độ tìm kiếm dựa trên radio button
            if self.search_type.get() == 1:  # Lựa chọn 1: Tìm kiếm theo trọng số
                # Lấy trọng số từ các slider
                weights = {
                    "color": self.weight_vars["color"].get(),
                    "texture": self.weight_vars["texture"].get(),
                    "edge": self.weight_vars["edge"].get(),
                }

                # Lấy loại độ đo
                metric = self.metric_var.get()

                # Tìm kiếm ảnh với trọng số
                results = search_similar_images_weighted(
                    self.query_image_path,
                    weights=weights,
                    metric=metric,
                    top_k=3
                )

            else:  # Lựa chọn 2: Tìm kiếm theo VGG16
                # Lấy loại độ đo từ combobox VGG16
                metric = self.vgg16_metric_var.get()

                # Tìm kiếm ảnh chỉ với VGG16
                results = search_similar_images_vgg16(
                    self.query_image_path,
                    metric=metric,
                    top_k=3
                )

            # Hiển thị kết quả
            for i, (image_path, similarity) in enumerate(results):
                if i < 3:  # Chỉ hiển thị 3 kết quả
                    img = Image.open(image_path)
                    img = img.resize((150, 150))  # Resize để hiển thị
                    img_tk = ImageTk.PhotoImage(img)

                    self.result_labels[i].configure(image=img_tk)
                    self.result_labels[i].image = img_tk  # Giữ tham chiếu

                    # Hiển thị độ tương tự
                    if self.metric_var.get() == "cosine" or self.vgg16_metric_var.get() == "cosine":
                        sim_text = f"Độ tương tự: {similarity:.4f}"
                    else:  # euclidean
                        sim_text = f"Khoảng cách: {-similarity:.4f}"

                    self.similarity_labels[i].configure(text=sim_text)

        except Exception as e:
            # Hiển thị thông báo lỗi
            tk.messagebox.showerror("Lỗi", f"Lỗi khi tìm kiếm: {e}")
            print(f"Lỗi chi tiết: {e}")


# Chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSearchApp(root)
    root.mainloop()