import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import cv2


class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống tìm kiếm ảnh đồi núi")

        # Tạo giao diện
        self.create_widgets()

        # Ảnh truy vấn hiện tại
        self.query_image_path = None

    def create_widgets(self):
        # Frame chứa ảnh truy vấn
        self.query_frame = tk.LabelFrame(self.root, text="Ảnh truy vấn")
        self.query_frame.grid(row=0, column=0, padx=10, pady=10)

        self.query_label = tk.Label(self.query_frame)
        self.query_label.pack(padx=10, pady=10)

        # Frame chứa các nút chức năng
        self.buttons_frame = tk.Frame(self.root)
        self.buttons_frame.grid(row=1, column=0, padx=10, pady=10)

        self.select_button = tk.Button(self.buttons_frame, text="Chọn ảnh", command=self.select_image)
        self.select_button.grid(row=0, column=0, padx=5)

        self.search_button = tk.Button(self.buttons_frame, text="Tìm kiếm", command=self.search_images)
        self.search_button.grid(row=0, column=1, padx=5)

        # Frame chứa kết quả
        self.results_frame = tk.LabelFrame(self.root, text="Kết quả tìm kiếm")
        self.results_frame.grid(row=2, column=0, padx=10, pady=10)

        # Tạo 3 label cho 3 ảnh kết quả
        self.result_labels = []
        self.similarity_labels = []
        for i in range(3):
            frame = tk.Frame(self.results_frame)
            frame.grid(row=0, column=i, padx=5, pady=5)

            label = tk.Label(frame)
            label.pack()
            self.result_labels.append(label)

            sim_label = tk.Label(frame, text="")
            sim_label.pack()
            self.similarity_labels.append(sim_label)

    def select_image(self):
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
        if not self.query_image_path:
            return

        # Thực hiện tìm kiếm
        results = search_similar_images(self.query_image_path, feature_type="vgg16", top_k=3)

        # Hiển thị kết quả
        for i, (image_path, similarity) in enumerate(results):
            if i < 3:  # Chỉ hiển thị 3 kết quả
                img = Image.open(image_path)
                img = img.resize((150, 150))  # Resize để hiển thị
                img_tk = ImageTk.PhotoImage(img)

                self.result_labels[i].configure(image=img_tk)
                self.result_labels[i].image = img_tk  # Giữ tham chiếu

                # Hiển thị độ tương tự
                if similarity >= 0:  # Cosine similarity
                    sim_text = f"Similarity: {similarity:.4f}"
                else:  # Euclidean distance
                    sim_text = f"Distance: {-similarity:.4f}"

                self.similarity_labels[i].configure(text=sim_text)


# Chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSearchApp(root)
    root.mainloop()