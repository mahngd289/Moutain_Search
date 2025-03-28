import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Khởi tạo model chỉ khi hàm được gọi lần đầu tiên (lazy loading)
base_model = None


def extract_features_vgg16(img_path):
    """Trích xuất đặc trưng từ ảnh sử dụng VGG16"""
    global base_model

    # Tải model khi cần thiết
    if base_model is None:
        print("Đang tải mô hình VGG16...")
        base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Trích xuất features từ mô hình
    features = base_model.predict(img_array, verbose=0)
    features_flatten = features.flatten()

    return features_flatten