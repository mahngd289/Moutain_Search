import tensorflow as tf
from tensorflow.keras.applications import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Tải mô hình VGG16 đã được training trước trên ImageNet
# Bỏ lớp fully connected cuối để lấy feature vector
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')


def extract_features_vgg16(img_path):
    """Trích xuất đặc trưng từ ảnh sử dụng VGG16"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Trích xuất features từ mô hình
    features = base_model.predict(img_array)
    features_flatten = features.flatten()

    return features_flatten