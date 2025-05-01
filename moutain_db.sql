CREATE TABLE mountain_images (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    image_path VARCHAR(255) NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tạo extension pgvector nếu chưa có
CREATE EXTENSION IF NOT EXISTS vector;

-- Bảng đặc trưng hợp nhất - chỉ lưu vector
CREATE TABLE image_features_unified (
    image_id INTEGER PRIMARY KEY REFERENCES mountain_images(id),
    color_vector vector(4096),  -- 16^3 bins từ histogram màu
    texture_vector vector(5),   -- 5 đặc trưng Haralick
    edge_vector vector(36),     -- 36 bins góc cạnh
    vgg16_vector vector(512),   -- 512 chiều từ VGG16
    enhanced_color_vector vector(361)
);