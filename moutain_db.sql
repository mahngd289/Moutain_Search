CREATE TABLE mountain_images (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    image_data BYTEA NOT NULL,  -- Lưu ảnh dưới dạng nhị phân
    image_path VARCHAR(255),    -- Hoặc lưu đường dẫn đến file
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bảng lưu trữ các đặc trưng của ảnh
CREATE TABLE image_features (
    id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES mountain_images(id),
    feature_type VARCHAR(50) NOT NULL,  -- Loại đặc trưng (color, texture, shape...)
    feature_value JSONB,                -- Lưu giá trị đặc trưng dạng JSON
    feature_vector vector(512)          -- Với extension pgvector (vector 512 chiều)
);