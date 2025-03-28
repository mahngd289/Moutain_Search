import psycopg2
import json
import numpy as np

# Tạo kết nối tới PostgreSQL
def get_connection():
    conn = psycopg2.connect(
        dbname="moutains_db",
        user="admin_user",
        password="123456",
        host="localhost"
    )
    return conn

# Biến toàn cục để lưu trữ kết nối
_connection = None

# Hàm lấy kết nối hiện tại hoặc tạo mới
def get_db_connection():
    global _connection
    if _connection is None or _connection.closed:
        _connection = get_connection()
    return _connection