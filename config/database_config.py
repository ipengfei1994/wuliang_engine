# 这个文件专门用于数据库连接的配置。

import sqlite3

def create_connection():
    # 连接到 SQLite 数据库（如果数据库不存在，它会被创建）
    conn = sqlite3.connect('wuliang_engine.db')  # 这里的文件名是你的数据库文件名
    return conn
