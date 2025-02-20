# 该文件主要存放一些通用的配置项，如日志、默认路径等。

# 项目常规配置
PROJECT_NAME = "无量引擎"
LOG_LEVEL = "INFO"  # 日志级别，可选 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

# 默认文件路径配置
RAW_DATA_PATH = "../data/raw/"  # 原始数据存放目录
PROCESSED_DATA_PATH = "../data/processed/"  # 处理后的数据存放目录

# 数据库配置
DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "root"
DB_PASSWORD = "password"  # 根据实际情况修改
DB_NAME = "wuliang_engine"  # 数据库名称
