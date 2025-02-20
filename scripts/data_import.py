# 数据导入
import pandas as pd
import os
import json

def import_csv_data(file_path):
    """导入CSV文件"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return None

def import_json_data(file_path):
    """导入JSON文件"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        return None

def load_data_from_folder(folder_path):
    """根据文件类型导入数据"""
    data = {}
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if file_name.endswith('.csv'):
            print(f"导入CSV文件: {file_name}")
            data[file_name] = import_csv_data(file_path)
        elif file_name.endswith('.json'):
            print(f"导入JSON文件: {file_name}")
            data[file_name] = import_json_data(file_path)
    
    return data
