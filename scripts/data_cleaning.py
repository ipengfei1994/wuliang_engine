# 数据清洗

import pandas as pd

def clean_data(df):
    """清洗数据"""
    # 删除重复行
    df = df.drop_duplicates()
    
    # 填充缺失值，可以根据情况选择填充方法（均值、中位数、前后填充等）
    df = df.fillna(method='ffill')
    
    # 数据类型转换，如果需要可以进行
    # df['column_name'] = df['column_name'].astype('int')
    
    return df

def clean_all_data(data):
    """对所有导入的数据进行清洗"""
    cleaned_data = {}
    
    for file_name, df in data.items():
        if isinstance(df, pd.DataFrame):  # 对于DataFrame类型数据
            cleaned_data[file_name] = clean_data(df)
        else:
            print(f"跳过文件 {file_name}，不支持的格式")
    
    return cleaned_data
