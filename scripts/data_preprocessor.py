import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re
import jieba
from datetime import datetime

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.word_dict = {}
        self.stopwords = set()

    def load_stopwords(self, filepath: str) -> None:
        """加载停用词"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.stopwords = set([line.strip() for line in f])

    def clean_text(self, text: str) -> str:
        """文本清洗"""
        # 移除特殊字符
        text = re.sub(r'[^\w\s]', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_text(self, text: str) -> List[str]:
        """分词"""
        words = jieba.cut(self.clean_text(text))
        return [w for w in words if w not in self.stopwords]

    def normalize_numeric(self, data: pd.DataFrame,
                        columns: List[str],
                        method: str = 'standard') -> pd.DataFrame:
        """数值归一化"""
        result = data.copy()
        for col in columns:
            if col not in self.scalers:
                self.scalers[col] = (
                    StandardScaler() if method == 'standard' else MinMaxScaler()
                )
                result[col] = self.scalers[col].fit_transform(
                    result[col].values.reshape(-1, 1)
                )
            else:
                result[col] = self.scalers[col].transform(
                    result[col].values.reshape(-1, 1)
                )
        return result

    def handle_missing_values(self, data: pd.DataFrame,
                            numeric_strategy: str = 'mean',
                            categorical_strategy: str = 'mode') -> pd.DataFrame:
        """处理缺失值"""
        result = data.copy()
        
        # 处理数值型列
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if numeric_strategy == 'mean':
                result[col].fillna(result[col].mean(), inplace=True)
            elif numeric_strategy == 'median':
                result[col].fillna(result[col].median(), inplace=True)
            elif numeric_strategy == 'zero':
                result[col].fillna(0, inplace=True)
        
        # 处理分类型列
        categorical_cols = result.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if categorical_strategy == 'mode':
                result[col].fillna(result[col].mode()[0], inplace=True)
            elif categorical_strategy == 'unknown':
                result[col].fillna('unknown', inplace=True)
        
        return result

    def encode_categorical(self, data: pd.DataFrame,
                         columns: List[str],
                         method: str = 'onehot') -> pd.DataFrame:
        """编码分类变量"""
        result = data.copy()
        
        if method == 'onehot':
            return pd.get_dummies(result, columns=columns)
        elif method == 'label':
            for col in columns:
                result[col] = pd.Categorical(result[col]).codes
        
        return result

    def process_datetime(self, data: pd.DataFrame,
                        datetime_columns: List[str]) -> pd.DataFrame:
        """处理时间特征"""
        result = data.copy()
        
        for col in datetime_columns:
            result[col] = pd.to_datetime(result[col])
            result[f'{col}_year'] = result[col].dt.year
            result[f'{col}_month'] = result[col].dt.month
            result[f'{col}_day'] = result[col].dt.day
            result[f'{col}_hour'] = result[col].dt.hour
            result[f'{col}_weekday'] = result[col].dt.weekday
        
        return result

    def remove_outliers(self, data: pd.DataFrame,
                       columns: List[str],
                       method: str = 'zscore',
                       threshold: float = 3.0) -> pd.DataFrame:
        """移除异常值"""
        result = data.copy()
        
        for col in columns:
            if method == 'zscore':
                z_scores = np.abs((result[col] - result[col].mean()) / result[col].std())
                result = result[z_scores < threshold]
            elif method == 'iqr':
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                result = result[
                    (result[col] >= Q1 - 1.5 * IQR) & 
                    (result[col] <= Q3 + 1.5 * IQR)
                ]
        
        return result

    def create_features(self, data: pd.DataFrame,
                       feature_config: Dict) -> pd.DataFrame:
        """特征工程"""
        result = data.copy()
        
        for feature_name, config in feature_config.items():
            if config['type'] == 'ratio':
                result[feature_name] = (
                    result[config['numerator']] / result[config['denominator']]
                )
            elif config['type'] == 'difference':
                result[feature_name] = (
                    result[config['minuend']] - result[config['subtrahend']]
                )
            elif config['type'] == 'aggregate':
                result[feature_name] = result[config['columns']].agg(config['function'])
        
        return result

    def save_preprocessor(self, filepath: str) -> None:
        """保存预处理器"""
        import joblib
        joblib.dump({
            'scalers': self.scalers,
            'word_dict': self.word_dict,
            'stopwords': self.stopwords
        }, filepath)

    def load_preprocessor(self, filepath: str) -> None:
        """加载预处理器"""
        import joblib
        data = joblib.load(filepath)
        self.scalers = data['scalers']
        self.word_dict = data['word_dict']
        self.stopwords = data['stopwords']