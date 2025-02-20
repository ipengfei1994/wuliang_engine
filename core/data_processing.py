import pandas as pd
import numpy as np
from typing import Dict, Any, Union

class DataProcessor:
    def __init__(self):
        self.data = None

    def load_data(self, data: Union[pd.DataFrame, Dict[str, Any]]):
        """加载数据"""
        self.data = data

    def handle_missing_values(self, strategy: str = 'mean'):
        """处理缺失值"""
        if isinstance(self.data, pd.DataFrame):
            if strategy == 'mean':
                self.data = self.data.fillna(self.data.mean())
            elif strategy == 'median':
                self.data = self.data.fillna(self.data.median())
            elif strategy == 'drop':
                self.data = self.data.dropna()

    def normalize_data(self, method: str = 'minmax'):
        """数据标准化"""
        if isinstance(self.data, pd.DataFrame):
            if method == 'minmax':
                self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
            elif method == 'zscore':
                self.data = (self.data - self.data.mean()) / self.data.std()

    def get_processed_data(self):
        """获取处理后的数据"""
        return self.data