import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

class FeatureEngineer:
    def __init__(self):
        self.features = None
        self.labels = None

    def set_data(self, features, labels=None):
        """设置特征和标签数据"""
        self.features = features
        self.labels = labels

    def create_polynomial_features(self, degree=2):
        """创建多项式特征"""
        if isinstance(self.features, pd.DataFrame):
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=degree)
            return pd.DataFrame(poly.fit_transform(self.features))

    def select_best_features(self, k=10):
        """选择最佳特征"""
        if self.labels is not None:
            selector = SelectKBest(f_classif, k=k)
            return selector.fit_transform(self.features, self.labels)

    def reduce_dimensions(self, n_components=0.95):
        """降维处理"""
        pca = PCA(n_components=n_components)
        return pca.fit_transform(self.features)