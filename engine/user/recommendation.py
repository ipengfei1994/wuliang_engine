import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from typing import Dict, List, Tuple, Optional

class RecommendationEngine:
    def __init__(self):
        self.user_item_matrix = None
        self.item_features = None
        self.user_features = None
        self.model = None

    def create_user_item_matrix(self, interactions_df: pd.DataFrame,
                              user_col: str = 'user_id',
                              item_col: str = 'item_id',
                              rating_col: str = 'rating') -> pd.DataFrame:
        """创建用户-物品交互矩阵"""
        self.user_item_matrix = interactions_df.pivot(
            index=user_col,
            columns=item_col,
            values=rating_col
        ).fillna(0)
        return self.user_item_matrix

    def train_collaborative_filtering(self, n_factors: int = 50) -> None:
        """训练协同过滤模型"""
        # 矩阵分解
        matrix = self.user_item_matrix.values
        user_means = np.mean(matrix, axis=1)
        matrix_centered = matrix - user_means.reshape(-1, 1)
        
        U, sigma, Vt = svds(matrix_centered, k=n_factors)
        sigma = np.diag(sigma)
        
        self.user_features = U
        self.item_features = Vt.T
        self.model = {
            'U': U,
            'sigma': sigma,
            'Vt': Vt,
            'user_means': user_means
        }

    def get_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """获取推荐结果"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train_collaborative_filtering()")
            
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        predicted_ratings = (
            self.model['U'][user_idx].dot(self.model['sigma']).dot(self.model['Vt']) +
            self.model['user_means'][user_idx]
        )
        
        # 获取用户未交互的物品
        user_items = self.user_item_matrix.iloc[user_idx].values
        candidate_items = np.where(user_items == 0)[0]
        
        # 为未交互物品排序
        candidate_ratings = predicted_ratings[candidate_items]
        top_items_idx = candidate_items[np.argsort(-candidate_ratings)[:n_recommendations]]
        top_ratings = predicted_ratings[top_items_idx]
        
        # 返回推荐结果
        item_ids = self.user_item_matrix.columns[top_items_idx]
        return list(zip(item_ids, top_ratings))

    def calculate_item_similarity(self, item_id: int, n_similar: int = 5) -> List[Tuple[int, float]]:
        """计算物品相似度"""
        if self.item_features is None:
            raise ValueError("模型未训练，请先调用 train_collaborative_filtering()")
            
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        item_similarities = cosine_similarity(
            self.item_features[item_idx].reshape(1, -1),
            self.item_features
        )[0]
        
        # 获取最相似的物品
        similar_items_idx = np.argsort(-item_similarities)[1:n_similar+1]
        similar_items = self.user_item_matrix.columns[similar_items_idx]
        similar_scores = item_similarities[similar_items_idx]
        
        return list(zip(similar_items, similar_scores))

    def evaluate_recommendations(self, test_interactions: pd.DataFrame,
                              user_col: str = 'user_id',
                              item_col: str = 'item_id',
                              rating_col: str = 'rating') -> Dict[str, float]:
        """评估推荐系统性能"""
        predictions = []
        actuals = []
        
        for _, row in test_interactions.iterrows():
            user_id = row[user_col]
            item_id = row[item_col]
            
            if user_id in self.user_item_matrix.index:
                user_idx = self.user_item_matrix.index.get_loc(user_id)
                item_idx = self.user_item_matrix.columns.get_loc(item_id)
                
                predicted_rating = (
                    self.model['U'][user_idx].dot(self.model['sigma']).dot(self.model['Vt'])[:, item_idx] +
                    self.model['user_means'][user_idx]
                )
                
                predictions.append(predicted_rating)
                actuals.append(row[rating_col])
        
        # 计算评估指标
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        
        return {
            'rmse': rmse,
            'mae': mae
        }

    def save_model(self, filepath: str) -> None:
        """保存模型"""
        import joblib
        joblib.dump({
            'model': self.model,
            'user_item_matrix': self.user_item_matrix,
            'user_features': self.user_features,
            'item_features': self.item_features
        }, filepath)

    def load_model(self, filepath: str) -> None:
        """加载模型"""
        import joblib
        data = joblib.load(filepath)
        self.model = data['model']
        self.user_item_matrix = data['user_item_matrix']
        self.user_features = data['user_features']
        self.item_features = data['item_features']