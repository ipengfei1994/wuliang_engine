import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime
import json

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.training_history = []
        self._setup_logging()

    def _setup_logging(self) -> None:
        """配置日志系统"""
        logging.basicConfig(
            filename='model_training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def prepare_data(self, data: pd.DataFrame,
                    target_col: str,
                    feature_cols: List[str],
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple:
        """准备训练数据"""
        X = data[feature_cols]
        y = data[target_col]
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test

    def train_model(self, model,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   model_params: Optional[Dict] = None) -> Dict:
        """训练模型"""
        try:
            # 设置模型参数
            if model_params:
                model.set_params(**model_params)
            
            # 训练模型
            model.fit(X_train, y_train)
            self.model = model
            
            # 评估模型
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            metrics = {
                'train': self._calculate_metrics(y_train, train_pred),
                'test': self._calculate_metrics(y_test, test_pred)
            }
            
            # 记录训练历史
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'model_params': model_params,
                'metrics': metrics
            })
            
            logging.info(f"Model training completed: {metrics}")
            return metrics
            
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise

    def _calculate_metrics(self, y_true: np.ndarray,
                         y_pred: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }

    def cross_validate(self, model,
                      X: np.ndarray,
                      y: np.ndarray,
                      cv: int = 5,
                      scoring: str = 'accuracy') -> Dict[str, float]:
        """交叉验证"""
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            results = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'all_scores': scores.tolist()
            }
            
            logging.info(f"Cross validation completed: {results}")
            return results
            
        except Exception as e:
            logging.error(f"Error during cross validation: {str(e)}")
            raise

    def save_model(self, model_path: str,
                  scaler_path: Optional[str] = None) -> None:
        """保存模型"""
        if self.model is None:
            raise ValueError("No model to save")
            
        try:
            joblib.dump(self.model, model_path)
            if scaler_path:
                joblib.dump(self.scaler, scaler_path)
                
            logging.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: str,
                  scaler_path: Optional[str] = None) -> None:
        """加载模型"""
        try:
            self.model = joblib.load(model_path)
            if scaler_path:
                self.scaler = joblib.load(scaler_path)
                
            logging.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, X: np.ndarray,
               scale_input: bool = True) -> np.ndarray:
        """使用模型进行预测"""
        if self.model is None:
            raise ValueError("No model loaded")
            
        try:
            if scale_input:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
                
            predictions = self.model.predict(X_scaled)
            return predictions
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise

    def export_training_history(self, filepath: str) -> None:
        """导出训练历史"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, ensure_ascii=False, indent=4)
                
            logging.info(f"Training history exported to {filepath}")
            
        except Exception as e:
            logging.error(f"Error exporting training history: {str(e)}")
            raise