from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self, model=None):
        self.model = model
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """分割训练集和测试集"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def preprocess_data(self, scale=True):
        """数据预处理"""
        if scale and self.X_train is not None:
            self.X_train = self.scaler.fit_transform(self.X_train)
            if self.X_test is not None:
                self.X_test = self.scaler.transform(self.X_test)

    def train(self, scale=True):
        """训练模型"""
        if self.model and self.X_train is not None:
            if scale:
                self.preprocess_data()
            self.model.fit(self.X_train, self.y_train)

    def predict(self, X):
        """预测新数据"""
        if self.model:
            if hasattr(self, 'scaler') and self.scaler is not None:
                X = self.scaler.transform(X)
            return self.model.predict(X)
        return None

    def evaluate(self, plot_confusion=True):
        """评估模型"""
        if self.model and self.X_test is not None:
            y_pred = self.model.predict(self.X_test)
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, average='weighted'),
                'recall': recall_score(self.y_test, y_pred, average='weighted'),
                'f1': f1_score(self.y_test, y_pred, average='weighted')
            }
            
            if plot_confusion:
                self.plot_confusion_matrix(y_pred)
                
            return metrics

    def plot_confusion_matrix(self, y_pred):
        """绘制混淆矩阵"""
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def grid_search(self, param_grid, cv=5):
        """网格搜索最优参数"""
        if self.model and self.X_train is not None:
            grid_search = GridSearchCV(
                self.model, 
                param_grid, 
                cv=cv, 
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }

    def plot_learning_curve(self, cv=5):
        """绘制学习曲线"""
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X_train, self.y_train,
            cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    def save_model(self, filepath):
        """保存模型"""
        import joblib
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """加载模型"""
        import joblib
        self.model = joblib.load(filepath)

    def cross_validate(self, cv=5):
        """交叉验证"""
        if self.model and self.X_train is not None:
            scores = cross_val_score(self.model, self.X_train, self.y_train, cv=cv)
            return {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            }