import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
import json

class UserBehaviorAnalyzer:
    def __init__(self):
        self.data = None
        self.user_profiles = None
        self.behavior_clusters = None
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)

    def load_data(self, data_source: Union[str, pd.DataFrame], 
              user_col: str = 'user_id',
              timestamp_col: str = 'timestamp',
              action_col: str = 'action') -> None:
        """加载用户行为数据
        Args:
            data_source: 数据源，可以是DataFrame或文件路径（支持.csv, .xlsx）
            user_col: 用户ID列名
            timestamp_col: 时间戳列名
            action_col: 行为列名
        """
        try:
            # 加载数据
            if isinstance(data_source, str):
                if data_source.endswith('.csv'):
                    self.data = pd.read_csv(data_source)
                elif data_source.endswith(('.xlsx', '.xls')):
                    self.data = pd.read_excel(data_source)
                else:
                    raise ValueError("不支持的文件格式，请使用CSV或Excel文件")
            elif isinstance(data_source, pd.DataFrame):
                self.data = data_source.copy()
            else:
                raise ValueError("数据源必须是DataFrame或文件路径")
    
            # 验证必要的列是否存在
            required_cols = [user_col, timestamp_col, action_col]
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"缺少必要的列: {', '.join(missing_cols)}")
    
            # 数据预处理
            self.data['timestamp'] = pd.to_datetime(self.data[timestamp_col])
            self.user_col = user_col
            self.timestamp_col = timestamp_col
            self.action_col = action_col
    
            self.logger.info(f"成功加载数据，共 {len(self.data)} 行")
    
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise
    def calculate_user_metrics(self) -> pd.DataFrame:
        """计算用户行为指标"""
        if self.data is None:
            raise ValueError("请先加载数据")

        metrics = {}
        for user in self.data[self.user_col].unique():
            user_data = self.data[self.data[self.user_col] == user]
            
            metrics[user] = {
                'total_actions': len(user_data),
                'unique_actions': user_data[self.action_col].nunique(),
                'first_action': user_data[self.timestamp_col].min(),
                'last_action': user_data[self.timestamp_col].max(),
                'active_days': user_data[self.timestamp_col].dt.date.nunique(),
                'avg_daily_actions': len(user_data) / user_data[self.timestamp_col].dt.date.nunique()
            }
        
        return pd.DataFrame.from_dict(metrics, orient='index')

    def segment_users(self, n_clusters: int = 3) -> Dict[str, List[str]]:
        """用户分群"""
        metrics_df = self.calculate_user_metrics()
        
        # 准备聚类特征
        features = metrics_df[['total_actions', 'unique_actions', 'active_days', 'avg_daily_actions']]
        scaled_features = self.scaler.fit_transform(features)
        
        # 执行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        metrics_df['cluster'] = kmeans.fit_predict(scaled_features)
        
        self.behavior_clusters = metrics_df['cluster']
        
        # 返回分群结果
        clusters = {}
        for i in range(n_clusters):
            clusters[f'cluster_{i}'] = metrics_df[metrics_df['cluster'] == i].index.tolist()
        
        return clusters

    def analyze_user_journey(self, user_id: str) -> Dict[str, Any]:
        """分析用户旅程"""
        try:
            if user_id not in self.data[self.user_col].unique():
                self.logger.error(f"用户 {user_id} 不存在")
                return {}
                
            user_data = self.data[self.data[self.user_col] == user_id].sort_values(self.timestamp_col)
            
            journey_data = {
                'first_action_time': user_data[self.timestamp_col].iloc[0],
                'last_action_time': user_data[self.timestamp_col].iloc[-1],
                'total_duration': (user_data[self.timestamp_col].iloc[-1] - 
                             user_data[self.timestamp_col].iloc[0]).total_seconds() / 3600,
                'action_sequence': user_data[self.action_col].tolist(),
                'action_frequencies': user_data[self.action_col].value_counts().to_dict()
            }
            
            self.logger.info(f"成功分析用户 {user_id} 的旅程数据")
            return journey_data
            
        except Exception as e:
            self.logger.error(str(e))
            return {}

    def get_user_retention(self, time_window: str = 'D') -> pd.DataFrame:
        """计算用户留存率"""
        user_first_action = self.data.groupby(self.user_col)[self.timestamp_col].min()
        
        retention_data = []
        for day in pd.date_range(start=user_first_action.min(), end=user_first_action.max()):
            # 获取该日期首次活跃的用户
            new_users = user_first_action[user_first_action.dt.date == day.date()].index
            
            if len(new_users) > 0:
                # 计算这些用户在后续日期的留存情况
                for i in range(7):  # 计算7天留存
                    active_users = self.data[
                        (self.data[self.user_col].isin(new_users)) &
                        (self.data[self.timestamp_col].dt.date == (day + timedelta(days=i)).date())
                    ][self.user_col].nunique()
                    
                    retention_data.append({
                        'cohort_date': day.date(),
                        'day': i,
                        'users': len(new_users),
                        'retained': active_users,
                        'retention_rate': active_users / len(new_users)
                    })
        
        return pd.DataFrame(retention_data)

    def plot_user_activity(self, time_unit: str = 'D') -> None:
        """可视化用户活动"""
        activity = self.data.set_index(self.timestamp_col).resample(time_unit).size()
        
        plt.figure(figsize=(15, 8))  # 增加图表大小
        activity.plot(linewidth=2)  # 增加线条宽度
        plt.title('User Activity Over Time', fontsize=14, pad=20)  # 调整标题大小和位置
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Number of Actions', fontsize=12)
        plt.xticks(rotation=45)  # 旋转x轴标签
        plt.grid(True, alpha=0.3)  # 调整网格透明度
        plt.tight_layout()  # 自动调整布局
        plt.show()

    def plot_retention_heatmap(self) -> None:
        """绘制留存热力图"""
        retention_data = self.get_user_retention()
        retention_matrix = retention_data.pivot(
            index='cohort_date',
            columns='day',
            values='retention_rate'
        )
        
        plt.figure(figsize=(12, 10))  # 增加图表大小
        sns.heatmap(retention_matrix, 
                    annot=True, 
                    fmt='.0%', 
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Retention Rate'},  # 添加颜色条标签
                    annot_kws={'size': 8})  # 调整注释字体大小
        
        plt.title('User Retention Heatmap', fontsize=14, pad=20)  # 调整标题大小和位置
        plt.xlabel('Days Since First Action', fontsize=12)
        plt.ylabel('Cohort Date', fontsize=12)
        plt.tight_layout()  # 自动调整布局
        plt.show()

    def export_analysis(self, filepath: str) -> None:
        """导出分析结果"""
        def convert_timestamps(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # 准备可序列化的数据
        metrics_df = self.calculate_user_metrics()
        retention_df = self.get_user_retention()
        segments = self.segment_users()

        # 确保数据可以被序列化，将所有数据转换为基本类型
        analysis_results = {
            'user_metrics': [
                {
                    'user_id': str(user_id),
                    'total_actions': int(metrics['total_actions']),
                    'unique_actions': int(metrics['unique_actions']),
                    'active_days': int(metrics['active_days']),
                    'avg_daily_actions': float(metrics['avg_daily_actions']),
                    'first_action': metrics['first_action'].isoformat(),
                    'last_action': metrics['last_action'].isoformat()
                }
                for user_id, metrics in metrics_df.to_dict('index').items()
            ],
            'user_segments': [
                {
                    'cluster_id': str(cluster_id),
                    'users': [str(uid) for uid in users]
                }
                for cluster_id, users in segments.items()
            ],
            'retention_data': [
                {
                    'cohort_date': row['cohort_date'].isoformat(),
                    'day': int(row['day']),
                    'users': int(row['users']),
                    'retained': int(row['retained']),
                    'retention_rate': float(row['retention_rate'])
                }
                for _, row in retention_df.iterrows()
            ]
        }

        # 导出为JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=4)