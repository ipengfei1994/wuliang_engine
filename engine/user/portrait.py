import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
from collections import Counter

class UserPortraitAnalyzer:
    def __init__(self):
        self.user_data = None
        self.behavior_data = None
        self.preference_data = None
        self.n_clusters = 5

    def load_data(self, user_data: pd.DataFrame = None,
                 behavior_data: pd.DataFrame = None,
                 preference_data: pd.DataFrame = None) -> None:
        """加载用户数据"""
        if user_data is not None:
            self.user_data = user_data.copy()
        if behavior_data is not None:
            self.behavior_data = behavior_data.copy()
        if preference_data is not None:
            self.preference_data = preference_data.copy()

    def analyze_demographic_features(self) -> Dict:
        """分析人口统计学特征"""
        if self.user_data is None:
            raise ValueError("请先加载用户数据")

        demographics = {}
        
        # 分析年龄分布
        if 'age' in self.user_data.columns:
            age_bins = [0, 18, 25, 35, 45, 55, 100]
            age_labels = ['18岁以下', '18-25岁', '26-35岁', '36-45岁', '46-55岁', '55岁以上']
            self.user_data['age_group'] = pd.cut(
                self.user_data['age'],
                bins=age_bins,
                labels=age_labels
            )
            demographics['age_distribution'] = self.user_data['age_group'].value_counts().to_dict()

        # 分析性别分布
        if 'gender' in self.user_data.columns:
            demographics['gender_distribution'] = self.user_data['gender'].value_counts().to_dict()

        # 分析地理分布
        if 'location' in self.user_data.columns:
            demographics['location_distribution'] = self.user_data['location'].value_counts().head(10).to_dict()

        # 分析职业分布
        if 'occupation' in self.user_data.columns:
            demographics['occupation_distribution'] = self.user_data['occupation'].value_counts().head(10).to_dict()

        return demographics

    def analyze_behavior_patterns(self) -> Dict:
        """分析行为模式"""
        if self.behavior_data is None:
            raise ValueError("请先加载行为数据")

        behavior_patterns = {}
        
        # 分析活跃时间
        if 'timestamp' in self.behavior_data.columns:
            self.behavior_data['hour'] = pd.to_datetime(self.behavior_data['timestamp']).dt.hour
            behavior_patterns['active_hours'] = self.behavior_data['hour'].value_counts().sort_index().to_dict()

        # 分析行为类型分布
        if 'behavior_type' in self.behavior_data.columns:
            behavior_patterns['behavior_distribution'] = (
                self.behavior_data['behavior_type'].value_counts().to_dict()
            )

        # 分析行为序列
        if 'user_id' in self.behavior_data.columns:
            behavior_sequences = self.behavior_data.groupby('user_id')['behavior_type'].agg(list)
            common_sequences = self._find_common_sequences(behavior_sequences)
            behavior_patterns['common_sequences'] = common_sequences

        return behavior_patterns

    def analyze_preferences(self) -> Dict:
        """分析用户偏好"""
        if self.preference_data is None:
            raise ValueError("请先加载偏好数据")

        preferences = {}
        
        # 分析内容偏好
        if 'content_type' in self.preference_data.columns:
            preferences['content_preferences'] = (
                self.preference_data.groupby('content_type')['interaction_count'].sum().to_dict()
            )

        # 分析标签偏好
        if 'tags' in self.preference_data.columns:
            all_tags = []
            for tags in self.preference_data['tags']:
                if isinstance(tags, str):
                    all_tags.extend(tags.split(','))
            preferences['tag_preferences'] = Counter(all_tags)

        # 分析时间偏好
        if 'interaction_time' in self.preference_data.columns:
            self.preference_data['hour'] = pd.to_datetime(
                self.preference_data['interaction_time']
            ).dt.hour
            preferences['time_preferences'] = (
                self.preference_data.groupby('hour')['interaction_count'].mean().to_dict()
            )

        return preferences

    def segment_users(self) -> Dict:
        """用户分群"""
        if self.user_data is None or self.behavior_data is None:
            raise ValueError("请先加载用户数据和行为数据")

        # 准备特征数据
        features = self._prepare_segmentation_features()
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # 使用K-means聚类
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)

        # 分析各群体特征
        segment_profiles = self._analyze_segments(features, clusters)

        return {
            'segment_profiles': segment_profiles,
            'segment_sizes': pd.Series(clusters).value_counts().to_dict()
        }

    def _prepare_segmentation_features(self) -> pd.DataFrame:
        """准备分群特征"""
        features = pd.DataFrame()
        
        # 合并用户基础特征
        if 'age' in self.user_data.columns:
            features['age'] = self.user_data['age']

        # 添加行为特征
        behavior_metrics = self.behavior_data.groupby('user_id').agg({
            'behavior_type': 'count',
            'duration': 'mean'
        }).reset_index()
        features = features.merge(behavior_metrics, on='user_id', how='left')

        # 添加偏好特征
        if self.preference_data is not None:
            preference_metrics = self.preference_data.groupby('user_id').agg({
                'interaction_count': 'sum',
                'content_type': lambda x: x.mode().iloc[0] if not x.empty else None
            }).reset_index()
            features = features.merge(preference_metrics, on='user_id', how='left')

        return features.fillna(0)

    def _analyze_segments(self, features: pd.DataFrame, clusters: np.ndarray) -> List[Dict]:
        """分析用户群体特征"""
        segment_profiles = []
        
        for i in range(self.n_clusters):
            segment_mask = clusters == i
            segment_features = features[segment_mask]
            
            profile = {
                'segment_id': i,
                'size': int(sum(segment_mask)),
                'characteristics': self._extract_segment_characteristics(segment_features)
            }
            segment_profiles.append(profile)

        return segment_profiles

    def _extract_segment_characteristics(self, segment_features: pd.DataFrame) -> Dict:
        """提取群体特征"""
        characteristics = {}
        
        # 计算数值特征的平均值
        numeric_cols = segment_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            characteristics[f'avg_{col}'] = float(segment_features[col].mean())

        # 计算分类特征的主要类别
        categorical_cols = segment_features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            characteristics[f'main_{col}'] = segment_features[col].mode().iloc[0]

        return characteristics

    def _find_common_sequences(self, sequences: pd.Series, min_support: float = 0.01) -> List[Dict]:
        """查找常见行为序列"""
        from collections import defaultdict
        
        sequence_counts = defaultdict(int)
        total_sequences = len(sequences)

        for seq in sequences:
            for i in range(len(seq)-1):
                pair = (seq[i], seq[i+1])
                sequence_counts[pair] += 1

        # 筛选满足最小支持度的序列
        common_sequences = [
            {
                'sequence': list(seq),
                'count': count,
                'support': count/total_sequences
            }
            for seq, count in sequence_counts.items()
            if count/total_sequences >= min_support
        ]

        return sorted(common_sequences, key=lambda x: x['support'], reverse=True)

    def generate_user_portraits(self) -> Dict:
        """生成用户画像报告"""
        report = {
            'demographic_features': self.analyze_demographic_features(),
            'behavior_patterns': self.analyze_behavior_patterns(),
            'preferences': self.analyze_preferences(),
            'user_segments': self.segment_users(),
            'timestamp': datetime.now().isoformat()
        }
        return report

    def export_report(self, filepath: str) -> None:
        """导出分析报告"""
        report = self.generate_user_portraits()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)