import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import networkx as nx
from collections import Counter
import json

class SocialMediaAnalyzer:
    def __init__(self):
        self.content_data = None
        self.interaction_data = None
        self.user_data = None
        self.metrics = {}

    def load_data(self, content_data: pd.DataFrame = None,
                 interaction_data: pd.DataFrame = None,
                 user_data: pd.DataFrame = None) -> None:
        """加载社交媒体数据"""
        if content_data is not None:
            self.content_data = content_data.copy()
        if interaction_data is not None:
            self.interaction_data = interaction_data.copy()
        if user_data is not None:
            self.user_data = user_data.copy()

    def analyze_content_performance(self) -> pd.DataFrame:
        """分析内容表现"""
        if self.content_data is None:
            raise ValueError("请先加载内容数据")

        performance = self.content_data.groupby('content_id').agg({
            'likes': 'sum',
            'comments': 'sum',
            'shares': 'sum',
            'views': 'sum'
        })

        # 计算互动率
        performance['engagement_rate'] = (
            (performance['likes'] + performance['comments'] * 2 + performance['shares'] * 3) /
            performance['views'] * 100
        ).round(2)

        # 计算内容影响力分数
        performance['impact_score'] = (
            performance['engagement_rate'] * 0.4 +
            (performance['shares'] / performance['views'] * 100) * 0.3 +
            (performance['comments'] / performance['views'] * 100) * 0.3
        ).round(2)

        return performance

    def analyze_posting_patterns(self) -> Dict:
        """分析发布模式"""
        if self.content_data is None:
            raise ValueError("请先加载内容数据")

        self.content_data['post_time'] = pd.to_datetime(self.content_data['post_time'])
        
        patterns = {
            'hourly_distribution': self.content_data['post_time'].dt.hour.value_counts().sort_index().to_dict(),
            'daily_distribution': self.content_data['post_time'].dt.dayofweek.value_counts().sort_index().to_dict(),
            'monthly_distribution': self.content_data['post_time'].dt.month.value_counts().sort_index().to_dict()
        }

        # 分析最佳发布时间
        performance = self.analyze_content_performance()
        self.content_data['engagement_rate'] = performance['engagement_rate']
        
        best_hours = self.content_data.groupby(self.content_data['post_time'].dt.hour)['engagement_rate'].mean()
        patterns['best_posting_hours'] = best_hours.nlargest(3).to_dict()

        return patterns

    def analyze_hashtag_performance(self) -> Dict:
        """分析话题标签效果"""
        if self.content_data is None or 'hashtags' not in self.content_data.columns:
            raise ValueError("请先加载包含hashtags的内容数据")

        # 展开所有标签
        all_hashtags = []
        for tags in self.content_data['hashtags']:
            if isinstance(tags, str):
                try:
                    tag_list = json.loads(tags)
                    all_hashtags.extend(tag_list)
                except json.JSONDecodeError:
                    all_hashtags.extend(tags.split(','))

        # 计算标签使用频率
        tag_frequency = Counter(all_hashtags)
        
        # 计算标签效果
        tag_performance = {}
        for tag in set(all_hashtags):
            tag_posts = self.content_data[self.content_data['hashtags'].str.contains(tag, na=False)]
            if not tag_posts.empty:
                tag_performance[tag] = {
                    'usage_count': tag_frequency[tag],
                    'avg_engagement': tag_posts['engagement_rate'].mean(),
                    'total_views': tag_posts['views'].sum()
                }

        return {
            'top_hashtags': dict(tag_frequency.most_common(10)),
            'hashtag_performance': tag_performance
        }

    def analyze_user_interactions(self) -> Dict:
        """分析用户互动"""
        if self.interaction_data is None:
            raise ValueError("请先加载互动数据")

        # 计算用户互动网络
        G = nx.Graph()
        for _, row in self.interaction_data.iterrows():
            G.add_edge(row['user_id'], row['target_user_id'], weight=row['interaction_weight'])

        # 计算网络指标
        metrics = {
            'total_interactions': len(self.interaction_data),
            'unique_users': len(G.nodes()),
            'interaction_density': nx.density(G),
            'top_influencers': sorted(nx.degree_centrality(G).items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:10]
        }

        return metrics

    def generate_content_recommendations(self) -> List[Dict]:
        """生成内容建议"""
        if self.content_data is None:
            raise ValueError("请先加载内容数据")

        performance = self.analyze_content_performance()
        patterns = self.analyze_posting_patterns()
        hashtag_analysis = self.analyze_hashtag_performance()

        recommendations = []

        # 基于最佳发布时间的建议
        recommendations.append({
            'type': '发布时间优化',
            'suggestion': f"建议在这些时间发布内容: {list(patterns['best_posting_hours'].keys())}点",
            'expected_improvement': '预计可提升15-30%的互动率'
        })

        # 基于高性能内容的建议
        top_content = performance.nlargest(5, 'impact_score')
        common_features = self._extract_content_features(top_content.index)
        recommendations.append({
            'type': '内容优化',
            'suggestion': f"建议采用以下特征: {common_features}",
            'expected_improvement': '预计可提升20-40%的内容影响力'
        })

        # 基于标签分析的建议
        top_tags = list(hashtag_analysis['top_hashtags'].keys())[:5]
        recommendations.append({
            'type': '标签策略',
            'suggestion': f"建议使用这些高效标签: {top_tags}",
            'expected_improvement': '预计可提升10-25%的触达率'
        })

        return recommendations

    def _extract_content_features(self, content_ids: List[str]) -> Dict:
        """提取内容特征"""
        top_content = self.content_data[self.content_data['content_id'].isin(content_ids)]
        
        features = {
            'avg_length': top_content['content_length'].mean(),
            'common_types': top_content['content_type'].mode().tolist(),
            'common_topics': top_content['topic'].mode().tolist()
        }
        
        return features

    def export_analysis_report(self, filepath: str) -> None:
        """导出分析报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'content_performance': self.analyze_content_performance().to_dict(),
            'posting_patterns': self.analyze_posting_patterns(),
            'hashtag_analysis': self.analyze_hashtag_performance(),
            'user_interactions': self.analyze_user_interactions(),
            'recommendations': self.generate_content_recommendations()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)