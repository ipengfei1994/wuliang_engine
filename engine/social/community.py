import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import networkx as nx
from collections import Counter
import json
from sklearn.metrics import silhouette_score
from textblob import TextBlob

class CommunityAnalyzer:
    def __init__(self):
        self.community_data = None
        self.interaction_data = None
        self.member_data = None
        self.event_data = None

    def load_data(self, community_data: pd.DataFrame = None,
                 interaction_data: pd.DataFrame = None,
                 member_data: pd.DataFrame = None,
                 event_data: pd.DataFrame = None) -> None:
        """加载社群数据"""
        if community_data is not None:
            self.community_data = community_data.copy()
        if interaction_data is not None:
            self.interaction_data = interaction_data.copy()
        if member_data is not None:
            self.member_data = member_data.copy()
        if event_data is not None:
            self.event_data = event_data.copy()

    def analyze_community_health(self) -> Dict:
        """分析社群健康度"""
        if self.interaction_data is None or self.member_data is None:
            raise ValueError("请先加载互动数据和成员数据")

        # 计算活跃度指标
        daily_active = self.interaction_data.groupby('date')['user_id'].nunique()
        total_members = len(self.member_data)
        
        health_metrics = {
            'dau': int(daily_active.mean()),
            'dau_ratio': round(daily_active.mean() / total_members * 100, 2),
            'interaction_density': round(len(self.interaction_data) / total_members, 2),
            'core_member_ratio': self._calculate_core_member_ratio()
        }

        return health_metrics

    def analyze_interaction_quality(self) -> Dict:
        """分析互动质量"""
        if self.interaction_data is None:
            raise ValueError("请先加载互动数据")

        # 分析互动深度
        interaction_depth = self.interaction_data.groupby('thread_id').agg({
            'user_id': 'nunique',
            'content': 'count',
            'interaction_type': lambda x: len(set(x))
        })

        quality_metrics = {
            'avg_participants': float(interaction_depth['user_id'].mean()),
            'avg_responses': float(interaction_depth['content'].mean()),
            'interaction_types': self._analyze_interaction_types(),
            'sentiment_scores': self._analyze_sentiment()
        }

        return quality_metrics

    def analyze_event_performance(self) -> Dict:
        """分析活动效果"""
        if self.event_data is None:
            raise ValueError("请先加载活动数据")

        event_metrics = self.event_data.groupby('event_id').agg({
            'participant_count': 'sum',
            'engagement_duration': 'mean',
            'satisfaction_score': 'mean'
        })

        return {
            'event_metrics': event_metrics.to_dict(),
            'total_events': len(self.event_data),
            'avg_participation': float(event_metrics['participant_count'].mean()),
            'avg_satisfaction': float(event_metrics['satisfaction_score'].mean())
        }

    def analyze_member_engagement(self) -> Dict:
        """分析成员参与度"""
        if self.member_data is None or self.interaction_data is None:
            raise ValueError("请先加载成员数据和互动数据")

        # 计算成员参与度指标
        member_engagement = self.interaction_data.groupby('user_id').agg({
            'content': 'count',
            'date': 'nunique',
            'interaction_type': 'nunique'
        })

        # 成员分层
        member_engagement['engagement_score'] = (
            member_engagement['content'] * 0.4 +
            member_engagement['date'] * 0.4 +
            member_engagement['interaction_type'] * 0.2
        )

        engagement_levels = pd.qcut(
            member_engagement['engagement_score'],
            q=4,
            labels=['低度参与', '一般参与', '活跃参与', '高度活跃']
        )

        return {
            'engagement_distribution': engagement_levels.value_counts().to_dict(),
            'avg_engagement_score': float(member_engagement['engagement_score'].mean()),
            'top_contributors': self._identify_top_contributors()
        }

    def _calculate_core_member_ratio(self) -> float:
        """计算核心成员比例"""
        if self.interaction_data is None or self.member_data is None:
            return 0.0

        interaction_counts = self.interaction_data['user_id'].value_counts()
        core_members = interaction_counts[interaction_counts > interaction_counts.median()]
        return round(len(core_members) / len(self.member_data) * 100, 2)

    def _analyze_interaction_types(self) -> Dict:
        """分析互动类型分布"""
        if self.interaction_data is None:
            return {}

        type_counts = self.interaction_data['interaction_type'].value_counts()
        return type_counts.to_dict()

    def _analyze_sentiment(self) -> Dict:
        """分析情感倾向"""
        if self.interaction_data is None or 'content' not in self.interaction_data.columns:
            return {}

        def get_sentiment(text):
            try:
                return TextBlob(str(text)).sentiment.polarity
            except:
                return 0

        sentiments = self.interaction_data['content'].apply(get_sentiment)
        return {
            'positive': float((sentiments > 0).mean()),
            'neutral': float((sentiments == 0).mean()),
            'negative': float((sentiments < 0).mean()),
            'avg_sentiment': float(sentiments.mean())
        }

    def _identify_top_contributors(self, top_n: int = 10) -> List[Dict]:
        """识别核心贡献者"""
        if self.interaction_data is None or self.member_data is None:
            return []

        contributor_stats = self.interaction_data.groupby('user_id').agg({
            'content': 'count',
            'interaction_type': 'nunique',
            'thread_id': 'nunique'
        })

        contributor_stats['contribution_score'] = (
            contributor_stats['content'] * 0.5 +
            contributor_stats['interaction_type'] * 0.3 +
            contributor_stats['thread_id'] * 0.2
        )

        top_contributors = contributor_stats.nlargest(top_n, 'contribution_score')
        return [
            {
                'user_id': user_id,
                'contribution_score': float(score),
                'interaction_count': int(contributor_stats.loc[user_id, 'content'])
            }
            for user_id, score in top_contributors['contribution_score'].items()
        ]

    def generate_community_report(self) -> Dict:
        """生成社群分析报告"""
        report = {
            'health_metrics': self.analyze_community_health(),
            'interaction_quality': self.analyze_interaction_quality(),
            'event_performance': self.analyze_event_performance(),
            'member_engagement': self.analyze_member_engagement(),
            'recommendations': self._generate_recommendations(),
            'timestamp': datetime.now().isoformat()
        }
        return report

    def _generate_recommendations(self) -> List[Dict]:
        """生成运营建议"""
        recommendations = []
        
        # 基于健康度指标的建议
        health_metrics = self.analyze_community_health()
        if health_metrics['dau_ratio'] < 30:
            recommendations.append({
                'area': '活跃度提升',
                'suggestion': '增加日常互动话题和活动频率',
                'expected_impact': '提高日活跃度'
            })

        # 基于互动质量的建议
        interaction_quality = self.analyze_interaction_quality()
        if interaction_quality['avg_responses'] < 5:
            recommendations.append({
                'area': '互动深度',
                'suggestion': '设计更多引发讨论的话题',
                'expected_impact': '提升互动深度'
            })

        # 基于成员参与度的建议
        engagement = self.analyze_member_engagement()
        if engagement['engagement_distribution'].get('低度参与', 0) > 0.4:
            recommendations.append({
                'area': '成员激活',
                'suggestion': '针对低参与度成员开展专项活动',
                'expected_impact': '提升整体参与度'
            })

        return recommendations

    def export_report(self, filepath: str) -> None:
        """导出分析报告"""
        report = self.generate_community_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)