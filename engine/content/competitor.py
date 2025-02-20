import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

class CompetitorAnalyzer:
    def __init__(self):
        self.competitor_data = None
        self.market_data = None
        self.benchmark_metrics = {}

    def load_data(self, competitor_data: pd.DataFrame = None,
                 market_data: pd.DataFrame = None) -> None:
        """加载竞品数据"""
        if competitor_data is not None:
            self.competitor_data = competitor_data.copy()
        if market_data is not None:
            self.market_data = market_data.copy()

    def analyze_market_share(self) -> Dict:
        """分析市场份额"""
        if self.market_data is None:
            raise ValueError("请先加载市场数据")

        market_share = self.market_data.groupby('competitor_name').agg({
            'user_count': 'sum',
            'engagement_count': 'sum',
            'revenue': 'sum'
        })

        total = market_share.sum()
        market_share['user_share'] = (market_share['user_count'] / total['user_count'] * 100).round(2)
        market_share['engagement_share'] = (market_share['engagement_count'] / total['engagement_count'] * 100).round(2)
        market_share['revenue_share'] = (market_share['revenue'] / total['revenue'] * 100).round(2)

        return market_share.to_dict()

    def analyze_content_strategy(self) -> Dict:
        """分析内容策略"""
        if self.competitor_data is None:
            raise ValueError("请先加载竞品数据")

        strategy_analysis = {}
        
        # 分析内容类型分布
        content_distribution = self.competitor_data.groupby(
            ['competitor_name', 'content_type']
        ).size().unstack(fill_value=0)
        
        # 分析发布频率
        posting_frequency = self.competitor_data.groupby('competitor_name').agg({
            'post_time': lambda x: len(x) / (max(x) - min(x)).days
        })

        # 分析互动效果
        engagement_analysis = self.competitor_data.groupby('competitor_name').agg({
            'likes': 'mean',
            'comments': 'mean',
            'shares': 'mean'
        })

        strategy_analysis = {
            'content_distribution': content_distribution.to_dict(),
            'posting_frequency': posting_frequency.to_dict(),
            'engagement_metrics': engagement_analysis.to_dict()
        }

        return strategy_analysis

    def analyze_user_growth(self) -> pd.DataFrame:
        """分析用户增长"""
        if self.competitor_data is None:
            raise ValueError("请先加载竞品数据")

        growth_data = self.competitor_data.pivot_table(
            index='date',
            columns='competitor_name',
            values='follower_count',
            aggfunc='first'
        )

        # 计算增长率
        growth_rates = growth_data.pct_change()
        
        return {
            'absolute_growth': growth_data.to_dict(),
            'growth_rates': growth_rates.to_dict()
        }

    def analyze_competitive_advantages(self) -> List[Dict]:
        """分析竞争优势"""
        if self.competitor_data is None:
            raise ValueError("请先加载竞品数据")

        advantages = []
        metrics = ['engagement_rate', 'user_growth_rate', 'content_quality_score']
        
        for competitor in self.competitor_data['competitor_name'].unique():
            competitor_data = self.competitor_data[
                self.competitor_data['competitor_name'] == competitor
            ]
            
            # 计算各项指标的优势
            advantages.append({
                'competitor': competitor,
                'strengths': self._identify_strengths(competitor_data),
                'weaknesses': self._identify_weaknesses(competitor_data),
                'unique_features': self._identify_unique_features(competitor_data)
            })

        return advantages

    def _identify_strengths(self, data: pd.DataFrame) -> List[str]:
        """识别竞争优势"""
        strengths = []
        avg_metrics = self.competitor_data.mean()
        
        if data['engagement_rate'].mean() > avg_metrics['engagement_rate']:
            strengths.append('高互动率')
        if data['user_growth_rate'].mean() > avg_metrics['user_growth_rate']:
            strengths.append('快速增长')
        if data['content_quality_score'].mean() > avg_metrics['content_quality_score']:
            strengths.append('优质内容')
            
        return strengths

    def _identify_weaknesses(self, data: pd.DataFrame) -> List[str]:
        """识别竞争劣势"""
        weaknesses = []
        avg_metrics = self.competitor_data.mean()
        
        if data['engagement_rate'].mean() < avg_metrics['engagement_rate']:
            weaknesses.append('低互动率')
        if data['user_growth_rate'].mean() < avg_metrics['user_growth_rate']:
            weaknesses.append('增长缓慢')
        if data['content_quality_score'].mean() < avg_metrics['content_quality_score']:
            weaknesses.append('内容质量待提升')
            
        return weaknesses

    def _identify_unique_features(self, data: pd.DataFrame) -> List[str]:
        """识别独特特征"""
        unique_features = []
        
        # 分析独特的内容类型
        content_types = data['content_type'].value_counts()
        if len(content_types) > 0:
            unique_features.append(f"主打{content_types.index[0]}类内容")
            
        # 分析特殊的用户群体
        if 'target_audience' in data.columns:
            audiences = data['target_audience'].value_counts()
            if len(audiences) > 0:
                unique_features.append(f"针对{audiences.index[0]}人群")
                
        return unique_features

    def generate_competitive_report(self) -> Dict:
        """生成竞争分析报告"""
        report = {
            'market_analysis': self.analyze_market_share(),
            'content_strategy': self.analyze_content_strategy(),
            'user_growth': self.analyze_user_growth(),
            'competitive_advantages': self.analyze_competitive_advantages(),
            'recommendations': self._generate_recommendations()
        }
        
        return report

    def _generate_recommendations(self) -> List[Dict]:
        """生成竞争策略建议"""
        recommendations = []
        
        # 基于市场份额的建议
        market_share = self.analyze_market_share()
        if market_share:
            leader = max(market_share['user_share'].items(), key=lambda x: x[1])[0]
            recommendations.append({
                'area': '市场策略',
                'observation': f"市场领导者为 {leader}",
                'suggestion': '建议关注其用户获取策略和内容运营方式'
            })

        # 基于内容策略的建议
        content_strategy = self.analyze_content_strategy()
        if content_strategy:
            top_performer = max(
                content_strategy['engagement_metrics']['likes'].items(),
                key=lambda x: x[1]
            )[0]
            recommendations.append({
                'area': '内容策略',
                'observation': f"{top_performer} 的内容互动性最强",
                'suggestion': '建议分析其内容形式和话题选择'
            })

        return recommendations

    def export_analysis(self, filepath: str) -> None:
        """导出分析结果"""
        report = self.generate_competitive_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)