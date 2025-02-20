import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class UserGrowthAnalyzer:
    def __init__(self):
        self.user_data = None
        self.activity_data = None
        self.acquisition_data = None

    def load_data(self, user_data: pd.DataFrame = None,
                 activity_data: pd.DataFrame = None,
                 acquisition_data: pd.DataFrame = None) -> None:
        """加载用户数据"""
        if user_data is not None:
            self.user_data = user_data.copy()
        if activity_data is not None:
            self.activity_data = activity_data.copy()
        if acquisition_data is not None:
            self.acquisition_data = acquisition_data.copy()

    def analyze_user_acquisition(self) -> Dict:
        """分析用户获取情况"""
        if self.acquisition_data is None:
            raise ValueError("请先加载获客数据")

        # 按渠道分析获客效果
        channel_metrics = self.acquisition_data.groupby('channel').agg({
            'user_id': 'count',
            'acquisition_cost': 'sum',
            'conversion_rate': 'mean'
        }).round(2)

        # 计算渠道ROI
        channel_metrics['roi'] = (
            (channel_metrics['user_id'] * self.acquisition_data['user_value'].mean()) /
            channel_metrics['acquisition_cost']
        ).round(2)

        return {
            'channel_performance': channel_metrics.to_dict(),
            'total_users': int(channel_metrics['user_id'].sum()),
            'total_cost': float(channel_metrics['acquisition_cost'].sum()),
            'avg_acquisition_cost': float(
                channel_metrics['acquisition_cost'].sum() / channel_metrics['user_id'].sum()
            )
        }

    def analyze_retention(self, periods: List[int] = [1, 7, 30, 90]) -> Dict:
        """分析用户留存"""
        if self.activity_data is None:
            raise ValueError("请先加载活动数据")

        retention_data = {}
        first_activities = self.activity_data.groupby('user_id')['activity_date'].min()

        for period in periods:
            # 计算每个用户在特定时期后是否仍然活跃
            retention_count = 0
            total_users = len(first_activities)

            for user_id, first_date in first_activities.items():
                check_date = first_date + timedelta(days=period)
                is_retained = self.activity_data[
                    (self.activity_data['user_id'] == user_id) &
                    (self.activity_data['activity_date'] == check_date)
                ].shape[0] > 0

                if is_retained:
                    retention_count += 1

            retention_data[f'{period}d'] = {
                'retention_rate': round(retention_count / total_users * 100, 2),
                'retained_users': retention_count,
                'total_users': total_users
            }

        return retention_data

    def analyze_user_activity(self) -> Dict:
        """分析用户活跃度"""
        if self.activity_data is None:
            raise ValueError("请先加载活动数据")

        # 计算用户活跃度指标
        user_activity = self.activity_data.groupby('user_id').agg({
            'activity_date': 'count',
            'session_duration': 'mean',
            'interaction_count': 'sum'
        })

        # 对用户进行分层
        user_activity['activity_score'] = (
            user_activity['activity_date'] * 0.4 +
            user_activity['session_duration'] * 0.3 +
            user_activity['interaction_count'] * 0.3
        )

        # 使用K-means进行用户分群
        kmeans = KMeans(n_clusters=4, random_state=42)
        user_activity['segment'] = kmeans.fit_predict(
            user_activity[['activity_score']]
        )

        segments = {
            'highly_active': len(user_activity[user_activity['segment'] == 3]),
            'active': len(user_activity[user_activity['segment'] == 2]),
            'moderate': len(user_activity[user_activity['segment'] == 1]),
            'inactive': len(user_activity[user_activity['segment'] == 0])
        }

        return {
            'activity_metrics': user_activity.to_dict(),
            'user_segments': segments,
            'total_active_users': len(user_activity)
        }

    def analyze_growth_metrics(self) -> Dict:
        """分析增长指标"""
        metrics = {}
        
        # 计算获客成本趋势
        if self.acquisition_data is not None:
            cac_trend = self.acquisition_data.groupby('date').agg({
                'acquisition_cost': 'sum',
                'user_id': 'count'
            })
            cac_trend['cac'] = cac_trend['acquisition_cost'] / cac_trend['user_id']
            metrics['cac_trend'] = cac_trend['cac'].to_dict()

        # 计算用户生命周期价值
        if self.user_data is not None and 'user_value' in self.user_data.columns:
            ltv = self.user_data['user_value'].mean()
            metrics['average_ltv'] = float(ltv)

        # 计算病毒系数
        if self.user_data is not None and 'referral_count' in self.user_data.columns:
            viral_coefficient = self.user_data['referral_count'].mean()
            metrics['viral_coefficient'] = float(viral_coefficient)

        return metrics

    def identify_growth_opportunities(self) -> List[Dict]:
        """识别增长机会"""
        opportunities = []

        # 分析获客机会
        if self.acquisition_data is not None:
            channel_performance = self.analyze_user_acquisition()
            best_channel = max(
                channel_performance['channel_performance']['roi'].items(),
                key=lambda x: x[1]
            )[0]
            opportunities.append({
                'area': '用户获取',
                'opportunity': f'扩大{best_channel}渠道的投入',
                'expected_impact': '提高ROI和获客效率'
            })

        # 分析留存机会
        retention_data = self.analyze_retention()
        if retention_data:
            critical_period = min(retention_data.keys(), key=lambda x: retention_data[x]['retention_rate'])
            opportunities.append({
                'area': '用户留存',
                'opportunity': f'优化{critical_period}期的用户体验',
                'expected_impact': '提高用户留存率'
            })

        # 分析活跃度提升机会
        activity_data = self.analyze_user_activity()
        if activity_data:
            inactive_ratio = activity_data['user_segments']['inactive'] / activity_data['total_active_users']
            if inactive_ratio > 0.3:
                opportunities.append({
                    'area': '用户活跃',
                    'opportunity': '激活沉睡用户',
                    'expected_impact': '提高整体活跃度'
                })

        return opportunities

    def generate_growth_report(self) -> Dict:
        """生成增长报告"""
        report = {
            'acquisition_analysis': self.analyze_user_acquisition(),
            'retention_analysis': self.analyze_retention(),
            'activity_analysis': self.analyze_user_activity(),
            'growth_metrics': self.analyze_growth_metrics(),
            'opportunities': self.identify_growth_opportunities(),
            'timestamp': datetime.now().isoformat()
        }

        return report

    def export_report(self, filepath: str) -> None:
        """导出分析报告"""
        report = self.generate_growth_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)