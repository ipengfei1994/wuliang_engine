import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

class AdPerformanceAnalyzer:
    def __init__(self):
        self.ad_data = None
        self.performance_metrics = None
        self.conversion_data = None

    def load_data(self, ad_data: pd.DataFrame) -> None:
        """加载广告数据"""
        self.ad_data = ad_data.copy()
        if 'timestamp' in self.ad_data.columns:
            self.ad_data['timestamp'] = pd.to_datetime(self.ad_data['timestamp'])

    def calculate_basic_metrics(self) -> pd.DataFrame:
        """计算基础指标"""
        metrics = self.ad_data.groupby('ad_id').agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'cost': 'sum',
            'revenue': 'sum'
        })
        
        # 计算派生指标
        metrics['ctr'] = (metrics['clicks'] / metrics['impressions'] * 100).round(2)
        metrics['cvr'] = (metrics['conversions'] / metrics['clicks'] * 100).round(2)
        metrics['cpc'] = (metrics['cost'] / metrics['clicks']).round(2)
        metrics['cpa'] = (metrics['cost'] / metrics['conversions']).round(2)
        metrics['roas'] = (metrics['revenue'] / metrics['cost']).round(2)
        metrics['profit'] = (metrics['revenue'] - metrics['cost']).round(2)
        
        return metrics

    def analyze_performance_trends(self, 
                                 time_unit: str = 'D',
                                 metrics: List[str] = None) -> pd.DataFrame:
        """分析性能趋势"""
        if metrics is None:
            metrics = ['impressions', 'clicks', 'conversions', 'cost', 'revenue']
            
        trends = self.ad_data.set_index('timestamp').resample(time_unit).agg({
            metric: 'sum' for metric in metrics
        })
        
        # 计算变化率
        for metric in metrics:
            trends[f'{metric}_growth'] = trends[metric].pct_change() * 100
            
        return trends.round(2)

    def segment_performance(self, 
                          segment_by: List[str] = None) -> pd.DataFrame:
        """分析细分市场表现"""
        if segment_by is None:
            segment_by = ['platform', 'campaign_type', 'target_audience']
            
        segments = []
        for segment in segment_by:
            if segment in self.ad_data.columns:
                segment_metrics = self.ad_data.groupby(segment).agg({
                    'impressions': 'sum',
                    'clicks': 'sum',
                    'conversions': 'sum',
                    'cost': 'sum',
                    'revenue': 'sum'
                })
                
                segment_metrics['ctr'] = (segment_metrics['clicks'] / 
                                        segment_metrics['impressions'] * 100).round(2)
                segment_metrics['cvr'] = (segment_metrics['conversions'] / 
                                        segment_metrics['clicks'] * 100).round(2)
                segment_metrics['roas'] = (segment_metrics['revenue'] / 
                                         segment_metrics['cost']).round(2)
                
                segments.append({
                    'segment_type': segment,
                    'metrics': segment_metrics.to_dict()
                })
                
        return segments

    def analyze_budget_allocation(self) -> Dict:
        """分析预算分配"""
        budget_analysis = {
            'total_spend': float(self.ad_data['cost'].sum()),
            'spend_by_platform': self.ad_data.groupby('platform')['cost'].sum().to_dict(),
            'spend_by_campaign': self.ad_data.groupby('campaign_id')['cost'].sum().to_dict(),
            'roas_by_platform': (self.ad_data.groupby('platform').agg({
                'revenue': 'sum',
                'cost': 'sum'
            }).apply(lambda x: x['revenue'] / x['cost'], axis=1) * 100).round(2).to_dict()
        }
        
        return budget_analysis

    def identify_top_performers(self, 
                              metric: str = 'roas',
                              top_n: int = 5) -> pd.DataFrame:
        """识别表现最佳的广告"""
        performance_metrics = self.calculate_basic_metrics()
        return performance_metrics.nlargest(top_n, metric)

    def calculate_roi_metrics(self) -> pd.DataFrame:
        """计算投资回报指标"""
        roi_metrics = self.ad_data.groupby('ad_id').agg({
            'cost': 'sum',
            'revenue': 'sum',
            'conversions': 'sum'
        })
        
        roi_metrics['roi'] = ((roi_metrics['revenue'] - roi_metrics['cost']) / 
                            roi_metrics['cost'] * 100).round(2)
        roi_metrics['cpa'] = (roi_metrics['cost'] / roi_metrics['conversions']).round(2)
        roi_metrics['value_per_conversion'] = (roi_metrics['revenue'] / 
                                             roi_metrics['conversions']).round(2)
        
        return roi_metrics

    def analyze_ad_fatigue(self, window_size: int = 7) -> pd.DataFrame:
        """分析广告疲劳度"""
        daily_metrics = self.ad_data.set_index('timestamp').resample('D').agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'ctr': 'mean'
        })
        
        # 计算移动平均和变化率
        daily_metrics['ctr_ma'] = daily_metrics['ctr'].rolling(window=window_size).mean()
        daily_metrics['ctr_change'] = daily_metrics['ctr'].pct_change() * 100
        
        return daily_metrics.round(2)

    def generate_optimization_recommendations(self) -> List[Dict]:
        """生成优化建议"""
        recommendations = []
        
        # 分析CTR表现
        performance = self.calculate_basic_metrics()
        low_ctr_ads = performance[performance['ctr'] < performance['ctr'].mean()]
        if not low_ctr_ads.empty:
            recommendations.append({
                'area': 'CTR优化',
                'issue': f'发现{len(low_ctr_ads)}个CTR低于平均水平的广告',
                'suggestion': '建议优化广告创意和定向，提高点击率'
            })
        
        # 分析ROI表现
        roi_metrics = self.calculate_roi_metrics()
        negative_roi_ads = roi_metrics[roi_metrics['roi'] < 0]
        if not negative_roi_ads.empty:
            recommendations.append({
                'area': 'ROI优化',
                'issue': f'发现{len(negative_roi_ads)}个ROI为负的广告',
                'suggestion': '建议调整出价策略，优化目标受众定向'
            })
        
        # 分析预算效率
        budget_analysis = self.analyze_budget_allocation()
        low_roas_platforms = {k: v for k, v in budget_analysis['roas_by_platform'].items() 
                            if v < 100}
        if low_roas_platforms:
            recommendations.append({
                'area': '预算分配',
                'issue': f'发现{len(low_roas_platforms)}个ROAS低于100%的平台',
                'suggestion': '建议重新分配预算，将资源集中在高效平台'
            })
        
        return recommendations

    def plot_performance_dashboard(self) -> None:
        """绘制性能仪表板"""
        plt.figure(figsize=(15, 10))
        
        # 绘制CTR趋势
        plt.subplot(2, 2, 1)
        trends = self.analyze_performance_trends()
        plt.plot(trends.index, trends['ctr'])
        plt.title('CTR Trend')
        plt.xticks(rotation=45)
        
        # 绘制转化率分布
        plt.subplot(2, 2, 2)
        performance = self.calculate_basic_metrics()
        sns.histplot(performance['cvr'])
        plt.title('Conversion Rate Distribution')
        
        # 绘制平台支出占比
        plt.subplot(2, 2, 3)
        budget = self.analyze_budget_allocation()
        plt.pie(budget['spend_by_platform'].values(), 
                labels=budget['spend_by_platform'].keys(),
                autopct='%1.1f%%')
        plt.title('Budget Distribution by Platform')
        
        # 绘制ROI分布
        plt.subplot(2, 2, 4)
        roi_metrics = self.calculate_roi_metrics()
        sns.boxplot(y=roi_metrics['roi'])
        plt.title('ROI Distribution')
        
        plt.tight_layout()
        plt.show()

    def export_performance_report(self, filepath: str) -> None:
        """导出性能分析报告"""
        report = {
            'overall_metrics': self.calculate_basic_metrics().to_dict(),
            'performance_trends': self.analyze_performance_trends().to_dict(),
            'segment_performance': self.segment_performance(),
            'budget_analysis': self.analyze_budget_allocation(),
            'top_performers': self.identify_top_performers().to_dict(),
            'recommendations': self.generate_optimization_recommendations()
        }
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)