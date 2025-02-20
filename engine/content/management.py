import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union
import json
from collections import Counter

class ContentManager:
    def __init__(self):
        self.content_data = None
        self.content_metrics = None
        self.content_categories = set()

    def load_content(self, content_df: pd.DataFrame) -> None:
        """加载内容数据"""
        self.content_data = content_df.copy()
        if 'category' in self.content_data.columns:
            self.content_categories = set(self.content_data['category'].unique())

    def add_content(self, content_dict: Dict) -> None:
        """添加新内容"""
        new_content = pd.DataFrame([content_dict])
        if self.content_data is None:
            self.content_data = new_content
        else:
            self.content_data = pd.concat([self.content_data, new_content], ignore_index=True)
        
        if 'category' in content_dict:
            self.content_categories.add(content_dict['category'])

    def analyze_content_performance(self) -> pd.DataFrame:
        """分析内容表现"""
        if self.content_data is None:
            raise ValueError("请先加载内容数据")

        metrics = []
        for _, content in self.content_data.iterrows():
            metric = {
                'content_id': content.get('content_id'),
                'title': content.get('title'),
                'category': content.get('category'),
                'views': content.get('views', 0),
                'likes': content.get('likes', 0),
                'comments': content.get('comments', 0),
                'shares': content.get('shares', 0)
            }
            
            # 计算互动率
            metric['engagement_rate'] = (
                (metric['likes'] + metric['comments'] + metric['shares']) / 
                max(metric['views'], 1) * 100
            )
            
            metrics.append(metric)
        
        self.content_metrics = pd.DataFrame(metrics)
        return self.content_metrics

    def get_top_performing_content(self, metric: str = 'engagement_rate',
                                 top_n: int = 10) -> pd.DataFrame:
        """获取表现最好的内容"""
        if self.content_metrics is None:
            self.analyze_content_performance()
            
        return self.content_metrics.nlargest(top_n, metric)

    def analyze_category_performance(self) -> pd.DataFrame:
        """分析类别表现"""
        if self.content_metrics is None:
            self.analyze_content_performance()
            
        return self.content_metrics.groupby('category').agg({
            'views': 'sum',
            'likes': 'sum',
            'comments': 'sum',
            'shares': 'sum',
            'engagement_rate': 'mean'
        }).round(2)

    def get_content_trends(self, time_col: str = 'publish_time',
                          freq: str = 'M') -> pd.DataFrame:
        """分析内容趋势"""
        if time_col not in self.content_data.columns:
            raise ValueError(f"数据中不存在 {time_col} 列")
            
        trends = self.content_data.set_index(time_col).resample(freq).agg({
            'views': 'sum',
            'likes': 'sum',
            'comments': 'sum',
            'shares': 'sum'
        }).fillna(0)
        
        # 计算环比增长率
        for col in trends.columns:
            trends[f'{col}_growth'] = trends[col].pct_change() * 100
            
        return trends

    def analyze_content_tags(self, tags_col: str = 'tags') -> Dict[str, int]:
        """分析内容标签"""
        if tags_col not in self.content_data.columns:
            raise ValueError(f"数据中不存在 {tags_col} 列")
            
        all_tags = []
        for tags in self.content_data[tags_col]:
            if isinstance(tags, str):
                try:
                    tag_list = json.loads(tags)
                    all_tags.extend(tag_list)
                except json.JSONDecodeError:
                    all_tags.extend(tags.split(','))
            elif isinstance(tags, list):
                all_tags.extend(tags)
                
        return dict(Counter(all_tags))

    def get_content_recommendations(self, content_id: str,
                                  n_recommendations: int = 5) -> pd.DataFrame:
        """获取相似内容推荐"""
        if 'category' not in self.content_data.columns:
            raise ValueError("数据中必须包含 category 列")
            
        target_content = self.content_data[self.content_data['content_id'] == content_id].iloc[0]
        same_category = self.content_data[
            (self.content_data['category'] == target_content['category']) &
            (self.content_data['content_id'] != content_id)
        ]
        
        return same_category.nlargest(n_recommendations, 'views')

    def export_content_report(self, filepath: str) -> None:
        """导出内容分析报告"""
        if self.content_metrics is None:
            self.analyze_content_performance()
            
        report = {
            'overall_metrics': {
                'total_content': len(self.content_data),
                'total_views': self.content_metrics['views'].sum(),
                'total_engagement': (
                    self.content_metrics['likes'].sum() +
                    self.content_metrics['comments'].sum() +
                    self.content_metrics['shares'].sum()
                ),
                'average_engagement_rate': self.content_metrics['engagement_rate'].mean()
            },
            'top_performing_content': self.get_top_performing_content().to_dict('records'),
            'category_performance': self.analyze_category_performance().to_dict(),
            'content_trends': self.get_content_trends().to_dict()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)