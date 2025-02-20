import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class OperationalEfficiencyAnalyzer:
    def __init__(self):
        self.operations_data = None
        self.efficiency_metrics = None
        self.performance_trends = None

    def load_data(self, operations_df: pd.DataFrame) -> None:
        """加载运营数据"""
        self.operations_data = operations_df.copy()
        self.operations_data['timestamp'] = pd.to_datetime(self.operations_data['timestamp'])

    def calculate_response_metrics(self, 
                                 user_id_col: str = 'user_id',
                                 timestamp_col: str = 'timestamp',
                                 response_time_col: str = 'response_time') -> pd.DataFrame:
        """计算响应指标"""
        metrics = self.operations_data.groupby(user_id_col).agg({
            response_time_col: ['mean', 'median', 'std', 'min', 'max', 'count']
        }).round(2)
        
        metrics.columns = ['avg_response_time', 'median_response_time', 
                         'std_response_time', 'min_response_time', 
                         'max_response_time', 'total_responses']
        
        return metrics

    def analyze_workload_distribution(self, 
                                    staff_col: str = 'staff_id',
                                    task_col: str = 'task_id') -> pd.DataFrame:
        """分析工作负载分布"""
        workload = self.operations_data.groupby(staff_col).agg({
            task_col: 'count',
            'completion_time': 'mean',
            'task_priority': 'mean'
        }).round(2)
        
        workload.columns = ['total_tasks', 'avg_completion_time', 'avg_priority']
        workload['workload_score'] = (
            workload['total_tasks'] * workload['avg_priority'] / 
            workload['avg_completion_time']
        ).round(2)
        
        return workload

    def calculate_efficiency_trends(self, 
                                  time_unit: str = 'D') -> pd.DataFrame:
        """计算效率趋势"""
        daily_metrics = self.operations_data.set_index('timestamp').resample(time_unit).agg({
            'task_id': 'count',
            'completion_time': 'mean',
            'response_time': 'mean',
            'success_rate': 'mean'
        })
        
        # 计算移动平均
        daily_metrics['tasks_ma'] = daily_metrics['task_id'].rolling(window=7).mean()
        daily_metrics['completion_ma'] = daily_metrics['completion_time'].rolling(window=7).mean()
        daily_metrics['response_ma'] = daily_metrics['response_time'].rolling(window=7).mean()
        
        return daily_metrics.round(2)

    def predict_workload(self, days_ahead: int = 7) -> pd.DataFrame:
        """预测未来工作负载"""
        daily_tasks = self.operations_data.set_index('timestamp')['task_id'].resample('D').count()
        
        # 准备特征
        X = np.arange(len(daily_tasks)).reshape(-1, 1)
        y = daily_tasks.values
        
        # 训练模型
        model = LinearRegression()
        model.fit(X, y)
        
        # 预测未来工作负载
        future_dates = pd.date_range(
            start=daily_tasks.index[-1] + timedelta(days=1),
            periods=days_ahead
        )
        future_X = np.arange(len(daily_tasks), len(daily_tasks) + days_ahead).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_tasks': np.round(predictions, 0)
        })

    def identify_bottlenecks(self) -> Dict[str, List[Dict]]:
        """识别运营瓶颈"""
        bottlenecks = {
            'high_response_time': [],
            'low_success_rate': [],
            'overloaded_staff': []
        }
        
        # 分析响应时间
        response_threshold = self.operations_data['response_time'].mean() + \
                           self.operations_data['response_time'].std()
        high_response = self.operations_data[
            self.operations_data['response_time'] > response_threshold
        ]
        
        for _, row in high_response.iterrows():
            bottlenecks['high_response_time'].append({
                'task_id': row['task_id'],
                'response_time': row['response_time'],
                'timestamp': row['timestamp']
            })
        
        # 分析成功率
        success_rate = self.operations_data.groupby('staff_id')['success_rate'].mean()
        low_success = success_rate[success_rate < 0.8]
        
        for staff_id, rate in low_success.items():
            bottlenecks['low_success_rate'].append({
                'staff_id': staff_id,
                'success_rate': rate
            })
        
        # 分析工作负载
        workload = self.analyze_workload_distribution()
        overloaded = workload[workload['workload_score'] > workload['workload_score'].mean() + \
                            workload['workload_score'].std()]
        
        for staff_id, row in overloaded.iterrows():
            bottlenecks['overloaded_staff'].append({
                'staff_id': staff_id,
                'total_tasks': row['total_tasks'],
                'workload_score': row['workload_score']
            })
        
        return bottlenecks

    def generate_optimization_suggestions(self) -> List[Dict]:
        """生成优化建议"""
        bottlenecks = self.identify_bottlenecks()
        suggestions = []
        
        # 基于瓶颈生成建议
        if bottlenecks['high_response_time']:
            suggestions.append({
                'area': '响应时间',
                'issue': f'发现 {len(bottlenecks["high_response_time"])} 个高响应时间任务',
                'suggestion': '建议优化任务分配机制，增加关键时段人员配置'
            })
        
        if bottlenecks['low_success_rate']:
            suggestions.append({
                'area': '成功率',
                'issue': f'发现 {len(bottlenecks["low_success_rate"])} 个低成功率员工',
                'suggestion': '建议进行针对性培训，优化工作流程'
            })
        
        if bottlenecks['overloaded_staff']:
            suggestions.append({
                'area': '工作负载',
                'issue': f'发现 {len(bottlenecks["overloaded_staff"])} 个工作超负荷员工',
                'suggestion': '建议重新分配工作任务，平衡团队工作负载'
            })
        
        return suggestions

    def export_efficiency_report(self, filepath: str) -> None:
        """导出效率分析报告"""
        report = {
            'overall_metrics': {
                'avg_response_time': float(self.operations_data['response_time'].mean()),
                'avg_completion_time': float(self.operations_data['completion_time'].mean()),
                'avg_success_rate': float(self.operations_data['success_rate'].mean()),
                'total_tasks': int(len(self.operations_data))
            },
            'workload_distribution': self.analyze_workload_distribution().to_dict(),
            'efficiency_trends': self.calculate_efficiency_trends().to_dict(),
            'bottlenecks': self.identify_bottlenecks(),
            'optimization_suggestions': self.generate_optimization_suggestions()
        }
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)