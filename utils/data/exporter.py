import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime
import json
import xlsxwriter
import csv
import os
from pathlib import Path

class DataExporter:
    def __init__(self, output_dir: str = "exports"):
        self.output_dir = output_dir
        self.supported_formats = ['csv', 'excel', 'json', 'html']
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def export_data(self, data: Union[pd.DataFrame, Dict],
                   filename: str,
                   format: str = 'csv',
                   sheet_name: str = 'Sheet1') -> str:
        """导出数据到指定格式"""
        if format not in self.supported_formats:
            raise ValueError(f"不支持的导出格式: {format}")

        filepath = os.path.join(self.output_dir, f"{filename}.{format}")
        
        if isinstance(data, pd.DataFrame):
            if format == 'csv':
                data.to_csv(filepath, index=False, encoding='utf-8-sig')
            elif format == 'excel':
                data.to_excel(filepath, sheet_name=sheet_name, index=False)
            elif format == 'json':
                data.to_json(filepath, orient='records', force_ascii=False)
            elif format == 'html':
                data.to_html(filepath, index=False)
        elif isinstance(data, dict):
            if format == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            else:
                df = pd.DataFrame.from_dict(data)
                return self.export_data(df, filename, format, sheet_name)

        return filepath

    def export_report(self, report_data: Dict,
                     report_type: str,
                     include_charts: bool = True) -> str:
        """导出分析报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = os.path.join(self.output_dir, f"{report_type}_report_{timestamp}")
        Path(report_dir).mkdir(parents=True, exist_ok=True)

        # 导出报告主体
        report_path = os.path.join(report_dir, "report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=4)

        # 生成Excel报告
        excel_path = os.path.join(report_dir, "report.xlsx")
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            for section, data in report_data.items():
                if isinstance(data, (list, dict)):
                    df = pd.DataFrame(data)
                    df.to_excel(writer, sheet_name=section[:31], index=False)

        return report_dir

    def export_dashboard(self, dashboard_data: Dict[str, pd.DataFrame],
                        dashboard_name: str) -> str:
        """导出数据仪表板"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dashboard_dir = os.path.join(self.output_dir, f"{dashboard_name}_{timestamp}")
        Path(dashboard_dir).mkdir(parents=True, exist_ok=True)

        # 导出各个数据表
        for table_name, df in dashboard_data.items():
            table_path = os.path.join(dashboard_dir, f"{table_name}.csv")
            df.to_csv(table_path, index=False, encoding='utf-8-sig')

        # 生成仪表板配置
        config = {
            'dashboard_name': dashboard_name,
            'created_at': timestamp,
            'tables': list(dashboard_data.keys())
        }
        
        config_path = os.path.join(dashboard_dir, "dashboard_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        return dashboard_dir

    def export_batch(self, data_list: List[Dict],
                    base_filename: str,
                    format: str = 'csv') -> List[str]:
        """批量导出数据"""
        exported_files = []
        
        for i, data in enumerate(data_list):
            filename = f"{base_filename}_{i+1}"
            filepath = self.export_data(data, filename, format)
            exported_files.append(filepath)

        return exported_files

    def export_time_series(self, time_series_data: pd.DataFrame,
                          filename: str,
                          time_col: str,
                          value_cols: List[str]) -> str:
        """导出时间序列数据"""
        # 确保时间列格式正确
        time_series_data[time_col] = pd.to_datetime(time_series_data[time_col])
        
        # 按时间排序
        time_series_data = time_series_data.sort_values(time_col)
        
        # 创建Excel文件
        filepath = os.path.join(self.output_dir, f"{filename}.xlsx")
        
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            # 导出原始数据
            time_series_data.to_excel(writer, sheet_name='原始数据', index=False)
            
            # 创建日期汇总
            daily_data = time_series_data.groupby(
                time_series_data[time_col].dt.date
            )[value_cols].mean()
            daily_data.to_excel(writer, sheet_name='日汇总')
            
            # 创建月度汇总
            monthly_data = time_series_data.groupby([
                time_series_data[time_col].dt.year,
                time_series_data[time_col].dt.month
            ])[value_cols].mean()
            monthly_data.to_excel(writer, sheet_name='月汇总')

        return filepath

    def export_comparison(self, comparison_data: Dict[str, pd.DataFrame],
                         filename: str) -> str:
        """导出对比分析数据"""
        filepath = os.path.join(self.output_dir, f"{filename}.xlsx")
        
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            # 导出各个对比组的数据
            for group_name, df in comparison_data.items():
                df.to_excel(writer, sheet_name=group_name[:31], index=False)
            
            # 创建汇总sheet
            summary_data = {}
            for group_name, df in comparison_data.items():
                summary_data[group_name] = df.describe()
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='汇总分析')

        return filepath

    def export_metrics(self, metrics_data: Dict[str, Union[float, int, str]],
                      filename: str) -> str:
        """导出指标数据"""
        filepath = os.path.join(self.output_dir, f"{filename}.json")
        
        # 添加导出时间戳
        metrics_data['export_timestamp'] = datetime.now().isoformat()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=4)

        return filepath

    def create_export_summary(self, exported_files: List[str]) -> str:
        """创建导出汇总报告"""
        summary = {
            'export_time': datetime.now().isoformat(),
            'total_files': len(exported_files),
            'files': []
        }

        for filepath in exported_files:
            file_info = {
                'filename': os.path.basename(filepath),
                'path': filepath,
                'size': os.path.getsize(filepath),
                'format': os.path.splitext(filepath)[1][1:],
                'created_at': datetime.fromtimestamp(
                    os.path.getctime(filepath)
                ).isoformat()
            }
            summary['files'].append(file_info)

        summary_path = os.path.join(self.output_dir, 'export_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)

        return summary_path