from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import re
import json
from pathlib import Path

class DataValidator:
    def __init__(self):
        self.validation_rules = {}
        self.validation_results = {}
        self._load_default_rules()

    def _load_default_rules(self) -> None:
        """加载默认验证规则"""
        self.validation_rules = {
            'content_data': {
                'required_columns': ['content_id', 'content', 'created_at', 'author_id'],
                'data_types': {
                    'content_id': str,
                    'content': str,
                    'created_at': 'datetime',
                    'author_id': str
                },
                'value_ranges': {
                    'content': {'min_length': 10, 'max_length': 10000}
                }
            },
            'user_data': {
                'required_columns': ['user_id', 'username', 'registration_date'],
                'data_types': {
                    'user_id': str,
                    'username': str,
                    'registration_date': 'datetime'
                },
                'unique_columns': ['user_id', 'username']
            },
            'interaction_data': {
                'required_columns': ['interaction_id', 'user_id', 'content_id', 'type', 'timestamp'],
                'data_types': {
                    'interaction_id': str,
                    'user_id': str,
                    'content_id': str,
                    'type': str,
                    'timestamp': 'datetime'
                },
                'value_ranges': {
                    'type': {'allowed_values': ['like', 'comment', 'share', 'view']}
                }
            }
        }

    def add_validation_rule(self, data_type: str, rule_type: str, rule: Any) -> None:
        """添加验证规则"""
        if data_type not in self.validation_rules:
            self.validation_rules[data_type] = {}
        
        if rule_type not in ['required_columns', 'data_types', 'value_ranges', 'unique_columns']:
            raise ValueError(f"不支持的规则类型: {rule_type}")
            
        self.validation_rules[data_type][rule_type] = rule

    def validate_dataframe(self, df: pd.DataFrame, data_type: str) -> Dict:
        """验证数据框"""
        if data_type not in self.validation_rules:
            raise ValueError(f"未定义的数据类型: {data_type}")

        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'null_counts': df.isnull().sum().to_dict()
            }
        }

        rules = self.validation_rules[data_type]

        # 验证必需列
        if 'required_columns' in rules:
            missing_columns = set(rules['required_columns']) - set(df.columns)
            if missing_columns:
                results['is_valid'] = False
                results['errors'].append(f"缺少必需列: {missing_columns}")

        # 验证数据类型
        if 'data_types' in rules:
            for col, expected_type in rules['data_types'].items():
                if col not in df.columns:
                    continue
                    
                if expected_type == 'datetime':
                    try:
                        pd.to_datetime(df[col])
                    except Exception:
                        results['is_valid'] = False
                        results['errors'].append(f"列 {col} 包含无效的日期时间格式")
                else:
                    if not all(isinstance(x, expected_type) for x in df[col].dropna()):
                        results['is_valid'] = False
                        results['errors'].append(f"列 {col} 包含错误的数据类型")

        # 验证值范围
        if 'value_ranges' in rules:
            for col, range_rules in rules['value_ranges'].items():
                if col not in df.columns:
                    continue
                    
                if 'min_length' in range_rules:
                    invalid_length = df[col].str.len() < range_rules['min_length']
                    if invalid_length.any():
                        results['is_valid'] = False
                        results['errors'].append(
                            f"列 {col} 包含长度小于 {range_rules['min_length']} 的值"
                        )
                
                if 'max_length' in range_rules:
                    invalid_length = df[col].str.len() > range_rules['max_length']
                    if invalid_length.any():
                        results['is_valid'] = False
                        results['errors'].append(
                            f"列 {col} 包含长度大于 {range_rules['max_length']} 的值"
                        )
                
                if 'allowed_values' in range_rules:
                    invalid_values = ~df[col].isin(range_rules['allowed_values'])
                    if invalid_values.any():
                        results['is_valid'] = False
                        results['errors'].append(
                            f"列 {col} 包含不允许的值"
                        )

        # 验证唯一性
        if 'unique_columns' in rules:
            for col in rules['unique_columns']:
                if col not in df.columns:
                    continue
                    
                duplicates = df[col].duplicated()
                if duplicates.any():
                    results['is_valid'] = False
                    results['errors'].append(f"列 {col} 包含重复值")

        self.validation_results[data_type] = results
        return results

    def validate_json(self, data: Dict, schema: Dict) -> Dict:
        """验证JSON数据"""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        def validate_field(field_data: Any, field_schema: Dict, path: str = '') -> None:
            if 'type' in field_schema:
                expected_type = field_schema['type']
                actual_type = type(field_data).__name__
                if expected_type != actual_type:
                    results['is_valid'] = False
                    results['errors'].append(
                        f"字段 {path} 类型错误: 期望 {expected_type}, 实际 {actual_type}"
                    )

            if 'required' in field_schema and field_schema['required']:
                if field_data is None:
                    results['is_valid'] = False
                    results['errors'].append(f"必需字段 {path} 为空")

            if 'pattern' in field_schema and isinstance(field_data, str):
                if not re.match(field_schema['pattern'], field_data):
                    results['is_valid'] = False
                    results['errors'].append(f"字段 {path} 格式不匹配")

            if 'min_length' in field_schema and isinstance(field_data, (str, list)):
                if len(field_data) < field_schema['min_length']:
                    results['is_valid'] = False
                    results['errors'].append(
                        f"字段 {path} 长度小于 {field_schema['min_length']}"
                    )

            if 'max_length' in field_schema and isinstance(field_data, (str, list)):
                if len(field_data) > field_schema['max_length']:
                    results['is_valid'] = False
                    results['errors'].append(
                        f"字段 {path} 长度大于 {field_schema['max_length']}"
                    )

            if 'enum' in field_schema and field_data not in field_schema['enum']:
                results['is_valid'] = False
                results['errors'].append(
                    f"字段 {path} 的值不在允许范围内"
                )

        for field_name, field_schema in schema.items():
            field_path = field_name
            field_data = data.get(field_name)
            validate_field(field_data, field_schema, field_path)

        return results

    def validate_time_series(self, data: pd.DataFrame,
                           time_column: str,
                           value_columns: List[str]) -> Dict:
        """验证时间序列数据"""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }

        # 验证时间列
        try:
            data[time_column] = pd.to_datetime(data[time_column])
        except Exception:
            results['is_valid'] = False
            results['errors'].append("时间列格式无效")
            return results

        # 检查时间序列的完整性
        time_diff = data[time_column].diff()
        if time_diff.nunique() > 1:
            results['warnings'].append("时间间隔不一致")

        # 检查缺失值
        for col in value_columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                results['warnings'].append(f"列 {col} 包含 {missing_count} 个缺失值")

        # 检查异常值
        for col in value_columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            outliers = data[col][(data[col] < q1 - 1.5 * iqr) | 
                                (data[col] > q3 + 1.5 * iqr)]
            if len(outliers) > 0:
                results['warnings'].append(f"列 {col} 包含 {len(outliers)} 个异常值")

        # 统计信息
        results['stats'] = {
            'time_range': {
                'start': data[time_column].min().isoformat(),
                'end': data[time_column].max().isoformat()
            },
            'value_stats': {
                col: {
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std())
                }
                for col in value_columns
            }
        }

        return results

    def export_validation_report(self, filepath: str) -> None:
        """导出验证报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': self.validation_results,
            'validation_rules': self.validation_rules
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)

    def get_validation_summary(self) -> Dict:
        """获取验证结果摘要"""
        summary = {
            'total_validations': len(self.validation_results),
            'passed_validations': 0,
            'failed_validations': 0,
            'error_counts': {},
            'warning_counts': {}
        }

        for data_type, result in self.validation_results.items():
            if result['is_valid']:
                summary['passed_validations'] += 1
            else:
                summary['failed_validations'] += 1
            
            summary['error_counts'][data_type] = len(result['errors'])
            summary['warning_counts'][data_type] = len(result['warnings'])

        return summary