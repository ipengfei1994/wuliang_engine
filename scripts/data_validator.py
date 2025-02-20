import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import re
import json
import logging

class DataValidator:
    def __init__(self):
        self.validation_rules = {}
        self.validation_results = {}
        self._setup_logging()

    def _setup_logging(self) -> None:
        """配置日志系统"""
        logging.basicConfig(
            filename='data_validation.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def add_rule(self, column: str, rule_type: str, 
                 parameters: Dict[str, Any]) -> None:
        """添加验证规则"""
        if column not in self.validation_rules:
            self.validation_rules[column] = []
        
        self.validation_rules[column].append({
            'type': rule_type,
            'parameters': parameters
        })

    def validate_not_null(self, data: pd.Series) -> pd.Series:
        """验证非空"""
        return ~data.isna()

    def validate_unique(self, data: pd.Series) -> pd.Series:
        """验证唯一性"""
        return ~data.duplicated()

    def validate_range(self, data: pd.Series,
                      min_value: Optional[float] = None,
                      max_value: Optional[float] = None) -> pd.Series:
        """验证数值范围"""
        mask = pd.Series(True, index=data.index)
        if min_value is not None:
            mask &= (data >= min_value)
        if max_value is not None:
            mask &= (data <= max_value)
        return mask

    def validate_regex(self, data: pd.Series,
                      pattern: str) -> pd.Series:
        """验证正则表达式"""
        return data.str.match(pattern).fillna(False)

    def validate_categorical(self, data: pd.Series,
                           allowed_values: List[Any]) -> pd.Series:
        """验证分类值"""
        return data.isin(allowed_values)

    def validate_date_format(self, data: pd.Series,
                           date_format: str = '%Y-%m-%d') -> pd.Series:
        """验证日期格式"""
        def is_valid_date(x):
            try:
                if pd.isna(x):
                    return False
                datetime.strptime(str(x), date_format)
                return True
            except ValueError:
                return False
        
        return data.apply(is_valid_date)

    def validate_relationship(self, data: pd.DataFrame,
                            parent_col: str,
                            child_col: str) -> pd.Series:
        """验证关系完整性"""
        parent_values = set(data[parent_col].dropna())
        return data[child_col].isin(parent_values)

    def validate_custom(self, data: pd.Series,
                       validation_func: callable) -> pd.Series:
        """自定义验证"""
        return data.apply(validation_func)

    def validate_dataframe(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """验证整个数据框"""
        results = {}
        
        for column, rules in self.validation_rules.items():
            if column not in data.columns:
                logging.warning(f"Column {column} not found in data")
                continue
                
            column_results = []
            for rule in rules:
                try:
                    if rule['type'] == 'not_null':
                        mask = self.validate_not_null(data[column])
                    elif rule['type'] == 'unique':
                        mask = self.validate_unique(data[column])
                    elif rule['type'] == 'range':
                        mask = self.validate_range(
                            data[column],
                            rule['parameters'].get('min_value'),
                            rule['parameters'].get('max_value')
                        )
                    elif rule['type'] == 'regex':
                        mask = self.validate_regex(
                            data[column],
                            rule['parameters']['pattern']
                        )
                    elif rule['type'] == 'categorical':
                        mask = self.validate_categorical(
                            data[column],
                            rule['parameters']['allowed_values']
                        )
                    elif rule['type'] == 'date_format':
                        mask = self.validate_date_format(
                            data[column],
                            rule['parameters'].get('format', '%Y-%m-%d')
                        )
                    elif rule['type'] == 'relationship':
                        mask = self.validate_relationship(
                            data,
                            rule['parameters']['parent_col'],
                            column
                        )
                    elif rule['type'] == 'custom':
                        mask = self.validate_custom(
                            data[column],
                            rule['parameters']['func']
                        )
                    
                    invalid_rows = data[~mask].index.tolist()
                    column_results.append({
                        'rule_type': rule['type'],
                        'valid_count': mask.sum(),
                        'invalid_count': (~mask).sum(),
                        'invalid_rows': invalid_rows
                    })
                    
                except Exception as e:
                    logging.error(f"Error validating {column} with rule {rule['type']}: {str(e)}")
                    column_results.append({
                        'rule_type': rule['type'],
                        'error': str(e)
                    })
            
            results[column] = column_results
        
        self.validation_results = results
        return results

    def generate_validation_report(self) -> Dict:
        """生成验证报告"""
        if not self.validation_results:
            return {}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_columns': len(self.validation_results),
                'total_rules': sum(len(rules) for rules in self.validation_results.values()),
                'columns_with_errors': []
            },
            'details': {}
        }
        
        for column, results in self.validation_results.items():
            column_summary = {
                'total_rules': len(results),
                'failed_rules': 0,
                'error_count': 0
            }
            
            rule_details = []
            for result in results:
                if 'error' in result:
                    column_summary['failed_rules'] += 1
                    rule_details.append({
                        'rule_type': result['rule_type'],
                        'status': 'error',
                        'error_message': result['error']
                    })
                else:
                    invalid_count = result['invalid_count']
                    if invalid_count > 0:
                        column_summary['error_count'] += invalid_count
                        rule_details.append({
                            'rule_type': result['rule_type'],
                            'status': 'failed',
                            'invalid_count': invalid_count,
                            'invalid_rows': result['invalid_rows']
                        })
                    else:
                        rule_details.append({
                            'rule_type': result['rule_type'],
                            'status': 'passed'
                        })
            
            if column_summary['failed_rules'] > 0 or column_summary['error_count'] > 0:
                report['summary']['columns_with_errors'].append(column)
            
            report['details'][column] = {
                'summary': column_summary,
                'rules': rule_details
            }
        
        return report

    def export_report(self, filepath: str) -> None:
        """导出验证报告"""
        report = self.generate_validation_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        
        logging.info(f"Validation report exported to {filepath}")