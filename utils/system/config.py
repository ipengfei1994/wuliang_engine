import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import os
from datetime import datetime
import logging
from copy import deepcopy

class ConfigManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.config_data = {}
        self.default_config = {
            'analysis': {
                'sample_size': 1000,
                'confidence_level': 0.95,
                'min_data_points': 100
            },
            'visualization': {
                'theme': 'plotly_white',
                'default_width': 800,
                'default_height': 600,
                'color_palette': 'Set3'
            },
            'export': {
                'default_format': 'excel',
                'encoding': 'utf-8-sig',
                'compress_output': False
            },
            'api': {
                'host': 'localhost',
                'port': 8000,
                'debug': False,
                'request_timeout': 30
            },
            'database': {
                'host': 'localhost',
                'port': 5432,
                'max_connections': 10,
                'timeout': 30
            }
        }
        
        Path(config_dir).mkdir(parents=True, exist_ok=True)
        self._load_config()

    def _load_config(self) -> None:
        """加载配置文件"""
        config_file = os.path.join(self.config_dir, 'config.yaml')
        
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
        else:
            self.config_data = deepcopy(self.default_config)
            self._save_config()

    def _save_config(self) -> None:
        """保存配置到文件"""
        config_file = os.path.join(self.config_dir, 'config.yaml')
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config_data, f, allow_unicode=True, default_flow_style=False)

    def get_config(self, section: str = None, key: str = None) -> Any:
        """获取配置值"""
        if section is None:
            return self.config_data
        
        if key is None:
            return self.config_data.get(section, {})
            
        return self.config_data.get(section, {}).get(key)

    def set_config(self, section: str, key: str, value: Any) -> None:
        """设置配置值"""
        if section not in self.config_data:
            self.config_data[section] = {}
            
        self.config_data[section][key] = value
        self._save_config()

    def reset_config(self, section: str = None) -> None:
        """重置配置到默认值"""
        if section is None:
            self.config_data = deepcopy(self.default_config)
        else:
            self.config_data[section] = deepcopy(self.default_config.get(section, {}))
            
        self._save_config()

    def validate_config(self) -> Dict[str, List[str]]:
        """验证配置有效性"""
        issues = {}
        
        # 验证分析配置
        if 'analysis' in self.config_data:
            analysis_issues = []
            analysis_config = self.config_data['analysis']
            
            if analysis_config.get('sample_size', 0) <= 0:
                analysis_issues.append("样本大小必须大于0")
            if not 0 < analysis_config.get('confidence_level', 0) <= 1:
                analysis_issues.append("置信水平必须在0到1之间")
                
            if analysis_issues:
                issues['analysis'] = analysis_issues

        # 验证可视化配置
        if 'visualization' in self.config_data:
            vis_issues = []
            vis_config = self.config_data['visualization']
            
            if vis_config.get('default_width', 0) <= 0:
                vis_issues.append("图表宽度必须大于0")
            if vis_config.get('default_height', 0) <= 0:
                vis_issues.append("图表高度必须大于0")
                
            if vis_issues:
                issues['visualization'] = vis_issues

        return issues

    def export_config(self, format: str = 'yaml') -> str:
        """导出配置"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'yaml':
            export_file = os.path.join(self.config_dir, f'config_export_{timestamp}.yaml')
            with open(export_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, allow_unicode=True, default_flow_style=False)
        elif format == 'json':
            export_file = os.path.join(self.config_dir, f'config_export_{timestamp}.json')
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, ensure_ascii=False, indent=4)
        else:
            raise ValueError(f"不支持的导出格式: {format}")

        return export_file

    def import_config(self, filepath: str) -> None:
        """导入配置"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"配置文件不存在: {filepath}")

        # 备份当前配置
        self.export_config(format='yaml')

        # 导入新配置
        file_ext = os.path.splitext(filepath)[1].lower()
        
        try:
            if file_ext == '.yaml' or file_ext == '.yml':
                with open(filepath, 'r', encoding='utf-8') as f:
                    new_config = yaml.safe_load(f)
            elif file_ext == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    new_config = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {file_ext}")

            # 验证新配置
            self.config_data = new_config
            issues = self.validate_config()
            
            if issues:
                raise ValueError(f"配置验证失败: {issues}")

            self._save_config()
        except Exception as e:
            # 如果导入失败，恢复默认配置
            self.reset_config()
            raise Exception(f"配置导入失败: {str(e)}")

    def get_config_history(self) -> List[Dict]:
        """获取配置修改历史"""
        history = []
        
        for file in os.listdir(self.config_dir):
            if file.startswith('config_export_'):
                filepath = os.path.join(self.config_dir, file)
                history.append({
                    'filename': file,
                    'timestamp': datetime.strptime(
                        file.split('_')[2].split('.')[0],
                        '%Y%m%d_%H%M%S'
                    ).isoformat(),
                    'size': os.path.getsize(filepath)
                })

        return sorted(history, key=lambda x: x['timestamp'], reverse=True)

    def merge_config(self, new_config: Dict) -> None:
        """合并配置"""
        def deep_merge(source: Dict, destination: Dict) -> Dict:
            for key, value in source.items():
                if key in destination:
                    if isinstance(value, dict) and isinstance(destination[key], dict):
                        deep_merge(value, destination[key])
                    else:
                        destination[key] = value
                else:
                    destination[key] = value
            return destination

        self.config_data = deep_merge(new_config, self.config_data)
        self._save_config()