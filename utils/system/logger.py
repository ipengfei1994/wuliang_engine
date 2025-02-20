import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import json
import traceback
import sys

class LogManager:
    def __init__(self, log_dir: str = "logs",
                 app_name: str = "wuliang_engine",
                 max_bytes: int = 10485760,  # 10MB
                 backup_count: int = 10):
        self.log_dir = log_dir
        self.app_name = app_name
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.loggers = {}

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._setup_default_loggers()

    def _setup_default_loggers(self) -> None:
        """设置默认日志记录器"""
        # 应用程序日志
        self.setup_logger('app', 'app.log')
        
        # 错误日志
        self.setup_logger('error', 'error.log', level=logging.ERROR)
        
        # 访问日志
        self.setup_logger('access', 'access.log')
        
        # 性能日志
        self.setup_logger('performance', 'performance.log')

    def setup_logger(self, name: str,
                    filename: str,
                    level: int = logging.INFO,
                    rotation: str = 'size') -> logging.Logger:
        """设置日志记录器"""
        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(f"{self.app_name}.{name}")
        logger.setLevel(level)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 设置控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 设置文件输出
        log_file = os.path.join(self.log_dir, filename)
        
        if rotation == 'size':
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
        else:  # time based rotation
            file_handler = TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=self.backup_count,
                encoding='utf-8'
            )

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        self.loggers[name] = logger
        return logger

    def get_logger(self, name: str) -> Optional[logging.Logger]:
        """获取日志记录器"""
        return self.loggers.get(name)

    def log_error(self, error: Exception,
                  context: Dict = None,
                  logger_name: str = 'error') -> None:
        """记录错误信息"""
        logger = self.get_logger(logger_name)
        if logger is None:
            return

        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }

        logger.error(json.dumps(error_info, ensure_ascii=False))

    def log_performance(self, operation: str,
                       execution_time: float,
                       details: Dict = None) -> None:
        """记录性能指标"""
        logger = self.get_logger('performance')
        if logger is None:
            return

        performance_info = {
            'operation': operation,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }

        logger.info(json.dumps(performance_info, ensure_ascii=False))

    def log_access(self, endpoint: str,
                   method: str,
                   status_code: int,
                   response_time: float,
                   user_id: str = None) -> None:
        """记录访问日志"""
        logger = self.get_logger('access')
        if logger is None:
            return

        access_info = {
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'response_time': response_time,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(json.dumps(access_info, ensure_ascii=False))

    def analyze_logs(self, logger_name: str,
                    start_time: datetime = None,
                    end_time: datetime = None) -> Dict:
        """分析日志数据"""
        log_file = os.path.join(self.log_dir, f"{logger_name}.log")
        if not os.path.exists(log_file):
            return {}

        analysis = {
            'total_entries': 0,
            'error_count': 0,
            'warning_count': 0,
            'info_count': 0,
            'level_distribution': {},
            'common_errors': [],
            'time_distribution': {}
        }

        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # 解析日志条目
                    if ' ERROR ' in line:
                        analysis['error_count'] += 1
                    elif ' WARNING ' in line:
                        analysis['warning_count'] += 1
                    elif ' INFO ' in line:
                        analysis['info_count'] += 1

                    analysis['total_entries'] += 1

                    # 提取时间信息
                    timestamp = line.split(' - ')[0]
                    hour = timestamp.split(' ')[1].split(':')[0]
                    analysis['time_distribution'][hour] = \
                        analysis['time_distribution'].get(hour, 0) + 1

                except Exception:
                    continue

        return analysis

    def cleanup_logs(self, days_to_keep: int = 30) -> None:
        """清理旧日志文件"""
        current_time = datetime.now()
        
        for file in os.listdir(self.log_dir):
            file_path = os.path.join(self.log_dir, file)
            file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            if (current_time - file_modified).days > days_to_keep:
                try:
                    os.remove(file_path)
                except Exception as e:
                    self.log_error(e, {'file': file_path})

    def export_logs(self, logger_name: str,
                   start_time: datetime = None,
                   end_time: datetime = None,
                   format: str = 'json') -> str:
        """导出日志数据"""
        log_file = os.path.join(self.log_dir, f"{logger_name}.log")
        if not os.path.exists(log_file):
            raise FileNotFoundError(f"日志文件不存在: {log_file}")

        export_data = []
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    timestamp = datetime.strptime(
                        line.split(' - ')[0],
                        '%Y-%m-%d %H:%M:%S,%f'
                    )
                    
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue
                        
                    export_data.append(line.strip())
                except Exception:
                    continue

        # 导出文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_file = os.path.join(
            self.log_dir,
            f"log_export_{logger_name}_{timestamp}.{format}"
        )

        if format == 'json':
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=4)
        else:  # text format
            with open(export_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(export_data))

        return export_file

    def get_log_summary(self) -> Dict:
        """获取日志概要信息"""
        summary = {
            'loggers': list(self.loggers.keys()),
            'log_files': {},
            'total_size': 0,
            'last_modified': {}
        }

        for logger_name in self.loggers:
            log_file = os.path.join(self.log_dir, f"{logger_name}.log")
            if os.path.exists(log_file):
                file_size = os.path.getsize(log_file)
                last_modified = datetime.fromtimestamp(
                    os.path.getmtime(log_file)
                ).isoformat()
                
                summary['log_files'][logger_name] = file_size
                summary['total_size'] += file_size
                summary['last_modified'][logger_name] = last_modified

        return summary