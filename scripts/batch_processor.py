import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
from tqdm import tqdm
import logging

class BatchProcessor:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers
        self._setup_logging()

    def _setup_logging(self) -> None:
        """配置日志"""
        logging.basicConfig(
            filename='batch_processor.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def process_file(self, filepath: str,
                    processor_func: callable,
                    **kwargs) -> Optional[pd.DataFrame]:
        """处理单个文件"""
        try:
            df = pd.read_csv(filepath)
            result = processor_func(df, **kwargs)
            logging.info(f"Successfully processed file: {filepath}")
            return result
        except Exception as e:
            logging.error(f"Error processing file {filepath}: {str(e)}")
            return None

    def process_directory(self, directory: str,
                         processor_func: callable,
                         file_pattern: str = '*.csv',
                         parallel: bool = True,
                         **kwargs) -> Dict[str, pd.DataFrame]:
        """处理目录下的所有文件"""
        import glob
        
        files = glob.glob(os.path.join(directory, file_pattern))
        results = {}
        
        if parallel:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.process_file, f, processor_func, **kwargs): f
                    for f in files
                }
                
                for future in tqdm(futures, desc="Processing files"):
                    filepath = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results[filepath] = result
                    except Exception as e:
                        logging.error(f"Error in file {filepath}: {str(e)}")
        else:
            for filepath in tqdm(files, desc="Processing files"):
                result = self.process_file(filepath, processor_func, **kwargs)
                if result is not None:
                    results[filepath] = result
        
        return results

    def save_results(self, results: Dict[str, pd.DataFrame],
                    output_dir: str,
                    prefix: str = 'processed_') -> None:
        """保存处理结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        for filepath, df in results.items():
            filename = os.path.basename(filepath)
            output_path = os.path.join(output_dir, f"{prefix}{filename}")
            df.to_csv(output_path, index=False)
            logging.info(f"Saved processed file to: {output_path}")

    def merge_results(self, results: Dict[str, pd.DataFrame],
                     merge_on: Optional[str] = None) -> pd.DataFrame:
        """合并处理结果"""
        if not results:
            return pd.DataFrame()
            
        if merge_on is None:
            return pd.concat(results.values(), ignore_index=True)
        
        merged = list(results.values())[0]
        for df in list(results.values())[1:]:
            merged = merged.merge(df, on=merge_on, how='outer')
        
        return merged