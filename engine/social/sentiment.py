import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Union, Optional
import pandas as pd
from tqdm import tqdm

class SentimentAnalyzer:
    def __init__(self, model_name: str = "bert-base-chinese"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def analyze_text(self, text: str) -> Dict[str, float]:
        """分析单条文本的情感"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            
        sentiment_scores = {
            'positive': float(probabilities[0][1]),
            'negative': float(probabilities[0][0])
        }
        return sentiment_scores

    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """批量分析文本情感"""
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                  max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
            
            batch_results = [
                {
                    'positive': float(prob[1]),
                    'negative': float(prob[0])
                }
                for prob in probabilities
            ]
            results.extend(batch_results)
        
        return results

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str, 
                         batch_size: int = 32) -> pd.DataFrame:
        """分析DataFrame中的文本列"""
        texts = df[text_column].tolist()
        sentiment_results = self.analyze_batch(texts, batch_size)
        
        # 添加情感分析结果到DataFrame
        df['sentiment_positive'] = [result['positive'] for result in sentiment_results]
        df['sentiment_negative'] = [result['negative'] for result in sentiment_results]
        df['sentiment_label'] = df.apply(
            lambda x: 'positive' if x['sentiment_positive'] > x['sentiment_negative'] else 'negative',
            axis=1
        )
        
        return df

    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict[str, Union[float, dict]]:
        """获取情感分析摘要"""
        if 'sentiment_label' not in df.columns:
            raise ValueError("DataFrame must contain sentiment analysis results")
            
        summary = {
            'positive_ratio': float((df['sentiment_label'] == 'positive').mean()),
            'negative_ratio': float((df['sentiment_label'] == 'negative').mean()),
            'average_scores': {
                'positive': float(df['sentiment_positive'].mean()),
                'negative': float(df['sentiment_negative'].mean())
            }
        }
        return summary

    def save_model(self, filepath: str) -> None:
        """保存模型"""
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)

    def load_model(self, filepath: str) -> None:
        """加载模型"""
        self.model = AutoModelForSequenceClassification.from_pretrained(filepath).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(filepath)