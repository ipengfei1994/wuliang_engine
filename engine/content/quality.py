import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import json
from collections import Counter

class ContentQualityAnalyzer:
    def __init__(self):
        self.content_data = None
        self.interaction_data = None
        self.quality_thresholds = {
            'min_length': 100,
            'min_engagement_rate': 0.02,
            'min_completion_rate': 0.3
        }

    def load_data(self, content_data: pd.DataFrame = None,
                 interaction_data: pd.DataFrame = None) -> None:
        """加载内容数据"""
        if content_data is not None:
            self.content_data = content_data.copy()
        if interaction_data is not None:
            self.interaction_data = interaction_data.copy()

    def analyze_content_quality(self) -> Dict:
        """分析内容质量"""
        if self.content_data is None:
            raise ValueError("请先加载内容数据")

        quality_metrics = {
            'readability': self._analyze_readability(),
            'originality': self._analyze_originality(),
            'engagement': self._analyze_engagement(),
            'completion_rate': self._analyze_completion_rate(),
            'content_depth': self._analyze_content_depth()
        }

        return quality_metrics

    def _analyze_readability(self) -> Dict:
        """分析可读性"""
        if 'content' not in self.content_data.columns:
            return {}

        readability_scores = []
        sentence_lengths = []
        word_counts = []

        for content in self.content_data['content']:
            words = list(jieba.cut(str(content)))
            sentences = content.split('。')
            
            word_counts.append(len(words))
            sentence_lengths.append(np.mean([len(s) for s in sentences]))
            
            # 计算可读性得分
            score = 100 - (len(words) * 0.1 + np.mean([len(s) for s in sentences]) * 0.5)
            readability_scores.append(max(0, min(100, score)))

        return {
            'avg_readability_score': float(np.mean(readability_scores)),
            'avg_sentence_length': float(np.mean(sentence_lengths)),
            'avg_word_count': float(np.mean(word_counts))
        }

    def _analyze_originality(self) -> Dict:
        """分析原创性"""
        if 'content' not in self.content_data.columns:
            return {}

        # 使用TF-IDF分析文本相似度
        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: list(jieba.cut(x)),
            max_features=1000
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(self.content_data['content'])
            unique_words = len(vectorizer.vocabulary_)
            
            # 计算文本相似度
            similarity_scores = []
            for i in range(len(self.content_data)):
                similarities = (tfidf_matrix[i] * tfidf_matrix.T).toarray()[0]
                # 排除自身的相似度
                similarities = np.delete(similarities, i)
                similarity_scores.append(np.max(similarities) if len(similarities) > 0 else 0)

            return {
                'unique_words_ratio': float(unique_words / len(self.content_data)),
                'avg_similarity_score': float(np.mean(similarity_scores)),
                'high_similarity_count': int(sum(np.array(similarity_scores) > 0.8))
            }
        except Exception as e:
            return {'error': str(e)}

    def _analyze_engagement(self) -> Dict:
        """分析互动效果"""
        if self.interaction_data is None:
            return {}

        engagement_metrics = self.interaction_data.groupby('content_id').agg({
            'likes': 'sum',
            'comments': 'sum',
            'shares': 'sum',
            'views': 'sum'
        })

        engagement_metrics['engagement_rate'] = (
            (engagement_metrics['likes'] + 
             engagement_metrics['comments'] * 2 + 
             engagement_metrics['shares'] * 3) /
            engagement_metrics['views']
        ).fillna(0)

        return {
            'avg_engagement_rate': float(engagement_metrics['engagement_rate'].mean()),
            'high_engagement_ratio': float(
                (engagement_metrics['engagement_rate'] > 
                 self.quality_thresholds['min_engagement_rate']).mean()
            ),
            'engagement_distribution': engagement_metrics['engagement_rate'].describe().to_dict()
        }

    def _analyze_completion_rate(self) -> Dict:
        """分析完成率"""
        if 'read_duration' not in self.interaction_data.columns:
            return {}

        completion_rates = (
            self.interaction_data['read_duration'] / 
            self.interaction_data['expected_duration']
        ).fillna(0)

        return {
            'avg_completion_rate': float(completion_rates.mean()),
            'high_completion_ratio': float(
                (completion_rates > self.quality_thresholds['min_completion_rate']).mean()
            ),
            'completion_distribution': completion_rates.describe().to_dict()
        }

    def _analyze_content_depth(self) -> Dict:
        """分析内容深度"""
        if 'content' not in self.content_data.columns:
            return {}

        # 计算内容深度指标
        depth_scores = []
        for content in self.content_data['content']:
            words = list(jieba.cut(str(content)))
            
            # 计算专业词汇比例
            professional_words = self._identify_professional_words(words)
            
            # 计算深度得分
            depth_score = (
                len(professional_words) / len(words) * 0.4 +
                (len(words) / 500) * 0.3 +
                (len(content.split('。')) / 10) * 0.3
            )
            depth_scores.append(min(1.0, depth_score))

        return {
            'avg_depth_score': float(np.mean(depth_scores)),
            'depth_distribution': pd.Series(depth_scores).describe().to_dict(),
            'high_depth_ratio': float((np.array(depth_scores) > 0.6).mean())
        }

    def _identify_professional_words(self, words: List[str]) -> List[str]:
        """识别专业词汇"""
        # 这里可以添加专业词汇库的判断逻辑
        # 当前使用词长作为简单判断标准
        return [w for w in words if len(w) > 2]

    def identify_quality_issues(self) -> List[Dict]:
        """识别质量问题"""
        if self.content_data is None:
            return []

        issues = []
        quality_metrics = self.analyze_content_quality()

        # 检查可读性问题
        if quality_metrics['readability']['avg_readability_score'] < 60:
            issues.append({
                'type': '可读性问题',
                'severity': 'high',
                'description': '内容可读性较差，建议优化句子结构和用词',
                'affected_content': self._get_low_readability_content()
            })

        # 检查原创性问题
        if quality_metrics['originality']['avg_similarity_score'] > 0.8:
            issues.append({
                'type': '原创性问题',
                'severity': 'high',
                'description': '存在内容重复或相似度过高的情况',
                'affected_content': self._get_similar_content()
            })

        # 检查互动问题
        if quality_metrics['engagement']['avg_engagement_rate'] < self.quality_thresholds['min_engagement_rate']:
            issues.append({
                'type': '互动性问题',
                'severity': 'medium',
                'description': '内容互动率较低，需要提升内容吸引力',
                'affected_content': self._get_low_engagement_content()
            })

        return issues

    def generate_quality_report(self) -> Dict:
        """生成质量报告"""
        report = {
            'quality_metrics': self.analyze_content_quality(),
            'quality_issues': self.identify_quality_issues(),
            'recommendations': self._generate_recommendations(),
            'timestamp': datetime.now().isoformat()
        }
        return report

    def _generate_recommendations(self) -> List[Dict]:
        """生成改进建议"""
        recommendations = []
        quality_metrics = self.analyze_content_quality()

        # 基于可读性的建议
        if quality_metrics['readability']['avg_readability_score'] < 60:
            recommendations.append({
                'area': '内容可读性',
                'suggestion': '简化句子结构，使用更通俗的表达',
                'expected_impact': '提升阅读体验和完成率'
            })

        # 基于互动率的建议
        if quality_metrics['engagement']['avg_engagement_rate'] < self.quality_thresholds['min_engagement_rate']:
            recommendations.append({
                'area': '内容互动性',
                'suggestion': '增加互动元素，优化内容结构',
                'expected_impact': '提升用户参与度'
            })

        # 基于内容深度的建议
        if quality_metrics['content_depth']['avg_depth_score'] < 0.5:
            recommendations.append({
                'area': '内容深度',
                'suggestion': '增加专业观点和数据支持',
                'expected_impact': '提升内容价值和专业性'
            })

        return recommendations

    def export_report(self, filepath: str) -> None:
        """导出分析报告"""
        report = self.generate_quality_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)