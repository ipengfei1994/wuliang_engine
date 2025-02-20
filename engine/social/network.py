import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from community import community_louvain

class SocialNetworkAnalyzer:
    def __init__(self):
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()
        self.communities = None

    def build_network(self, interactions_df: pd.DataFrame,
                     source_col: str = 'from_user',
                     target_col: str = 'to_user',
                     weight_col: Optional[str] = None) -> None:
        """构建社交网络"""
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()
        
        for _, row in interactions_df.iterrows():
            source = row[source_col]
            target = row[target_col]
            weight = row[weight_col] if weight_col else 1.0
            
            self.graph.add_edge(source, target, weight=weight)
            self.directed_graph.add_edge(source, target, weight=weight)

    def get_centrality_metrics(self) -> Dict[str, Dict]:
        """计算中心性指标"""
        metrics = {
            'degree': nx.degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph),
            'closeness': nx.closeness_centrality(self.graph),
            'eigenvector': nx.eigenvector_centrality(self.graph, max_iter=1000)
        }
        return metrics

    def detect_communities(self, resolution: float = 1.0) -> Dict:
        """检测社区"""
        self.communities = community_louvain.best_partition(
            self.graph, resolution=resolution
        )
        return self.communities

    def get_influential_users(self, top_n: int = 10) -> pd.DataFrame:
        """识别有影响力的用户"""
        metrics = self.get_centrality_metrics()
        
        df = pd.DataFrame({
            'user': list(self.graph.nodes()),
            'degree': list(metrics['degree'].values()),
            'betweenness': list(metrics['betweenness'].values()),
            'closeness': list(metrics['closeness'].values()),
            'eigenvector': list(metrics['eigenvector'].values())
        })
        
        df['influence_score'] = df[['degree', 'betweenness', 'closeness', 'eigenvector']].mean(axis=1)
        return df.nlargest(top_n, 'influence_score')

    def analyze_network_structure(self) -> Dict:
        """分析网络结构"""
        return {
            'density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
            'average_shortest_path': nx.average_shortest_path_length(self.graph),
            'diameter': nx.diameter(self.graph),
            'number_of_nodes': self.graph.number_of_nodes(),
            'number_of_edges': self.graph.number_of_edges()
        }

    def get_user_recommendations(self, user_id: str, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """基于网络结构的用户推荐"""
        if user_id not in self.graph:
            return []
            
        # 计算与目标用户的相似度
        similarities = []
        for node in self.graph.nodes():
            if node != user_id:
                # 计算共同邻居数
                common_neighbors = list(nx.common_neighbors(self.graph, user_id, node))
                # 计算Jaccard系数
                jaccard = len(common_neighbors) / len(set(self.graph[user_id]).union(self.graph[node]))
                similarities.append((node, jaccard))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:n_recommendations]

    def visualize_network(self, figsize: Tuple[int, int] = (12, 8),
                         with_communities: bool = False) -> None:
        """可视化社交网络"""
        plt.figure(figsize=figsize)
        
        if with_communities and self.communities:
            colors = [self.communities[node] for node in self.graph.nodes()]
            nx.draw(self.graph, node_color=colors, with_labels=True,
                   node_size=500, font_size=8, cmap=plt.cm.rainbow)
        else:
            nx.draw(self.graph, with_labels=True, node_size=500, font_size=8)
        
        plt.title("Social Network Visualization")
        plt.show()

    def export_network_data(self, filepath: str) -> None:
        """导出网络数据"""
        data = {
            'nodes': list(self.graph.nodes()),
            'edges': list(self.graph.edges()),
            'metrics': self.get_centrality_metrics(),
            'communities': self.communities if self.communities else {},
            'structure': self.analyze_network_structure()
        }
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)