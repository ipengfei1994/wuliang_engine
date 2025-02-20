import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import typing as tp
import plotly.express as px
import pandas as pd
from typing import Optional

class DataVisualizer:
    def __init__(self, style: str = 'darkgrid'):
        """初始化数据可视化器"""
        sns.set_style(style)
        self.default_figsize = (12, 6)
        self.color_palette = sns.color_palette("husl", 8)

    def plot_time_series(self, data: pd.DataFrame,
                        x_col: str,
                        y_col: str,
                        title: str = None,
                        interactive: bool = False) -> None:
        """时间序列可视化"""
        if interactive:
            fig = px.line(data, x=x_col, y=y_col, title=title)
            fig.show()
        else:
            plt.figure(figsize=self.default_figsize)
            plt.plot(data[x_col], data[y_col])
            plt.title(title or f'{y_col} over {x_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def plot_distribution(self, data: pd.Series,
                         title: str = None,
                         bins: int = 30,
                         kde: bool = True) -> None:
        """分布可视化"""
        plt.figure(figsize=self.default_figsize)
        sns.histplot(data=data, bins=bins, kde=kde)
        plt.title(title or f'Distribution of {data.name}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, data: pd.DataFrame,
                              method: str = 'pearson',
                              figsize: Tuple[int, int] = (10, 8)) -> None:
        """相关性矩阵可视化"""
        corr_matrix = data.corr(method=method)
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, 
                   cmap='coolwarm', center=0, square=True)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

    def plot_scatter_matrix(self, data: pd.DataFrame,
                          columns: List[str] = None) -> None:
        """散点矩阵可视化"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        sns.pairplot(data[columns], diag_kind='kde')
        plt.tight_layout()
        plt.show()

    def plot_box_plots(self, data: pd.DataFrame,
                      numeric_cols: List[str] = None,
                      figsize: Tuple[int, int] = None) -> None:
        """箱线图可视化"""
        if numeric_cols is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        n_cols = len(numeric_cols)
        figsize = figsize or (4 * n_cols, 6)
        
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        if n_cols == 1:
            axes = [axes]
        
        for ax, col in zip(axes, numeric_cols):
            sns.boxplot(y=data[col], ax=ax)
            ax.set_title(f'Distribution of {col}')
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_pie_chart(self, data: pd.Series,
                      title: str = None,
                      interactive: bool = False) -> None:
        """饼图可视化"""
        if interactive:
            fig = px.pie(values=data.values, names=data.index, title=title)
            fig.show()
        else:
            plt.figure(figsize=(10, 10))
            plt.pie(data.values, labels=data.index, autopct='%1.1f%%')
            plt.title(title or f'Distribution of {data.name}')
            plt.axis('equal')
            plt.show()

    def plot_bar_chart(self, data: pd.Series,
                      title: str = None,
                      orientation: str = 'vertical',
                      interactive: bool = False) -> None:
        """柱状图可视化"""
        if interactive:
            fig = px.bar(x=data.index, y=data.values, title=title)
            fig.show()
        else:
            plt.figure(figsize=self.default_figsize)
            if orientation == 'vertical':
                plt.bar(data.index, data.values)
                plt.xticks(rotation=45)
            else:
                plt.barh(data.index, data.values)
            
            plt.title(title or f'Bar Chart of {data.name}')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def plot_3d_scatter(self, data: pd.DataFrame,
                       x_col: str, y_col: str, z_col: str,
                       color_col: Optional[str] = None) -> None:
        """3D散点图可视化"""
        fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col,
                           color=color_col)
        fig.show()

    def create_dashboard(self, plots: List[Dict]) -> None:
        """创建交互式仪表板"""
        n_plots = len(plots)
        rows = (n_plots + 1) // 2
        cols = min(n_plots, 2)
        
        fig = make_subplots(rows=rows, cols=cols,
                           subplot_titles=[p.get('title', '') for p in plots])
        
        for i, plot in enumerate(plots, 1):
            row = (i - 1) // 2 + 1
            col = (i - 1) % 2 + 1
            
            if plot['type'] == 'scatter':
                trace = go.Scatter(x=plot['x'], y=plot['y'],
                                 name=plot.get('name', ''))
            elif plot['type'] == 'bar':
                trace = go.Bar(x=plot['x'], y=plot['y'],
                             name=plot.get('name', ''))
            elif plot['type'] == 'pie':
                trace = go.Pie(values=plot['values'], labels=plot['labels'],
                             name=plot.get('name', ''))
            
            fig.add_trace(trace, row=row, col=col)
        
        fig.update_layout(height=400*rows, showlegend=True)
        fig.show()

    def save_plot(self, filepath: str, dpi: int = 300) -> None:
        """保存当前图表"""
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')