import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class DataVisualizer:
    def __init__(self):
        self.theme = 'plotly_white'
        self.color_palette = px.colors.qualitative.Set3
        self.default_height = 600
        self.default_width = 800

    def create_time_series(self, data: pd.DataFrame,
                          x_col: str,
                          y_col: str,
                          title: str = '',
                          color_col: str = None) -> go.Figure:
        """创建时间序列图"""
        fig = px.line(data, x=x_col, y=y_col, color=color_col,
                      title=title, template=self.theme)
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        return fig

    def create_bar_chart(self, data: pd.DataFrame,
                        x_col: str,
                        y_col: str,
                        title: str = '',
                        orientation: str = 'v') -> go.Figure:
        """创建柱状图"""
        fig = px.bar(data, x=x_col, y=y_col,
                    title=title, template=self.theme,
                    orientation=orientation)
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        return fig

    def create_pie_chart(self, data: pd.DataFrame,
                        names: str,
                        values: str,
                        title: str = '') -> go.Figure:
        """创建饼图"""
        fig = px.pie(data, names=names, values=values,
                    title=title, template=self.theme)
        fig.update_layout(
            height=self.default_height,
            width=self.default_width
        )
        return fig

    def create_scatter_plot(self, data: pd.DataFrame,
                           x_col: str,
                           y_col: str,
                           title: str = '',
                           color_col: str = None,
                           size_col: str = None) -> go.Figure:
        """创建散点图"""
        fig = px.scatter(data, x=x_col, y=y_col,
                        color=color_col, size=size_col,
                        title=title, template=self.theme)
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        return fig

    def create_heatmap(self, data: pd.DataFrame,
                      title: str = '') -> go.Figure:
        """创建热力图"""
        fig = px.imshow(data, template=self.theme,
                       title=title)
        fig.update_layout(
            height=self.default_height,
            width=self.default_width
        )
        return fig

    def create_box_plot(self, data: pd.DataFrame,
                       x_col: str,
                       y_col: str,
                       title: str = '') -> go.Figure:
        """创建箱线图"""
        fig = px.box(data, x=x_col, y=y_col,
                    title=title, template=self.theme)
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        return fig

    def create_histogram(self, data: pd.DataFrame,
                        x_col: str,
                        title: str = '',
                        nbins: int = 30) -> go.Figure:
        """创建直方图"""
        fig = px.histogram(data, x=x_col,
                          nbins=nbins,
                          title=title,
                          template=self.theme)
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            xaxis_title=x_col,
            yaxis_title='Count'
        )
        return fig

    def create_dashboard(self, charts: List[Dict]) -> go.Figure:
        """创建仪表板"""
        n_charts = len(charts)
        rows = (n_charts + 1) // 2
        cols = min(2, n_charts)
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[chart.get('title', '') for chart in charts]
        )

        for i, chart in enumerate(charts):
            row = i // 2 + 1
            col = i % 2 + 1
            
            if chart['type'] == 'line':
                trace = go.Scatter(
                    x=chart['data'][chart['x']],
                    y=chart['data'][chart['y']],
                    name=chart.get('name', '')
                )
            elif chart['type'] == 'bar':
                trace = go.Bar(
                    x=chart['data'][chart['x']],
                    y=chart['data'][chart['y']],
                    name=chart.get('name', '')
                )
            elif chart['type'] == 'pie':
                trace = go.Pie(
                    labels=chart['data'][chart['names']],
                    values=chart['data'][chart['values']],
                    name=chart.get('name', '')
                )
            else:
                continue

            fig.add_trace(trace, row=row, col=col)

        fig.update_layout(
            height=self.default_height * rows,
            width=self.default_width,
            template=self.theme,
            showlegend=True
        )
        return fig

    def create_trend_analysis(self, data: pd.DataFrame,
                            date_col: str,
                            value_col: str,
                            title: str = '') -> go.Figure:
        """创建趋势分析图"""
        # 计算移动平均
        data = data.sort_values(date_col)
        ma7 = data[value_col].rolling(window=7).mean()
        ma30 = data[value_col].rolling(window=30).mean()

        fig = go.Figure()
        
        # 添加原始数据
        fig.add_trace(go.Scatter(
            x=data[date_col],
            y=data[value_col],
            name='原始数据',
            mode='lines'
        ))
        
        # 添加7日移动平均
        fig.add_trace(go.Scatter(
            x=data[date_col],
            y=ma7,
            name='7日移动平均',
            line=dict(dash='dash')
        ))
        
        # 添加30日移动平均
        fig.add_trace(go.Scatter(
            x=data[date_col],
            y=ma30,
            name='30日移动平均',
            line=dict(dash='dot')
        ))

        fig.update_layout(
            title=title,
            height=self.default_height,
            width=self.default_width,
            template=self.theme,
            xaxis_title=date_col,
            yaxis_title=value_col
        )
        return fig

    def create_comparison_chart(self, data: pd.DataFrame,
                              category_col: str,
                              value_cols: List[str],
                              title: str = '') -> go.Figure:
        """创建对比图表"""
        fig = go.Figure()
        
        for col in value_cols:
            fig.add_trace(go.Bar(
                name=col,
                x=data[category_col],
                y=data[col]
            ))

        fig.update_layout(
            barmode='group',
            title=title,
            height=self.default_height,
            width=self.default_width,
            template=self.theme,
            xaxis_title=category_col,
            yaxis_title='Value'
        )
        return fig

    def save_figure(self, fig: go.Figure,
                   filepath: str,
                   format: str = 'html') -> None:
        """保存图表"""
        if format == 'html':
            fig.write_html(filepath)
        elif format == 'png':
            fig.write_image(filepath)
        elif format == 'json':
            fig.write_json(filepath)

    def create_report(self, data_dict: Dict[str, pd.DataFrame],
                     report_config: List[Dict],
                     output_path: str) -> None:
        """生成可视化报告"""
        figures = []
        
        for chart_config in report_config:
            data = data_dict[chart_config['data_source']]
            
            if chart_config['type'] == 'time_series':
                fig = self.create_time_series(
                    data,
                    chart_config['x_col'],
                    chart_config['y_col'],
                    chart_config.get('title', '')
                )
            elif chart_config['type'] == 'bar':
                fig = self.create_bar_chart(
                    data,
                    chart_config['x_col'],
                    chart_config['y_col'],
                    chart_config.get('title', '')
                )
            elif chart_config['type'] == 'pie':
                fig = self.create_pie_chart(
                    data,
                    chart_config['names'],
                    chart_config['values'],
                    chart_config.get('title', '')
                )
            else:
                continue
                
            figures.append(fig)

        # 创建HTML报告
        html_content = """
        <html>
        <head>
            <title>数据分析报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .chart-container { margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <h1>数据分析报告</h1>
            <p>生成时间: {}</p>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        for i, fig in enumerate(figures):
            chart_div = f'chart_{i}'
            html_content += f'<div class="chart-container" id="{chart_div}"></div>'
            fig.write_html(f'{output_path}/{chart_div}.html')

        html_content += """
        </body>
        </html>
        """

        with open(f'{output_path}/report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)