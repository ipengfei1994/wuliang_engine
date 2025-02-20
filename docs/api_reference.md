# 无量引擎 API 参考文档

## 用户行为分析模块

### 用户行为分析

```python
from engine.user.behavior import UserBehaviorAnalyzer

# 初始化分析器
analyzer = UserBehaviorAnalyzer()

# 分析单个用户行为
user_behavior = analyzer.analyze_behavior("user_id")

# 获取用户分群结果
user_segments = analyzer.get_segments()
```

### 用户增长分析

```python
from engine.user.growth import UserGrowthAnalyzer

# 初始化分析器
growth_analyzer = UserGrowthAnalyzer()

# 分析用户增长趋势
growth_trends = growth_analyzer.analyze_growth_trends()
```

## 内容分析模块

### 内容质量评估

```python
from engine.content.quality import ContentQualityAnalyzer

# 初始化分析器
quality_analyzer = ContentQualityAnalyzer()

# 评估内容质量
quality_score = quality_analyzer.evaluate_content(content_id)
```

### 竞品分析

```python
from engine.content.competitor import CompetitorAnalyzer

# 初始化分析器
competitor_analyzer = CompetitorAnalyzer()

# 分析竞品内容
competitor_insights = competitor_analyzer.analyze_competitor(competitor_id)
```

## 社交网络分析模块

### 社区分析

```python
from engine.social.community import CommunityAnalyzer

# 初始化分析器
community_analyzer = CommunityAnalyzer()

# 分析社区结构
community_structure = community_analyzer.analyze_community()
```

### 情感分析

```python
from engine.social.sentiment import SentimentAnalyzer

# 初始化分析器
sentiment_analyzer = SentimentAnalyzer()

# 分析文本情感
sentiment = sentiment_analyzer.analyze_sentiment(text)
```

## 运营分析模块

### 运营效率分析

```python
from engine.operation.efficiency import OperationalEfficiencyAnalyzer

# 初始化分析器
efficiency_analyzer = OperationalEfficiencyAnalyzer()

# 分析运营效率
efficiency_metrics = efficiency_analyzer.analyze_efficiency()
```

### 广告效果分析

```python
from engine.operation.ad_performance import AdPerformanceAnalyzer

# 初始化分析器
ad_analyzer = AdPerformanceAnalyzer()

# 分析广告效果
ad_performance = ad_analyzer.analyze_ad_performance(ad_id)
```

## 数据导入导出

### 数据导出

```python
from utils.data.user_report_exporter import UserReportExporter

# 初始化导出器
exporter = UserReportExporter()

# 导出用户分析报告
exporter.export_user_report(user_id, format='pdf')
```

## 数据可视化

### 图表生成

```python
from visualization.charts import ChartGenerator

# 初始化图表生成器
chart_generator = ChartGenerator()

# 生成用户行为趋势图
chart_generator.plot_user_behavior_trend(user_id)
```

### 仪表盘

```python
from visualization.dashboard import Dashboard

# 初始化仪表盘
dashboard = Dashboard()

# 生成运营分析仪表盘
dashboard.generate_operation_dashboard()
```