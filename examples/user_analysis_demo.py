import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from engine.user.behavior import UserBehaviorAnalyzer

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from engine.user.behavior import UserBehaviorAnalyzer
import matplotlib.font_manager as fm
import os
import sys

# 设置中文字体和样式配置
plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 设置绘图样式
sns.set_style('whitegrid')
plt.style.use('seaborn-v0_8-whitegrid')


# 创建分析器实例
analyzer = UserBehaviorAnalyzer()

# 创建示例数据
data = {
    'user_id': [],
    'timestamp': [],
    'action': []
}

# 生成1000个用户的行为数据
import random
from datetime import datetime, timedelta

# 定义用户类型及其比例
user_types = {
    'high_active': 0.1,    # 高活跃用户 10%
    'medium_active': 0.3,   # 中等活跃用户 30%
    'low_active': 0.35,     # 低活跃用户 35%
    'churned': 0.15,        # 流失用户 15%
    'returned': 0.1         # 回归用户 10%
}

# 定义可能的用户行为
actions = ['浏览', '加入购物车', '购买', '评论', '分享', '点赞', '收藏']

# 设定时间范围
start_date = datetime(2021, 1, 1)
end_date = datetime(2024, 1, 31)

# 为每种类型的用户生成数据
for user_type, ratio in user_types.items():
    num_users = int(1000 * ratio)
    
    for user_idx in range(num_users):
        user_id = f'user{user_idx + 1}'
        
        # 根据用户类型确定行为频率和时间分布
        if user_type == 'high_active':
            num_actions = random.randint(100, 150)
            active_period = (end_date - start_date).days
        elif user_type == 'medium_active':
            num_actions = random.randint(50, 80)
            active_period = (end_date - start_date).days
        elif user_type == 'low_active':
            num_actions = random.randint(20, 40)
            active_period = random.randint(180, 365)
        elif user_type == 'churned':
            num_actions = random.randint(30, 50)
            active_period = random.randint(90, 180)
        else:  # returned users
            num_actions = random.randint(40, 60)
            active_period = (end_date - start_date).days
        
        # 生成用户行为序列
        for _ in range(num_actions):
            if user_type == 'churned':
                # 流失用户只在前期活跃
                action_date = start_date + timedelta(days=random.randint(0, active_period))
            elif user_type == 'returned':
                # 回归用户有两个活跃期
                if random.random() < 0.3:
                    # 早期活跃
                    action_date = start_date + timedelta(days=random.randint(0, 180))
                else:
                    # 晚期回归
                    action_date = end_date - timedelta(days=random.randint(0, 180))
            else:
                # 其他用户在其活跃期内随机分布
                action_date = start_date + timedelta(days=random.randint(0, active_period))
            
            # 根据用户类型设定行为权重
            if user_type in ['high_active', 'medium_active']:
                # 高活跃和中活跃用户更可能进行深度交互
                action_weights = [0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1]
            else:
                # 其他用户以浏览为主
                action_weights = [0.4, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05]
            
            action = random.choices(actions, weights=action_weights)[0]
            
            # 添加数据
            data['user_id'].append(user_id)
            data['timestamp'].append(action_date.strftime('%Y-%m-%d %H:%M:%S'))
            data['action'].append(action)

# 转换为DataFrame并按时间排序
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# 加载数据
analyzer.load_data(df)

# 验证字体设置
try:
    plt.rcParams['font.family']
except Exception as e:
    print(f"警告：字体设置失败：{str(e)}")
    # 如果设置失败，尝试使用备选方案
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

# 创建一个统一的可视化界面
fig = plt.figure(figsize=(20, 15))

# 1. 用户活动趋势图 (左上)
plt.subplot(2, 2, 1)
activity = df.set_index('timestamp').resample('D').size()
activity.plot(linewidth=2)
plt.title('用户活动趋势', fontsize=14, pad=20)
plt.xlabel('时间', fontsize=12)
plt.ylabel('活动次数', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# 2. 用户行为分布图 (右上)
plt.subplot(2, 2, 2)
action_counts = df['action'].value_counts()
plt.pie(action_counts, labels=action_counts.index, autopct='%1.1f%%', textprops={'fontsize': 10})
plt.title('用户行为分布', fontsize=14)

# 3. 留存热图 (左下)
plt.subplot(2, 2, 3)
retention_data = analyzer.get_user_retention()
retention_matrix = retention_data.pivot(
    index='cohort_date',
    columns='day',
    values='retention_rate'
)
sns.heatmap(retention_matrix, 
            annot=True, 
            fmt='.0%', 
            cmap='YlOrRd',
            cbar_kws={'label': '留存率'},
            annot_kws={'size': 8})
plt.title('用户留存分析', fontsize=14)
plt.xlabel('留存天数', fontsize=12)
plt.ylabel('用户群时间', fontsize=12)

# 4. 用户分群结果 (右下)
plt.subplot(2, 2, 4)
user_metrics = analyzer.calculate_user_metrics()
segments = analyzer.segment_users(n_clusters=2)

# 使用散点图展示分群结果，增加可读性
colors = ['#FF9999', '#66B2FF']
for i, (cluster_id, users) in enumerate(segments.items()):
    cluster_metrics = user_metrics.loc[users]
    plt.scatter(cluster_metrics['total_actions'].tolist(), 
                cluster_metrics['active_days'].tolist(),
                c=colors[i],
                alpha=0.6,
                s=100,
                label=f'群组 {cluster_id}')

plt.xlabel('总活动次数', fontsize=12)
plt.ylabel('活跃天数', fontsize=12)
plt.title('用户分群分析', fontsize=14)
plt.legend(fontsize=10, title='用户群组', title_fontsize=12)
plt.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()

# 显示整体标题
fig.suptitle('用户行为分析报告', fontsize=16, y=1.02)

# 保存图表
plt.savefig('用户分析结果.png', bbox_inches='tight', dpi=300)

# 导出详细分析结果
analyzer.export_analysis('用户分析结果.json')

# 显示图表
plt.show()

# 打印关键指标
print('\n=== 用户行为分析报告 ===')
print('\n1. 用户活跃度指标：')
print(f'总用户数：{len(df["user_id"].unique())}')
print(f'平均每日活动次数：{df.groupby(df["timestamp"].dt.date).size().mean():.2f}')

print('\n2. 用户分群结果：')
for cluster_id, users in segments.items():
    cluster_metrics = user_metrics.loc[users]
    print(f'群组 {cluster_id}：{len(users)} 用户')
    print(f'  - 平均活动次数：{cluster_metrics["total_actions"].mean():.1f}')
    print(f'  - 平均活跃天数：{cluster_metrics["active_days"].mean():.1f}')

print('\n3. 用户行为分布：')
for action, count in action_counts.items():
    print(f'{action}: {count} ({count/len(df)*100:.1f}%)')