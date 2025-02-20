import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

class UserReportExporter:
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def export_user_report(self, report_data: Dict, include_charts: bool = True) -> str:
        """导出用户分析报告
        
        Args:
            report_data: 包含用户分析数据的字典
            include_charts: 是否包含图表

        Returns:
            str: 报告保存的目录路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = os.path.join(self.output_dir, f"user_analysis_report_{timestamp}")
        Path(report_dir).mkdir(parents=True, exist_ok=True)

        # 导出JSON格式报告
        json_path = os.path.join(report_dir, "user_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=4)

        # 导出Excel格式报告
        excel_path = os.path.join(report_dir, "user_analysis.xlsx")
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            # 用户行为指标
            if 'user_metrics' in report_data:
                pd.DataFrame(report_data['user_metrics']).to_excel(
                    writer, sheet_name='用户行为指标', index=False
                )

            # 用户分群结果
            if 'user_segments' in report_data:
                pd.DataFrame(report_data['user_segments']).to_excel(
                    writer, sheet_name='用户分群', index=False
                )

            # 留存分析
            if 'retention_analysis' in report_data:
                pd.DataFrame(report_data['retention_analysis']).to_excel(
                    writer, sheet_name='留存分析', index=False
                )

            # 增长指标
            if 'growth_metrics' in report_data:
                pd.DataFrame([report_data['growth_metrics']]).to_excel(
                    writer, sheet_name='增长指标', index=False
                )

            # 增长机会
            if 'opportunities' in report_data:
                pd.DataFrame(report_data['opportunities']).to_excel(
                    writer, sheet_name='增长机会', index=False
                )

        # 导出PDF格式报告
        pdf_path = os.path.join(report_dir, "user_analysis.pdf")
        self._generate_pdf_report(report_data, pdf_path)

        return report_dir

    def _generate_pdf_report(self, report_data: Dict, output_path: str) -> None:
        """生成PDF格式的用户分析报告

        Args:
            report_data: 包含用户分析数据的字典
            output_path: PDF文件保存路径
        """
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        # 添加标题
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        elements.append(Paragraph("用户分析报告", title_style))
        elements.append(Spacer(1, 20))

        # 添加各个部分的内容
        sections = [
            ('用户行为指标', 'user_metrics'),
            ('用户分群', 'user_segments'),
            ('留存分析', 'retention_analysis'),
            ('增长指标', 'growth_metrics'),
            ('增长机会', 'opportunities')
        ]

        for section_title, section_key in sections:
            if section_key in report_data:
                # 添加小节标题
                elements.append(Paragraph(section_title, styles['Heading2']))
                elements.append(Spacer(1, 12))

                # 将数据转换为DataFrame并创建表格
                if section_key == 'growth_metrics':
                    df = pd.DataFrame([report_data[section_key]])
                else:
                    df = pd.DataFrame(report_data[section_key])

                # 创建表格数据
                table_data = [df.columns.tolist()] + df.values.tolist()
                table = Table(table_data, repeatRows=1)

                # 设置表格样式
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))

                elements.append(table)
                elements.append(Spacer(1, 20))

        # 生成PDF文件
        doc.build(elements)