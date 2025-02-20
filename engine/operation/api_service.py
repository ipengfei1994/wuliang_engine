from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Union
import pandas as pd
import json
from datetime import datetime
import uvicorn
from pathlib import Path

from engine.social.analyzer import SocialMediaAnalyzer
from engine.content.competitor import CompetitorAnalyzer
from engine.user.growth import UserGrowthAnalyzer
from engine.social.community import CommunityAnalyzer
from engine.content.quality import ContentQualityAnalyzer
from engine.operation.efficiency import OperationalEfficiencyAnalyzer
from visualization.data_visualizer import DataVisualizer

class APIService:
    def __init__(self):
        self.app = FastAPI(
            title="五粮引擎API",
            description="社交媒体数据分析引擎API接口",
            version="1.0.0"
        )
        
        # 实例化分析器
        self.social_analyzer = SocialMediaAnalyzer()
        self.competitor_analyzer = CompetitorAnalyzer()
        self.user_analyzer = UserGrowthAnalyzer()
        self.community_analyzer = CommunityAnalyzer()
        self.content_analyzer = ContentQualityAnalyzer()
        self.efficiency_analyzer = OperationalEfficiencyAnalyzer()
        self.visualizer = DataVisualizer()
        
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.post("/api/analyze", response_model=AnalysisResponse)
        async def analyze_data(request: AnalysisRequest):
            """执行数据分析"""
            try:
                result = {}
                
                if request.data_type == "social_media":
                    result = self.social_analyzer.analyze_content_performance()
                elif request.data_type == "competitor":
                    result = self.competitor_analyzer.analyze_market_share()
                elif request.data_type == "user_growth":
                    result = self.user_analyzer.analyze_user_acquisition()
                elif request.data_type == "community":
                    result = self.community_analyzer.analyze_community_health()
                elif request.data_type == "content":
                    result = self.content_analyzer.analyze_content_quality()
                elif request.data_type == "efficiency":
                    result = self.efficiency_analyzer.analyze_task_efficiency()
                else:
                    raise HTTPException(status_code=400, detail="不支持的数据类型")

                return AnalysisResponse(
                    status="success",
                    data=result,
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/{data_type}")
async def upload_data(data_type: str, file: UploadFile = File(...)):
    """上传数据文件"""
    try:
        content = await file.read()
        df = pd.read_csv(file.file)
        
        if data_type == "social_media":
            social_analyzer.load_data(content_data=df)
        elif data_type == "competitor":
            competitor_analyzer.load_data(competitor_data=df)
        elif data_type == "user_growth":
            user_analyzer.load_data(user_data=df)
        elif data_type == "community":
            community_analyzer.load_data(community_data=df)
        elif data_type == "content":
            content_analyzer.load_data(content_data=df)
        elif data_type == "efficiency":
            efficiency_analyzer.load_data(task_data=df)
        else:
            raise HTTPException(status_code=400, detail="不支持的数据类型")

        return JSONResponse(
            content={"status": "success", "message": "数据上传成功"},
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/visualize/{chart_type}")
async def create_visualization(
    chart_type: str,
    data_source: str,
    x_col: str,
    y_col: str,
    title: Optional[str] = ""
):
    """生成可视化图表"""
    try:
        # 获取数据
        data = None
        if data_source == "social_media":
            data = social_analyzer.content_data
        elif data_source == "competitor":
            data = competitor_analyzer.competitor_data
        elif data_source == "user_growth":
            data = user_analyzer.user_data
        elif data_source == "community":
            data = community_analyzer.community_data
        else:
            raise HTTPException(status_code=400, detail="不支持的数据源")

        # 创建图表
        if chart_type == "line":
            fig = visualizer.create_time_series(data, x_col, y_col, title)
        elif chart_type == "bar":
            fig = visualizer.create_bar_chart(data, x_col, y_col, title)
        elif chart_type == "pie":
            fig = visualizer.create_pie_chart(data, x_col, y_col, title)
        elif chart_type == "scatter":
            fig = visualizer.create_scatter_plot(data, x_col, y_col, title)
        else:
            raise HTTPException(status_code=400, detail="不支持的图表类型")

        # 保存图表
        output_path = f"temp/charts/{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        visualizer.save_figure(fig, output_path)

        return FileResponse(output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report/{report_type}")
async def generate_report(report_type: str, parameters: Dict = None):
    """生成分析报告"""
    try:
        report = None
        if report_type == "social_media":
            report = social_analyzer.generate_content_recommendations()
        elif report_type == "competitor":
            report = competitor_analyzer.generate_competitive_report()
        elif report_type == "user_growth":
            report = user_analyzer.generate_growth_report()
        elif report_type == "community":
            report = community_analyzer.generate_community_report()
        elif report_type == "content":
            report = content_analyzer.generate_quality_report()
        elif report_type == "efficiency":
            report = efficiency_analyzer.generate_efficiency_report()
        else:
            raise HTTPException(status_code=400, detail="不支持的报告类型")

        return JSONResponse(content=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """启动API服务器"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()