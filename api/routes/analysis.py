from fastapi import APIRouter, HTTPException
from typing import Dict, List
from engine.analysis.data_analyzer import DataAnalyzer
from utils.system.logger import LogManager

router = APIRouter()
analyzer = DataAnalyzer()
logger = LogManager()

@router.get("/overview", response_model=Dict)
async def get_analysis_overview():
    """获取数据分析概览"""
    try:
        return analyzer.get_overview()
    except Exception as e:
        logger.log_error(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends", response_model=Dict)
async def get_analysis_trends(time_range: str = "7d"):
    """获取趋势分析"""
    try:
        return analyzer.get_trends(time_range)
    except Exception as e:
        logger.log_error(e)
        raise HTTPException(status_code=500, detail=str(e))