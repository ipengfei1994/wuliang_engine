from fastapi import APIRouter, HTTPException
from typing import Dict
from engine.content.management import ContentManager
from utils.system.logger import LogManager

router = APIRouter()
manager = ContentManager()
logger = LogManager()

@router.get("/quality-metrics", response_model=Dict)
async def get_content_quality():
    """获取内容质量指标"""
    try:
        return manager.analyze_quality()
    except Exception as e:
        logger.log_error(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends", response_model=Dict)
async def get_content_trends():
    """获取内容趋势分析"""
    try:
        return manager.analyze_trends()
    except Exception as e:
        logger.log_error(e)
        raise HTTPException(status_code=500, detail=str(e))