from fastapi import APIRouter, HTTPException
from typing import Dict, List
from engine.social.analyzer import SocialMediaAnalyzer
from utils.system.logger import LogManager

router = APIRouter()
analyzer = SocialMediaAnalyzer()
logger = LogManager()

@router.get("/metrics", response_model=Dict)
async def get_social_metrics():
    """获取社交媒体指标"""
    try:
        return analyzer.get_metrics()
    except Exception as e:
        logger.log_error(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/influential-users", response_model=List)
async def get_influential_users(top_n: int = 10):
    """获取有影响力的用户"""
    try:
        return analyzer.get_influential_users(top_n)
    except Exception as e:
        logger.log_error(e)
        raise HTTPException(status_code=500, detail=str(e))