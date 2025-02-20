from fastapi import APIRouter, HTTPException
from typing import Dict, List
from engine.user.behavior import UserBehaviorAnalyzer
from utils.system.logger import LogManager

router = APIRouter()
analyzer = UserBehaviorAnalyzer()
logger = LogManager()

@router.get("/behavior", response_model=Dict)
async def get_user_behavior(user_id: str):
    """获取用户行为分析"""
    try:
        return analyzer.analyze_behavior(user_id)
    except Exception as e:
        logger.log_error(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/segments", response_model=List)
async def get_user_segments():
    """获取用户分群"""
    try:
        return analyzer.get_segments()
    except Exception as e:
        logger.log_error(e)
        raise HTTPException(status_code=500, detail=str(e))