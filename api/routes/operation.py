from fastapi import APIRouter, HTTPException
from typing import Dict
from engine.operation.efficiency import OperationalEfficiencyAnalyzer
from engine.operation.ad_performance import AdPerformanceAnalyzer
from utils.system.logger import LogManager

router = APIRouter()
efficiency_analyzer = OperationalEfficiencyAnalyzer()
ad_analyzer = AdPerformanceAnalyzer()
logger = LogManager()

@router.get("/efficiency", response_model=Dict)
async def get_operational_efficiency():
    """获取运营效率指标"""
    try:
        return efficiency_analyzer.get_metrics()
    except Exception as e:
        logger.log_error(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ad-performance", response_model=Dict)
async def get_ad_performance():
    """获取广告效果分析"""
    try:
        return ad_analyzer.get_performance()
    except Exception as e:
        logger.log_error(e)
        raise HTTPException(status_code=500, detail=str(e))