from fastapi import APIRouter
from .social import router as social_router
from .content import router as content_router
from .user import router as user_router
from .operation import router as operation_router
from .analysis import router as analysis_router

router = APIRouter()

router.include_router(social_router, prefix="/social", tags=["社交分析"])
router.include_router(content_router, prefix="/content", tags=["内容分析"])
router.include_router(user_router, prefix="/user", tags=["用户分析"])
router.include_router(operation_router, prefix="/operation", tags=["运营分析"])
router.include_router(analysis_router, prefix="/analysis", tags=["数据分析"])