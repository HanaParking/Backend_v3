# app/api/v1/routers.py
from fastapi import APIRouter

# 엔드포인트들
from app.api.v1.endpoints import parkingLot
from app.api.v1.endpoints import files
from app.api.v1.endpoints import imgUpload
from app.api.v1.endpoints import redisDetailPage
from app.api.v1.endpoints import report

api_router = APIRouter()

# ===== 기존 =====
api_router.include_router(files.router,     prefix="/files",  tags=["files"])
api_router.include_router(imgUpload.router, prefix="/upload", tags=["Imgs"])
api_router.include_router(parkingLot.router, prefix="/lot",   tags=["lots"])
api_router.include_router(redisDetailPage.router, prefix="/redis/detail", tags=["detail"]) 
api_router.include_router(report.router, prefix="/report", tags=["Report"])

