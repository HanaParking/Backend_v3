from pydantic import BaseModel, ConfigDict, Field, conint, model_validator
from typing import List, Literal, Optional
from datetime import datetime

Bit = conint(ge=0, le=1)  # 0 또는 1만 허용

class GetParkingLot(BaseModel):
    lot_code: str        # 주차장 코드
    lot_name: str        # 주차장 이름
    capacity: int       # 전체 주차 가능 수
    status_cd: str       # 상태 코드
    available: Optional[int] = 0      # 현재 이용 가능 수

    # ORM 객체에서 속성 읽어오기 허용
    model_config = ConfigDict(from_attributes=True)

# 실시간 자리 조회
class ParkingSpotOut(BaseModel):
    lot_code: str
    spot_id: str
    occupied_cd: str   
    
    #ORM 객체에서 속성 읽어오기 허용
    model_config = ConfigDict(from_attributes=True)
    
class ParkingSpotBasicOut(BaseModel):
    spot_id: str
    spot_row: int
    spot_column: int

    model_config = ConfigDict(from_attributes=True)
    
class RealtimePayload(BaseModel):
    positions: List[List[int]]
    carExists: List[List[int]]
    ts: str  # ISO 문자열을 그대로 쓸 경우 str

    model_config = ConfigDict(from_attributes=True)
    
class ParkingLotHistoryOut(BaseModel):
    lot_code: str
    lot_name: str | None = None
    status_cd: str
    occupied: int
    capacity: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
