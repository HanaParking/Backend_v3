from typing import List
from fastapi import Depends, APIRouter, HTTPException, Query
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas.parkingLot import GetParkingLot , ParkingSpotOut, ParkingSpotBasicOut,RealtimePayload, ParkingLotHistoryOut
from app.crud import parkingLot as crud_parkingLot  # crud 모듈 임포트
from datetime import datetime, timezone

# 실제 경로에 맞게 수정해줘 (img_upload에서 import하던 것과 동일하게!)
from app.api.v1.endpoints.imgUpload import get_spot_matrix_map, build_positions_from_db  


router = APIRouter()

@router.get("/", response_model=List[GetParkingLot])
def get_parking_lots(db: Session = Depends(get_db)):
    # REDIS로 부터 실시간 자리수 데이터 조회
    
    # DB로 부터 주차장 기본 정보 조회
    parking_lots: List[GetParkingLot] = crud_parkingLot.get_parkingLots(db)
    return parking_lots

@router.get("/recent", response_model=RealtimePayload)
def get_parking_lots_real(
    db: Session = Depends(get_db),
    lot_code: str = Query("A1", description="주차장 코드 (예: A1)"),
):
    # 1) DB에서 해당 lot_code의 좌표 정보 가져오기
    spot_map, all_coords = get_spot_matrix_map(db, lot_code)
    if not spot_map:
        raise HTTPException(
            status_code=404,
            detail=f"해당 주차장({lot_code}) 슬롯 정보가 없습니다.",
        )

    # 2) positions (슬롯 존재 여부) 2D 배열 생성
    positions = build_positions_from_db(all_coords)  # 예: 38x28, 슬롯 위치만 1

    # 3) 최근 주차 상태(차량 존재 여부)를 DB에서 가져오기
    #    (spot_id, spot_row, spot_column, occupied_cd, created_at) 튜플/Row 리스트
    recent_spots = crud_parkingLot.get_recent_parking_spots_with_coords(db, lot_code=lot_code)

    # positions와 동일한 shape의 carExists 그리드 초기화 (0으로)
    car_exists = [[0 for _ in row] for row in positions]

    # 4) recent_spots를 2D 배열에 매핑
    for row in recent_spots:
        # row.spot_row, row.spot_column, row.occupied_cd 형태로 접근 가능
        r = int(row.spot_row)-1
        c = int(row.spot_column)-1

        if 0 <= r < len(car_exists) and 0 <= c < len(car_exists[0]):
            # occupied_cd가 '1' / 1 / True 등일 수 있으니 문자열 기준으로 처리 예시
            occupied = str(row.occupied_cd)
            has_car = 1 if occupied in ("1", "Y", "T", "true", "True") else 0
            car_exists[r][c] = has_car

    # 5) SSE/pubsub payload와 동일한 구조로 반환
    return RealtimePayload(
        positions=positions,
        carExists=car_exists,
        ts=datetime.now(timezone.utc).isoformat(),
    )



# ★ 추가: 특정 Lot의 자리 좌표 조회
@router.get("/spots", response_model=List[ParkingSpotBasicOut])
def get_parking_spots(
    lot_code: str = Query(..., min_length=1, max_length=50, description="주차장 코드 (예: A1)"),
    db: Session = Depends(get_db),
):
    rows = crud_parkingLot.get_parking_spots_by_lot(db, lot_code)
    return rows

@router.get("/lots", response_model=List[ParkingLotHistoryOut])
def get_parking_lots_history(db: Session = Depends(get_db)):
    """
    주차장별 최신 이력 조회 (lot_code별 created_at 가장 최근 1건)
    """
    rows = crud_parkingLot.get_parking_lots_history_latest(db)
    return rows