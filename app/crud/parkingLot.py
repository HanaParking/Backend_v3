from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from app.models.parkingLot import ParkingLot, ParkingSpotHistory, ParkingSpot, ParkingLotHistory

# 주차장(Lot)관련 CRUD 작업

# 구역별 주차자리 특성 테이블
def get_parkingLots(db: Session):
    return db.query(ParkingLot).all()

def get_RecentParkingSpot(db: Session):
    # 자리별(= lot_code + spot_id) 최신 1개: 날짜 최신 → 같은 날짜 내 시퀀스 최신
    rn = func.row_number().over(
        partition_by=(ParkingSpotHistory.lot_code, ParkingSpotHistory.spot_id),
        order_by=(desc(ParkingSpotHistory.history_dt),
                  desc(ParkingSpotHistory.history_seq))
    ).label("rn")

    subq = (
        db.query(
            ParkingSpotHistory.lot_code.label("lot_code"),
            ParkingSpotHistory.spot_id.label("spot_id"),
            ParkingSpotHistory.history_dt.label("history_dt"),
            ParkingSpotHistory.history_seq.label("history_seq"),
            rn
        )
        # 필요 시 특정 주차장만: .filter(ParkingSpotHistory.lot_code == 'A1')
        .subquery()
    )

    # 복합키로 원본과 조인 → ORM 객체 그대로 반환
    return (
        db.query(ParkingSpotHistory)
        .join(
            subq,
            and_(
                ParkingSpotHistory.lot_code == subq.c.lot_code,
                ParkingSpotHistory.spot_id == subq.c.spot_id,
                ParkingSpotHistory.history_dt == subq.c.history_dt,
                ParkingSpotHistory.history_seq == subq.c.history_seq,
            ),
        )
        .filter(subq.c.rn == 1)
        .all()
    )

def get_parking_spots_by_lot(db: Session, lot_code: str):
    """SELECT spot_id, spot_row, spot_column FROM hanaparking.parking_spot WHERE lot_code=:lot_code"""
    return (
        db.query(
            ParkingSpot.spot_id,
            ParkingSpot.spot_row,
            ParkingSpot.spot_column,
        )
        .filter(ParkingSpot.lot_code == lot_code)
        .order_by(ParkingSpot.spot_row.asc(), ParkingSpot.spot_column.asc(), ParkingSpot.spot_id.asc())
        .all()
    )

def get_recent_parking_spots_with_coords(
    db: Session,
    lot_code: str | None = None,
):
    """
SELECT a.spot_id, a.spot_row, a.spot_column, b.occupied_cd, b.created_at
FROM hanaparking.parking_spot a
JOIN (
    SELECT *
    FROM (
        SELECT 
            spot_id,
            occupied_cd,
            created_at,
            ROW_NUMBER() OVER (PARTITION BY spot_id ORDER BY history_seq DESC) AS rn
        FROM hanaparking.parking_spot_history
        -- WHERE lot_code = :lot_code   -- (history에도 lot_code가 있다면)
    ) t
    WHERE t.rn = 1
) b
ON a.spot_id = b.spot_id
WHERE a.lot_code = :lot_code
ORDER BY a.spot_row, a.spot_column;
"""


    # 1) spot_id 별 최신 이력 뽑는 서브쿼리
    rn = func.row_number().over(
        partition_by=ParkingSpotHistory.spot_id,
        order_by=ParkingSpotHistory.created_at.desc(),
    ).label("rn")

    history_subq = (
        db.query(
            ParkingSpotHistory.spot_id.label("spot_id"),
            ParkingSpotHistory.occupied_cd.label("occupied_cd"),
            ParkingSpotHistory.created_at.label("created_at"),
            rn,
        )
        .subquery()
    )

    # 2) parking_spot(a) 와 조인 + rn = 1만 사용
    query = (
        db.query(
            ParkingSpot.spot_id,
            ParkingSpot.spot_row,
            ParkingSpot.spot_column,
            history_subq.c.occupied_cd,
            history_subq.c.created_at,
        )
        .join(history_subq, ParkingSpot.spot_id == history_subq.c.spot_id)
        .filter(history_subq.c.rn == 1)
    )

    # lot_code로 제한하고 싶으면 옵션으로
    if lot_code is not None:
        query = query.filter(ParkingSpot.lot_code == lot_code)

    # 결과는 튜플 리스트:
    # (spot_id, spot_row, spot_column, occupied_cd, created_at)
    return query.all()

def get_parking_lots_history_latest(db: Session):
    """
    각 lot_code별 가장 최근 이력 1건 조회.

    SELECT h.*
    FROM hanaparking.parking_lot_history h
    JOIN (
        SELECT lot_code, MAX(created_at) AS max_created_at
        FROM hanaparking.parking_lot_history
        GROUP BY lot_code
    ) sub
      ON h.lot_code = sub.lot_code
     AND h.created_at = sub.max_created_at
    ORDER BY h.lot_code;
    """

    # 1) lot_code별 max(created_at) 구하는 서브쿼리
    subq = (
        db.query(
            ParkingLotHistory.lot_code.label("lot_code"),
            func.max(ParkingLotHistory.created_at).label("max_created_at"),
        )
        .group_by(ParkingLotHistory.lot_code)
        .subquery()
    )

    # 2) 원본 테이블과 조인해서 최신 이력만 가져오기
    return (
        db.query(ParkingLotHistory)
        .join(
            subq,
            and_(
                ParkingLotHistory.lot_code == subq.c.lot_code,
                ParkingLotHistory.created_at == subq.c.max_created_at,
            ),
        )
        .order_by(ParkingLotHistory.lot_code.asc())
        .all()
    )