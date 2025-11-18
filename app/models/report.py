from sqlalchemy.orm import Session
from sqlalchemy import text


# -------------------------
# 1) 요약 summary 조회
# -------------------------
def get_commute_summary(db: Session, stat_date: str, lot_code: str):
    sql = """
        SELECT 
            MAX(occupied * 100.0 / capacity) AS max_rate,
            MIN(occupied * 100.0 / capacity) AS min_rate,
            AVG(occupied * 100.0 / capacity) AS avg_rate,
            (SELECT to_char(created_at, 'HH24:MI')
             FROM hanaparking.parking_lot_history
             WHERE lot_code = :lot_code
               AND created_at::date = :date
             ORDER BY occupied / capacity DESC
             LIMIT 1) AS peak_time
        FROM hanaparking.parking_lot_history
        WHERE lot_code = :lot_code
          AND created_at::date = :date
    """

    row = db.execute(
        text(sql),
        {"date": stat_date, "lot_code": lot_code}
    ).mappings().first()

    return row


# -------------------------
# 2) 시간대별 그래프 조회
# -------------------------
def get_hourly_data(db: Session, stat_date: str, lot_code: str):

    sql = """
        SELECT 
            to_char(created_at, 'HH24:MI') AS time_slot,
            ROUND(occupied * 100.0 / capacity, 2) AS occupancy_rate
        FROM hanaparking.parking_lot_history
        WHERE lot_code = :lot_code
          AND created_at::date = :date
        ORDER BY created_at
    """

    rows = db.execute(
        text(sql),
        {"date": stat_date, "lot_code": lot_code}
    ).mappings().all()

    return rows


# -------------------------
# 3) 상세 테이블 조회
# -------------------------
def get_commute_detail(db: Session, stat_date: str, lot_code: str):

    sql = """
        SELECT 
            to_char(created_at, 'HH24:MI') AS time_slot,
            occupied,
            capacity,
            ROUND(occupied * 100.0 / capacity, 2) AS occupancy_rate,
            CASE
                WHEN capacity = 0 THEN '정보없음'
                WHEN (occupied * 100.0 / capacity) < 30 THEN '여유'
                WHEN (occupied * 100.0 / capacity) < 70 THEN '여유 부족'
                ELSE '혼잡'
            END AS congestion_level
        FROM hanaparking.parking_lot_history
        WHERE lot_code = :lot_code
          AND created_at::date = :date
        ORDER BY created_at
    """

    rows = db.execute(
        text(sql),
        {"date": stat_date, "lot_code": lot_code}
    ).mappings().all()

    return rows
