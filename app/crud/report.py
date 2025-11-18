from sqlalchemy.orm import Session
from sqlalchemy import text

# -------------------------
# 1) 요약 summary 조회
# -------------------------
def get_commute_summary(db: Session, stat_date: str, lot_code: str, start_t: str, end_t: str):
    sql = """
        WITH filtered AS (
            SELECT *,
                   (capacity - occupied) AS available
            FROM hanaparking.parking_lot_history
            WHERE lot_code = :lot_code
              AND created_at::date = :date
              AND to_char(created_at, 'HH24:MI') BETWEEN :start_t AND :end_t
        )
        SELECT 
            -- 최대 / 최소 / 평균 점유율
            MAX(occupied * 100.0 / capacity) AS max_rate,
            MIN(occupied * 100.0 / capacity) AS min_rate,
            AVG(occupied * 100.0 / capacity) AS avg_rate,

            -- 최대 점유율 시간
            (
                SELECT to_char(created_at, 'HH24:MI')
                FROM filtered
                ORDER BY (occupied * 100.0 / capacity) DESC
                LIMIT 1
            ) AS peak_time,

            -- 최소 남은 자리 수
            (
                SELECT available
                FROM filtered
                ORDER BY available ASC
                LIMIT 1
            ) AS min_available,

            -- 최소 남은 자리 발생 시간
            (
                SELECT to_char(created_at, 'HH24:MI')
                FROM filtered
                ORDER BY available ASC
                LIMIT 1
            ) AS min_time,

            -- 혼잡도 판단 (가장 최근 값 기준)
            (
                SELECT CASE
                    WHEN (occupied * 100.0 / capacity) < 30 THEN '여유'
                    WHEN (occupied * 100.0 / capacity) < 70 THEN '여유 부족'
                    ELSE '혼잡'
                END
                FROM filtered
                ORDER BY created_at DESC
                LIMIT 1
            ) AS congestion

        FROM filtered;
    """

    row = db.execute(
        text(sql),
        {
            "date": stat_date,
            "lot_code": lot_code,
            "start_t": start_t,
            "end_t": end_t,
        }
    ).mappings().first()

    return row




# -------------------------
# 2) 시간대별 그래프 조회
# -------------------------
def get_hourly_data(db: Session, stat_date: str, lot_code: str, start_t: str, end_t: str):

    sql = """
        WITH base AS (
            SELECT
                date_trunc('minute', created_at)
                + floor(extract(minute FROM created_at)::int / 5) * interval '5 min'
                  AS time_slot,
                occupied,
                capacity
            FROM hanaparking.parking_lot_history
            WHERE lot_code = :lot_code
              AND created_at::date = :date
              AND to_char(created_at, 'HH24:MI') BETWEEN :start_t AND :end_t
        )
        SELECT
            to_char(time_slot, 'HH24:MI') AS time_slot,
            ROUND(AVG(occupied * 100.0 / capacity), 2) AS occupancy_rate
        FROM base
        GROUP BY time_slot
        ORDER BY time_slot;
    """

    rows = db.execute(
        text(sql),
        {
            "date": stat_date,
            "lot_code": lot_code,
            "start_t": start_t,
            "end_t": end_t,
        }
    ).mappings().all()

    return rows




# -------------------------
# 3) 상세 테이블 조회
# -------------------------
def get_commute_detail(db: Session, stat_date: str, lot_code: str, start_t: str, end_t: str):

    sql = """
        WITH base AS (
            SELECT
                date_trunc('minute', created_at)
                + floor(extract(minute FROM created_at)::int / 5) * interval '5 min'
                  AS time_slot,
                occupied,
                capacity
            FROM hanaparking.parking_lot_history
            WHERE lot_code = :lot_code
              AND created_at::date = :date
              AND to_char(created_at, 'HH24:MI') BETWEEN :start_t AND :end_t
        )
        SELECT
            to_char(time_slot, 'HH24:MI') AS time_slot,
            ROUND(AVG(occupied), 0) AS occupied,
            ROUND(AVG(capacity), 0) AS capacity,
            ROUND(AVG(occupied * 100.0 / capacity), 2) AS occupancy_rate,
            CASE
                WHEN AVG(capacity) = 0 THEN '정보없음'
                WHEN AVG(occupied * 100.0 / capacity) < 30 THEN '여유'
                WHEN AVG(occupied * 100.0 / capacity) < 70 THEN '여유 부족'
                ELSE '혼잡'
            END AS congestion_level
        FROM base
        GROUP BY time_slot
        ORDER BY time_slot;
    """

    rows = db.execute(
        text(sql),
        {
            "date": stat_date,
            "lot_code": lot_code,
            "start_t": start_t,
            "end_t": end_t,
        }
    ).mappings().all()

    return rows

def get_least_congested_day(db: Session, lot_code: str, period: str):
    # period → 시간 범위
    if period == "morning":
        start_t, end_t = "07:30", "09:30"
    else:
        start_t, end_t = "17:00", "19:00"

    sql = """
        WITH recent AS (
            SELECT 
                created_at::date AS dt,
                AVG(occupied * 100.0 / capacity) AS avg_rate
            FROM hanaparking.parking_lot_history
            WHERE lot_code = :lot_code
              AND created_at >= NOW() - interval '7 days'
              AND created_at::time BETWEEN (:start_t)::time AND (:end_t)::time
              AND EXTRACT(ISODOW FROM created_at) BETWEEN 1 AND 5   -- 월(1)~금(5)
            GROUP BY created_at::date
        )
        SELECT dt, avg_rate
        FROM recent
        ORDER BY avg_rate ASC
        LIMIT 1;
    """

    row = db.execute(
        text(sql),
        {
            "lot_code": lot_code,
            "start_t": start_t,
            "end_t": end_t,
        }
    ).mappings().first()

    return row
