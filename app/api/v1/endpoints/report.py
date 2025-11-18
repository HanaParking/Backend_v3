from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from app.core.openai import client
from fastapi import Body
from app.dependencies import get_db
from app.schemas.report import (
    CommuteSummaryOut,
    CommuteDetailOut,
    HourlyChartOut
)
from app.crud import report as crud_report

router = APIRouter()

# 공통: period → 시간 범위 계산
def get_period_range(period: str):
    if period == "morning":
        return ("07:30", "09:30")
    elif period == "evening":
        return ("17:00", "19:00")
    else:
        raise HTTPException(400, "period는 morning 또는 evening 이어야 합니다.")


# -------------------------
# 1) 요약 summary
# -------------------------
@router.get("/summary", response_model=CommuteSummaryOut)
def get_commute_summary(
    date: str = Query(...),
    lot_code: str = Query(...),
    period: str = Query("morning"),
    db: Session = Depends(get_db)
):
    start_t, end_t = get_period_range(period)

    summary = crud_report.get_commute_summary(db, date, lot_code, start_t, end_t)

    if not summary:
        raise HTTPException(404, "데이터가 없습니다.")

    # ⭐ RowMapping → dict 변환
    summary = dict(summary)

    # ⭐ 지난 7일 중 가장 한가한 날 가져오기
    least_day = crud_report.get_least_congested_day(db, lot_code, period)
    summary["least_congested_day"] = (
    least_day["dt"].isoformat() if least_day and least_day["dt"] else None
)

    return summary




# -------------------------
# 2) 시간대별 그래프
# -------------------------
@router.get("/hourly", response_model=list[HourlyChartOut])
def get_hourly_chart(
    date: str = Query(...),
    lot_code: str = Query(...),
    period: str = Query("morning"),
    db: Session = Depends(get_db),
):
    start_t, end_t = get_period_range(period)
    rows = crud_report.get_hourly_data(db, date, lot_code, start_t, end_t)
    return rows


# -------------------------
# 3) 상세 테이블
# -------------------------
@router.get("/detail", response_model=list[CommuteDetailOut])
def get_commute_detail(
    date: str = Query(...),
    lot_code: str = Query(...),
    period: str = Query("morning"),
    db: Session = Depends(get_db)
):
    start_t, end_t = get_period_range(period)
    rows = crud_report.get_commute_detail(db, date, lot_code, start_t, end_t)
    return rows


@router.post("/ai")
def analyze_with_gpt(
    today: list[dict] = Body(..., description="오늘 시간대별 점유 데이터"),
    yesterday: dict = Body(..., description="어제 summary 데이터"),
):
    """
    GPT에게 주차 패턴 분석을 요청하는 엔드포인트
    """

    prompt = f"""
당신은 주차장 혼잡도 분석 전문가입니다.

다음은 오늘의 실시간 주차 데이터(today)와 어제의 요약 데이터(yesterday_summary)입니다.

오늘(today):
{today}

어제(yesterday_summary):
{yesterday}

주차상황의 핵심은 2~3 문장으로 요약해준 후 
이에 기반한 추천 주차 전략을 1~2 문장으로
    - 어떤 시간대에 여유 있는지
    - 몇 시까지 도착하면 좋은지
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 데이터 분석 및 주차 전략 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
    )

    # 🔥 여기 수정!
    return {"analysis": response.choices[0].message.content}

