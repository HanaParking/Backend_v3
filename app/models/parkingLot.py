from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, TIMESTAMP, DateTime, Date, Index, text, func
from app.db.database import Base
from typing import Optional

# 구역별 주차장 자리 특성 테이블
class ParkingLot(Base):
    __tablename__ = "parking_lot"
    __table_args__ = {"schema": "hanaparking"}  # 스키마 지정

    lot_code: Mapped[str] = mapped_column(String(50), primary_key=True, index=True)
    lot_name: Mapped[str | None] = mapped_column(String(100), unique=True, nullable=True)
    capacity: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    status_cd: Mapped[str] = mapped_column(String(1), nullable=False, server_default=text("'1'"))
    created_at: Mapped[str] = mapped_column(TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[str | None] = mapped_column(TIMESTAMP, nullable=True)

# 주차장 자리 이력 테이블
class ParkingSpotHistory(Base):
    __tablename__ = "parking_spot_history"
    __table_args__ = {"schema": "hanaparking"}  # 스키마 지정

    # PK: (history_dt, history_seq)
    history_dt: Mapped[str] = mapped_column(Date, primary_key=True, nullable=False)
    history_seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False)
    lot_code: Mapped[str] = mapped_column(String(50), nullable=False)
    spot_id: Mapped[str] = mapped_column(String(10), nullable=False)
    occupied_cd: Mapped[str] = mapped_column(String(1), nullable=False)
    created_at: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.current_timestamp())
    
class ParkingSpot(Base):
    __tablename__ = "parking_spot"
    __table_args__ = (
        Index("ix_parking_spot_lot_code", "lot_code"),
        {"schema": "hanaparking"},
    )

    # PK는 spot_id 하나라고 가정 (필요하면 복합키로 변경)
    spot_id: Mapped[str]   = mapped_column(String(10), primary_key=True)
    lot_code: Mapped[str]  = mapped_column(String(50), nullable=False)
    spot_row: Mapped[int]  = mapped_column(Integer, nullable=False)
    spot_column: Mapped[int]= mapped_column(Integer, nullable=False)
    
class ParkingLotHistory(Base):
    __tablename__ = "parking_lot_history"
    __table_args__ = {"schema": "hanaparking"}

    # PK: history_seq (serial4)
    history_seq: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True,
    )

    # Lot code (예: "A1")
    lot_code: Mapped[str] = mapped_column(String(50), nullable=False)

    # 주차장 이름 (예: "본사 지하주차장")
    lot_name: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # 상태 코드 (1 = 정상 등)
    status_cd: Mapped[str] = mapped_column(
        String(1),
        nullable=False,
        server_default="1"
    )

    # 점유 차량 수 (occupied)
    occupied: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="0"
    )
    
        # 점유 차량 수 (occupied)
    capacity: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="0"
    )

    # 등록 시각
    created_at: Mapped[DateTime] = mapped_column(
        DateTime,
        server_default=func.current_timestamp()
    )