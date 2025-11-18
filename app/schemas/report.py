from pydantic import BaseModel, ConfigDict, Field, conint, model_validator
from typing import List, Literal, Optional
from datetime import datetime

from pydantic import BaseModel, ConfigDict

class CommuteSummaryOut(BaseModel):
    max_rate: float | None = None
    min_rate: float | None = None
    avg_rate: float | None = None
    peak_time: str | None = None
    min_available: int | None = None
    min_time: str | None = None
    congestion: str | None = None

    # ⭐ 새로운 필드
    least_congested_day: str | None = None  

    model_config = ConfigDict(from_attributes=True)



class HourlyChartOut(BaseModel):
    time_slot: str
    occupancy_rate: float
    
    model_config = ConfigDict(from_attributes=True)


class CommuteDetailOut(BaseModel):
    time_slot: str
    occupied: int
    capacity: int
    occupancy_rate: float
    congestion_level: str
    
    model_config = ConfigDict(from_attributes=True)
