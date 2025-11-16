from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from pathlib import Path
import os, uuid, json, numpy as np, cv2
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any
from app.dependencies import get_db, get_redis
from app.schemas.imgUpload import UploadOut
from app.models.parkingLot import ParkingSpotHistory
from ultralytics import YOLO
from zoneinfo import ZoneInfo
from app.crud import parkingLot as crud_parkingLot
from fastapi.responses import FileResponse
import mimetypes

router = APIRouter()

UPLOAD_DIR = "upload_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
AI_DIR = BASE_DIR / "ai"

ROI_JSON = AI_DIR / "roi_points.json"
MODEL_PATH = AI_DIR / "best_hana.pt"
CROP_SIZE = (200, 300)
ROWS, COLS = 38, 28

# ---------- 전역 로드 ----------
with open(ROI_JSON, "r") as f:
    ROI_DATA = json.load(f)

MODEL = YOLO(MODEL_PATH)

# ---------- 유틸리티 ----------
def sort_points_clockwise(pts):
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    pts = sorted(pts, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
    return np.array(pts, dtype=np.float32)

def imdecode_upload(file_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")
    return img

def blank_grids():
    positions = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    car_exists = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    return positions, car_exists

def get_spot_matrix_map(db: Session, lot_code: str):
    rows = crud_parkingLot.get_parking_spots_by_lot(db, lot_code)
    spot_map = {}
    coords = []

    for r in rows:
        i = int(r.spot_row) - 1
        j = int(r.spot_column) - 1
        if 0 <= i < ROWS and 0 <= j < COLS:
            sid = str(r.spot_id).strip()
            spot_map[sid] = (i, j)
            coords.append((i, j))
    return spot_map, coords

def build_positions_from_db(all_coords: List[Tuple[int, int]]):
    positions, _ = blank_grids()
    for (i, j) in all_coords:
        positions[i][j] = 1
    return positions

# ========== 🔥 핵심 기능: infer + DB 저장 + 시각화 이미지 생성 ==========
def infer_and_map(
    db: Session,
    lot_code: str,
    img_bgr: np.ndarray,
    ROI_DATA: List[dict],
    spot_map: Dict[str, Tuple[int, int]],
    positions: List[List[int]],
) -> Tuple[List[List[int]], np.ndarray]:

    ROWS = len(positions)
    COLS = len(positions[0])
    car_exists = [[0 for _ in range(COLS)] for _ in range(ROWS)]

    img_draw = img_bgr.copy()  # 시각화용 이미지

    dst_pts = np.float32([
        [0, 0],
        [CROP_SIZE[0], 0],
        [CROP_SIZE[0], CROP_SIZE[1]],
        [0, CROP_SIZE[1]],
    ])

    today = datetime.now(ZoneInfo("Asia/Seoul")).date()
    rows_to_insert = []

    for roi in ROI_DATA:
        spot_id = str(roi.get("name", "")).strip()
        pts = roi.get("points")

        if not spot_id or spot_id not in spot_map or not pts or len(pts) != 4:
            continue

        (i, j) = spot_map[spot_id]
        if not (0 <= i < ROWS and 0 <= j < COLS and positions[i][j] == 1):
            continue

        try:
            src_pts = sort_points_clockwise(pts)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img_bgr, M, CROP_SIZE)

            result = MODEL(warped, verbose=False)
            label_idx = int(result[0].probs.top1)
            label_name = MODEL.names[label_idx]

            occupied = 0 if label_name.lower() == "empty" else 1
            car_exists[i][j] = occupied

            # 🔵 시각화 색상
            color = (0, 255, 0) if occupied == 0 else (0, 0, 255)

            # ROI 영역 폴리곤
            cv2.polylines(img_draw, [src_pts.astype(int)], True, color, 3)

            # 텍스트 표현
            text = f"{spot_id}: {'empty' if occupied == 0 else 'car'}"
            pos = tuple(np.mean(src_pts, axis=0).astype(int))
            cv2.putText(img_draw, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # DB insert
            rows_to_insert.append({
                "history_dt": today,
                "lot_code": lot_code,
                "spot_id": spot_id,
                "occupied_cd": "1" if occupied else "0",
            })

        except Exception:
            continue

    if rows_to_insert:
        db.bulk_insert_mappings(ParkingSpotHistory, rows_to_insert)
        db.commit()

    return car_exists, img_draw

# ========== 🔥 이미지 업로드 엔드포인트 ==========
@router.post("/img_upload", response_model=UploadOut, status_code=201)
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    redis = Depends(get_redis),
):
    lot_code = "A1"

    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    # 1️⃣ 이미지 디코드
    try:
        content = await file.read()
        img = imdecode_upload(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 디코딩 실패: {e}")

    # 2️⃣ DB → slot 좌표
    spot_map, all_coords = get_spot_matrix_map(db, lot_code)
    if not spot_map:
        raise HTTPException(status_code=404, detail=f"슬롯 정보 없음")

    positions = build_positions_from_db(all_coords)

    # 3️⃣ infer + DB 저장 + 시각화 이미지 생성
    try:
        car_exists, img_draw = infer_and_map(db, lot_code, img, ROI_DATA, spot_map, positions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {e}")

    # 4️⃣ 원본 대신 ‘시각화된 이미지’를 저장
    try:
        cv2.imwrite(file_path, img_draw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 저장 실패: {e}")

    # 5️⃣ Redis 발행
    realtime_payload = {
        "positions": positions,
        "carExists": car_exists,
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    try:
        await redis.set("parking_detail_data", json.dumps(realtime_payload))
        await redis.publish("parking_detail_channel", "updated")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redis 처리 실패: {e}")

    return {
        "filename": safe_name,
        "url": f"/upload_images/{safe_name}",
        "message": f"분석 및 시각화 완료",
    }

# ========== 🔥 최신 이미지 확인 ==========
def _get_latest_image_path(upload_dir: str) -> Path | None:
    p = Path(upload_dir)
    if not p.exists():
        return None
    files = [f for f in p.iterdir() if f.is_file()]
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)

@router.get("/img_latest", response_class=HTMLResponse)
def view_latest_image():
    latest = _get_latest_image_path(UPLOAD_DIR)
    if latest is None:
        return HTMLResponse("<h1>이미지가 없습니다.</h1>")

    img_url = f"/upload_images/{latest.name}"

    html = f"""
    <html>
    <body style="background:#000;display:flex;justify-content:center;align-items:center;height:100vh;">
        <img src="{img_url}" style="max-width:90vw;max-height:90vh;border-radius:12px;" />
    </body>
    </html>
    """
    return HTMLResponse(html)

@router.get("/img_files", response_class=HTMLResponse, tags=["Imgs"])
def list_images():
    """
    upload_images 폴더 안의 이미지 파일 목록을 HTML로 보여줌.
    각 이미지 썸네일 + 다운로드 링크 제공.
    """
    p = Path(UPLOAD_DIR)
    if not p.exists():
        return HTMLResponse("<h3>업로드 폴더가 없습니다.</h3>", status_code=200)

    # jpg/png 등만 대상으로
    exts = {".jpg", ".jpeg", ".png", ".gif"}
    files = [
        f for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    ]

    if not files:
        return HTMLResponse("<h3>저장된 이미지가 없습니다.</h3>", status_code=200)

    # 최근순 정렬 (수정시간 기준 최신 → 오래된 순)
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    # StaticFiles로 /upload_images mount 되어 있다고 가정
    rows_html = []
    for f in files:
        img_url = f"/upload_images/{f.name}"
        download_url = f"/api/v1/img/img_download?filename={f.name}"  # 라우팅 prefix에 따라 수정
        created = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        rows_html.append(f"""
        <div class="card">
            <div class="thumb-wrap">
                <a href="{img_url}" target="_blank">
                    <img src="{img_url}" alt="{f.name}" />
                </a>
            </div>
            <div class="info">
                <div class="name">{f.name}</div>
                <div class="time">{created}</div>
                <a class="btn" href="{download_url}">다운로드</a>
            </div>
        </div>
        """)

    body = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>이미지 목록</title>
        <style>
            body {{
                margin: 0;
                padding: 16px;
                background: #0b0b0d;
                color: #eee;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                             Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
            }}
            h1 {{
                margin-bottom: 16px;
                font-size: 20px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                gap: 12px;
            }}
            .card {{
                background: #15151a;
                border-radius: 10px;
                padding: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
                display: flex;
                flex-direction: column;
                gap: 8px;
            }}
            .thumb-wrap {{
                width: 100%;
                aspect-ratio: 4 / 3;
                overflow: hidden;
                border-radius: 8px;
                background: #222;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .thumb-wrap img {{
                max-width: 100%;
                max-height: 100%;
                object-fit: cover;
            }}
            .info {{
                font-size: 12px;
                display: flex;
                flex-direction: column;
                gap: 4px;
            }}
            .name {{
                font-weight: 600;
                word-break: break-all;
            }}
            .time {{
                opacity: 0.7;
            }}
            .btn {{
                margin-top: 4px;
                display: inline-block;
                padding: 4px 8px;
                border-radius: 6px;
                background: #1f6feb;
                color: #fff;
                text-decoration: none;
                font-size: 12px;
                text-align: center;
            }}
            .btn:hover {{
                filter: brightness(1.1);
            }}
        </style>
    </head>
    <body>
        <h1>업로드된 이미지 목록</h1>
        <div class="grid">
            {''.join(rows_html)}
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=body, status_code=200)
