from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.orm import Session
from pathlib import Path
import os, uuid, json, numpy as np, cv2
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any
from zoneinfo import ZoneInfo
import mimetypes

from app.dependencies import get_db, get_redis
from app.schemas.imgUpload import UploadOut
from app.models.parkingLot import ParkingSpotHistory, ParkingLotHistory
from app.crud import parkingLot as crud_parkingLot
from ultralytics import YOLO

router = APIRouter()

UPLOAD_DIR = "upload_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
AI_DIR = BASE_DIR / "ai"

ROI_JSON = AI_DIR / "roi_points.json"
MODEL_PATH = AI_DIR / "hana_model_v2.pt"
CROP_SIZE = (200, 300)
ROWS, COLS = 38, 28

# ---------- 전역 로드 ----------
print(f"[INFO] ROI_JSON 경로: {ROI_JSON}")
print(f"[INFO] MODEL_PATH 경로: {MODEL_PATH}")

with open(ROI_JSON, "r") as f:
    ROI_DATA = json.load(f)
    print(f"[INFO] ROI_DATA 로드 완료: {len(ROI_DATA)}개 구역")

MODEL = YOLO(MODEL_PATH)
print("[INFO] YOLO 모델 로드 완료")

# ---------- 유틸리티 ----------
def sort_points_clockwise(pts):
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    pts = sorted(pts, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
    return np.array(pts, dtype=np.float32)

def imdecode_upload(file_bytes: bytes) -> np.ndarray:
    print(f"[DEBUG] 업로드된 파일 바이트 길이: {len(file_bytes)}")
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("[ERROR] 이미지 디코딩 실패")
        raise ValueError("이미지 디코딩 실패")
    print(f"[INFO] 이미지 디코딩 성공: shape={img.shape}")
    return img

def blank_grids():
    positions = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    car_exists = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    return positions, car_exists

def get_spot_matrix_map(db: Session, lot_code: str):
    print(f"[INFO] get_spot_matrix_map 호출 lot_code={lot_code}")
    rows = crud_parkingLot.get_parking_spots_by_lot(db, lot_code)
    print(f"[INFO] DB에서 불러온 슬롯 개수: {len(rows)}")

    spot_map: Dict[str, Tuple[int, int]] = {}
    coords: List[Tuple[int, int]] = []

    for r in rows:
        i = int(r.spot_row) - 1
        j = int(r.spot_column) - 1
        sid = str(r.spot_id).strip()
        #print(f"[DEBUG] 슬롯 로드 → spot_id={sid}, row={r.spot_row}, col={r.spot_column}")

        if 0 <= i < ROWS and 0 <= j < COLS:
            spot_map[sid] = (i, j)
            coords.append((i, j))
        else:
            print(f"[WARN] 좌표 범위 밖 → spot_id={sid}, (i,j)=({i},{j})")

    print(f"[INFO] spot_map 크기: {len(spot_map)}, coords 개수: {len(coords)}")
    return spot_map, coords

def build_positions_from_db(all_coords: List[Tuple[int, int]]):
    positions, _ = blank_grids()
    for (i, j) in all_coords:
        positions[i][j] = 1
    print("[INFO] positions 그리드 생성 완료")
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

    print("[INFO] infer_and_map 시작")
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

    # 🔹 한국 시간 기준 날짜
    today = datetime.now(ZoneInfo("Asia/Seoul")).date()
    rows_to_insert: List[Dict[str, Any]] = []

    # ✅ 이번 추론에서 실제로 YOLO를 태운 자리 목록 (spot_id 기준)
    processed_spot_ids: set[str] = set()

    for roi in ROI_DATA:
        spot_id = str(roi.get("name", "")).strip()
        pts = roi.get("points")

        print(f"[DEBUG] ROI 체크 → id={spot_id}, pts={pts}")

        # ROI에 정의됐지만 DB에 자리코드가 없거나, pts 이상하면 스킵
        if not spot_id or spot_id not in spot_map or not pts or len(pts) != 4:
            print(f"[WARN] ROI 스킵됨 → spot_id={spot_id}, spot_map에 없거나 pts 이상")
            continue

        (i, j) = spot_map[spot_id]
        if not (0 <= i < ROWS and 0 <= j < COLS and positions[i][j] == 1):
            print(f"[WARN] ROI 스킵됨 → spot_id={spot_id}, positions 매핑 안됨 또는 범위 밖")
            continue

        try:
            src_pts = sort_points_clockwise(pts)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img_bgr, M, CROP_SIZE)

            result = MODEL(warped, verbose=False)
            label_idx = int(result[0].probs.top1)
            label_name = MODEL.names[label_idx]

            # empty → 0, 나머지(차 있음) → 1
            occupied = 0 if label_name.lower() == "empty" else 1
            car_exists[i][j] = occupied

            print(f"[DEBUG] YOLO 결과 → spot={spot_id}, label={label_name}, occupied={occupied}")

            # 🔵 시각화 색상
            color = (0, 255, 0) if occupied == 0 else (0, 0, 255)

            # ROI 영역 폴리곤
            cv2.polylines(img_draw, [src_pts.astype(int)], True, color, 3)

            # 텍스트 표현
            text = f"{spot_id}: {'empty' if occupied == 0 else 'car'}"
            pos = tuple(np.mean(src_pts, axis=0).astype(int))
            cv2.putText(img_draw, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ✅ YOLO를 돌린 자리만 우선 rows_to_insert에 추가
            rows_to_insert.append({
                "history_dt": today,
                "lot_code": lot_code,
                "spot_id": spot_id,
                "occupied_cd": "1" if occupied else "0",
            })
            processed_spot_ids.add(spot_id)

        except Exception as e:
            print(f"[ERROR] ROI 처리 중 예외 → spot_id={spot_id}, error={e}")
            continue

    # ✅ 여기서부터가 핵심!
    # DB에 존재하는 모든 슬롯(spot_map 기준)을 훑으면서,
    # 이번 infer 과정에서 처리되지 않은 자리들은
    #  - car_exists: 2 (ROI 없음 / 비활성)
    #  - DB 저장: occupied_cd = '0' (빈 자리 취급)
    for sid, (i, j) in spot_map.items():
        if sid in processed_spot_ids:
            # 이미 위에서 YOLO 돌려서 rows_to_insert에 들어간 자리면 패스
            continue

        if not (0 <= i < ROWS and 0 <= j < COLS):
            print(f"[WARN] spot_map 좌표 범위 밖 → spot_id={sid}, (i,j)=({i},{j})")
            continue

        if positions[i][j] != 1:
            # positions에 표시되지 않은 좌석이면 스킵(안전용)
            print(f"[WARN] positions에 표시되지 않은 슬롯 → spot_id={sid}, (i,j)=({i},{j})")
            continue

        # 🔸 ROI/모델 미적용 슬롯 → 프론트에는 2 (ROI 없음)으로 전달
        car_exists[i][j] = 2

        # 🔸 DB에는 0(빈 자리)로 저장
        rows_to_insert.append({
            "history_dt": today,
            "lot_code": lot_code,
            "spot_id": sid,
            "occupied_cd": "0",
        })
        print(f"[INFO] ROI/모델 미적용 슬롯 0으로 추가 → spot_id={sid}, occupied=0 (car_exists=2)")

    print(f"[INFO] ParkingSpotHistory rows_to_insert 개수: {len(rows_to_insert)}")

    if rows_to_insert:
        try:
            db.bulk_insert_mappings(ParkingSpotHistory, rows_to_insert)
            db.commit()
            print("[INFO] ParkingSpotHistory bulk insert 성공")
        except Exception as e:
            db.rollback()
            print(f"[ERROR] ParkingSpotHistory bulk insert 실패: {e}")
    else:
        print("[WARN] ParkingSpotHistory에 INSERT할 데이터가 없습니다.")

    return car_exists, img_draw



# ========== 🔥 이미지 업로드 엔드포인트 ==========
@router.post("/img_upload", response_model=UploadOut, status_code=201)
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    redis = Depends(get_redis),
):
    lot_code = "A1"
    print(f"[INFO] /img_upload 호출됨, lot_code={lot_code}, filename={file.filename}")

    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    print(f"[INFO] 저장 예정 파일명: {safe_name}")

    # 1️⃣ 이미지 디코드
    try:
        content = await file.read()
        print(f"[DEBUG] 업로드 파일 크기: {len(content)} bytes")
        img = imdecode_upload(content)
    except Exception as e:
        print(f"[ERROR] 이미지 디코딩 실패: {e}")
        raise HTTPException(status_code=400, detail=f"이미지 디코딩 실패: {e}")

    # 2️⃣ DB → slot 좌표
    spot_map, all_coords = get_spot_matrix_map(db, lot_code)
    if not spot_map:
        print("[ERROR] 슬롯 정보 없음, spot_map 비어있음")
        raise HTTPException(status_code=404, detail=f"슬롯 정보 없음")

    positions = build_positions_from_db(all_coords)
    print("[INFO] positions / spot_map 로딩 완료")

    # 3️⃣ infer + DB 저장 + 시각화 이미지 생성
    try:
        car_exists, img_draw = infer_and_map(
            db=db,
            lot_code=lot_code,
            img_bgr=img,
            ROI_DATA=ROI_DATA,
            spot_map=spot_map,
            positions=positions,
        )
        print("[INFO] infer_and_map 완료")
    except Exception as e:
        print(f"[ERROR] 추론 실패: {e}")
        raise HTTPException(status_code=500, detail=f"추론 실패: {e}")

    # ⭐ 3-1️⃣ 현재 점유한 자리 수(occupied) + 실제 인식된 capacity 계산
    try:
        occupied_count = 0
        capacity = 0  # ✅ ROI가 있어서 실제로 인식 가능한 슬롯 수 (occupied + empty)

        rows = len(positions)
        cols = len(positions[0]) if rows > 0 else 0

        for i in range(rows):
            for j in range(cols):
                # positions[i][j] == 1 인 곳만 "주차 슬롯"으로 간주
                if positions[i][j] != 1:
                    continue

                status = car_exists[i][j]  # 0/1/2

                if status == 1:
                    # 차가 있는 자리 → occupied + capacity 둘 다 증가
                    occupied_count += 1
                    capacity += 1
                elif status == 0:
                    # 차는 없지만 ROI로 인식된 빈자리 → capacity만 증가
                    capacity += 1
                elif status == 2:
                    # ROI 없음 → 이번 스캔에서는 capacity에 포함하지 않음
                    # (실제 인식 불가능한 자리이므로 무시)
                    continue

        print(f"[INFO] 집계된 occupied_count = {occupied_count}, capacity = {capacity}")

        lot_name = '옥외주차장'
        status_cd = "1"

        history_row = ParkingLotHistory(
            lot_code=lot_code,
            lot_name=lot_name,
            status_cd=status_cd,
            capacity=capacity,          # ✅ 실제 인식된 슬롯 수로 반영
            occupied=occupied_count,    # ✅ 실제 차가 있는 슬롯 수
        )
        db.add(history_row)
        db.commit()
        print("[INFO] ParkingLotHistory insert 성공")
    except Exception as e:
        db.rollback()
        print(f"[WARN] ParkingLotHistory insert 실패: {e}")

    # 4️⃣ 원본 대신 ‘시각화된 이미지’를 저장
    try:
        cv2.imwrite(file_path, img_draw)
        print(f"[INFO] 시각화 이미지 저장 완료: {file_path}")
    except Exception as e:
        print(f"[ERROR] 이미지 저장 실패: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 저장 실패: {e}")

    # 5️⃣ Redis 발행
    realtime_payload = {
        "positions": positions,
        "carExists": car_exists,
        "ts": datetime.now(timezone.utc).isoformat(),  # ← 원하면 여기도 Asia/Seoul로 바꿀 수 있음
    }

    try:
        await redis.set("parking_detail_data", json.dumps(realtime_payload))
        await redis.publish("parking_detail_channel", "updated")
        print(f"[INFO] Redis 발행 완료: channel=parking_detail_channel, payload_ts={realtime_payload['ts']}")
    except Exception as e:
        print(f"[ERROR] Redis 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Redis 처리 실패: {e}")

    return {
        "filename": safe_name,
        "url": f"/upload_images/{safe_name}",
        "message": "분석 및 시각화 완료",
    }

# ========== 🔥 최신 이미지 확인 ==========
def _get_latest_image_path(upload_dir: str) -> Path | None:
    p = Path(upload_dir)
    if not p.exists():
        print("[WARN] _get_latest_image_path: 업로드 폴더 없음")
        return None
    files = [f for f in p.iterdir() if f.is_file()]
    if not files:
        print("[WARN] _get_latest_image_path: 파일 없음")
        return None
    latest = max(files, key=lambda f: f.stat().st_mtime)
    print(f"[INFO] 최신 이미지 파일: {latest}")
    return latest

@router.get("/img_latest", response_class=HTMLResponse)
def view_latest_image():
    latest = _get_latest_image_path(UPLOAD_DIR)
    if latest is None:
        return HTMLResponse("<h1>이미지가 없습니다.</h1>")

    img_url = f"/upload_images/{latest.name}"
    print(f"[INFO] /img_latest → {img_url}")

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
        print("[WARN] img_files: 업로드 폴더 없음")
        return HTMLResponse("<h3>업로드 폴더가 없습니다.</h3>", status_code=200)

    exts = {".jpg", ".jpeg", ".png", ".gif"}
    files = [
        f for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    ]

    print(f"[INFO] img_files: 이미지 파일 개수 = {len(files)}")

    if not files:
        return HTMLResponse("<h3>저장된 이미지가 없습니다.</h3>", status_code=200)

    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    rows_html = []
    for f in files:
        img_url = f"/upload_images/{f.name}"
        download_url = f"/api/v1/img/img_download?filename={f.name}"
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
