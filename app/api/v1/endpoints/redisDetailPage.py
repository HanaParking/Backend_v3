import asyncio
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.dependencies import get_redis
import json

router = APIRouter()

@router.post("/set")
async def set_data(redis = Depends(get_redis)) :
    data = {
        "positions" : [[1,1,1,1,1,1,1,1,1,1,0,1],[1,1,1,1,1,1,1,1,1,1,0,0]],
        "carExists" : [[True,False,False,False,False,False,False,True,False,False,False,False]
                    ,[True,True,False,False,False,False,False,True,False,False,False,True]]
    }
    # redis에 data 저장
    await redis.set("parking_detail_data", json.dumps(data))
    # Pub/Sub 채널에 알림 발행
    await redis.publish("parking_detail_channel", "updated")
    return {"message" : "data set ok"}

@router.get("/subscribe")
async def subscribe(redis = Depends(get_redis)):
    async def event_generator():
        pubsub = redis.pubsub()
        await pubsub.subscribe("parking_detail_channel")

        try:
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=5.0)

                if message:
                    data = await redis.get("parking_detail_data")
                    if data:
                        yield f"data: {data}\n\n"
                    else:
                        yield 'data: {"error": "no data"}\n\n'
                else:
                    # keep-alive
                    yield ":\n\n"

                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            print("SSE 연결 해제 (CancelledError)")
            # 필요하면 여기서도 unsubscribe
            await pubsub.unsubscribe("parking_detail_channel")
            raise
        finally:
            # ✅ 진짜 커넥션 닫기
            await pubsub.close()
            print("PubSub 커넥션 close 완료")

    return StreamingResponse(event_generator(), media_type="text/event-stream")



# Pub/Sub 아닌 단순 Redis에 세팅된 data값 가져오기
@router.get("/get")
async def get_data(redis = Depends(get_redis)) :
    data = await redis.get("parking_detail_data")
    if data :
        return json.loads(data)
    return {"error" : "can not be found"}