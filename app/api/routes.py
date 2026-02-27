from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import numpy as np
from app.core.dependencies import get_ocr_service, get_yolo_service
from app.schemas.models import OfferRequest
from app.services.webrtc_service import WebRTCService
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webrtc", tags=["WebRTC"])

_webrtc_service: Optional[WebRTCService] = None
_main_loop: Optional[asyncio.AbstractEventLoop] = None  # ✅ simpan loop di module level


def get_webrtc_service() -> WebRTCService:
    global _webrtc_service
    if _webrtc_service is None:
        _webrtc_service = WebRTCService()
    return _webrtc_service


# ─── Connection Manager ───────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info("WebSocket terhubung. Total: %d", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info("WebSocket terputus. Total: %d", len(self._connections))

    async def broadcast(self, message: dict) -> None:
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()

class YOLOThrottle:
    INTERVAL = 0.3

    def __init__(self) -> None:
        self._last_ts: float = 0.0

    def should_run(self) -> bool:
        return (time.perf_counter() - self._last_ts) >= self.INTERVAL

    def mark(self) -> None:
        self._last_ts = time.perf_counter()


# ─── WebRTC Offer — KTP ───────────────────────────────────────────────────────

@router.post("/offer")
async def offer(payload: OfferRequest) -> dict:
    global _main_loop
    _main_loop = asyncio.get_running_loop()

    service = get_webrtc_service()
    throttle = YOLOThrottle()

    def on_frame(frame: np.ndarray) -> None:
        svc = get_yolo_service()
        if svc is None or _main_loop is None:
            logger.warning("on_frame: service atau loop belum siap")
            return

        svc.store_frame(frame)

        if not throttle.should_run():
            return

        throttle.mark()
        logger.debug("on_frame: scheduling _run_yolo")
        asyncio.run_coroutine_threadsafe(
            _run_yolo(frame, svc), _main_loop  # ✅ pakai _main_loop
        )

    try:
        answer = await service.handle_offer(
            sdp=payload.sdp,
            type_=payload.type,
            on_frame=on_frame,
        )
        return answer
    except Exception as e:
        logger.error("Gagal handle offer: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ─── WebRTC Offer — Face ──────────────────────────────────────────────────────

@router.post("/offer/face")
async def offer_face(payload: OfferRequest) -> dict:
    global _main_loop
    _main_loop = asyncio.get_running_loop()

    service = get_webrtc_service()

    def on_frame(frame: np.ndarray) -> None:
        if _main_loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            manager.broadcast({"event": "frame_received"}),
            _main_loop,
        )

    try:
        answer = await service.handle_offer(
            sdp=payload.sdp,
            type_=payload.type,
            on_frame=on_frame,
        )
        return answer
    except Exception as e:
        logger.error("Gagal handle offer face: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ─── YOLO Runner ─────────────────────────────────────────────────────────────

async def _run_yolo(frame: np.ndarray, yolo_service) -> None:
    loop = asyncio.get_running_loop()

    try:
        boxes = await loop.run_in_executor(None, yolo_service.predict, frame)

        if boxes:
            yolo_service.store_box(boxes[0])
            await manager.broadcast({
                "event": "yolo_result",
                "boxes": [b.to_dict() for b in boxes],
            })
        else:
            yolo_service.store_box(None)
            await manager.broadcast({"event": "no_ktp"})

    except Exception as e:
        logger.error("YOLO predict error: %s", e)


# ─── WebSocket Notify ─────────────────────────────────────────────────────────

@router.websocket("/ws/notify")
async def notify(ws: WebSocket) -> None:
    await manager.connect(ws)

    try:
        await ws.send_json({"event": "connected", "message": "Siap menerima notifikasi."})

        while True:
            data = await ws.receive_json()
            event = data.get("event")

            if event == "capture":
                await _handle_capture(ws)
            elif event == "ping":
                await ws.send_json({"event": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        manager.disconnect(ws)


async def _handle_capture(ws: WebSocket) -> None:
    ocr_service = get_ocr_service()
    yolo_service = get_yolo_service()
    loop = asyncio.get_running_loop()

    if yolo_service.last_frame is None:
        await ws.send_json({
            "event": "capture_failed",
            "reason": "Belum ada frame yang diterima.",
        })
        return

    if yolo_service.last_box is None:
        await ws.send_json({
            "event": "capture_failed",
            "reason": "KTP belum terdeteksi. Arahkan KTP ke kamera.",
        })
        return

    await ws.send_json({
        "event": "capture_processing",
        "message": "Memproses OCR...",
    })

    try:
        frame = yolo_service.last_frame
        box = yolo_service.last_box
        cropped = yolo_service.crop(frame, box)

        ktp_data = await loop.run_in_executor(
            None, ocr_service.extract_from_array, cropped
        )

        await ws.send_json({
            "event": "ktp_result",
            "data": ktp_data.to_dict(),
        })

        logger.info(
            "Capture selesai | completeness=%.0f%% | NIK=%s",
            ktp_data.completeness * 100,
            ktp_data.nik or "NOT FOUND",
        )

    except Exception as e:
        logger.error("Capture OCR error: %s", e)
        await ws.send_json({
            "event": "capture_failed",
            "reason": f"OCR error: {str(e)}",
        })
