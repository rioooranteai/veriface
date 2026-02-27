from __future__ import annotations

import asyncio
import logging
from typing import Callable

import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole

logger = logging.getLogger(__name__)


class WebRTCService:

    def __init__(self) -> None:
        self._peer_connections: set[RTCPeerConnection] = set()

    async def handle_offer(
            self,
            sdp: str,
            type_: str,
            on_frame: Callable[[np.ndarray], None],
    ) -> dict:
        pc = RTCPeerConnection()
        self._peer_connections.add(pc)
        sink = MediaBlackhole()

        async def _consume_track(
                track: MediaStreamTrack,
                callback: Callable[[np.ndarray], None],
        ) -> None:

            frame_count = 0
            while True:
                try:
                    frame = await track.recv()
                    img = frame.to_ndarray(format="bgr24")
                    frame_count += 1

                    callback(img)
                except Exception as e:
                    logger.info("Track ended atau error: %s", e)
                    break

        @pc.on("track")
        async def on_track(track: MediaStreamTrack) -> None:
            if track.kind != "video":
                await sink.addTrack(track)
                return

            logger.info("Video track diterima dari peer.")
            asyncio.ensure_future(_consume_track(track, on_frame))

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            logger.info("WebRTC state: %s", pc.connectionState)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self._cleanup(pc)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
        await sink.start()

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }

    async def _cleanup(self, pc: RTCPeerConnection) -> None:
        await pc.close()
        self._peer_connections.discard(pc)
        logger.info("Peer connection ditutup dan dibersihkan.")

    async def close_all(self) -> None:
        await asyncio.gather(*[pc.close() for pc in self._peer_connections])
        self._peer_connections.clear()
