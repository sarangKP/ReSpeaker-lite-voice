# ─── server.py ────────────────────────────────────────────
# WebSocket server entry point.
# Clients connect, stream raw S16_LE 16kHz mono audio (binary),
# and receive JSON messages back in real-time.
#
# Usage:
#   python3 server.py
#
# Install extra dependency:
#   pip install websockets
#
# ── WebSocket Protocol ────────────────────────────────────
#
# Client → Server (binary):
#   Raw audio bytes — S16_LE, 16kHz, mono, any chunk size
#   The server accumulates and processes internally.
#
# Server → Client (JSON text):
#   {"type": "transcript", "text": "...",  "ts": "HH:MM:SS"}
#   {"type": "state",      "state": "waiting|listening|thinking"}
#   {"type": "wake",       "word": "alexa"}
#   {"type": "level",      "db": -32.5}
#   {"type": "ready"}   — sent once on connect
#   {"type": "error",      "message": "..."}
#
# ── Multi-client ─────────────────────────────────────────
# Each client gets its own pipeline instance.
# Multiple clients can connect simultaneously and independently.

import asyncio
import json
import time
import threading
import numpy as np

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets")
    raise

import config
from pipeline import Pipeline, State

STATE_NAMES = {
    State.WAITING:   "waiting",
    State.LISTENING: "listening",
    State.THINKING:  "thinking",
}


# ─── PER-CLIENT HANDLER ───────────────────────────────────
async def handle_client(websocket):
    remote = websocket.remote_address
    print(f"  [+] Client connected: {remote}")

    send_q = asyncio.Queue()
    loop   = asyncio.get_event_loop()

    def send(msg: dict):
        """Thread-safe — called from pipeline worker threads."""
        asyncio.run_coroutine_threadsafe(send_q.put(json.dumps(msg)), loop)

    # wire pipeline callbacks to websocket sends
    pipeline = Pipeline(
        on_transcript   = lambda text: send({"type": "transcript", "text": text,
                                              "ts": time.strftime("%H:%M:%S")}),
        on_state_change = lambda s:    send({"type": "state", "state": STATE_NAMES[s]}),
        on_wake         = lambda:      send({"type": "wake",  "word": config.WAKE_WORD}),
        on_level        = lambda db:   send({"type": "level", "db": round(db, 1)}),
    )

    try:
        pipeline.start()
    except RuntimeError as e:
        await websocket.send(json.dumps({"type": "error", "message": str(e)}))
        print(f"  [!] Pipeline failed for {remote}: {e}")
        return

    await websocket.send(json.dumps({"type": "ready"}))

    # sender coroutine — drains send_q and writes to websocket
    async def sender():
        while True:
            msg = await send_q.get()
            try:
                await websocket.send(msg)
            except websockets.ConnectionClosed:
                break

    # receiver coroutine — reads audio bytes from client, feeds pipeline
    async def receiver():
        async for message in websocket:
            if isinstance(message, bytes):
                # incoming audio: S16_LE mono — feed into pipeline audio queue
                chunk = np.frombuffer(message, dtype=np.int16)
                # duplicate to stereo (both channels same) since pipeline expects stereo
                # if client sends only CH0 mono, we duplicate for wake word channel too
                stereo = np.stack([chunk, chunk], axis=1) if chunk.ndim == 1 else chunk
                ch0 = stereo[:, 0].copy()
                ch1 = stereo[:, 1].copy() if stereo.shape[1] > 1 else ch0.copy()
                try:
                    pipeline._audio_q.put_nowait((ch0, ch1))
                except Exception:
                    pass

    sender_task   = asyncio.create_task(sender())
    receiver_task = asyncio.create_task(receiver())

    try:
        await asyncio.gather(sender_task, receiver_task)
    except websockets.ConnectionClosed:
        pass
    finally:
        pipeline.stop()
        sender_task.cancel()
        receiver_task.cancel()
        print(f"  [-] Client disconnected: {remote}")


# ─── MAIN ─────────────────────────────────────────────────
async def main():
    print(f"\n  ReSpeaker Lite — WebSocket STT Server")
    print(f"  Listening on ws://{config.WS_HOST}:{config.WS_PORT}")
    print(f"  Wake word: {config.WAKE_WORD}  |  STT: faster-whisper {config.MODEL_SIZE}")
    print(f"  Ctrl+C to stop\n")

    async with websockets.serve(handle_client, config.WS_HOST, config.WS_PORT):
        await asyncio.Future()   # run forever


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\n\n  Stopped.')
