# ─── pipeline.py ──────────────────────────────────────────
# The audio engine. Handles device detection, audio routing,
# wake word detection, VAD segmentation, and STT transcription.
#
# Zero knowledge of UI or transport — all output goes through callbacks:
#   on_transcript(text: str)
#   on_state_change(state: int)   — State.WAITING / LISTENING / THINKING
#   on_wake()
#   on_level(db: float)           — called every chunk for level metering

import pyaudio
import numpy as np
import queue
import threading
import time
import sys

from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel

import config


# ─── STATE ────────────────────────────────────────────────
class State:
    WAITING   = 0
    LISTENING = 1
    THINKING  = 2


# ─── PIPELINE CLASS ───────────────────────────────────────
class Pipeline:
    def __init__(self,
                 on_transcript=None,
                 on_state_change=None,
                 on_wake=None,
                 on_level=None):

        self.on_transcript   = on_transcript   or (lambda text: None)
        self.on_state_change = on_state_change or (lambda s: None)
        self.on_wake         = on_wake         or (lambda: None)
        self.on_level        = on_level        or (lambda db: None)

        # internal state
        self._state      = State.WAITING
        self._state_lock = threading.Lock()
        self._wake_event = threading.Event()
        self._seg_event  = threading.Event()

        # timing
        self._last_activity = 0.0

        # queues
        self._audio_q  = queue.Queue(maxsize=50)
        self._ch0_q    = queue.Queue(maxsize=200)
        self._ww_q     = queue.Queue(maxsize=25)
        self._level_q  = queue.Queue(maxsize=8)
        self._seg_q    = queue.Queue()

        # pyaudio
        self._pa     = None
        self._stream = None

        # models (loaded in start())
        self._stt_model = None
        self._oww_model = None

    # ── STATE HELPERS ─────────────────────────────────────
    def _get_state(self):
        return self._state

    def _set_state(self, s):
        with self._state_lock:
            self._state = s
        self.on_state_change(s)

    # ── DEVICE DETECTION ──────────────────────────────────
    @staticmethod
    def find_device(pa):
        """Find ReSpeaker Lite device index. Returns (index, name) or (None, None)."""
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if ('ReSpeaker' in d['name'] or 'Lite' in d['name']) and d['maxInputChannels'] >= 2:
                return i, d['name']
        return None, None

    # ── AUDIO HELPERS ─────────────────────────────────────
    @staticmethod
    def rms_db(signal):
        rms = np.sqrt(np.mean(signal.astype(np.float32) ** 2))
        return 20 * np.log10(rms / 32768 + 1e-9)

    @staticmethod
    def is_speech(chunk):
        rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2)) / 32768
        return rms > config.SILENCE_THRESHOLD

    @staticmethod
    def is_hallucination(text):
        return (text.strip().lower().rstrip('.').strip() in config.HALLUCINATIONS
                or len(text.strip()) < 3)

    # ── THREADS ───────────────────────────────────────────
    def _dispatcher(self):
        """Fan-out: routes CH0 and CH1 to the right queues."""
        while True:
            try:
                ch0, ch1 = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            # level meter — always
            try:
                self._level_q.put_nowait(ch0)
            except queue.Full:
                pass

            state = self._get_state()

            if state == State.WAITING:
                # CH1 → wake word engine (bounded, drop oldest if full)
                try:
                    self._ww_q.put_nowait(ch1)
                except queue.Full:
                    try:
                        self._ww_q.get_nowait()
                        self._ww_q.put_nowait(ch1)
                    except queue.Empty:
                        pass

            elif state == State.LISTENING:
                # CH0 → segmenter
                try:
                    self._ch0_q.put_nowait(ch0)
                except queue.Full:
                    pass

    def _level_thread(self):
        while True:
            try:
                chunk = self._level_q.get(timeout=0.5)
                self.on_level(self.rms_db(chunk))
            except queue.Empty:
                pass

    def _wake_word_thread(self):
        last_detection = 0.0
        while True:
            try:
                chunk = self._ww_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if self._get_state() != State.WAITING:
                continue

            now = time.time()
            if now - last_detection < config.WAKE_COOLDOWN_S:
                continue

            pred  = self._oww_model.predict(chunk)
            score = float(pred.get(config.WAKE_WORD, 0.0))

            if score > config.WAKE_THRESHOLD:
                last_detection      = now
                self._last_activity = now

                # flush stale CH0 audio
                while not self._ch0_q.empty():
                    try:
                        self._ch0_q.get_nowait()
                    except queue.Empty:
                        break

                self._set_state(State.LISTENING)
                self.on_wake()
                self._wake_event.set()

    def _segmenter(self):
        silence_needed = max(1, round((config.SPEECH_PAD_MS / 1000) / (config.CHUNK / config.RATE)))
        max_chunks     = round((config.MAX_SPEECH_MS / 1000) / (config.CHUNK / config.RATE))

        while True:
            self._wake_event.wait()
            self._wake_event.clear()

            if self._get_state() != State.LISTENING:
                continue

            buffer        = []
            speaking      = False
            silence_count = 0
            listen_start  = time.time()

            while True:
                if self._get_state() == State.WAITING:
                    break

                # timeout handling
                if not speaking and (time.time() - listen_start) > config.LISTEN_TIMEOUT_S:
                    if (self._last_activity > 0 and
                            (time.time() - self._last_activity) > config.INACTIVITY_TIMEOUT):
                        self._set_state(State.WAITING)
                    else:
                        listen_start = time.time()   # reset — still in active window
                    break

                try:
                    chunk = self._ch0_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self.is_speech(chunk):
                    speaking = True
                    silence_count = 0
                    buffer.append(chunk)
                    listen_start = time.time()
                else:
                    if speaking:
                        silence_count += 1
                        buffer.append(chunk)
                        if silence_count >= silence_needed:
                            seg = np.concatenate(buffer)
                            if len(seg) / config.RATE * 1000 >= config.MIN_SPEECH_MS:
                                self._set_state(State.THINKING)
                                self._seg_q.put(seg)
                                self._seg_event.set()
                            else:
                                self._set_state(State.WAITING)
                            break

                if speaking and len(buffer) >= max_chunks:
                    self._set_state(State.THINKING)
                    self._seg_q.put(np.concatenate(buffer))
                    self._seg_event.set()
                    break

    def _transcriber(self):
        while True:
            self._seg_event.wait()
            self._seg_event.clear()

            try:
                segment = self._seg_q.get_nowait()
            except queue.Empty:
                continue

            audio_f32 = segment.astype(np.float32) / 32768.0
            segments, _ = self._stt_model.transcribe(
                audio_f32,
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False
            )
            text = " ".join(s.text.strip() for s in segments).strip()

            if text and not self.is_hallucination(text):
                self._last_activity = time.time()
                self.on_transcript(text)

            # flush stale audio from during transcription
            while not self._ch0_q.empty():
                try:
                    self._ch0_q.get_nowait()
                except queue.Empty:
                    break

            # stay active for next command
            self._set_state(State.LISTENING)
            self._wake_event.set()

    # ── PUBLIC API ────────────────────────────────────────
    def start(self, device_index=None):
        """Load models, open audio stream, start all threads."""
        self._pa = pyaudio.PyAudio()

        if device_index is None:
            device_index, name = self.find_device(self._pa)
            if device_index is None:
                self._pa.terminate()
                raise RuntimeError("ReSpeaker Lite not found. Is it plugged in? Run: arecord -l")
            print(f"  Mic: [{device_index}] {name}")

        print("  Loading STT model...")
        self._stt_model = WhisperModel(
            config.MODEL_SIZE,
            device=config.DEVICE_TYPE,
            compute_type=config.COMPUTE_TYPE
        )
        print("  STT model loaded.")

        print(f'  Loading wake word model "{config.WAKE_WORD}"...')
        self._oww_model = WakeWordModel(
            wakeword_models=[config.WAKE_WORD],
            inference_framework='onnx'
        )
        print("  Wake word model loaded.")

        # open single stereo stream
        self._stream = self._pa.open(
            rate=config.RATE,
            channels=config.CHANNELS,
            format=pyaudio.paInt16,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=config.CHUNK
        )

        # start worker threads
        for target, args in [
            (self._dispatcher,       ()),
            (self._level_thread,     ()),
            (self._wake_word_thread, ()),
            (self._segmenter,        ()),
            (self._transcriber,      ()),
        ]:
            threading.Thread(target=target, args=args, daemon=True).start()

        print("  Pipeline started.\n")

    def run(self):
        """Block and feed audio into the pipeline. Call after start()."""
        try:
            while True:
                raw    = self._stream.read(config.CHUNK, exception_on_overflow=False)
                stereo = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
                ch0    = stereo[:, 0].copy()
                ch1    = stereo[:, 1].copy()
                try:
                    self._audio_q.put_nowait((ch0, ch1))
                except queue.Full:
                    pass
        finally:
            self.stop()

    def stop(self):
        """Clean up stream and PyAudio."""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()

    @property
    def last_activity(self):
        return self._last_activity
