# ─── pipeline.py ──────────────────────────────────────────
# The audio engine. Handles device detection, audio routing,
# wake word detection, VAD segmentation, and STT transcription.

import os
import contextlib

# Suppress C-level stderr noise (onnxruntime GPU probe, ALSA JACK errors)
@contextlib.contextmanager
def _quiet_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved   = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)

import numpy as np
import queue
import threading
import time

with _quiet_stderr():
    import pyaudio
    from faster_whisper import WhisperModel
    from openwakeword.model import Model as WakeWordModel

import config


# ─── STATE ────────────────────────────────────────────────
class State:
    WAITING   = 0
    LISTENING = 1
    THINKING  = 2


# ─── HELPERS ──────────────────────────────────────────────
def _get_oww_model_path(name: str) -> str:
    """
    Resolve a wake word name to its .onnx file path.
    Accepts a bare name ("alexa"), versioned name ("alexa_v0.1"), or an
    absolute file path. Searches the openwakeword bundled dir first, then
    auto-downloads from GitHub if not found (handles openwakeword 0.6+
    which no longer bundles models).
    """
    if os.path.isfile(name):
        return name

    base_name = name.split('_v')[0]  # "alexa_v0.1" → "alexa"

    try:
        import openwakeword as _oww
        import openwakeword.utils
        models_dir = os.path.join(os.path.dirname(_oww.__file__), 'resources', 'models')

        def _search():
            if not os.path.isdir(models_dir):
                return None
            for fname in os.listdir(models_dir):
                if fname.lower().startswith(base_name.lower()) and fname.endswith('.onnx'):
                    return os.path.join(models_dir, fname)
            return None

        path = _search()
        if path:
            print(f"  Wake word model: {os.path.basename(path)}")
            return path

        if os.path.isdir(models_dir):
            bundled = sorted(
                f[:-5] for f in os.listdir(models_dir)
                if f.endswith('.onnx') and not f.startswith(('embedding', 'melspec', 'silero'))
            )
            if bundled:
                print(f"  Bundled wake words: {bundled}")

        print(f"  Downloading wake word model '{base_name}' (first-run, one-time)...")
        openwakeword.utils.download_models([base_name])
        path = _search()
        if path:
            print(f"  Wake word model: {os.path.basename(path)}")
            return path

    except FileNotFoundError:
        raise
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load wake word model '{name}': {e}\n"
            f"Try manually: python3 -c \"import openwakeword; "
            f"openwakeword.utils.download_models(['{base_name}'])\""
        ) from e

    raise FileNotFoundError(
        f"Wake word model '{name}' not found.\n"
        f"Built-in options: alexa, hey_jarvis, hey_mycroft, hey_marvin\n"
        f"Custom model: set WAKE_WORD to an absolute .onnx file path in config.py"
    )


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

        self._state      = State.WAITING
        self._state_lock = threading.Lock()
        self._wake_event = threading.Event()
        self._seg_event  = threading.Event()

        self._last_activity = 0.0

        self._audio_q  = queue.Queue(maxsize=50)
        self._ch0_q    = queue.Queue(maxsize=200)
        self._ww_q     = queue.Queue(maxsize=25)
        self._level_q  = queue.Queue(maxsize=8)
        self._seg_q    = queue.Queue()

        self._pa     = None
        self._stream = None

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
        while True:
            try:
                ch0, ch1 = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._level_q.put_nowait(ch0)
            except queue.Full:
                pass

            state = self._get_state()

            if state == State.WAITING:
                try:
                    self._ww_q.put_nowait(ch1)
                except queue.Full:
                    try:
                        self._ww_q.get_nowait()
                        self._ww_q.put_nowait(ch1)
                    except queue.Empty:
                        pass

            elif state == State.LISTENING:
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
            score = max(
                (float(v) for k, v in pred.items() if config.WAKE_WORD.split('_v')[0] in k),
                default=0.0
            )

            if score > config.WAKE_THRESHOLD:
                last_detection      = now
                self._last_activity = now

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

                if not speaking and (time.time() - listen_start) > config.LISTEN_TIMEOUT_S:
                    if (self._last_activity > 0 and
                            (time.time() - self._last_activity) > config.INACTIVITY_TIMEOUT):
                        self._set_state(State.WAITING)
                    else:
                        listen_start = time.time()
                    break

                try:
                    chunk = self._ch0_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self.is_speech(chunk):
                    speaking      = True
                    silence_count = 0
                    buffer.append(chunk)
                    listen_start  = time.time()
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

            while not self._ch0_q.empty():
                try:
                    self._ch0_q.get_nowait()
                except queue.Empty:
                    break

            self._set_state(State.LISTENING)
            self._wake_event.set()

    # ── PUBLIC API ────────────────────────────────────────
    def start(self, device_index=None):
        with _quiet_stderr():
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

        # ── FIX: resolve name → .onnx file path for openWakeWord 0.4.0 ──
        model_path = _get_oww_model_path(config.WAKE_WORD)
        self._oww_model = WakeWordModel(
            wakeword_model_paths=[model_path]
        )
        print("  Wake word model loaded.")

        try:
            with _quiet_stderr():  # silence ALSA mmap-probe warnings on Pi 5
                self._stream = self._pa.open(
                    rate=config.RATE,
                    channels=config.CHANNELS,
                    format=pyaudio.paInt16,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=config.CHUNK
                )
        except OSError as e:
            self._pa.terminate()
            raise RuntimeError(
                f"Could not open audio stream at {config.RATE}Hz: {e}\n"
                f"Check that the ReSpeaker Lite USB mic is plugged in and not in use by another app."
            )

        for target in [
            self._dispatcher,
            self._level_thread,
            self._wake_word_thread,
            self._segmenter,
            self._transcriber,
        ]:
            threading.Thread(target=target, daemon=True).start()

        print("  Pipeline started.\n")

    def run(self):
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
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()

    @property
    def last_activity(self):
        return self._last_activity