import pyaudio
import numpy as np
import sys
import os
import queue
import threading
import time
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel

# ─── CONFIG ───────────────────────────────────────────────
RATE              = 16000
CHANNELS          = 2
CHUNK             = 1280          # 80ms — required by openWakeWord
MODEL_SIZE        = "tiny"
DEVICE_TYPE       = "cpu"
COMPUTE_TYPE      = "int8"

SILENCE_THRESHOLD = 0.025
SPEECH_PAD_MS     = 200
MIN_SPEECH_MS     = 200
MAX_SPEECH_MS     = 8000

WAKE_WORD         = "alexa"
WAKE_THRESHOLD     = 0.5
WAKE_COOLDOWN_S    = 2.0
LISTEN_TIMEOUT_S   = 8.0    # seconds to wait for speech after activation
INACTIVITY_TIMEOUT = 60.0   # seconds before requiring wake word again
# ──────────────────────────────────────────────────────────

HALLUCINATIONS = {
    "thank you", "thanks", "thank you.", "thanks.", "huh", "huh?",
    "huh.", "uh", "uh.", "um", "um.", "hmm", "hmm.", "bye", "bye.",
    "goodbye", "goodbye.", "you", "you.", ".", "..", "...", "okay",
    "okay.", "ok", "ok.", "no", "no.", "yes", "yes.", "sure", "sure.",
    "i don't know", "i'm sorry", "sorry", "sorry.", "right", "right.",
    "oh", "oh.", "ah", "ah.", "what?", "what", "really", "really.",
    "wow", "wow.", "please", "please.", "hello", "hello.",
    "so", "so.", "well", "well.", "and", "and.",
}

def is_hallucination(text):
    return text.strip().lower().rstrip('.').strip() in HALLUCINATIONS or len(text.strip()) < 3

# ─── PIPELINE STATE ───────────────────────────────────────
class State:
    WAITING   = 0
    LISTENING = 1
    THINKING  = 2

_state      = [State.WAITING]
_state_lock = threading.Lock()
_wake_event = threading.Event()
_seg_event  = threading.Event()

def get_state():
    return _state[0]

def set_state(s):
    with _state_lock:
        _state[0] = s

# ─── QUEUES ───────────────────────────────────────────────
# Single audio queue — raw stereo int16 frames from pyaudio
# Each item: (ch0_int16, ch1_int16) tuple
audio_q      = queue.Queue(maxsize=50)
ch0_q        = queue.Queue(maxsize=200)
level_q      = queue.Queue(maxsize=8)
transcript_q = queue.Queue()

# ─── SHARED DISPLAY ───────────────────────────────────────
_disp_lock = threading.Lock()
_disp = {"db": -60.0, "last_wake": 0.0, "last_activity": 0.0}

def update_disp(**kw):
    with _disp_lock:
        _disp.update(kw)

def read_disp():
    with _disp_lock:
        return dict(_disp)

# ─── DEVICE DETECTION ─────────────────────────────────────
def find_respeaker(pa):
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if 'ReSpeaker' in d['name'] or 'Lite' in d['name']:
            if d['maxInputChannels'] >= 2:
                return i, d['name']
    return None, None

# ─── AUDIO HELPERS ────────────────────────────────────────
def rms_db(signal):
    rms = np.sqrt(np.mean(signal.astype(np.float32) ** 2))
    return 20 * np.log10(rms / 32768 + 1e-9)

def is_speech(chunk):
    rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2)) / 32768
    return rms > SILENCE_THRESHOLD

def level_bar(db_val, width=30):
    level = int(max(0, (db_val + 60) / 60 * width))
    empty = width - level
    if db_val > -6:
        color = '\033[91m'
    elif db_val > -20:
        color = '\033[93m'
    else:
        color = '\033[92m'
    return color + '█' * level + '\033[90m' + '░' * empty + '\033[0m'

# ─── AUDIO DISPATCHER ─────────────────────────────────────
# Reads raw stereo frames from audio_q and routes:
#   CH0 → level meter + STT queue (when LISTENING)
#   CH1 → wake word queue (always, as mono)
# This is the single fan-out point — everything else is downstream

ww_q = queue.Queue(maxsize=25)   # wake word — bounded, drop stale

def audio_dispatcher():
    while True:
        try:
            ch0, ch1 = audio_q.get(timeout=0.5)
        except queue.Empty:
            continue

        # always feed level meter
        try:
            level_q.put_nowait(ch0)
        except queue.Full:
            pass

        state = get_state()

        # wake word always gets audio when waiting
        if state == State.WAITING:
            try:
                ww_q.put_nowait(ch1)
            except queue.Full:
                # drop oldest to keep latency low
                try:
                    ww_q.get_nowait()
                    ww_q.put_nowait(ch1)
                except queue.Empty:
                    pass

        # STT gets CH0 when listening
        elif state == State.LISTENING:
            try:
                ch0_q.put_nowait(ch0)
            except queue.Full:
                pass

# ─── LEVEL METER THREAD ───────────────────────────────────
def level_thread():
    while True:
        try:
            chunk = level_q.get(timeout=0.5)
            update_disp(db=rms_db(chunk))
        except queue.Empty:
            pass

# ─── WAKE WORD THREAD ─────────────────────────────────────
def wake_word_thread(oww_model):
    last_detection = 0.0

    while True:
        try:
            # CH1 as mono int16 — exactly what openWakeWord needs
            chunk = ww_q.get(timeout=0.5)
        except queue.Empty:
            continue

        if get_state() != State.WAITING:
            continue

        now = time.time()
        if now - last_detection < WAKE_COOLDOWN_S:
            continue

        pred  = oww_model.predict(chunk)
        score = float(pred.get(WAKE_WORD, 0.0))

        if score > WAKE_THRESHOLD:
            last_detection = now
            update_disp(last_wake=now, last_activity=now)

            # flush stale ch0 audio
            while not ch0_q.empty():
                try:
                    ch0_q.get_nowait()
                except queue.Empty:
                    break

            set_state(State.LISTENING)
            _wake_event.set()

# ─── SEGMENTER ────────────────────────────────────────────
def segmenter(segment_q):
    silence_chunks_needed = max(1, round((SPEECH_PAD_MS / 1000) / (CHUNK / RATE)))
    max_chunks            = round((MAX_SPEECH_MS / 1000) / (CHUNK / RATE))

    while True:
        _wake_event.wait()
        _wake_event.clear()

        if get_state() != State.LISTENING:
            continue

        buffer        = []
        speaking      = False
        silence_count = 0
        listen_start  = time.time()

        while True:
            if get_state() == State.WAITING:
                break

            if not speaking and (time.time() - listen_start) > LISTEN_TIMEOUT_S:
                # check inactivity — if exceeded, require wake word again
                last_activity = read_disp()["last_activity"]
                if last_activity > 0 and (time.time() - last_activity) > INACTIVITY_TIMEOUT:
                    set_state(State.WAITING)
                else:
                    # still within active window — reset and keep listening
                    listen_start = time.time()
                break

            try:
                chunk = ch0_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if is_speech(chunk):
                speaking = True
                silence_count = 0
                buffer.append(chunk)
                listen_start = time.time()
            else:
                if speaking:
                    silence_count += 1
                    buffer.append(chunk)
                    if silence_count >= silence_chunks_needed:
                        seg = np.concatenate(buffer)
                        if len(seg) / RATE * 1000 >= MIN_SPEECH_MS:
                            set_state(State.THINKING)
                            segment_q.put(seg)
                            _seg_event.set()
                        else:
                            set_state(State.WAITING)
                        break

            if speaking and len(buffer) >= max_chunks:
                set_state(State.THINKING)
                segment_q.put(np.concatenate(buffer))
                _seg_event.set()
                break

# ─── TRANSCRIBER ──────────────────────────────────────────
def transcriber(segment_q, model):
    while True:
        _seg_event.wait()
        _seg_event.clear()

        try:
            segment = segment_q.get_nowait()
        except queue.Empty:
            continue

        audio_f32 = segment.astype(np.float32) / 32768.0
        segments, _ = model.transcribe(
            audio_f32,
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if text and not is_hallucination(text):
            transcript_q.put(text)
            update_disp(last_activity=time.time())

        while not ch0_q.empty():
            try:
                ch0_q.get_nowait()
            except queue.Empty:
                break

        # stay active for more commands — only go back to WAITING
        # if inactivity timeout exceeded (checked in segmenter)
        set_state(State.LISTENING)
        _wake_event.set()

# ─── UI THREAD ────────────────────────────────────────────
def ui_loop():
    transcripts = []
    os.system('clear')

    STATE_DISPLAY = {
        State.WAITING:   '\033[90m◉ waiting for wake word\033[0m      ',
        State.LISTENING: '\033[92m● listening...\033[0m               ',
        State.THINKING:  '\033[93m◎ transcribing...\033[0m            ',
    }

    while True:
        try:
            while True:
                t = transcript_q.get_nowait()
                ts = time.strftime("%H:%M:%S")
                transcripts.append((ts, t))
                if len(transcripts) > 12:
                    transcripts.pop(0)
        except queue.Empty:
            pass

        d             = read_disp()
        db0           = d["db"]
        current       = get_state()
        now           = time.time()
        last_activity = d["last_activity"]
        inactive_s    = (now - last_activity) if last_activity > 0 else None

        # session line
        if current == State.WAITING:
            session_str = '\033[90msay "Alexa" to activate\033[0m          '
        elif inactive_s is not None:
            remaining = max(0, INACTIVITY_TIMEOUT - inactive_s)
            if remaining > 10:
                session_str = f'\033[92m● active · resets in {int(remaining)}s\033[0m        '
            else:
                session_str = f'\033[93m● active · resets in {int(remaining)}s\033[0m        '
        else:
            session_str = '\033[92m● active\033[0m                          '

        sys.stdout.write('\033[H')
        print('\033[1m  ReSpeaker Lite — Wake Word + STT\033[0m')
        print(f'  Wake: "{WAKE_WORD}"  →  STT: faster-whisper {MODEL_SIZE} · int8')
        print('  ' + '─' * 62)
        print()
        print(f'  CH0     [{level_bar(db0)}]  {db0:6.1f} dBFS')
        print(f'  STT     {STATE_DISPLAY[current]}')
        print(f'  Session {session_str}')
        print()
        print('  ' + '─' * 62)
        print('  \033[1mTranscript\033[0m')
        print()

        if not transcripts:
            print('  \033[90m  (nothing yet)\033[0m')
        else:
            for ts, text in transcripts[-12:]:
                print(f'  \033[90m[{ts}]\033[0m  {text}')

        lines_used = 13 + len(transcripts)
        for _ in range(max(0, 26 - lines_used)):
            print()

        print('  ' + '─' * 62)
        print('  \033[90mCtrl+C to stop\033[0m')
        sys.stdout.flush()
        time.sleep(0.05)

# ─── MAIN ─────────────────────────────────────────────────
def main():
    pa = pyaudio.PyAudio()
    dev_idx, dev_name = find_respeaker(pa)
    if dev_idx is None:
        print("ERROR: ReSpeaker not found.")
        pa.terminate()
        sys.exit(1)

    os.system('clear')
    print(f"  Mic: [{dev_idx}] {dev_name}")
    print("  Loading faster-whisper model...")
    stt_model = WhisperModel(MODEL_SIZE, device=DEVICE_TYPE, compute_type=COMPUTE_TYPE)
    print("  STT model loaded.")
    print("  Loading openWakeWord model...")
    oww_model = WakeWordModel(wakeword_models=[WAKE_WORD], inference_framework='onnx')
    print(f'  Wake word ready: "{WAKE_WORD}"')
    print("  Starting...\n")
    time.sleep(1)

    # single stereo pyaudio stream
    stream = pa.open(
        rate=RATE,
        channels=CHANNELS,
        format=pyaudio.paInt16,
        input=True,
        input_device_index=dev_idx,
        frames_per_buffer=CHUNK
    )

    segment_q = queue.Queue()

    threading.Thread(target=audio_dispatcher,                              daemon=True).start()
    threading.Thread(target=level_thread,                                  daemon=True).start()
    threading.Thread(target=wake_word_thread, args=(oww_model,),           daemon=True).start()
    threading.Thread(target=segmenter,        args=(segment_q,),           daemon=True).start()
    threading.Thread(target=transcriber,      args=(segment_q, stt_model), daemon=True).start()
    threading.Thread(target=ui_loop,                                       daemon=True).start()

    print("  Pipeline running. Reading audio...")

    try:
        while True:
            # read stereo frames, split into CH0 and CH1
            raw = stream.read(CHUNK, exception_on_overflow=False)
            stereo = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
            ch0 = stereo[:, 0].copy()
            ch1 = stereo[:, 1].copy()
            try:
                audio_q.put_nowait((ch0, ch1))
            except queue.Full:
                pass
    except KeyboardInterrupt:
        print('\n\n  Stopped.')
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

if __name__ == '__main__':
    main()