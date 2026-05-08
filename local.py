# ─── local.py ─────────────────────────────────────────────
# Terminal UI entry point.
# Run this on a Pi or laptop with the ReSpeaker plugged in via USB.
#
# Usage:
#   python3 local.py

import sys
import os
import time
import threading

import config
from pipeline import Pipeline, State


# ─── DISPLAY HELPERS ──────────────────────────────────────
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


# ─── SHARED UI STATE ──────────────────────────────────────
_ui_lock      = threading.Lock()
_transcripts  = []
_current_db   = -60.0
_current_state = State.WAITING


def _add_transcript(text):
    global _transcripts
    ts = time.strftime("%H:%M:%S")
    with _ui_lock:
        _transcripts.append((ts, text))
        if len(_transcripts) > 12:
            _transcripts.pop(0)

def _set_db(db):
    global _current_db
    with _ui_lock:
        _current_db = db

def _set_state(state):
    global _current_state
    with _ui_lock:
        _current_state = state


# ─── UI LOOP ──────────────────────────────────────────────
def ui_loop(pipeline):
    STATE_DISPLAY = {
        State.WAITING:   '\033[90m◉ waiting for wake word\033[0m      ',
        State.LISTENING: '\033[92m● listening...\033[0m               ',
        State.THINKING:  '\033[93m◎ transcribing...\033[0m            ',
    }

    os.system('clear')

    while True:
        with _ui_lock:
            db0      = _current_db
            current  = _current_state
            transcripts = list(_transcripts)

        now           = time.time()
        last_activity = pipeline.last_activity
        inactive_s    = (now - last_activity) if last_activity > 0 else None

        if current == State.WAITING:
            session_str = f'\033[90msay "{config.WAKE_WORD}" to activate\033[0m     '
        elif inactive_s is not None:
            remaining = max(0, config.INACTIVITY_TIMEOUT - inactive_s)
            color = '\033[92m' if remaining > 10 else '\033[93m'
            session_str = f'{color}● active · resets in {int(remaining)}s\033[0m       '
        else:
            session_str = '\033[92m● active\033[0m                          '

        sys.stdout.write('\033[H')
        print('\033[1m  ReSpeaker Lite — Wake Word + STT\033[0m')
        print(f'  Wake: "{config.WAKE_WORD}"  →  STT: faster-whisper {config.MODEL_SIZE} · {config.COMPUTE_TYPE}')
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
    pipeline = Pipeline(
        on_transcript=_add_transcript,
        on_state_change=_set_state,
        on_wake=lambda: None,
        on_level=_set_db,
    )

    pipeline.start()

    threading.Thread(target=ui_loop, args=(pipeline,), daemon=True).start()

    try:
        pipeline.run()
    except KeyboardInterrupt:
        print('\n\n  Stopped.')


if __name__ == '__main__':
    main()
