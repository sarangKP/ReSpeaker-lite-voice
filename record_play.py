#!/usr/bin/env python3
# record_play.py — Record from ReSpeaker Lite and play back on your laptop via SSH.
#
# ── Usage ────────────────────────────────────────────────────────────────────
#
#  On the Pi (saves wav locally, then plays it there):
#    python3 record_play.py --duration 5
#
#  From your LAPTOP (records on Pi, audio comes out of your laptop speakers):
#    ssh pi@<pi-ip> "cd ~/ReSpeaker-lite-voice && .venv/bin/python record_play.py --stream --duration 5" | aplay -f S16_LE -r 16000 -c 1
#
#  If your laptop is macOS, replace aplay with:
#    | sox -t raw -r 16000 -e signed -b 16 -c 1 - -d
#
# ── Options ──────────────────────────────────────────────────────────────────
#  --duration N     seconds to record (default: 5)
#  --stream         write raw S16_LE mono PCM to stdout instead of saving/playing locally
#  --output FILE    save recording to FILE (default: recording.wav)
#  --device INDEX   force a specific PyAudio device index

import os
os.environ.setdefault('ALSA_CARD', 'Lite')

import sys
import wave
import argparse
import contextlib
import time

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

with _quiet_stderr():
    import pyaudio

RATE     = 16000
CHANNELS = 2       # ReSpeaker USB is stereo; we'll mix down to mono
CHUNK    = 1024
FORMAT   = pyaudio.paInt16


def find_respeaker(pa):
    """Return (index, name) of the ReSpeaker USB mic."""
    candidates = []
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if ('ReSpeaker' in d['name'] or 'Lite' in d['name']) and d['maxInputChannels'] >= 2:
            candidates.append((i, d))
    for i, d in candidates:
        if 'USB' in d['name'] and d['maxInputChannels'] == 2:
            return i, d['name']
    if candidates:
        return candidates[0][0], candidates[0][1]['name']
    return None, None


def record(duration, device_index=None, stream_to_stdout=False, output_file='recording.wav'):
    with _quiet_stderr():
        pa = pyaudio.PyAudio()

    if device_index is None:
        device_index, name = find_respeaker(pa)
        if device_index is None:
            pa.terminate()
            sys.exit('ReSpeaker Lite not found. Is it plugged in?')
        print(f'Mic: [{device_index}] {name}', file=sys.stderr)

    with _quiet_stderr():
        stream = pa.open(
            rate=RATE,
            channels=CHANNELS,
            format=FORMAT,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
        )

    n_chunks = int(RATE / CHUNK * duration)

    if stream_to_stdout:
        # Write raw S16_LE mono PCM directly to stdout for SSH piping
        print(f'Recording {duration}s → streaming to stdout...', file=sys.stderr)
        stdout_buf = sys.stdout.buffer
        for _ in range(n_chunks):
            raw    = stream.read(CHUNK, exception_on_overflow=False)
            import struct, array as arr
            # Mix stereo → mono by averaging the two channels
            stereo = arr.array('h', raw)
            mono   = arr.array('h', (
                (stereo[i] + stereo[i + 1]) // 2
                for i in range(0, len(stereo), 2)
            ))
            stdout_buf.write(mono.tobytes())
        stdout_buf.flush()
    else:
        # Record to file
        print(f'Recording {duration}s to {output_file} ...', file=sys.stderr)
        frames = []
        for i in range(n_chunks):
            raw    = stream.read(CHUNK, exception_on_overflow=False)
            import array as arr
            stereo = arr.array('h', raw)
            mono   = arr.array('h', (
                (stereo[j] + stereo[j + 1]) // 2
                for j in range(0, len(stereo), 2)
            ))
            frames.append(mono.tobytes())
            # simple progress bar
            pct = int((i + 1) / n_chunks * 30)
            print(f'\r  [{"█" * pct}{"░" * (30 - pct)}] {i * CHUNK / RATE:.1f}s',
                  end='', file=sys.stderr)
        print('', file=sys.stderr)

        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(pa.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        print(f'Saved: {output_file}', file=sys.stderr)

    stream.stop_stream()
    stream.close()
    pa.terminate()

    return output_file


def play(wav_file):
    """Play a WAV file through the local default output device."""
    with _quiet_stderr():
        pa = pyaudio.PyAudio()

    with wave.open(wav_file, 'rb') as wf:
        with _quiet_stderr():
            stream = pa.open(
                format=pa.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
            )
        print(f'Playing {wav_file} ...', file=sys.stderr)
        data = wf.readframes(CHUNK)
        while data:
            stream.write(data)
            data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
    pa.terminate()


def main():
    ap = argparse.ArgumentParser(description='Record from ReSpeaker Lite and play back.')
    ap.add_argument('--duration', type=float, default=5.0, metavar='N',
                    help='seconds to record (default: 5)')
    ap.add_argument('--stream', action='store_true',
                    help='stream raw PCM to stdout instead of saving/playing locally')
    ap.add_argument('--output', default='recording.wav', metavar='FILE',
                    help='output WAV file (default: recording.wav)')
    ap.add_argument('--device', type=int, default=None, metavar='INDEX',
                    help='force a specific PyAudio device index')
    args = ap.parse_args()

    wav = record(
        duration=args.duration,
        device_index=args.device,
        stream_to_stdout=args.stream,
        output_file=args.output,
    )

    if not args.stream:
        play(wav)


if __name__ == '__main__':
    main()
