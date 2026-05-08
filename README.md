# respeaker-lite-voice

Offline wake word detection + real-time speech-to-text pipeline for the Seeed Studio ReSpeaker Lite (XMOS XU316) on Linux and Raspberry Pi 5. A practical foundation for building voice assistants — with every gotcha documented.

---

## Files

| File | Purpose |
|---|---|
| `config.py` | All tunable settings — edit this, nothing else |
| `pipeline.py` | Audio engine — wake word, VAD, STT, state machine |
| `local.py` | Run locally with terminal UI (Pi or laptop) |
| `server.py` | Run as WebSocket server (cloud or remote) |

---

## Hardware

| Component | Detail |
|---|---|
| Microphone | Seeed Studio ReSpeaker Lite (XMOS XU316) |
| Target | Raspberry Pi 5 or any Linux machine |
| Connection | USB-C (ReSpeaker) → USB-A (host) |

---

## Firmware

Flash the **USB firmware** — the board ships with I2S firmware which does not work for this pipeline.

```bash
# Use the XMOS USB-C port (closest to 3.5mm jack)
sudo dfu-util -R -e -a 1 -D respeaker_lite_usb_dfu_firmware_v2.0.7.bin
```

Confirm it worked:

```bash
arecord -l
# Should show: card X: Lite [ReSpeaker Lite]
```

---

## Installation

```bash
# 1. System dependency for pyaudio
sudo apt install portaudio19-dev

# 2. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install in order — do NOT do pip install -r requirements.txt in one shot
pip install "numpy<2"
pip install onnxruntime==1.16.3
pip install pyaudio faster-whisper openwakeword websockets
```

> See requirements.txt for the pinned versions and the reason each is needed.

---

## Usage

### Local (Pi or laptop with mic attached)

```bash
python3 local.py
```

Say **"Alexa"** to activate, then speak. The session stays active for 60 seconds of inactivity before requiring the wake word again.

### WebSocket server (cloud / remote)

```bash
python3 server.py
```

Connects on `ws://0.0.0.0:8765` by default. Change `WS_HOST` and `WS_PORT` in `config.py`.

**Protocol:**

```
Client → Server  binary  raw audio bytes — S16_LE 16kHz mono
Server → Client  JSON    {"type": "transcript", "text": "...", "ts": "HH:MM:SS"}
                         {"type": "state",      "state": "waiting|listening|thinking"}
                         {"type": "wake",       "word": "alexa"}
                         {"type": "level",      "db": -32.5}
                         {"type": "ready"}
                         {"type": "error",      "message": "..."}
```

---

## Configuration

All settings live in `config.py` — nothing else needs editing.

| Setting | Default | Description |
|---|---|---|
| `MODEL_SIZE` | `"tiny"` | Whisper model size — `tiny` for Pi 5, `base` for cloud |
| `WAKE_WORD` | `"alexa"` | Wake word — must match a downloaded model name |
| `WAKE_THRESHOLD` | `0.5` | Detection sensitivity 0–1 |
| `SILENCE_THRESHOLD` | `0.025` | RMS threshold for voice activity |
| `INACTIVITY_TIMEOUT` | `60.0` | Seconds before session resets and wake word required again |
| `LISTEN_TIMEOUT_S` | `8.0` | Seconds to wait for speech after wake before giving up |
| `WS_HOST` | `"0.0.0.0"` | WebSocket bind address |
| `WS_PORT` | `8765` | WebSocket port |
| `HALLUCINATIONS` | see config | Set of phrases Whisper commonly hallucinates on silence |

---

## Available Wake Words

| Model name | Say |
|---|---|
| `alexa` | "Alexa" ✅ confirmed working |
| `hey_jarvis` | "Hey Jarvis" |
| `hey_mycroft` | "Hey Mycroft" |
| `hey_rhasspy` | "Hey Rhasspy" |

For a custom wake word (e.g. "Elara"), train with the [openWakeWord training script](https://github.com/dscripka/openWakeWord#training-new-models).

---

## What the XU316 does automatically

The XMOS XU316 runs these AI algorithms in hardware — you get clean audio from CH0 with no extra work:

- Acoustic Echo Cancellation (AEC)
- Noise Suppression
- Automatic Gain Control (AGC)
- Interference Cancellation

**CH0** — processed ASR mic → fed to STT  
**CH1** — wake word mic → fed to openWakeWord

---

## Pipeline Architecture

```
ReSpeaker Lite (XU316)
  CH0 — AEC + noise suppressed ASR mic
  CH1 — wake word mic
       │
       │ USB-C → USB-A
       ▼
  pyaudio stereo stream (16kHz S16_LE)
       │
  pipeline.py dispatcher
   ┌───┴────────────┐
   ▼                ▼
 CH0              CH1 (mono int16)
 level meter      openWakeWord
 STT queue        score > 0.5 → wake
 (when active)         │
       ◄───────────────┘
       │
  segmenter (VAD)
       │
  faster-whisper tiny · int8
       │
  on_transcript(text) callback
       │
   ┌───┴──────────┐
   ▼              ▼
local.py       server.py
terminal UI    WebSocket JSON
```

---

## ⚠️ Known Issues & Gotchas

### 1. System audio hijacked on plug-in

Every time the ReSpeaker is plugged in, PulseAudio sets it as the default output. Sounds play through the ReSpeaker 3.5mm jack instead of laptop speakers.

**Fix:**

```bash
mkdir -p ~/.config/pulse
echo ".include /etc/pulse/default.pa
set-default-sink alsa_output.pci-0000_00_1f.3.analog-stereo
unload-module module-switch-on-connect" > ~/.config/pulse/default.pa
pulseaudio -k && pulseaudio --start
```

Replace the sink name with your actual laptop sink from `pactl list short sinks`.

---

### 2. Raspberry Pi 5 has no speaker output

The Pi 5 has no built-in audio output. `aplay` will produce no sound unless something is physically connected.

**Options:**
- Plug a speaker into the **ReSpeaker 3.5mm jack** — this is the intended setup
- Use a USB speaker
- Use Bluetooth

When playing back through the ReSpeaker on the Pi:

```bash
aplay -D plughw:1,0 your_file.wav
```

---

### 3. SSH does not forward audio

Audio from `aplay` on the Pi plays on the Pi's hardware — not forwarded to your laptop over SSH. You must have a speaker physically connected to the Pi to hear anything.

---

### 4. Always use `plughw` not `hw`

```bash
# Wrong — rate mismatch, corrupted WAV
arecord -D hw:1,0 -r 48000 -f S16_LE -c 2 test.wav

# Correct
arecord -D plughw:1,0 -r 16000 -f S16_LE -c 2 test.wav
```

The USB firmware runs at **16000 Hz**. Requesting 48000 Hz with `hw:` writes a broken WAV header.

---

### 5. openWakeWord version requirements

| Package | Required | Problem if wrong |
|---|---|---|
| `numpy` | `< 2.0` | onnxruntime segfaults on import |
| `onnxruntime` | `== 1.16.3` | 1.17+ causes frozen scores (`0.001703`) or segfault |

---

### 6. openWakeWord needs mono int16 — not stereo float32

The model returns near-zero scores (`0.00005` max) if fed stereo or float32 audio. It must receive **mono int16** directly from pyaudio. This is handled automatically in `pipeline.py`.

---

### 7. Cannot open two ALSA streams on the same device

ALSA does not allow two simultaneous `open()` calls on the same device. Opening sounddevice and pyaudio on the same ReSpeaker causes:

```
Error opening InputStream: Device unavailable [PaErrorCode -9985]
```

`pipeline.py` uses a **single pyaudio stereo stream** and splits channels internally.

---

### 8. RGB LED and I2C not available over USB

The WS2812 RGB LED and chip configuration (volume, mute, VNR) are on the I2C bus on the physical pin headers — not exposed over USB. To control these, connect SDA, SCL, and GND jumper wires from the Pi to the ReSpeaker headers.

---

## Performance

| Stage | Laptop | Raspberry Pi 5 |
|---|---|---|
| Wake word detection | ~80ms | ~80ms |
| Silence pad | 200ms | 200ms |
| STT tiny · int8 | 0.3–0.6s | 0.8–1.5s |
| **Total end-to-end** | **~0.6–0.8s** | **~1.1–1.8s** |
