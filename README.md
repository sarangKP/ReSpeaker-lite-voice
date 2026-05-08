# ReSpeaker Lite — Voice Pipeline

Wake word detection + real-time speech-to-text using the Seeed Studio ReSpeaker Lite (XMOS XU316) over USB. Built for Raspberry Pi 5 but developed and tested on Ubuntu laptop.

---

## Hardware

| Component | Detail |
|---|---|
| Microphone | Seeed Studio ReSpeaker Lite (XMOS XU316) |
| Target host | Raspberry Pi 5 |
| Connection | USB-C (ReSpeaker) → USB-A (Pi 5 or laptop) |

---

## Firmware

The board ships with I2S firmware. You must flash the **USB firmware** for this pipeline to work.

```bash
# Flash USB firmware (do this from a laptop, not the Pi)
sudo dfu-util -R -e -a 1 -D respeaker_lite_usb_dfu_firmware_v2.0.7.bin
```

Use the **XMOS USB-C port** (the one closest to the 3.5mm jack) — not the data port.

After flashing, confirm the device is detected:

```bash
arecord -l
# Should show: card X: Lite [ReSpeaker Lite]
```

### USB firmware vs I2S firmware

| | USB Firmware | I2S Firmware |
|---|---|---|
| Connection | USB-C | Pin headers |
| Setup complexity | Plug and play | Device Tree overlay required |
| Sample rate | 16000 Hz | 48000 Hz |
| Format | S16_LE | 32-bit left-justified |
| Recommended for Pi 5 | ✅ Yes | ❌ Avoid |

---

## What the XU316 chip does automatically

The XU316 is a professional DSP — it runs these AI algorithms in hardware before audio ever reaches your code. You do not need to enable or configure them:

- Acoustic Echo Cancellation (AEC)
- Noise Suppression
- Automatic Gain Control (AGC)
- Voice-to-Noise Ratio (VNR)
- Interference Cancellation

**CH0 (left)** — processed ASR mic → feed to STT / Whisper  
**CH1 (right)** — wake word mic → feed to wake word engine

---

## Installation

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install portaudio (required by pyaudio)
sudo apt install portaudio19-dev python3-pyaudio

# 3. Install Python packages in order
pip install "numpy<2"
pip install onnxruntime==1.16.3
pip install pyaudio sounddevice faster-whisper openwakeword
```

> **Do not** run `pip install -r requirements.txt` in one shot — numpy must be installed and pinned before onnxruntime or you will get a segfault.

---

## Usage

```bash
python3 stt_live.py
```

Say **"Alexa"** to activate, then speak your command. The session stays active for 60 seconds of inactivity before requiring the wake word again.

---

## ⚠️ Known Issues & Things to Watch Out For

### 1. System default audio switches to ReSpeaker on plug-in

Every time you plug in the ReSpeaker, Linux/PulseAudio sets it as the default audio output. This means system sounds, music, and any `aplay` commands without `-D` specified will play through the ReSpeaker's 3.5mm jack — not your laptop speakers.

**Fix (permanent):**

```bash
mkdir -p ~/.config/pulse
echo ".include /etc/pulse/default.pa
set-default-sink alsa_output.pci-0000_00_1f.3.analog-stereo
unload-module module-switch-on-connect" > ~/.config/pulse/default.pa

pulseaudio -k && pulseaudio --start
```

Replace `alsa_output.pci-0000_00_1f.3.analog-stereo` with your actual laptop sink from `pactl list short sinks`.

---

### 2. Raspberry Pi 5 has no speaker or headphone jack

The Pi 5 has **no built-in audio output**. If you run `aplay` on the Pi you will hear nothing unless something is physically connected.

**Options for audio output on Pi 5:**
- Plug a speaker or headphones into the **ReSpeaker's 3.5mm jack** — this is the intended setup
- Use a USB speaker
- Use a Bluetooth speaker

When playing back through the ReSpeaker jack on the Pi:
```bash
aplay -D plughw:1,0 your_file.wav
```

---

### 3. SSH audio does not forward

If you are connected to the Pi over SSH (e.g. from VS Code), `aplay` and any audio output plays on the **Pi's hardware**, not on your laptop. You will not hear anything on your laptop no matter what you play on the Pi over SSH.

To test audio over SSH you must have a speaker physically connected to the Pi or ReSpeaker.

---

### 4. Always use `plughw` not `hw` for arecord/aplay

```bash
# Wrong — causes rate mismatch, corrupted WAV header
arecord -D hw:1,0 -r 48000 -f S16_LE -c 2 test.wav

# Correct — plug layer handles rate conversion automatically
arecord -D plughw:1,0 -r 16000 -f S16_LE -c 2 test.wav
```

The USB firmware runs at **16000 Hz**. Requesting 48000 Hz with `hw:` produces a corrupted WAV file that plays at 3x speed or sounds like silence.

---

### 5. openWakeWord requires specific package versions

The wake word engine is sensitive to dependency versions. Using the wrong versions causes either a segfault or frozen scores (model returns `0.001703` constantly regardless of what you say).

| Package | Required version | Problem with wrong version |
|---|---|---|
| numpy | `< 2.0` | onnxruntime segfaults on import |
| onnxruntime | `== 1.16.3` | 1.17+ causes frozen scores or segfault |
| openwakeword | latest | fine |

---

### 6. openWakeWord needs mono int16 audio, not stereo float32

The wake word engine will silently return near-zero scores (`0.000062` max) if fed stereo or float32 audio. It must receive **mono int16** audio exactly as read from pyaudio.

```python
# Wrong — stereo float32 from sounddevice
chunk = indata[:, 0].astype(np.float32) / 32768.0  # scores stuck at 0.00005

# Correct — mono int16 directly from pyaudio stream
audio_bytes = stream.read(1280)
audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)  # scores reach 0.999
```

---

### 7. Cannot open two audio streams to the same ALSA device

ALSA does not allow two simultaneous `open()` calls on the same device. Opening one stream with sounddevice and another with pyaudio for the same ReSpeaker will fail with:

```
Error opening InputStream: Device unavailable [PaErrorCode -9985]
```

The solution is a **single pyaudio stereo stream** that reads both channels, then splits them internally in software.

---

### 8. RGB LED and I2C features not available over USB

The WS2812 RGB LED and chip configuration (volume, mute, VNR readout) are connected to the **I2C bus on the physical pin headers** — they are not exposed over USB.

To control the LED or configure the chip, you need 3 jumper wires from the Pi GPIO to the ReSpeaker's SDA, SCL, and GND pins.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────┐
│           ReSpeaker Lite (XU316)            │
│  CH0 — AEC + noise suppressed ASR mic       │
│  CH1 — wake word mic                        │
└────────────────┬────────────────────────────┘
                 │ USB-C → USB-A
                 ▼
      pyaudio stereo stream (16kHz S16_LE)
                 │
         audio_dispatcher
         ┌───────┴────────┐
         ▼                ▼
      CH0 (int16)     CH1 (int16 mono)
         │                │
    level meter      openWakeWord
    STT queue        "alexa" → 0.99
    (when active)         │
         │           wake detected
         ▼                │
      segmenter ◄─────────┘
      (VAD — silence detection)
         │
      faster-whisper tiny · int8
         │
      transcript
```

---

## Configuration

All tunable parameters are at the top of `stt_live.py`:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_SIZE` | `tiny` | Whisper model — `tiny` recommended for Pi 5 |
| `WAKE_WORD` | `alexa` | Wake word — must match a downloaded model |
| `WAKE_THRESHOLD` | `0.5` | Detection sensitivity (0–1) |
| `SILENCE_THRESHOLD` | `0.025` | RMS threshold for voice activity |
| `INACTIVITY_TIMEOUT` | `60.0` | Seconds before session resets |
| `LISTEN_TIMEOUT_S` | `8.0` | Seconds to wait for speech after wake |
| `SPEECH_PAD_MS` | `200` | Silence padding before cutting utterance |

---

## Available Wake Words (built-in)

| Model name | Say |
|---|---|
| `alexa` | "Alexa" ✅ confirmed working |
| `hey_jarvis` | "Hey Jarvis" |
| `hey_mycroft` | "Hey Mycroft" |
| `hey_rhasspy` | "Hey Rhasspy" |

To use a custom wake word like "Elara", train a model with the [openWakeWord training script](https://github.com/dscripka/openWakeWord#training-new-models).

---

## Performance (approximate)

| Stage | Laptop | Raspberry Pi 5 |
|---|---|---|
| Wake word detection | ~80ms | ~80ms |
| Silence pad | 200ms | 200ms |
| STT — tiny int8 | 0.3–0.6s | 0.8–1.5s |
| **Total end-to-end** | **~0.6–0.8s** | **~1.1–1.8s** |