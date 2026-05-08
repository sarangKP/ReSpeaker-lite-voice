# ─── config.py ────────────────────────────────────────────
# Central configuration for the ReSpeaker Lite voice pipeline.
# Edit values here — no need to touch pipeline.py, local.py, or server.py.

# ── Audio ──────────────────────────────────────────────────
RATE       = 16000   # Hz — fixed by USB firmware, do not change
CHANNELS   = 2       # stereo: CH0 = ASR mic, CH1 = wake word mic
CHUNK      = 1280    # frames per read — 80ms, required by openWakeWord

# ── STT model ──────────────────────────────────────────────
# Options: "tiny", "base", "small", "medium", "large-v2"
MODEL_SIZE   = "base"
DEVICE_TYPE  = "cpu"    # "cpu" or "cuda" (if GPU available)
COMPUTE_TYPE = "int8"   # "int8" fastest on CPU, "float16" for GPU

# ── Wake word ──────────────────────────────────────────────
# Built-in options: "alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy"
# "alexa" confirmed working — others may need accent tuning
# For a custom word (e.g. "Elara"), train via openWakeWord training script
WAKE_WORD        = "alexa_v0.1"
WAKE_THRESHOLD   = 0.5    # 0.0–1.0 — lower = more sensitive, more false positives
WAKE_COOLDOWN_S  = 2.0    # seconds to ignore after a detection fires

# ── VAD (Voice Activity Detection) ─────────────────────────
SILENCE_THRESHOLD = 0.025  # RMS threshold — raise if background noise causes false triggers
SPEECH_PAD_MS     = 200    # ms of silence before cutting an utterance
MIN_SPEECH_MS     = 200    # discard utterances shorter than this
MAX_SPEECH_MS     = 8000   # force-flush utterances longer than this

# ── Session ────────────────────────────────────────────────
LISTEN_TIMEOUT_S   = 8.0   # seconds to wait for speech after wake word before giving up
INACTIVITY_TIMEOUT = 60.0  # seconds of silence before requiring wake word again

# ── WebSocket server ───────────────────────────────────────
WS_HOST = "0.0.0.0"   # listen on all interfaces — change to "127.0.0.1" for local only
WS_PORT = 8765

# ── Hallucinations ─────────────────────────────────────────
# Whisper "tiny" commonly hallucinates these on silence/noise.
# Any transcript matching an entry here is silently discarded.
# Add more if you notice repeated false outputs in your environment.
HALLUCINATIONS = {
    "thank you", "thanks", "thank you.", "thanks.",
    "huh", "huh?", "huh.", "uh", "uh.", "um", "um.",
    "hmm", "hmm.", "bye", "bye.", "goodbye", "goodbye.",
    "you", "you.", ".", "..", "...",
    "okay", "okay.", "ok", "ok.",
    "no", "no.", "yes", "yes.",
    "sure", "sure.", "i don't know", "i'm sorry",
    "sorry", "sorry.", "right", "right.",
    "oh", "oh.", "ah", "ah.",
    "what?", "what", "really", "really.",
    "wow", "wow.", "please", "please.",
    "hello", "hello.",
    "so", "so.", "well", "well.", "and", "and.",
}
