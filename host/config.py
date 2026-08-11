"""Every tunable, one place. Phase 2 retunes these; nothing else."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Link ──────────────────────────────────────────────────────────────────
# How the device is reached. "serial" is the USB cable it is already plugged
# into for power; "ws" is WiFi, which is where this ends up once the piece is
# untethered. The firmware has a matching switch in firmware/gradi_remark/
# config.h — change both or nothing connects.
LINK = "serial"
SERIAL_PORT = None                 # None = first /dev/cu.usbmodem*

# Host -> device pacing, as a multiple of realtime audio. USB will happily
# deliver an utterance faster than the device can parse it, and the ESP32's
# CDC driver drops bytes on a full buffer rather than pushing back — which
# sounds like crackle and loses the UTT_END that ends playback. The device has
# a 10.9 s ring and a 300 ms prebuffer, so a modest lead is all it ever needs.
SERIAL_PACE = 3.0
# Bytes allowed through at full USB speed before pacing bites. Must stay well
# under the device's SERIAL_RX_BUFFER (16 KB) or the burst is itself the
# overflow. The prebuffer costs ~100 ms to fill at SERIAL_PACE, so there is no
# reason to want a big one.
SERIAL_BURST_BYTES = 4096

# Used when LINK == "ws"
BIND_HOST = "0.0.0.0"
BIND_PORT = 8765

# ── Audio ─────────────────────────────────────────────────────────────────
# Kokoro's native rate. Matching it end to end is what keeps a resampler out
# of the chain — do not change one of these without the other.
SAMPLE_RATE = 24000
PCM_CHUNK_MS = 40                  # slice size when streaming PCM to the device

# ── Motion ────────────────────────────────────────────────────────────────
IMU_HZ = 100
FRAMES_PER_MESSAGE = 5             # device batches this many MOTION frames

# ── Segmentation (PLAN 5.1) ───────────────────────────────────────────────
# Validated against Phase 0 captures: at rest |w| median 0.11 deg/s, p99 1.73,
# so ARM_OFF sits ~14x above the noise floor. Arcs peaked 93-100, wiggles
# 112-133, jabs 300-481, shakes 142-496. 12/12 gestures segmented correctly.
ARM_ON_DPS = 60.0                  # IDLE -> ARMED
ARM_OFF_DPS = 25.0                 # GESTURING -> SETTLING
ONSET_HOLD_MS = 80                 # sustained above ARM_ON to arm
MIN_DURATION_MS = 250              # shorter than this is a twitch, discarded
SETTLE_HOLD_MS = 250               # quiet this long and the gesture is over
MAX_DURATION_MS = 4000             # force-cut
REFRACTORY_MS = 800                # after responding, before arming again
PREBUFFER_MS = 500                 # rolling history kept so onset isn't clipped

# ── Features (PLAN 5.2) ───────────────────────────────────────────────────
# Which sensor axis runs along the limb. Pose naming assumes the board is
# mounted with +X pointing away from the body along the forearm; change this
# if the enclosure orientation changes.
# Measured, not assumed — the BNO085 sits rotated on the breadboard and the
# breadboard sits at an angle on the back of the hand, so this is not
# guessable from the silkscreen. Re-measure if the enclosure changes.
# Sensor -X points along the arm toward the fingers: with the hand flat, the
# world-Z component of -X reads -0.98 hanging, -0.18 horizontal, +0.98 overhead.
LIMB_AXIS = 0                      # 0=x, 1=y, 2=z
LIMB_SIGN = -1                     # so the axis points toward the fingers
POSE_ELEVATION_DEG = 35.0          # +/- band that counts as "horizontal"
REVERSAL_MIN_DPS = 30.0            # ignore sign flips in the noise

# Directness is net rotation / integrated path. A gesture that ends somewhere
# new scores high; one that returns to where it started scores near zero.
# Measured on 29 real gestures: median 0.23, max 0.91 — the old 0.85 gate
# fired almost never, which is why everything fell into one bucket.
DIRECT_RATIO = 0.70                # ended somewhere else: "single arc"
RETURN_RATIO = 0.30                # came back to the start: "out and back"

EARLY_PEAK_FRAC = 0.35             # time-to-peak below this is "early"
LATE_PEAK_FRAC = 0.65              # above this is "late"
PERCUSSIVE_DPS = 250.0             # peak rate that reads as a strike
PERCUSSIVE_MAX_S = 1.0             # ...but only if it's brief. A 2.4 s gesture
                                   # is not percussive however fast it got.
REPEAT_MIN_PERIODS = 2.0           # need this many cycles to call it repetition

# ── Voice (PLAN 5.3) ──────────────────────────────────────────────────────
# PLAN §5.3 names llama3.2:3b. Swapped after an A/B on 30 logged gestures:
# the 3b invents objects that were never in the descriptor ("reaching for a
# light switch") and describes rather than judges. The 8b holds the register
# and compares against previous gestures. Costs ~300 ms, still inside budget.
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://127.0.0.1:11434"
PERSONA_PATH = ROOT / "persona" / "persona.txt"
CONTEXT_PAIRS = 5                  # recent descriptor/utterance pairs sent
MAX_WORDS = 22                     # hard truncation only. Nobody waits for the
                                   # device to finish before gesturing again, so
                                   # speech length costs nothing and the room to
                                   # say something is worth more than brevity.
OVERLAP_THRESHOLD = 0.6            # content-word overlap that triggers one retry

# ── TTS (PLAN 5.4) ────────────────────────────────────────────────────────
TTS_MODEL = "prince-canuma/Kokoro-82M"
TTS_VOICE = "af_heart"
TTS_SPEED = 1.0
TTS_LANG = "a"

# Kokoro comes out around -11 dBFS, which wastes most of the amp's range on a
# small speaker. Each utterance arrives as a single segment before the first
# chunk ships, so normalising costs no latency and beats a fixed gain — a fixed
# gain would have to be timid enough never to clip the loud ones.
# Drop TTS_PEAK if it sounds strained; that is distortion, not volume.
TTS_PEAK = 0.95                    # normalise each utterance to this peak
TTS_MAX_GAIN = 8.0                 # never haul a near-silent segment up this far

# ── Corpus ────────────────────────────────────────────────────────────────
# Everything the machine has watched, across every person who has picked it up.
# Delete this file to start an evening fresh; it is what makes "nobody has done
# that" a true statement rather than a bluff.
CORPUS_PATH = ROOT / "logs" / "corpus.jsonl"

# ── Logging (PLAN 5.6) ────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
