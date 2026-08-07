# gradi-remark — Build Plan

A wearable that watches how you move and says something about it.

The participant performs a gesture. The device notices it start, notices it stop, and responds out loud through its own speaker. A local LLM writes the line; a local TTS speaks it. Nothing leaves the LAN.

This document is the complete build specification. It is self-contained — do not consult or port from any prior repository.

---

## 0. Instructions for the implementing agent

**Read this section fully before writing code.**

### Working agreement

- Build **one phase at a time**, in order. At the end of each phase, **stop**. Write a short report (what was built, how it was verified, anything that surprised you) and wait for the human to test on hardware and reply. Do not begin the next phase unsolicited.
- **Stop and ask** the moment you hit any of: a required credential or secret, an API that doesn't match what this document describes, a hardware behaviour that contradicts §2, a library that appears unmaintained or renamed, or any design decision this document does not cover. A wrong guess costs an hour of hardware debugging; a question costs a minute.
- **Do not silently substitute.** If a specified library or model is unavailable, stop and say so with the alternatives you found. Do not swap it and continue.
- **Do not add hardware.** The wiring in §2 is fixed. If something seems to need another component, that's a question, not a change.
- **Do not add features** beyond the current phase. Phase 3 ideas do not belong in Phase 1.
- Verify library APIs against current documentation before use. The ESP32 Arduino core, `mlx-audio`, and Ollama's Python client all move quickly, and this document may lag them. Where reality differs from this spec, follow reality and flag the difference in your phase report.

### Environment

- Host: **macOS on Apple Silicon (M3 Max, 36 GB)**. Metal-accelerated inference. Not CUDA, not Linux.
- Firmware: Arduino, targeting **Seeed XIAO ESP32S3**.
- Assume nothing is installed. Include setup steps in the README as you go.

### Secrets

WiFi credentials go in `firmware/gradi_remark/secrets.h`, which is gitignored. Commit `secrets.h.example` with placeholders. **Ask the human for the SSID, password, and host IP — never invent or hardcode them.**

---

## 1. Architecture

Two halves, one socket between them.

```
┌────────────────────────────┐         ┌───────────────────────────────────┐
│  XIAO ESP32S3  (wearable)  │         │  MacBook Pro  (host)              │
│                            │         │                                   │
│  BNO085 ──I2C──► firmware  │  WiFi   │  segmenter → features → LLM → TTS │
│                     │      │◄───────►│                                   │
│  MAX98357A ◄─I2S─── ┘      │   WS    │  session log (JSONL)              │
└────────────────────────────┘         └───────────────────────────────────┘
     streams motion, plays audio            makes every decision
```

**The device is deliberately dumb.** Its entire job: read the IMU at 100 Hz, stream it, buffer and play whatever PCM arrives, report whether it is currently speaking. It holds no thresholds, no state machine, no interpretation. Every judgement lives in Python, where it can be changed without a reflash.

This is the load-bearing decision of the whole build. Preserve it. If you find yourself adding logic to the firmware, stop and ask.

**No tare, no dock, no calibration ritual.** Power on → connect → running. Arm pose is derived from the gravity direction in the sensor frame, which is absolute and needs no reference capture.

**No magnetometer.** Heading is irrelevant now, so the IMU runs in Game Rotation Vector mode — no magnetic declination, no figure-eight calibration, no drift near the speaker magnet or gallery steel.

---

## 2. Hardware

### Bill of materials

| Item | Notes |
|---|---|
| Seeed XIAO ESP32S3 | 8 MB PSRAM, 8 MB flash, B+/B− LiPo pads |
| Adafruit BNO085 breakout | 9-DoF IMU, running in I2C mode |
| Adafruit MAX98357A breakout | I2S mono class-D amp |
| Small 4 Ω or 8 Ω speaker | |
| Breadboard + jumpers | |

### Wiring

| Signal | XIAO pin | GPIO | To |
|---|---|---|---|
| I2C SDA | D4 | 5 | BNO085 SDA |
| I2C SCL | D5 | 6 | BNO085 SCL |
| I2S BCLK | D10 | 9 | MAX98357A BCLK |
| I2S LRCLK | D9 | 8 | MAX98357A LRC |
| I2S DIN | D8 | 7 | MAX98357A DIN |
| Amp enable | — | — | MAX98357A SD → **3V3** (hardwired, 12 dB gain) |
| Power | 3V3, GND | | both breakouts |

BNO085 I2C address is `0x4A` (Adafruit default; `0x4B` if the ADR jumper is bridged). Both PS0 and PS1 must be **LOW** for I2C — this is the Adafruit breakout's default state. If the board was previously strapped for UART-RVC, that strap must be removed. Phase 0's I2C scan is what confirms this; if the scan finds nothing, say so rather than trying software workarounds.

### IMU configuration

Adafruit BNO08x library, SH-2 protocol. Enable exactly two reports:

- `SH2_GAME_ROTATION_VECTOR` @ 100 Hz — quaternion, no magnetometer
- `SH2_GYROSCOPE_CALIBRATED` @ 100 Hz — rad/s, three axes

### Audio format

**24 kHz, 16-bit, mono, little-endian.** This matches Kokoro's native output rate, so no resampling exists anywhere in the chain. Do not introduce a resampling step.

---

## 3. Wire protocol

Single WebSocket. The ESP32 is the **client**; the Mac runs the server at `ws://<host-ip>:8765`. Host IP is hardcoded in `secrets.h` for now — no mDNS, no discovery.

All frames are **binary**, first byte is the message type.

### Device → host

| Type | Name | Payload |
|---|---|---|
| `0x01` | `MOTION` | `seq` u32, `t_ms` u32, `qw qx qy qz` f32×4, `gx gy gz` f32×3 — 37 bytes, 100 Hz (~3.7 kB/s) |
| `0x02` | `STATE` | u8: `0` = idle, `1` = playing |
| `0x03` | `LOG` | UTF-8 string, diagnostics only |

Batch 5 MOTION frames per WebSocket message (20 messages/sec) to cut per-frame overhead. All multi-byte fields little-endian.

### Host → device

| Type | Name | Payload |
|---|---|---|
| `0x10` | `UTT_BEGIN` | `sample_rate` u32, `utt_id` u16 |
| `0x11` | `PCM` | `utt_id` u16, then raw int16 LE samples |
| `0x12` | `UTT_END` | `utt_id` u16 |
| `0x13` | `FLUSH` | none — drop buffered audio, return to idle |

### Playback contract

- Device allocates a **512 KB PSRAM ring buffer** (≈10 s of audio) and prebuffers **300 ms** before starting playback. With that much slack, WiFi jitter is a non-issue.
- **I2S runs continuously**, feeding silence when idle. Never stop the clock — starting and stopping it makes the MAX98357A pop on every utterance. Apply a **5 ms linear ramp** at each utterance edge.
- Device sends `STATE playing` on first sample out and `STATE idle` when the ring drains after `UTT_END`. **The host treats this as authoritative** for knowing when speech has finished.
- `WiFi.setSleep(false)` and BLE disabled. Modem sleep otherwise injects 100–200 ms of random latency.
- Reconnect with exponential backoff (0.5 s → 8 s cap). On disconnect, flush the ring and go silent. A power cycle must never be required.

---

## 4. Repository layout

```
gradi-remark/
  PLAN.md
  README.md
  .gitignore
  pyproject.toml
  firmware/
    bringup/bringup.ino              # Phase 0 only
    gradi_remark/
      gradi_remark.ino
      config.h
      secrets.h.example
  host/
    __init__.py
    config.py                        # all tunables, one place
    protocol.py                      # pack/unpack, single source of truth
    server.py                        # websocket lifecycle + orchestration
    segmenter.py                     # motion stream → gesture segments
    features.py                      # segment → descriptor string
    voice.py                         # descriptor → utterance (Ollama)
    tts.py                           # utterance → PCM chunks
    session_log.py                   # JSONL writer
    replay.py                        # re-run a log through the pipeline
  persona/
    persona.txt                      # the LLM system prompt — hot-reloaded
  logs/                              # gitignored
```

Use `uv` for the Python environment.

---

## 5. The pipeline

### 5.1 Segmentation (`segmenter.py`)

Gesture boundaries come from gyro magnitude with dual-threshold hysteresis — structurally the same as voice activity detection, and it fails the same way if you use a single threshold.

```
IDLE      → ARMED:      |ω| > 60°/s sustained 80 ms
ARMED     → GESTURING:  duration > 250 ms          (else → IDLE; twitch rejected)
GESTURING → SETTLING:   |ω| < 25°/s
SETTLING  → GESTURING:  |ω| > 60°/s again          (it wasn't over)
SETTLING  → COMPLETE:   held 250 ms
GESTURING → COMPLETE:   duration > 4000 ms         (force-cut)
COMPLETE  → RESPONDING → IDLE + 800 ms refractory
```

Maintain a rolling **500 ms pre-buffer** so the onset hold doesn't clip the attack. The first 80 ms of a gesture carries most of its character; discarding it costs you the difference between a stab and a drift.

Every threshold above lives in `config.py`. They will all be retuned in Phase 2.

### 5.2 Features (`features.py`)

Input: the motion frames spanning one gesture. Output: **one English line**.

Compute at minimum:

- duration
- integrated angular path vs. net rotation → **directness ratio** (separates a sweep from a wiggle)
- peak |ω| and normalized time-to-peak (early vs. late acceleration)
- dominant rotation axis (PCA over ω)
- direction-reversal count
- repetition period (autocorrelation of ω), if any
- peak linear acceleration — percussiveness, independent of rotation
- start pose and end pose from gravity direction: `hanging` / `horizontal` / `overhead` / `across-body`

Descriptor format — compact, human-readable, no raw numbers beyond a couple of round figures:

```
1.4s single arc, 130° path, very direct, late acceleration, hanging → overhead
0.6s sharp jab, 40° path, percussive, horizontal → horizontal
2.1s repeated shake, 4 reversals, ~2.5 Hz, hanging → hanging
```

### 5.3 Voice (`voice.py`)

- **Ollama**, model `llama3.2:3b`, `OLLAMA_KEEP_ALIVE=-1`, streaming enabled.
- System prompt read from `persona/persona.txt`, **hot-reloaded on file change** — the human will edit this constantly and must never restart the server to do it.
- Context: the current descriptor plus the last 5 descriptor/utterance pairs.
- Output cap ~12 words. Reject and regenerate once on high token overlap with a recent utterance.

**The LLM sees only the descriptor string, never raw numbers.** This boundary is the point: it lets what the device *notices* be retuned independently of how it *talks*. Do not pass telemetry through it.

Write a starter `persona.txt` that produces short, dry, slightly unsettling observations about how someone moved. Keep it plain and editable. It is a placeholder — the human owns this file.

### 5.4 TTS (`tts.py`)

- **`mlx-audio` running Kokoro-82M** on Metal. Fall back to `kokoro-onnx` only if `mlx-audio` fails, and report if you do.
- Output 24 kHz int16 mono, streamed to the device in chunks as synthesis proceeds — do not wait for full synthesis before sending the first chunk.

### 5.5 Orchestration (`server.py`)

**Single-slot policy: one utterance in flight, ever.** No queue, no backlog. If a gesture completes while the device is still speaking, drop it. A device narrating gestures from thirty seconds ago is broken in a way that is very hard to un-see.

Gate the decision in exactly one function, so that changing the policy later is a change to one predicate:

```python
def should_respond(state) -> bool:
    return (state.device == "idle"
            and not state.inflight
            and state.now - state.last_end >= REFRACTORY)
```

Latency budget, end of gesture to first audio: **~1.0–1.3 s**. If it lands materially worse, report the breakdown rather than optimizing on your own initiative.

### 5.6 Logging (`session_log.py`)

Append one JSONL record per gesture: raw motion frames for the segment, extracted features, descriptor, utterance, and stage-by-stage timings.

This is not instrumentation, it is the development loop. Without it, every tuning change requires standing up and waving an arm. With it, Phase 2 happens at a desk.

---

## 6. Phases

Each phase ends with a **hard stop**: report, then wait.

### Phase 0 — Bring-up

`firmware/bringup/bringup.ino`, standalone. Three checks:

1. I2C scan — report the address found.
2. Quaternion + gyro printing at 100 Hz over serial.
3. 440 Hz tone through the speaker for 1 s, every 5 s.

Then, separately: WiFi connect, print the assigned IP.

**Stop.** The human flashes and confirms. Every later failure is ambiguous until this passes — do not proceed on the assumption that it will.

### Phase 1 — End-to-end loop

Everything in §3–§5. Real Ollama, real Kokoro, JSONL logging from the first commit. USB power, hardcoded host IP.

Also deliver a README with full setup: `uv` env, Ollama install and model pull, `mlx-audio` install, Arduino board and library versions, and how to run.

**Acceptance:** perform a gesture, hear a response about it within ~1.3 s. Thirty consecutive gestures with no backlog, no audio pops, no dropped connection, and no degradation in segmentation.

**Three macOS gotchas to handle in code or document in the README:**

- **Local Network permission** — macOS 15+ prompts once. If Python doesn't get it, the ESP32 connection fails *silently* with no error on either end. Document how to grant it manually under System Settings → Privacy & Security → Local Network.
- **Sleep kills the server** — document `caffeinate -i python -m host.server`.
- **Getting the LAN IP** — `ipconfig getifaddr en0`, noting that `en0` is Ethernet on some machines and both should be checked.

**Stop.**

### Phase 2 — Tuning tools

`replay.py`: re-run a logged session through segmentation, features, and voice, with no hardware attached. Support swapping the persona file and the config thresholds per run, and diffing outputs across runs.

No firmware changes in this phase. If you think you need one, that's a question.

**Stop.**

### Phase 3 — Refinement

Only after the human has spent real time in Phase 2:

- **Thinking particles** — a small library of pre-synthesized non-lexical sounds (an intake of breath, a short *hm*), fired instantly at gesture end while the real utterance generates. This masks latency, but the actual reason is that a device which pauses before evaluating you is doing something more interesting than one that answers instantly.
- **Session memory** — recognition of repeated gestures across a session, rather than fresh commentary each time.
- **LiPo** — 500 mAh on B+/B−, expect ~2 h. WiFi TX peaks around 350 mA.

**Stop.**

---

## 7. Open questions — raise, don't decide

These are the human's calls. Surface them when they become relevant; do not resolve them in code.

1. **Who prompts the participant?** If the device itself asks for a gesture, the piece becomes a closed loop — solicitation, performance, judgement, re-solicitation — and the question of whether it escalates or tires across a session becomes live. If wall text does it, the device is purely reactive. This determines whether the persona has a trajectory or only a tone.
2. **Voice and register** — the persona file is a placeholder written to be replaced.
3. **Failure behaviour in front of an audience** — what the piece does when WiFi drops or the LLM stalls mid-session.
