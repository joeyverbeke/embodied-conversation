// Firmware tunables. Anything that is a *judgement* belongs on the host —
// this file holds only what the hardware itself needs to know.
#pragma once

// ── Pins (PLAN §2) ────────────────────────────────────────────────────────
#define PIN_SDA    5    // D4
#define PIN_SCL    6    // D5
#define PIN_BCLK   9    // D10
#define PIN_LRCLK  8    // D9
#define PIN_DIN    7    // D8

// ── Audio (PLAN §2 — 24 kHz / 16-bit / mono) ──────────────────────────────
#define SAMPLE_RATE     24000
#define I2S_CHUNK       240     // 10 ms per I2S write
#define RAMP_SAMPLES    120     // 5 ms anti-pop ramp at each utterance edge

// 512 KB of PSRAM ≈ 10.9 s of audio. Power of two so the ring wraps with a
// mask instead of a modulo.
#define RING_BYTES      (512 * 1024)
#define RING_SAMPLES    (RING_BYTES / 2)
#define RING_MASK       (RING_SAMPLES - 1)

// Prebuffer before playback starts. With ~10 s of slack behind it, WiFi
// jitter is a non-issue.
#define PREBUFFER_MS    300
#define PREBUFFER_SAMPLES ((SAMPLE_RATE * PREBUFFER_MS) / 1000)

// ── Motion (PLAN §3) ──────────────────────────────────────────────────────
#define IMU_HZ              100
#define IMU_INTERVAL_MS     (1000 / IMU_HZ)
#define IMU_REPORT_US       10000
#define FRAMES_PER_MESSAGE  5      // 20 messages/sec

// seq u32, t_ms u32, quat f32x4, gyro f32x3, linaccel f32x3, accel f32x3.
// Must match MOTION_FRAME_SIZE in host/protocol.py. A message is
// 1 + 5*60 = 301 bytes, comfortably inside MAX_FRAME below.
#define MOTION_FRAME_BYTES  60

// Do not raise IMU_HZ to buy resolution. Human movement lives below ~10 Hz,
// so 100 is already 5x oversampled; bandwidth is better spent on more report
// streams than on more samples of the same ones.

// ── Link ──────────────────────────────────────────────────────────────────
// How the host is reached. LINK_SERIAL is the USB cable the board is already
// on for power; LINK_WIFI is where this goes once the piece is untethered.
// host/config.py has the matching switch — change both or nothing connects.
#define LINK_WIFI    0
#define LINK_SERIAL  1
#define LINK         LINK_SERIAL

// Framing for LINK_SERIAL: MAGIC0 MAGIC1 | length u16 LE | payload. Serial is
// a byte stream, so the message boundaries WebSocket gave for free have to be
// put back. Must match host/framing.py.
#define FRAME_MAGIC0 0xA7
#define FRAME_MAGIC1 0x5E
// Largest real payload is a PCM chunk: PCM_CHUNK_MS at 24 kHz mono int16 is
// 1920 bytes, plus a 3-byte header. Anything bigger is a bad length read out
// of a desynced stream.
#define MAX_FRAME    2048
// 256 is the default. The CDC driver discards bytes when this fills rather
// than pushing back on the host, and a dropped byte costs a whole frame — so
// this is sized generously and the host paces itself on top (config.SERIAL_PACE).
#define SERIAL_RX_BUFFER 16384
// Must comfortably exceed one MOTION message (1 + 5*60 + 4 header = 305 bytes).
// link_send() refuses to write a frame that will not fit whole, so a buffer too
// small to hold one does not corrupt the stream — it silences it.
#define SERIAL_TX_BUFFER 8192

// ── Protocol (PLAN §3) ────────────────────────────────────────────────────
#define MSG_MOTION     0x01
#define MSG_STATE      0x02
#define MSG_LOG        0x03
#define MSG_UTT_BEGIN  0x10
#define MSG_PCM        0x11
#define MSG_UTT_END    0x12
#define MSG_FLUSH      0x13

#define STATE_IDLE     0
#define STATE_PLAYING  1

// ── Reconnect (PLAN §3) ───────────────────────────────────────────────────
#define RECONNECT_MIN_MS  500
#define RECONNECT_MAX_MS  8000
