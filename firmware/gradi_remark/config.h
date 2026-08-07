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
