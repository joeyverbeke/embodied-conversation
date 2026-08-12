// gradi-remark — wearable firmware
//
// The device is deliberately dumb. Its entire job: read the IMU at 100 Hz,
// stream it, buffer and play whatever PCM arrives, and report whether it is
// currently speaking. It holds no thresholds, no state machine, no
// interpretation. Every judgement lives in Python.
//
// If you find yourself adding logic here, stop and ask.
//
// Board: Seeed XIAO ESP32S3, PSRAM enabled. See README.

#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <ESP_I2S.h>

#include "config.h"
#include "secrets.h"
#include "link.h"

Adafruit_BNO08x bno08x(-1);
sh2_SensorValue_t sensorValue;
I2SClass i2s;

// ── Audio ring (PSRAM). Single producer: the link. Single consumer: the
// audio task. Lock-free by construction — do not add a second writer without
// adding a mutex. ──────────────────────────────────────────────────────────
static int16_t *ring = nullptr;
static volatile uint32_t ring_w = 0;      // producer index
static volatile uint32_t ring_r = 0;      // consumer index

static volatile bool utt_open = false;    // UTT_BEGIN seen, UTT_END not yet
static volatile bool utt_closing = false; // UTT_END seen, ring still draining
static volatile bool playing = false;
static volatile bool report_playing = false;   // audio task -> main loop
static volatile bool report_idle = false;
static volatile uint32_t overflow_samples = 0;

// ── IMU ───────────────────────────────────────────────────────────────────
// Four reports, not two. The ball is held, not strapped to a limb, so where it
// *went* matters more than how it spun — and rotation alone cannot see a lift,
// a carry, or a gentle throw. See host/protocol.py for why both linear and raw
// acceleration are carried.
static float qw = 1, qx = 0, qy = 0, qz = 0;
static float gx = 0, gy = 0, gz = 0;
static float lax = 0, lay = 0, laz = 0;
static float ax = 0, ay = 0, az = 0;

// Frames are stamped with millis(), not with the sensor hub's own report time.
// The hub timestamp would be better in principle — orientation and gyro arrive
// as separate reports a moment apart and get fused into one frame as though
// simultaneous, which matters for jerk. But sh2_SensorValue_t.timestamp is not
// populated by this library: it reads back as ~UINT32_MAX, which divided down
// pins every frame to the same millisecond and destroys the timebase the
// segmenter runs on. Measured, not assumed. Do not reinstate it without
// checking the field actually advances.
static uint32_t seq = 0;
static uint32_t last_sample_ms = 0;
static uint8_t batch[1 + FRAMES_PER_MESSAGE * MOTION_FRAME_BYTES];
static int batch_count = 0;

static bool enableReports();

// ──────────────────────────────────────────────────────────────────────────
// Ring helpers

static inline uint32_t ring_available() { return ring_w - ring_r; }

static void ring_reset() {
  ring_r = ring_w = 0;
}

// src is raw bytes, not int16_t*, because PCM samples land at an odd offset
// inside the received frame (opcode + utt_id = 3 bytes) and a misaligned
// int16_t load is not something to gamble a live show on. memcpy of 2 bytes
// costs nothing here.
static void ring_write(const uint8_t *src, uint32_t n) {
  uint32_t free_space = RING_SAMPLES - ring_available();
  if (n > free_space) {
    overflow_samples += (n - free_space);
    n = free_space;             // drop the tail rather than corrupt the ring
  }
  for (uint32_t i = 0; i < n; i++) {
    int16_t s;
    memcpy(&s, src + 2 * i, 2);
    ring[(ring_w + i) & RING_MASK] = s;
  }
  ring_w += n;
}

// ──────────────────────────────────────────────────────────────────────────
// Audio task — pinned to core 0, runs forever.
//
// I2S never stops. When there is nothing to say the task writes silence.
// Starting and stopping the clock makes the MAX98357A pop on every utterance.

static void audioTask(void *arg) {
  static int16_t out[I2S_CHUNK];
  uint32_t ramp_in = 0;          // samples into the fade-in

  for (;;) {
    uint32_t avail = ring_available();

    if (!playing) {
      // Start once the prebuffer is satisfied, or once the host has closed
      // the utterance and this is all we are ever going to get.
      bool ready = utt_open && avail >= PREBUFFER_SAMPLES;
      bool short_utterance = utt_closing && avail > 0;
      if (ready || short_utterance) {
        playing = true;
        ramp_in = 0;
        report_playing = true;
      }
    }

    if (playing) {
      for (int i = 0; i < I2S_CHUNK; i++) {
        if (ring_available() == 0) {
          out[i] = 0;
          continue;
        }
        int16_t s = ring[ring_r & RING_MASK];
        ring_r++;

        // 5 ms in
        if (ramp_in < RAMP_SAMPLES) {
          s = (int16_t)((int32_t)s * ramp_in / RAMP_SAMPLES);
          ramp_in++;
        }
        // 5 ms out, applied to the final samples of a closing utterance
        uint32_t left = ring_available();
        if (utt_closing && left < RAMP_SAMPLES) {
          s = (int16_t)((int32_t)s * left / RAMP_SAMPLES);
        }
        out[i] = s;
      }

      if (utt_closing && ring_available() == 0) {
        playing = false;
        utt_open = false;
        utt_closing = false;
        report_idle = true;
      }
    } else {
      memset(out, 0, sizeof(out));
    }

    i2s.write((uint8_t *)out, sizeof(out));   // blocks on DMA; paces the task
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Messages. Identical whichever carrier delivered them — see link.h.

static void sendState(uint8_t s) {
  uint8_t msg[2] = {MSG_STATE, s};
  link_send(msg, 2);
}

static void sendLog(const char *text) {
  size_t n = strlen(text);
  uint8_t msg[128];
  msg[0] = MSG_LOG;
  if (n > sizeof(msg) - 1) n = sizeof(msg) - 1;
  memcpy(msg + 1, text, n);
  link_send(msg, n + 1);
}

static void onLinkUp() {
  sendLog("device up");
  sendState(playing ? STATE_PLAYING : STATE_IDLE);
}

// drop everything buffered and go silent; a power cycle must never be
// required to recover
static void onLinkDown() {
  utt_open = utt_closing = false;
  playing = false;
  ring_reset();
}

static void onMessage(const uint8_t *payload, size_t length) {
  if (length < 1) return;
  switch (payload[0]) {
    case MSG_UTT_BEGIN: {
      if (length < 7) return;
      uint32_t sr;
      memcpy(&sr, payload + 1, 4);
      if (sr != SAMPLE_RATE) {
        Serial.printf("[link] host sent %lu Hz, expected %d — ignoring\n",
                      (unsigned long)sr, SAMPLE_RATE);
        return;
      }
      ring_reset();
      overflow_samples = 0;
      utt_open = true;
      utt_closing = false;
      break;
    }
    case MSG_PCM: {
      if (length < 3) return;
      ring_write(payload + 3, (length - 3) / 2);
      break;
    }
    case MSG_UTT_END:
      utt_closing = true;
      break;

    case MSG_FLUSH:
      utt_open = utt_closing = false;
      playing = false;
      ring_reset();
      report_idle = true;
      break;
  }
}

// ──────────────────────────────────────────────────────────────────────────
// IMU

// Four streams at 100 Hz is roughly a third of a 400 kHz I2C bus. There is
// room, but not unlimited room — adding a fifth is a measurement, not a guess.
static bool enableReports() {
  bool ok = bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, IMU_REPORT_US);
  ok &= bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, IMU_REPORT_US);
  ok &= bno08x.enableReport(SH2_LINEAR_ACCELERATION, IMU_REPORT_US);
  ok &= bno08x.enableReport(SH2_ACCELEROMETER, IMU_REPORT_US);
  return ok;
}

// A reset that lands mid-transaction leaves the BNO085 holding SDA low, and
// the bus stays wedged until it is clocked out. Observed once in testing
// immediately after a flash; cheap to recover from, confusing if you don't.
static void i2cRecover() {
  pinMode(PIN_SCL, OUTPUT);
  pinMode(PIN_SDA, INPUT_PULLUP);
  for (int i = 0; i < 9 && digitalRead(PIN_SDA) == LOW; i++) {
    digitalWrite(PIN_SCL, LOW);
    delayMicroseconds(5);
    digitalWrite(PIN_SCL, HIGH);
    delayMicroseconds(5);
  }
  pinMode(PIN_SDA, OUTPUT);      // manual STOP
  digitalWrite(PIN_SDA, LOW);
  delayMicroseconds(5);
  digitalWrite(PIN_SCL, HIGH);
  delayMicroseconds(5);
  digitalWrite(PIN_SDA, HIGH);
  delayMicroseconds(5);
}

static bool startIMU() {
  for (int attempt = 1; attempt <= 5; attempt++) {
    i2cRecover();
    Wire.begin(PIN_SDA, PIN_SCL);
    Wire.setClock(400000);
    if (bno08x.begin_I2C(0x4A, &Wire) && enableReports()) return true;
    Serial.printf("[imu] attempt %d/5 failed, recovering bus\n", attempt);
    delay(250);
  }
  return false;
}

static void pollIMU() {
  if (bno08x.wasReset()) {
    Serial.println(F("[imu] sensor reset — re-enabling"));
    enableReports();
  }
  while (bno08x.getSensorEvent(&sensorValue)) {
    switch (sensorValue.sensorId) {
      case SH2_GAME_ROTATION_VECTOR:
        qw = sensorValue.un.gameRotationVector.real;
        qx = sensorValue.un.gameRotationVector.i;
        qy = sensorValue.un.gameRotationVector.j;
        qz = sensorValue.un.gameRotationVector.k;
        break;
      case SH2_GYROSCOPE_CALIBRATED:
        gx = sensorValue.un.gyroscope.x;
        gy = sensorValue.un.gyroscope.y;
        gz = sensorValue.un.gyroscope.z;
        break;
      case SH2_LINEAR_ACCELERATION:
        lax = sensorValue.un.linearAcceleration.x;
        lay = sensorValue.un.linearAcceleration.y;
        laz = sensorValue.un.linearAcceleration.z;
        break;
      case SH2_ACCELEROMETER:
        ax = sensorValue.un.accelerometer.x;
        ay = sensorValue.un.accelerometer.y;
        az = sensorValue.un.accelerometer.z;
        break;
      default:
        break;
    }
  }
}

// Pack one MOTION frame: seq u32, t_ms u32, quat f32x4, gyro f32x3,
// linear accel f32x3, accel f32x3. Little-endian, which is native here — must
// match host/protocol.py exactly.
static void appendFrame(uint32_t t_ms) {
  uint8_t *p = batch + 1 + batch_count * MOTION_FRAME_BYTES;
  memcpy(p +  0, &seq,  4);
  memcpy(p +  4, &t_ms, 4);
  memcpy(p +  8, &qw,   4);
  memcpy(p + 12, &qx,   4);
  memcpy(p + 16, &qy,   4);
  memcpy(p + 20, &qz,   4);
  memcpy(p + 24, &gx,   4);
  memcpy(p + 28, &gy,   4);
  memcpy(p + 32, &gz,   4);
  memcpy(p + 36, &lax,  4);
  memcpy(p + 40, &lay,  4);
  memcpy(p + 44, &laz,  4);
  memcpy(p + 48, &ax,   4);
  memcpy(p + 52, &ay,   4);
  memcpy(p + 56, &az,   4);
  seq++;
  batch_count++;
}

// ──────────────────────────────────────────────────────────────────────────

void setup() {
  link_begin();                  // owns Serial; must run before anything prints

  Serial.println(F("\n=== gradi-remark ==="));

  ring = (int16_t *)ps_malloc(RING_BYTES);
  if (!ring) {
    Serial.println(F("[fatal] no PSRAM for the ring buffer."));
    Serial.println(F("[fatal] Build with PSRAM=opi — see README."));
    while (true) delay(1000);
  }
  Serial.printf("[ring] %d KB in PSRAM (%.1f s)\n",
                RING_BYTES / 1024, (float)RING_SAMPLES / SAMPLE_RATE);

  if (!startIMU()) {
    // Rebooting beats halting: a halted device needs a human with a cable,
    // and PLAN §3 is explicit that a power cycle must never be required.
    Serial.println(F("[fatal] BNO085 unreachable after 5 tries."));
    Serial.println(F("[fatal] Check wiring with bringup.ino. Rebooting in 5 s."));
    delay(5000);
    ESP.restart();
  }
  Serial.println(F("[imu] game rotation vector + gyro @ 100 Hz"));

  i2s.setPins(PIN_BCLK, PIN_LRCLK, PIN_DIN, -1, -1);
  if (!i2s.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT,
                 I2S_SLOT_MODE_MONO)) {
    Serial.println(F("[fatal] I2S begin failed"));
    while (true) delay(1000);
  }
  Serial.println(F("[i2s] 24 kHz / 16-bit / mono, running continuously"));

  batch[0] = MSG_MOTION;
  link_start();

  // Audio on core 0, everything else on core 1. The blocking I2S write must
  // never be able to stall the IMU or the link.
  xTaskCreatePinnedToCore(audioTask, "audio", 4096, nullptr, 3, nullptr, 0);

  last_sample_ms = millis();
}

void loop() {
  link_loop();
  pollIMU();

  uint32_t now = millis();
  if (now - last_sample_ms >= IMU_INTERVAL_MS) {
    last_sample_ms += IMU_INTERVAL_MS;
    appendFrame(now);
    if (batch_count >= FRAMES_PER_MESSAGE) {
      if (link_connected()) link_send(batch, sizeof(batch));
      batch_count = 0;
    }
  }

  // STATE is reported from here, not from the audio task — neither carrier is
  // thread-safe.
  if (report_playing) {
    report_playing = false;
    if (link_connected()) sendState(STATE_PLAYING);
  }
  if (report_idle) {
    report_idle = false;
    if (link_connected()) sendState(STATE_IDLE);
    if (overflow_samples) {
      Serial.printf("[audio] ring overflowed by %lu samples\n",
                    (unsigned long)overflow_samples);
      overflow_samples = 0;
    }
    if (link_resync_bytes) {
      Serial.printf("[link] lost %lu bytes — host is outrunning the parser\n",
                    (unsigned long)link_resync_bytes);
      link_resync_bytes = 0;
    }
    if (link_dropped_frames) {
      Serial.printf("[link] dropped %lu frames — TX buffer full\n",
                    (unsigned long)link_dropped_frames);
      link_dropped_frames = 0;
    }
  }
}
