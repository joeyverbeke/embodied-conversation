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
#include <WiFi.h>
#include <WebSocketsClient.h>

#include "config.h"
#include "secrets.h"

Adafruit_BNO08x bno08x(-1);
sh2_SensorValue_t sensorValue;
I2SClass i2s;
WebSocketsClient webSocket;

// ── Audio ring (PSRAM). Single producer: the WebSocket task. Single
// consumer: the audio task. Lock-free by construction — do not add a second
// writer without adding a mutex. ───────────────────────────────────────────
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
static float qw = 1, qx = 0, qy = 0, qz = 0, gx = 0, gy = 0, gz = 0;
static uint32_t seq = 0;
static uint32_t last_sample_ms = 0;
static uint8_t batch[1 + FRAMES_PER_MESSAGE * 36];
static int batch_count = 0;

// ── Connection ────────────────────────────────────────────────────────────
static bool ws_connected = false;
static uint32_t backoff_ms = RECONNECT_MIN_MS;

static bool enableReports();

// ──────────────────────────────────────────────────────────────────────────
// Ring helpers

static inline uint32_t ring_available() { return ring_w - ring_r; }

static void ring_reset() {
  ring_r = ring_w = 0;
}

static void ring_write(const int16_t *src, uint32_t n) {
  uint32_t free_space = RING_SAMPLES - ring_available();
  if (n > free_space) {
    overflow_samples += (n - free_space);
    n = free_space;             // drop the tail rather than corrupt the ring
  }
  for (uint32_t i = 0; i < n; i++) {
    ring[(ring_w + i) & RING_MASK] = src[i];
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
// WebSocket

static void sendState(uint8_t s) {
  uint8_t msg[2] = {MSG_STATE, s};
  webSocket.sendBIN(msg, 2);
}

static void sendLog(const char *text) {
  size_t n = strlen(text);
  uint8_t msg[128];
  msg[0] = MSG_LOG;
  if (n > sizeof(msg) - 1) n = sizeof(msg) - 1;
  memcpy(msg + 1, text, n);
  webSocket.sendBIN(msg, n + 1);
}

static void onWsEvent(WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      ws_connected = true;
      backoff_ms = RECONNECT_MIN_MS;
      webSocket.setReconnectInterval(backoff_ms);
      Serial.println(F("[ws] connected"));
      sendLog("device up");
      sendState(playing ? STATE_PLAYING : STATE_IDLE);
      break;

    case WStype_DISCONNECTED:
      ws_connected = false;
      // drop everything buffered and go silent; a power cycle must never be
      // required to recover
      utt_open = utt_closing = false;
      playing = false;
      ring_reset();
      backoff_ms = min<uint32_t>(backoff_ms * 2, RECONNECT_MAX_MS);
      webSocket.setReconnectInterval(backoff_ms);
      Serial.printf("[ws] disconnected, retry in %lu ms\n",
                    (unsigned long)backoff_ms);
      break;

    case WStype_BIN: {
      if (length < 1) return;
      switch (payload[0]) {
        case MSG_UTT_BEGIN: {
          if (length < 7) return;
          uint32_t sr;
          memcpy(&sr, payload + 1, 4);
          if (sr != SAMPLE_RATE) {
            Serial.printf("[ws] host sent %lu Hz, expected %d — ignoring\n",
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
          ring_write((const int16_t *)(payload + 3), (length - 3) / 2);
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
      break;
    }

    default:
      break;
  }
}

// ──────────────────────────────────────────────────────────────────────────
// IMU

static bool enableReports() {
  bool ok = bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, IMU_REPORT_US);
  ok &= bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, IMU_REPORT_US);
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
      default:
        break;
    }
  }
}

// Pack one MOTION frame: seq u32, t_ms u32, quat f32x4, gyro f32x3.
// Little-endian, which is native here — must match host/protocol.py exactly.
static void appendFrame(uint32_t t_ms) {
  uint8_t *p = batch + 1 + batch_count * 36;
  memcpy(p +  0, &seq,  4);
  memcpy(p +  4, &t_ms, 4);
  memcpy(p +  8, &qw,   4);
  memcpy(p + 12, &qx,   4);
  memcpy(p + 16, &qy,   4);
  memcpy(p + 20, &qz,   4);
  memcpy(p + 24, &gx,   4);
  memcpy(p + 28, &gy,   4);
  memcpy(p + 32, &gz,   4);
  seq++;
  batch_count++;
}

// ──────────────────────────────────────────────────────────────────────────

// 15 = 4-way handshake timeout (bad password), 201 = no AP found,
// 202 = auth fail. Without these a failure is just a row of dots.
static void onWiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info) {
  if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
    Serial.printf(" [reason %d]", info.wifi_sta_disconnected.reason);
  }
}

static void connectWiFi() {
  WiFi.onEvent(onWiFiEvent);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);        // modem sleep injects 100-200 ms of jitter

  for (int attempt = 1; ; attempt++) {
    WiFi.disconnect(true);
    delay(100);

    // Phase 0 finding: an iPhone hotspot doesn't beacon reliably until the
    // radio has swept the band. Without this scan the first connect fails
    // with a handshake timeout that looks exactly like a wrong password.
    int n = WiFi.scanNetworks(false, true);
    bool visible = false;
    for (int i = 0; i < n; i++) {
      if (WiFi.SSID(i) == WIFI_SSID) visible = true;
    }
    Serial.printf("[wifi] attempt %d: %d networks, \"%s\" %s\n",
                  attempt, n, WIFI_SSID, visible ? "visible" : "NOT visible");
    WiFi.scanDelete();

    WiFi.begin(WIFI_SSID, WIFI_PASS);
    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
      delay(250);
      Serial.print('.');
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
      Serial.printf("[wifi] %s  RSSI %d dBm  ch %d\n",
                    WiFi.localIP().toString().c_str(), WiFi.RSSI(),
                    WiFi.channel());
      return;
    }
    // Keep trying forever — a wearable that gives up needs a human.
    Serial.printf("[wifi] attempt %d failed (status %d), retrying\n",
                  attempt, WiFi.status());
    delay(2000);
  }
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 2000) delay(10);

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

  connectWiFi();

  batch[0] = MSG_MOTION;
  webSocket.begin(HOST_IP, HOST_PORT, "/");
  webSocket.onEvent(onWsEvent);
  webSocket.setReconnectInterval(RECONNECT_MIN_MS);
  Serial.printf("[ws] target ws://%s:%d\n", HOST_IP, HOST_PORT);

  // Audio on core 0, everything else on core 1. The blocking I2S write must
  // never be able to stall the IMU or the socket.
  xTaskCreatePinnedToCore(audioTask, "audio", 4096, nullptr, 3, nullptr, 0);

  last_sample_ms = millis();
}

void loop() {
  webSocket.loop();
  pollIMU();

  uint32_t now = millis();
  if (now - last_sample_ms >= IMU_INTERVAL_MS) {
    last_sample_ms += IMU_INTERVAL_MS;
    appendFrame(now);
    if (batch_count >= FRAMES_PER_MESSAGE) {
      if (ws_connected) webSocket.sendBIN(batch, sizeof(batch));
      batch_count = 0;
    }
  }

  // STATE is reported from here, not from the audio task — the WebSocket
  // library is not thread-safe.
  if (report_playing) {
    report_playing = false;
    if (ws_connected) sendState(STATE_PLAYING);
  }
  if (report_idle) {
    report_idle = false;
    if (ws_connected) sendState(STATE_IDLE);
    if (overflow_samples) {
      Serial.printf("[audio] ring overflowed by %lu samples\n",
                    (unsigned long)overflow_samples);
      overflow_samples = 0;
    }
  }
}
