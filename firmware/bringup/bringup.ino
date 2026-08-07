// gradi-remark — Phase 0 bring-up
//
// Standalone. Proves the hardware before any of the real build exists:
//   1. I2C bus scan, reports the address found
//   2. BNO085 game rotation vector + calibrated gyro, printed at 100 Hz
//   3. 440 Hz tone through the MAX98357A for 1 s, every 5 s
//   4. WiFi connect, prints the assigned IP
//
// Board: Seeed XIAO ESP32S3.  Build with PSRAM enabled — see README.

#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <ESP_I2S.h>
#include <WiFi.h>

#include "secrets.h"

// ── Pins (PLAN §2) ────────────────────────────────────────────────────────
static const int PIN_SDA    = 5;   // D4
static const int PIN_SCL    = 6;   // D5
static const int PIN_BCLK   = 9;   // D10
static const int PIN_LRCLK  = 8;   // D9
static const int PIN_DIN    = 7;   // D8

// ── Audio (PLAN §2 — 24 kHz / 16-bit / mono, no resampling anywhere) ──────
static const uint32_t SAMPLE_RATE   = 24000;
static const int      CHUNK_SAMPLES = 128;    // 5.3 ms; paces the main loop
static const float    TONE_HZ       = 440.0f;
static const float    TONE_AMP      = 0.85f;  // near full scale, like real speech
static const uint32_t TONE_MS       = 1000;
static const uint32_t TONE_PERIOD_MS= 5000;
static const uint32_t RAMP_MS       = 5;      // anti-pop edge ramp (PLAN §3)

static const uint32_t TONE_SAMPLES = (uint64_t)SAMPLE_RATE * TONE_MS / 1000;
static const uint32_t RAMP_SAMPLES = (uint64_t)SAMPLE_RATE * RAMP_MS / 1000;

// ── IMU ───────────────────────────────────────────────────────────────────
static const uint32_t REPORT_INTERVAL_US = 10000;  // 100 Hz
static const uint32_t PRINT_INTERVAL_MS  = 10;     // 100 Hz

Adafruit_BNO08x bno08x(-1);   // no reset pin wired
sh2_SensorValue_t sensorValue;
I2SClass i2s;

static uint8_t  imu_addr    = 0;
static bool     imu_ok      = false;
static bool     i2s_ok      = false;

static float    qw = 1, qx = 0, qy = 0, qz = 0;
static float    gx = 0, gy = 0, gz = 0;
static uint32_t quat_count = 0, gyro_count = 0;

static bool     tone_active   = false;
static uint32_t tone_idx      = 0;
static uint32_t last_tone_ms  = 0;
static uint32_t last_print_ms = 0;
static uint32_t last_rate_ms  = 0;

// ──────────────────────────────────────────────────────────────────────────

static bool enableReports();

static void scanI2C() {
  Serial.println(F("[i2c] scanning 0x08..0x77"));
  int found = 0;
  for (uint8_t addr = 0x08; addr < 0x78; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("[i2c] device at 0x%02X\n", addr);
      found++;
      if (addr == 0x4A || addr == 0x4B) imu_addr = addr;
    }
  }
  if (!found) {
    Serial.println(F("[i2c] NOTHING FOUND — check wiring, power, and that"));
    Serial.println(F("[i2c] PS0/PS1 are both LOW (I2C mode, not UART-RVC)."));
  } else if (!imu_addr) {
    Serial.println(F("[i2c] devices found, but none at 0x4A/0x4B."));
  }
}

static bool startIMU() {
  uint8_t addr = imu_addr ? imu_addr : 0x4A;
  if (!bno08x.begin_I2C(addr, &Wire)) {
    Serial.printf("[imu] begin_I2C(0x%02X) failed\n", addr);
    return false;
  }
  Serial.printf("[imu] up at 0x%02X — part %lu, sw %u.%u.%lu\n",
                addr,
                (unsigned long)bno08x.prodIds.entry[0].swPartNumber,
                bno08x.prodIds.entry[0].swVersionMajor,
                bno08x.prodIds.entry[0].swVersionMinor,
                (unsigned long)bno08x.prodIds.entry[0].swBuildNumber);
  return enableReports();
}

static bool enableReports() {
  bool ok = true;
  if (!bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, REPORT_INTERVAL_US)) {
    Serial.println(F("[imu] could not enable GAME_ROTATION_VECTOR"));
    ok = false;
  }
  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, REPORT_INTERVAL_US)) {
    Serial.println(F("[imu] could not enable GYROSCOPE_CALIBRATED"));
    ok = false;
  }
  if (ok) Serial.println(F("[imu] reports enabled @ 100 Hz"));
  return ok;
}

static const char *authName(wifi_auth_mode_t m) {
  switch (m) {
    case WIFI_AUTH_OPEN:            return "open";
    case WIFI_AUTH_WEP:             return "WEP";
    case WIFI_AUTH_WPA_PSK:         return "WPA";
    case WIFI_AUTH_WPA2_PSK:        return "WPA2";
    case WIFI_AUTH_WPA_WPA2_PSK:    return "WPA/WPA2";
    case WIFI_AUTH_WPA2_ENTERPRISE: return "WPA2-ent";
    case WIFI_AUTH_WPA3_PSK:        return "WPA3";
    case WIFI_AUTH_WPA2_WPA3_PSK:   return "WPA2/WPA3";
    default:                        return "?";
  }
}

// Reason codes tell wrong-password apart from can't-see-the-AP.
// 15 = 4-way handshake timeout (bad password), 201 = no AP found.
static void onWiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info) {
  if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
    Serial.printf("\n[wifi] disconnected, reason %d\n",
                  info.wifi_sta_disconnected.reason);
  }
}

static void scanWiFi() {
  Serial.println(F("[wifi] scanning..."));
  int n = WiFi.scanNetworks(false, true);   // include hidden
  if (n <= 0) { Serial.printf("[wifi] scan returned %d (0=none, -1=failed, -2=running)\n", n); return; }
  Serial.printf("[wifi] %d networks\n", n);
  for (int i = 0; i < n; i++) {
    Serial.printf("[wifi]   \"%s\"  ch %d  %d dBm  %s%s\n",
                  WiFi.SSID(i).c_str(), WiFi.channel(i), WiFi.RSSI(i),
                  authName(WiFi.encryptionType(i)),
                  WiFi.SSID(i) == WIFI_SSID ? "   <-- target" : "");
  }
  WiFi.scanDelete();
}

static void connectWiFi() {
  Serial.printf("[wifi] connecting to \"%s\"\n", WIFI_SSID);
  WiFi.onEvent(onWiFiEvent);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);          // PLAN §3 — modem sleep adds 100-200 ms jitter
  scanWiFi();                    // see the air before blaming the password
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(250);
    Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[wifi] connected — IP %s, RSSI %d dBm, ch %d\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI(), WiFi.channel());
    Serial.printf("[wifi] host target ws://%s:%d (not used in Phase 0)\n",
                  HOST_IP, HOST_PORT);
  } else {
    Serial.printf("[wifi] FAILED after 20 s (status %d)\n", WiFi.status());
    scanWiFi();
  }
}

// Fill one chunk: tone if we're inside a burst, silence otherwise.
// I2S never stops — silence between bursts is what keeps the amp from popping.
static void fillAudio(int16_t *buf, int n) {
  for (int i = 0; i < n; i++) {
    if (!tone_active) { buf[i] = 0; continue; }

    float env = 1.0f;
    if (tone_idx < RAMP_SAMPLES) {
      env = (float)tone_idx / RAMP_SAMPLES;
    } else if (tone_idx > TONE_SAMPLES - RAMP_SAMPLES) {
      env = (float)(TONE_SAMPLES - tone_idx) / RAMP_SAMPLES;
    }

    float phase = 2.0f * PI * TONE_HZ * (float)tone_idx / (float)SAMPLE_RATE;
    buf[i] = (int16_t)(sinf(phase) * env * TONE_AMP * 32767.0f);

    if (++tone_idx >= TONE_SAMPLES) { tone_active = false; tone_idx = 0; }
  }
}

// ──────────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);

  Serial.println();
  Serial.println(F("=== gradi-remark bring-up ==="));
  Serial.printf("[sys] heap %lu, psram %lu\n",
                (unsigned long)ESP.getFreeHeap(),
                (unsigned long)ESP.getPsramSize());
  if (ESP.getPsramSize() == 0) {
    Serial.println(F("[sys] PSRAM not detected — build with PSRAM=opi (see README)"));
  }

  // 1 — I2C
  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000);
  scanI2C();

  // 2 — IMU
  imu_ok = startIMU();

  // 3 — audio
  i2s.setPins(PIN_BCLK, PIN_LRCLK, PIN_DIN, -1, -1);
  i2s_ok = i2s.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT,
                     I2S_SLOT_MODE_MONO);
  Serial.println(i2s_ok ? F("[i2s] up — 24 kHz / 16-bit / mono")
                        : F("[i2s] begin() FAILED"));

  // 4 — WiFi
  connectWiFi();

  Serial.println(F("=== running: quat/gyro @100 Hz, tone every 5 s ==="));
  last_tone_ms = millis();
  last_rate_ms = millis();
}

void loop() {
  uint32_t now = millis();

  // audio — always writing, so the clock never stops
  if (i2s_ok) {
    if (!tone_active && now - last_tone_ms >= TONE_PERIOD_MS) {
      tone_active  = true;
      tone_idx     = 0;
      last_tone_ms = now;
    }
    static int16_t buf[CHUNK_SAMPLES];
    fillAudio(buf, CHUNK_SAMPLES);
    i2s.write((uint8_t *)buf, sizeof(buf));   // blocks on DMA; paces the loop
  }

  // imu — drain everything queued
  if (imu_ok) {
    if (bno08x.wasReset()) {
      Serial.println(F("[imu] sensor reset — re-enabling reports"));
      enableReports();
    }
    while (bno08x.getSensorEvent(&sensorValue)) {
      switch (sensorValue.sensorId) {
        case SH2_GAME_ROTATION_VECTOR:
          qw = sensorValue.un.gameRotationVector.real;
          qx = sensorValue.un.gameRotationVector.i;
          qy = sensorValue.un.gameRotationVector.j;
          qz = sensorValue.un.gameRotationVector.k;
          quat_count++;
          break;
        case SH2_GYROSCOPE_CALIBRATED:
          gx = sensorValue.un.gyroscope.x;
          gy = sensorValue.un.gyroscope.y;
          gz = sensorValue.un.gyroscope.z;
          gyro_count++;
          break;
        default:
          break;
      }
    }
  }

  // one combined line at 100 Hz
  if (now - last_print_ms >= PRINT_INTERVAL_MS) {
    last_print_ms = now;
    Serial.printf("q % .4f % .4f % .4f % .4f  w % 7.3f % 7.3f % 7.3f%s\n",
                  qw, qx, qy, qz, gx, gy, gz, tone_active ? "  [tone]" : "");
  }

  // honest rate check once a second
  if (now - last_rate_ms >= 1000) {
    last_rate_ms = now;
    Serial.printf("[rate] quat %lu Hz, gyro %lu Hz\n",
                  (unsigned long)quat_count, (unsigned long)gyro_count);
    quat_count = gyro_count = 0;
  }
}
