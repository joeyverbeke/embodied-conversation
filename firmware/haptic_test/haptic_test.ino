// gradi-remark — DRV2605L bring-up
//
// Standalone. Confirms the haptic driver on its own I2C bus without disturbing
// the IMU on the original one:
//   1. scans Wire  (D4/D5) — expects the BNO085 at 0x4A
//   2. scans Wire1 (D2/D3) — expects the DRV2605L at 0x5A
//   3. fires a rotating set of waveform effects, one every 2 s
//
// Board: Seeed XIAO ESP32S3.  Same FQBN as the other sketches.

#include <Wire.h>
#include <Adafruit_DRV2605.h>

// ── Pins ──────────────────────────────────────────────────────────────────
static const int PIN_SDA   = 5;   // D4 — existing bus, BNO085
static const int PIN_SCL   = 6;   // D5
static const int PIN_SDA1  = 3;   // D2 — new bus, DRV2605L
static const int PIN_SCL1  = 4;   // D3

static const uint8_t ADDR_BNO = 0x4A;
static const uint8_t ADDR_DRV = 0x5A;

static const uint32_t EFFECT_PERIOD_MS = 2000;

Adafruit_DRV2605 drv;

static bool     drv_ok        = false;
static uint32_t last_effect_ms = 0;
static uint8_t  effect_idx     = 0;

// A spread across the ERM library: sharp, soft, buzzing, long. If the motor is
// wired and driven correctly these feel obviously different from each other.
struct Effect { uint8_t id; const char *name; };
static const Effect EFFECTS[] = {
  {  1, "strong click 100%"    },
  { 10, "double click 100%"    },
  { 14, "triple click 100%"    },
  { 24, "sharp tick 100%"      },
  { 47, "buzz 100%"            },
  { 70, "transition ramp up"   },
  { 118, "long buzz 100%"      },
};
static const uint8_t EFFECT_COUNT = sizeof(EFFECTS) / sizeof(EFFECTS[0]);

// ──────────────────────────────────────────────────────────────────────────

// Returns the number of devices answering, and prints each address found.
static uint8_t scan(TwoWire &bus, const char *label) {
  uint8_t found = 0;
  Serial.printf("[i2c] scanning %s ...\n", label);
  for (uint8_t addr = 0x08; addr < 0x78; addr++) {
    bus.beginTransmission(addr);
    if (bus.endTransmission() == 0) {
      Serial.printf("[i2c]   found 0x%02X\n", addr);
      found++;
    }
  }
  if (!found) Serial.printf("[i2c]   nothing on %s\n", label);
  return found;
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);
  Serial.println(F("\n[boot] DRV2605L bring-up"));

  // ── Original bus. Scanned but not driven — this is only here to prove the
  // new wiring did not disturb the IMU.
  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000);
  scan(Wire, "Wire (D4/D5)");
  Wire.beginTransmission(ADDR_BNO);
  if (Wire.endTransmission() == 0) Serial.println(F("[imu] BNO085 present at 0x4A"));
  else                             Serial.println(F("[imu] WARNING: no BNO085 at 0x4A"));

  // ── New bus.
  Wire1.begin(PIN_SDA1, PIN_SCL1);
  Wire1.setClock(400000);
  scan(Wire1, "Wire1 (D2/D3)");
  Wire1.beginTransmission(ADDR_DRV);
  if (Wire1.endTransmission() != 0) {
    Serial.println(F("[drv] FAIL: nothing at 0x5A — check SDA→D2, SCL→D3, VIN→3V3, GND→GND"));
    return;
  }

  if (!drv.begin(&Wire1)) {
    Serial.println(F("[drv] FAIL: chip answered but begin() failed"));
    return;
  }

  drv.selectLibrary(1);        // library 1 = ERM
  drv.setMode(DRV2605_MODE_INTTRIG);
  drv.useERM();                // swap to drv.useLRA() if the motor is an LRA

  drv_ok = true;
  Serial.println(F("[drv] ready — firing an effect every 2 s"));
}

void loop() {
  if (!drv_ok) { delay(1000); return; }

  uint32_t now = millis();
  if (now - last_effect_ms < EFFECT_PERIOD_MS) return;
  last_effect_ms = now;

  const Effect &e = EFFECTS[effect_idx];
  Serial.printf("[drv] effect %u — %s\n", e.id, e.name);

  drv.setWaveform(0, e.id);    // slot 0: the effect
  drv.setWaveform(1, 0);       // slot 1: end of sequence
  drv.go();

  effect_idx = (effect_idx + 1) % EFFECT_COUNT;
}
