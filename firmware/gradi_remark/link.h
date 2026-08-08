// Transport. One interface, two carriers — the USB cable or WiFi.
//
// The sketch above this file never learns which one it got. It calls
// link_send() and link_connected(), and defines three callbacks the carrier
// invokes: onMessage(), onLinkUp(), onLinkDown(). LINK in config.h picks.
//
// WiFi is where this ends up once the piece is untethered, so that branch is
// the code that already worked, moved rather than rewritten.
#pragma once

#include "config.h"
#include "secrets.h"

// Defined in the sketch. onMessage() receives one whole protocol message,
// payload[0] being the opcode — identical either way, which is the point.
static void onMessage(const uint8_t *payload, size_t length);
static void onLinkUp();
static void onLinkDown();

static void link_begin();      // call first in setup(), before anything prints
static void link_start();      // call last in setup(), once the hardware is up
static void link_loop();
static void link_send(const uint8_t *payload, size_t length);
static bool link_connected();

// Bytes discarded while hunting for a frame start. Zero in steady state;
// anything else means bytes were lost, which corrupts audio and can swallow
// the UTT_END that ends playback. Worth saying out loud.
static uint32_t link_resync_bytes = 0;


#if LINK == LINK_SERIAL
// ──────────────────────────────────────────────────────────────────────────
// USB cable. Serial is native USB CDC on this board (USBMode=hwcdc,
// CDCOnBoot=Enabled are the XIAO_ESP32S3 defaults), so the 115200 is cosmetic
// and the real ceiling is USB full speed — far above the ~48 kB/s of audio.
//
// Diagnostics share this port. That works because every frame is magic-
// prefixed: the host treats anything that isn't a frame as device log text.
// So Serial.printf stays exactly as useful as it was.

static void link_begin() {
  // Both of these must happen before begin(). The default 256-byte receive
  // buffer drops audio at 48 kB/s, and without the timeout every write blocks
  // forever whenever the Mac isn't draining the port.
  Serial.setRxBufferSize(SERIAL_RX_BUFFER);
  Serial.setTxTimeoutMs(0);
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 2000) delay(10);
}

static void link_start() {
  Serial.println(F("[link] serial (USB)"));
  onLinkUp();
}

static void link_send(const uint8_t *payload, size_t length) {
  if (length == 0 || length > MAX_FRAME) return;
  const uint8_t hdr[4] = {FRAME_MAGIC0, FRAME_MAGIC1,
                          (uint8_t)(length & 0xFF), (uint8_t)(length >> 8)};
  Serial.write(hdr, sizeof(hdr));
  Serial.write(payload, length);
}

// The host sends FLUSH the moment it opens the port, so there is nothing to
// detect here — a stale ring is cleared by that, not by a connect event.
static bool link_connected() { return true; }

static void link_loop() {
  static uint8_t frame[MAX_FRAME];
  static uint8_t hdr = 0;          // header bytes matched: 0..3, then 4 = body
  static uint16_t need = 0, got = 0;

  // Bounded so a fast host can never starve the IMU's 100 Hz slot.
  int budget = SERIAL_RX_BUFFER;

  while (budget > 0) {
    int avail = Serial.available();
    if (avail <= 0) break;

    if (hdr < 4) {
      uint8_t b = Serial.read();
      budget--;
      switch (hdr) {
        case 0:
          hdr = (b == FRAME_MAGIC0) ? 1 : 0;
          if (!hdr) link_resync_bytes++;
          break;
        case 1:
          // A repeated magic0 is still a candidate start.
          hdr = (b == FRAME_MAGIC1) ? 2 : (b == FRAME_MAGIC0 ? 1 : 0);
          break;
        case 2:
          need = b;
          hdr = 3;
          break;
        case 3:
          need |= (uint16_t)b << 8;
          // An impossible length means we synced on magic-looking bytes inside
          // a payload. Drop it and hunt for the next magic.
          hdr = (need == 0 || need > MAX_FRAME) ? 0 : 4;
          got = 0;
          break;
      }
      continue;
    }

    int want = need - got;
    int n = avail < want ? avail : want;
    if (n > budget) n = budget;
    n = Serial.readBytes(frame + got, n);
    if (n <= 0) break;
    got += n;
    budget -= n;
    if (got >= need) {
      onMessage(frame, need);
      hdr = 0;
    }
  }
}


#elif LINK == LINK_WIFI
// ──────────────────────────────────────────────────────────────────────────
// WiFi. Unchanged from the version that works — the device is the client, the
// Mac runs the server, the host IP is compile-time in secrets.h.

#include <WiFi.h>
#include <WebSocketsClient.h>

static WebSocketsClient webSocket;
static bool ws_connected = false;
static uint32_t backoff_ms = RECONNECT_MIN_MS;

static void link_begin() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 2000) delay(10);
}

static void link_send(const uint8_t *payload, size_t length) {
  webSocket.sendBIN((uint8_t *)payload, length);
}

static bool link_connected() { return ws_connected; }

static void link_loop() { webSocket.loop(); }

static void onWsEvent(WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      ws_connected = true;
      backoff_ms = RECONNECT_MIN_MS;
      webSocket.setReconnectInterval(backoff_ms);
      Serial.println(F("[ws] connected"));
      onLinkUp();
      break;

    case WStype_DISCONNECTED:
      ws_connected = false;
      onLinkDown();
      backoff_ms = min<uint32_t>(backoff_ms * 2, RECONNECT_MAX_MS);
      webSocket.setReconnectInterval(backoff_ms);
      Serial.printf("[ws] disconnected, retry in %lu ms\n",
                    (unsigned long)backoff_ms);
      break;

    case WStype_BIN:
      onMessage(payload, length);
      break;

    default:
      break;
  }
}

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

static void link_start() {
  connectWiFi();
  webSocket.begin(HOST_IP, HOST_PORT, "/");
  webSocket.onEvent(onWsEvent);
  webSocket.setReconnectInterval(RECONNECT_MIN_MS);
  Serial.printf("[link] target ws://%s:%d\n", HOST_IP, HOST_PORT);
}

#else
#error "config.h: LINK must be LINK_SERIAL or LINK_WIFI"
#endif
