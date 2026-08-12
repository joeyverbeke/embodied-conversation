# gradi-remark

A wearable that watches how you move and says something about it.

See [PLAN.md](PLAN.md) for the full specification. This README covers setup, and
grows as phases land.

**Current state: Phase 1 — host validated end to end, firmware on-hardware test in progress.**

---

## Hardware

| Signal | XIAO pin | GPIO | To |
|---|---|---|---|
| I2C SDA | D4 | 5 | BNO085 SDA |
| I2C SCL | D5 | 6 | BNO085 SCL |
| I2S BCLK | D10 | 9 | MAX98357A BCLK |
| I2S LRCLK | D9 | 8 | MAX98357A LRC |
| I2S DIN | D8 | 7 | MAX98357A DIN |
| Amp enable | — | — | MAX98357A SD → 3V3 |
| Power | 3V3, GND | | both breakouts |

BNO085 must be in I2C mode: PS0 and PS1 both LOW (the Adafruit breakout's
default). Address `0x4A`, or `0x4B` with the ADR jumper bridged.

---

## Firmware toolchain

Verified with:

| | version |
|---|---|
| `arduino-cli` | 1.4.1 |
| `esp32:esp32` core | 3.3.1 |
| Adafruit BNO08x | 1.2.5 |
| Adafruit BusIO | 1.17.4 |
| Adafruit Unified Sensor | 1.1.15 |

Install, if starting from nothing:

```bash
brew install arduino-cli
```

```bash
arduino-cli config init && arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json && arduino-cli core update-index && arduino-cli core install esp32:esp32 && arduino-cli lib install "Adafruit BNO08x"
```

I2S comes from `ESP_I2S`, bundled with the esp32 core — nothing to install.

### Board options

PSRAM is **disabled by default** on this board and must be turned on — Phase 1
puts a 512 KB audio ring buffer there. The FQBN below carries the setting, so
building from the command line needs no menu fiddling. In the Arduino IDE, set
**Tools → PSRAM → OPI PSRAM** by hand.

---

## Phase 0 — bring-up

Checks, in order: I2C scan, BNO085 quaternion + gyro at 100 Hz, a 440 Hz tone
every 5 s, and a WiFi connect that prints the assigned IP.

### 1. WiFi credentials

```bash
cp firmware/bringup/secrets.h.example firmware/bringup/secrets.h
```

Then edit `firmware/bringup/secrets.h` with the network SSID, password, and the
IP of the Mac that will run the host server. `secrets.h` is gitignored; the
`.example` is not.

Find the Mac's LAN IP with:

```bash
ipconfig getifaddr en0 || ipconfig getifaddr en1
```

`en0` is Wi-Fi on most Apple Silicon laptops but Ethernet on some machines —
check both, and use the one on the same network as the device.

### 2. Build and flash

Plug the XIAO in and find its port:

```bash
arduino-cli board list
```

```bash
arduino-cli compile --upload -p /dev/cu.usbmodem101 -b esp32:esp32:XIAO_ESP32S3:PSRAM=opi firmware/bringup
```

Substitute the port from the previous command. If the board doesn't appear,
hold **B** while tapping **R** to enter bootloader mode, then re-run.

### 3. Watch

```bash
arduino-cli monitor -p /dev/cu.usbmodem101 -c baudrate=115200
```

Expected output:

Actual output from a passing run:

```
=== gradi-remark bring-up ===
[sys] heap 314276, psram 8388608
[i2c] scanning 0x08..0x77
[i2c] device at 0x4A
[imu] up at 0x4A — part 10004148, sw 3.2.6
[imu] reports enabled @ 100 Hz
[i2s] up — 24 kHz / 16-bit / mono
[wifi] connecting to "the.Phone"
[wifi] scanning...
[wifi] 4 networks
[wifi]   "TorranceIOT"  ch 1  -83 dBm  WPA2
[wifi] connected — IP 172.20.10.2, RSSI -31 dBm, ch 6
=== running: quat/gyro @100 Hz, tone every 5 s ===
[imu] sensor reset — re-enabling reports
[imu] reports enabled @ 100 Hz
q  0.9998 -0.0173 -0.0104  0.0026  w  -0.006   0.000   0.000
[rate] quat 100 Hz, gyro 100 Hz
```

### What counts as passing

- The I2C scan finds `0x4A` (or `0x4B`).
- `[rate]` reports roughly 100 Hz for both quat and gyro, steadily.
- Quaternion values change when the board is rotated; gyro values spike when
  it's moved and return near zero when still.
- A clean 440 Hz tone once every 5 s, with no click at either edge.
- WiFi prints an IP on the expected subnet.

If the I2C scan finds nothing, that is a wiring or strapping problem — check
power, SDA/SCL, and the PS0/PS1 straps rather than reaching for software
workarounds.

### WiFi troubleshooting

The sketch runs a full scan before connecting and prints the reason code on
every disconnect, so failures name themselves:

| reason | meaning |
|---|---|
| 201 | no AP found — wrong SSID, or the AP isn't beaconing yet |
| 202 | auth fail |
| 204 | 4-way handshake timeout — usually a wrong password |

On an **iPhone Personal Hotspot**, turn **Maximize Compatibility** on so the
2.4 GHz band appears; the ESP32 has no 5 GHz radio. A cold hotspot can throw one
201 before it settles — the connect loop rides that out. Note the hotspot hands
out `172.20.10.x` and the Mac's address can move between sessions, so re-check
`ipconfig getifaddr en0` and update `HOST_IP` after reconnecting.

---

## Phase 1 — the piece

### Host setup

Needs [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

```bash
brew install uv ollama
```

Pull the model and keep it resident:

```bash
ollama serve
```

```bash
ollama pull llama3.2:3b
```

Create the Python environment (installs `mlx-audio`, `misaki`, `websockets`,
`ollama`):

```bash
uv sync
```

Kokoro-82M downloads on first run, and the very first synthesis also fetches
spaCy's `en_core_web_sm` and warms Metal — about 40 s, paid once. The server
does this at startup on purpose, so no participant ever waits for it.

### Choosing the link

The device reaches the host over **the USB cable** or **WiFi**. Two switches,
which must agree:

| | cable (default) | WiFi |
|---|---|---|
| `firmware/gradi_remark/config.h` | `#define LINK LINK_SERIAL` | `#define LINK LINK_WIFI` |
| `host/config.py` | `LINK = "serial"` | `LINK = "ws"` |

The cable is the default because the board is already plugged into the Mac for
power, and it needs no network at all — which is the whole point in a room whose
only WiFi is behind a captive portal. WiFi is where this ends up once the piece
is untethered; that path is unchanged and still builds, it just isn't the
default right now.

Nothing about the wire format differs between them (`host/protocol.py`). Serial
adds message framing, since a byte stream has none — see `host/framing.py`.

Two consequences of the cable carrying data:

- **The server owns the port**, so `arduino-cli monitor` can't run alongside it.
  It doesn't need to: the firmware's `Serial.printf` output now appears in the
  server log prefixed `[device]`.
- **Stop the server before flashing**, or the upload can't open the port.

`secrets.h` only matters in WiFi mode; in cable mode its contents are ignored.

#### If the voice crackles

USB will deliver audio faster than the device can parse it, and the ESP32's CDC
driver drops bytes on a full buffer rather than pushing back on the host. Lost
bytes sound like crackle, and can swallow the `UTT_END` that ends playback —
which shows up as `no STATE idle within Ns; flushing device` in the log.

The device says so when it happens:

```
[device] [link] lost 2508 bytes — host is outrunning the parser
```

Two knobs in `host/config.py`, both host-side (no reflash):

| | meaning |
|---|---|
| `SERIAL_PACE` | sustained rate, as a multiple of realtime audio. 3.0 is comfortable. |
| `SERIAL_BURST_BYTES` | how much goes at full USB speed before pacing bites. **Must stay well under** the device's `SERIAL_RX_BUFFER`, or the burst is itself the overflow. |

Lower `SERIAL_BURST_BYTES` first — it is the usual culprit. In steady state that
`lost N bytes` line should never appear at all.

### Firmware

```bash
cp firmware/gradi_remark/secrets.h.example firmware/gradi_remark/secrets.h
```

Fill in SSID, password, and the Mac's IP (WiFi mode only), then:

```bash
arduino-cli compile --upload -p /dev/cu.usbmodem2101 -b esp32:esp32:XIAO_ESP32S3:PSRAM=opi firmware/gradi_remark
```

PSRAM **must** be on — the 512 KB audio ring lives there and the firmware
refuses to start without it. The board's other defaults (`USBMode=hwcdc`,
`CDCOnBoot=Enabled`) are what make `Serial` a native USB port rather than a
115200 UART, so the cable has ample room for 24 kHz audio.

Libraries beyond Phase 0: `WebSockets` 2.7.2 (Markus Sattler), needed for WiFi
mode.

```bash
arduino-cli lib install "WebSockets"
```

### Run

```bash
caffeinate -i uv run python -m host.server
```

`caffeinate -i` matters: if the Mac sleeps, the server dies mid-session.

Power the device. Over the cable the host picks up the first
`/dev/cu.usbmodem*` and waits for the board — resetting or replugging it is
fine, the port is reopened automatically. Over WiFi the device joins the network
and connects. Either way: perform a gesture; it responds.

### Replaying a session

Every logged gesture carries its raw frames, so segmentation, trajectory and
features can all be retuned at a desk against real movement:

```bash
uv run python -m host.replay logs/session-20260808-163514.jsonl --verbose
```

By default each stored segment is re-featurised in place. `--resegment` feeds
every frame back through a fresh `Segmenter` instead, which is the only way to
test the gates themselves — including whether movements that used to fall below
them are now caught.

Logs recorded before the accelerometer was enabled carry nine fields per frame
instead of fifteen. They replay, and the rotation features still mean what they
always did, but nothing depending on the trajectory can be computed from them.
Those descriptors come out as `distance unclear` rather than being guessed at.

### Checking the trajectory against a tape measure

The reconstruction is the one thing in the pipeline that can be confidently
wrong, so verify it directly rather than trusting it. Hold the ball at a marked
height, lift it exactly 30 cm, hold, and replay:

```bash
uv run python -m host.replay logs/<the session>.jsonl --verbose
```

It should report 25–35 cm. If it does not, the zero-velocity correction or the
world-frame rotation is wrong, and nothing downstream is worth tuning until it
is fixed. There is no calibration step to perform here — Game Rotation Vector's
world frame already has +Z up, so "rose" and "fell" are absolute without anyone
having to hold poses first.

### Editing the voice

`persona/persona.txt` is hot-reloaded on every gesture — edit and save, and the
next response uses it. No restart. This file is a placeholder and is meant to
be replaced.

Thresholds live in `host/config.py`; those need a restart.

### If the board won't take a new sketch

`arduino-cli` sometimes can't reset a running sketch into the bootloader.
Hold **B** (BOOT), tap **R** (RESET), release **B**, then upload again.

## macOS notes (relevant from Phase 1 on)

- **Local Network permission.** macOS 15+ prompts once when a process first
  touches the LAN. If Python never gets it, the ESP32's connection fails
  *silently* — no error on either side. Grant it under **System Settings →
  Privacy & Security → Local Network**, and enable the entry for Terminal (or
  whichever app runs Python).
- **Sleep kills the server.** Run it as `caffeinate -i uv run python -m host.server`.
- **Getting the LAN IP.** `ipconfig getifaddr en0`, falling back to `en1` —
  `en0` is Wi-Fi on most Apple Silicon laptops but Ethernet on some, so check
  both and use the one on the device's network.
