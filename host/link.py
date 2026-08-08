"""Transport. One interface, two carriers: the USB cable or WiFi.

Everything above this file talks to a link through exactly two operations —
`async for message in link` and `await link.send(payload)` — which is all
`Session` ever asked of a WebSocket. So swapping the carrier changes nothing
about the pipeline, and `protocol.py` is untouched either way.

`config.LINK` picks one. WiFi is the long-term transport and stays whole; the
cable is what works in a room with no network you can join.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import websockets

from . import config, framing, protocol

log = logging.getLogger("gradi")


class LinkClosed(Exception):
    """The device went away. Mirrors websockets.ConnectionClosed."""


_EOF = object()


# ── Serial (USB cable) ────────────────────────────────────────────────────

class SerialLink:
    """A framed message stream over a serial port.

    pyserial is blocking, so reads live on a daemon thread and writes go
    through a single-worker executor — single-worker because frame order is
    the protocol, and a general pool could interleave two writes.
    """

    remote_address = ("usb",)

    def __init__(self, ser, loop: asyncio.AbstractEventLoop) -> None:
        self._ser = ser
        self._loop = loop
        self._queue: asyncio.Queue = asyncio.Queue()
        self._stop = threading.Event()
        self._writer = ThreadPoolExecutor(max_workers=1,
                                          thread_name_prefix="serial-tx")
        # Token bucket, refilled at SERIAL_PACE x realtime audio. See config.
        self._rate = config.SERIAL_PACE * config.SAMPLE_RATE * 2
        self._tokens = float(config.SERIAL_BURST_BYTES)
        self._tokens_t = time.monotonic()
        self._thread = threading.Thread(target=self._read_loop,
                                        name="serial-rx", daemon=True)
        self._thread.start()

    # inbound
    def _read_loop(self) -> None:
        unframer = framing.Unframer(on_text=self._log_device_text)
        try:
            while not self._stop.is_set():
                data = self._ser.read(max(1, self._ser.in_waiting))
                if not data:
                    continue
                for payload in unframer.feed(data):
                    self._loop.call_soon_threadsafe(self._queue.put_nowait,
                                                    payload)
        except Exception as exc:            # unplugged, or reset mid-read
            if not self._stop.is_set():
                self._loop.call_soon_threadsafe(log.info, "serial read ended: %s",
                                                exc)
        finally:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, _EOF)

    def _log_device_text(self, line: str) -> None:
        # The firmware's own Serial.printf output, riding the same port. It
        # used to need a second terminal; now it lands in the session log.
        self._loop.call_soon_threadsafe(log.info, "[device] %s", line)

    def __aiter__(self) -> "SerialLink":
        return self

    async def __anext__(self) -> bytes:
        item = await self._queue.get()
        if item is _EOF:
            raise StopAsyncIteration
        return item

    # outbound
    async def send(self, data: bytes) -> None:
        try:
            await self._loop.run_in_executor(self._writer, self._write, data)
        except Exception as exc:
            raise LinkClosed(str(exc)) from exc

    def _write(self, data: bytes) -> None:
        buf = framing.frame(data)
        self._pace(len(buf))
        self._ser.write(buf)

    def _pace(self, n: int) -> None:
        """Block until the cable has budget for n bytes.

        Runs on the writer thread, never the event loop. Sleeping here is the
        point: it is what stops the host outrunning the device's parser.
        """
        now = time.monotonic()
        self._tokens = min(float(config.SERIAL_BURST_BYTES),
                           self._tokens + (now - self._tokens_t) * self._rate)
        self._tokens_t = now
        if self._tokens < n:
            time.sleep((n - self._tokens) / self._rate)
            self._tokens_t = time.monotonic()
            self._tokens = 0.0
        else:
            self._tokens -= n

    async def close(self) -> None:
        self._stop.set()
        self._writer.shutdown(wait=False)
        try:
            self._ser.close()
        except Exception:
            pass
        self._queue.put_nowait(_EOF)


def _find_port() -> str | None:
    if config.SERIAL_PORT:
        return config.SERIAL_PORT
    ports = sorted(glob.glob("/dev/cu.usbmodem*"))
    return ports[0] if ports else None


def _open(port: str):
    import serial

    ser = serial.Serial()
    ser.port = port
    ser.baudrate = 115200          # ignored by native USB CDC; pyserial wants it
    ser.timeout = 0.1
    ser.write_timeout = 5.0
    # Software flow control would eat 0x11 and 0x13 — which are PCM and FLUSH.
    ser.xonxoff = False
    ser.rtscts = False
    ser.dsrdtr = False
    ser.open()
    # Leave the modem lines idle. Toggling DTR/RTS is how esptool puts an S3
    # into its bootloader, and opening the port should never do that. Not every
    # tty has these lines (a pty doesn't), and their absence is not a problem.
    try:
        ser.dtr = False
        ser.rts = False
    except OSError as exc:
        log.debug("%s has no modem control lines (%s)", port, exc)
    ser.reset_input_buffer()
    return ser


async def _serve_serial(handler) -> None:
    loop = asyncio.get_running_loop()
    waiting = False

    while True:
        port = _find_port()
        if port is None:
            if not waiting:
                log.info("waiting for a device on /dev/cu.usbmodem*")
                waiting = True
            await asyncio.sleep(0.5)
            continue
        waiting = False

        try:
            ser = await loop.run_in_executor(None, _open, port)
        except Exception as exc:
            log.warning("cannot open %s: %s", port, exc)
            await asyncio.sleep(1.0)
            continue

        log.info("listening on %s", port)
        link = SerialLink(ser, loop)
        try:
            # The cable has no connect event, so the device may still be
            # holding audio from a session that died. Start it silent.
            await link.send(protocol.pack_flush())
            await handler(link)
        except LinkClosed:
            pass
        except Exception:
            log.exception("serial session failed")
        finally:
            await link.close()

        # A reset re-enumerates USB, so the port really does vanish and return.
        log.info("device gone; waiting for it to come back")
        await asyncio.sleep(1.0)


# ── WebSocket (WiFi) ──────────────────────────────────────────────────────

async def _serve_ws(handler) -> None:
    async with websockets.serve(handler, config.BIND_HOST, config.BIND_PORT,
                                max_size=None, ping_interval=20):
        log.info("listening on ws://%s:%d", config.BIND_HOST, config.BIND_PORT)
        await asyncio.Future()


async def serve(handler) -> None:
    """Accept one device at a time and hand it to `handler`. Runs forever."""
    if config.LINK == "serial":
        await _serve_serial(handler)
    elif config.LINK == "ws":
        await _serve_ws(handler)
    else:
        raise ValueError(f"config.LINK must be 'serial' or 'ws', got "
                         f"{config.LINK!r}")
