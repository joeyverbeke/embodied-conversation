"""Message framing for the serial link.

WebSocket hands us message boundaries; a serial port hands us a byte stream.
This adds the boundaries back:

    0xA7 0x5E | length u16 LE | payload

The payload is an untouched protocol.py message — its first byte is already the
opcode, so nothing about the wire format changes, only what carries it.

The magic prefix earns its keep twice. It lets the firmware's Serial.printf
diagnostics share the one port: anything that isn't a frame is device text, and
gets surfaced rather than swallowed. And it makes desync self-correcting — a
lost byte costs one frame, not the session.
"""

import struct
from typing import Callable, Iterator

MAGIC = b"\xa7\x5e"
_HEADER = struct.Struct("<H")
HEADER_SIZE = len(MAGIC) + _HEADER.size

# Nothing legitimate is bigger: PCM payloads are PCM_CHUNK_MS at 24 kHz mono
# int16 (1920 bytes) plus a 3-byte header. Anything larger is a bad length read
# out of a desynced stream, and resyncing beats trusting it.
MAX_PAYLOAD = 2048


def frame(payload: bytes) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload {len(payload)} exceeds {MAX_PAYLOAD}")
    return MAGIC + _HEADER.pack(len(payload)) + payload


class Unframer:
    """Incremental parser. Feed it whatever the port gave you.

    Bytes that aren't part of a frame are handed to `on_text` a line at a time —
    that's the device's own logging, and losing it would mean debugging this
    thing through a port already held by the server.
    """

    def __init__(self, on_text: Callable[[str], None] | None = None) -> None:
        self._buf = bytearray()
        self._text = bytearray()
        self._on_text = on_text

    def feed(self, data: bytes) -> Iterator[bytes]:
        self._buf += data

        while True:
            start = self._buf.find(MAGIC)
            if start == -1:
                # No frame in flight, so this is device text — except a
                # trailing MAGIC[0], which may be the first half of a magic
                # split across two reads. Hold that one byte and nothing else,
                # or the last log line before a quiet port never prints.
                keep = 1 if self._buf.endswith(MAGIC[:1]) else 0
                if len(self._buf) > keep:
                    self._drain_text(self._buf[:len(self._buf) - keep])
                    del self._buf[:len(self._buf) - keep]
                return

            if start:
                self._drain_text(self._buf[:start])
                del self._buf[:start]

            if len(self._buf) < HEADER_SIZE:
                return

            (length,) = _HEADER.unpack_from(self._buf, len(MAGIC))
            if length > MAX_PAYLOAD:
                # Almost certainly a magic-shaped pair inside a payload we lost
                # sync on. Step past it and look for the next one.
                self._drain_text(self._buf[:len(MAGIC)])
                del self._buf[:len(MAGIC)]
                continue

            end = HEADER_SIZE + length
            if len(self._buf) < end:
                return

            yield bytes(self._buf[HEADER_SIZE:end])
            del self._buf[:end]

    def _drain_text(self, chunk: bytes) -> None:
        if self._on_text is None:
            return
        self._text += chunk
        while True:
            nl = self._text.find(b"\n")
            if nl == -1:
                break
            line = self._text[:nl].decode("utf-8", "replace").strip()
            del self._text[:nl + 1]
            if line:
                self._on_text(line)
