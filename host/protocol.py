"""Wire format (PLAN 3). Single source of truth for both directions.

All frames binary, first byte is the message type, all multi-byte fields
little-endian. The firmware's packing must match this file exactly; if you
change one, change both.
"""

import struct
from dataclasses import dataclass
from typing import Iterator

# ── Message types ─────────────────────────────────────────────────────────
MOTION = 0x01      # device -> host
STATE = 0x02
LOG = 0x03

UTT_BEGIN = 0x10   # host -> device
PCM = 0x11
UTT_END = 0x12
FLUSH = 0x13

# Device states carried by STATE
IDLE = 0
PLAYING = 1

# seq u32, t_ms u32, qw qx qy qz f32x4, gx gy gz f32x3, lax lay laz f32x3,
# ax ay az f32x3
#
# Linear acceleration and raw accelerometer are both carried, which is
# algebraically redundant — one is the other minus gravity. It is deliberate.
# Linear acceleration is what the trajectory integrates, and the sensor hub's
# own fusion produces a better one than we could by subtracting a quaternion
# gravity estimate. But that estimate is worthless in free fall, which is
# exactly when we most need to know: the raw accelerometer going to ~0 is the
# only honest way to see that the ball is airborne. 12 extra bytes for an
# unambiguous throw is a bargain.
_MOTION_FRAME = struct.Struct("<IIfffffffffffff")
MOTION_FRAME_SIZE = _MOTION_FRAME.size          # 60; +1 type byte per message

_UTT_BEGIN = struct.Struct("<IH")               # sample_rate, utt_id
_UTT_ID = struct.Struct("<H")


@dataclass(slots=True)
class MotionFrame:
    seq: int
    t_ms: int
    qw: float
    qx: float
    qy: float
    qz: float
    gx: float                       # rad/s, sensor frame
    gy: float
    gz: float
    lax: float                      # m/s^2, sensor frame, gravity removed
    lay: float
    laz: float
    ax: float                       # m/s^2, sensor frame, gravity included
    ay: float
    az: float


class ProtocolError(ValueError):
    pass


# ── Device -> host ────────────────────────────────────────────────────────

def unpack(data: bytes):
    """Return (msg_type, payload). Payload type depends on msg_type:
    MOTION -> list[MotionFrame], STATE -> int, LOG -> str."""
    if not data:
        raise ProtocolError("empty message")
    kind = data[0]
    body = data[1:]

    if kind == MOTION:
        if len(body) % MOTION_FRAME_SIZE:
            raise ProtocolError(
                f"MOTION payload {len(body)} is not a multiple of "
                f"{MOTION_FRAME_SIZE}")
        return MOTION, list(_iter_frames(body))

    if kind == STATE:
        if len(body) != 1:
            raise ProtocolError(f"STATE payload must be 1 byte, got {len(body)}")
        return STATE, body[0]

    if kind == LOG:
        return LOG, body.decode("utf-8", "replace")

    raise ProtocolError(f"unknown message type 0x{kind:02x}")


def _iter_frames(body: bytes) -> Iterator[MotionFrame]:
    for off in range(0, len(body), MOTION_FRAME_SIZE):
        yield MotionFrame(*_MOTION_FRAME.unpack_from(body, off))


def pack_motion(frames) -> bytes:
    """Only used by tests and the fake device — the firmware packs its own."""
    out = bytearray([MOTION])
    for f in frames:
        out += _MOTION_FRAME.pack(f.seq, f.t_ms, f.qw, f.qx, f.qy, f.qz,
                                  f.gx, f.gy, f.gz,
                                  f.lax, f.lay, f.laz, f.ax, f.ay, f.az)
    return bytes(out)


def pack_state(state: int) -> bytes:
    return bytes([STATE, state])


def pack_log(text: str) -> bytes:
    return bytes([LOG]) + text.encode("utf-8")


# ── Host -> device ────────────────────────────────────────────────────────

def pack_utt_begin(sample_rate: int, utt_id: int) -> bytes:
    return bytes([UTT_BEGIN]) + _UTT_BEGIN.pack(sample_rate, utt_id)


def pack_pcm(utt_id: int, samples: bytes) -> bytes:
    """samples is raw int16 little-endian."""
    return bytes([PCM]) + _UTT_ID.pack(utt_id) + samples


def pack_utt_end(utt_id: int) -> bytes:
    return bytes([UTT_END]) + _UTT_ID.pack(utt_id)


def pack_flush() -> bytes:
    return bytes([FLUSH])
