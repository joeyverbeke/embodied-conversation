"""Where the ball actually went.

Orientation tells you how the ball spun. For a ball, that is the less
interesting half — a ball has no front, so a 90 degree rotation is not a fact
anyone can perceive about themselves. What a person experiences is *where they
moved it*: up, across, away, and how far. That is translation, and it is not in
the quaternion.

Recovering it means double-integrating acceleration, which is normally hopeless
because the error grows as t^2. Two things rescue it here:

  1. The window is bounded. The segmenter hands over one gesture, typically one
     to three seconds, not a continuous stream.
  2. Both ends are at rest. The segmenter only opens a gesture once the device
     starts moving and only closes it once it has gone quiet again. So the true
     velocity at both ends is zero, and any velocity left at the end of the
     integration is accumulated bias that can be subtracted back out.

That second point is a zero-velocity update, and it is the whole trick. What
comes out is good to roughly 5-15 cm over a couple of seconds, which is not
survey-grade and is entirely sufficient for "raised it about a foot".

It degrades, so `confidence` is returned alongside and must be honoured. A
distance nobody should assert is worse than no distance at all — the persona
is built on only claiming what it was actually given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(slots=True)
class Path:
    """A reconstructed trajectory, in metres, in the world frame (+Z is up)."""

    position: np.ndarray            # (N, 3) relative to the start point
    velocity: np.ndarray            # (N, 3) after drift correction
    speed: np.ndarray               # (N,) magnitude of velocity
    t: np.ndarray                   # (N,) seconds from the start of the gesture
    confidence: float               # 0..1 — see _confidence()
    residual_ms: float              # velocity left over before correction

    @property
    def trusted(self) -> bool:
        return self.confidence >= config.TRAJECTORY_MIN_CONFIDENCE


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v from the sensor frame into the world frame.

    q is (N,4) as (w,x,y,z), v is (N,3). Game Rotation Vector's world frame has
    +Z up, so the vertical component of the result is absolute and needs no
    tare — which is what makes "rose" and "fell" honest without a calibration
    step the audience would have to perform.
    """
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _confidence(duration: float, residual: float, peak_accel: float) -> float:
    """How much of this reconstruction is worth asserting.

    Three independent ways for it to be wrong, and the worst one wins.

    Long windows accumulate error. A large leftover velocity means the at-rest
    assumption itself broke — usually a force-cut that landed mid-flight, or a
    gesture that never really stopped.

    The third is the one that actually bites, and the one a clean ZUPT hides:
    if the movement never rose meaningfully above the tremor of a hand holding
    the device, there is no displacement signal to recover, only integrated
    noise. That failure looks *good* by the other two measures — short window,
    small residual, tidy arithmetic — and returns a confident wrong number.
    Slow movements fail here and always will.
    """
    if duration <= 0:
        return 0.0
    over = max(0.0, duration - config.TRAJECTORY_FULL_S)
    span = max(config.TRAJECTORY_MAX_S - config.TRAJECTORY_FULL_S, 1e-6)
    by_time = 1.0 - (over / span) ** 2
    by_drift = 1.0 - residual / max(config.TRAJECTORY_MAX_DRIFT_MS, 1e-6)

    snr = peak_accel / max(config.TREMOR_MS2, 1e-6)
    lo, hi = config.TRAJECTORY_MIN_SNR, config.TRAJECTORY_GOOD_SNR
    by_snr = (snr - lo) / max(hi - lo, 1e-6)

    return float(np.clip(min(by_time, by_drift, by_snr), 0.0, 1.0))


def reconstruct(frames) -> Path | None:
    """Integrate a segment's frames into a path through space.

    Returns None if there is not enough to integrate at all.
    """
    if len(frames) < 3:
        return None

    t_ms = np.array([f.t_ms for f in frames], dtype=np.float64)
    t = (t_ms - t_ms[0]) / 1000.0

    # Real per-frame intervals rather than a nominal 1/IMU_HZ. Frames are
    # stamped by the sensor hub and can arrive a millisecond or two apart;
    # over a few hundred samples that adds up in a term being integrated twice.
    dt = np.diff(t, prepend=t[0] - 1.0 / config.IMU_HZ)
    dt = np.clip(dt, 1e-4, 0.1)

    quats = np.array([[f.qw, f.qx, f.qy, f.qz] for f in frames])
    accel_body = np.array([[f.lax, f.lay, f.laz] for f in frames])

    # Identically zero means no accelerometer data, not a stationary ball —
    # frames replayed from a log recorded before the sensor was enabled. A real
    # reading is never exactly zero. Integrating it would produce a confident
    # 0 cm, which is a fabricated measurement wearing a plausible face.
    if not accel_body.any():
        return None

    accel_world = rotate(quats, accel_body)

    # First integration. Starts at zero because the segmenter guarantees the
    # gesture began from rest.
    velocity = np.cumsum(accel_world * dt[:, None], axis=0)

    duration = float(t[-1])
    residual = float(np.linalg.norm(velocity[-1]))

    # Zero-velocity update. The device is at rest at both ends, so whatever
    # velocity is left at the end is bias accumulated along the way. Removing it
    # as a linear ramp is equivalent to subtracting the mean acceleration, which
    # is exactly the constant offset a biased accelerometer contributes.
    if duration > 0:
        ramp = (t / t[-1])[:, None]
        velocity = velocity - ramp * velocity[-1]

    position = np.cumsum(velocity * dt[:, None], axis=0)
    position = position - position[0]

    peak_accel = float(np.linalg.norm(accel_body, axis=1).max())

    return Path(
        position=position,
        velocity=velocity,
        speed=np.linalg.norm(velocity, axis=1),
        t=t,
        confidence=_confidence(duration, residual, peak_accel),
        residual_ms=residual,
    )


# ── What the path is worth saying ─────────────────────────────────────────

def measure(path: Path) -> dict:
    """Distances and directions, in centimetres and metres per second.

    Every value here depends on the integration, so nothing in this dict should
    be asserted when `path.trusted` is false. Callers get the numbers either
    way — deciding what to claim is the feature layer's job, not this one's.
    """
    p = path.position
    from_start = np.linalg.norm(p, axis=1)
    steps = np.linalg.norm(np.diff(p, axis=0), axis=1)

    path_m = float(steps.sum())
    net_m = float(np.linalg.norm(p[-1]))
    vertical = p[:, 2]

    # Which way it went, as a share of the total distance travelled. Kept as a
    # ratio rather than a label so the feature layer can say "mostly upward"
    # and "straight up" differently.
    net = p[-1]
    net_len = max(net_m, 1e-9)

    return {
        "span_cm": round(float(from_start.max()) * 100, 1),
        "net_cm": round(net_m * 100, 1),
        "path_cm": round(path_m * 100, 1),
        "rise_cm": round(float(vertical.max()) * 100, 1),
        "drop_cm": round(float(vertical.min()) * 100, 1),
        "end_height_cm": round(float(vertical[-1]) * 100, 1),
        "travel_directness": round(net_m / path_m, 3) if path_m > 1e-6 else 0.0,
        "peak_speed_ms": round(float(path.speed.max()), 2),
        "vertical_share": round(float(net[2] / net_len), 3),
        "confidence": round(path.confidence, 2),
    }


def _name_direction(vec: np.ndarray, distance_cm: float) -> str:
    """A word for a displacement, from its share of vertical travel."""
    if distance_cm < config.SMALL_CM * 0.5:
        return "nowhere"
    share = float(vec[2]) / max(float(np.linalg.norm(vec)), 1e-9)
    if share > 0.7:
        return "up"
    if share < -0.7:
        return "down"
    if share > 0.35:
        return "up and out"
    if share < -0.35:
        return "down and out"
    return "sideways"


def direction(measured: dict) -> str:
    """A word for where it ended up, relative to where it started."""
    if measured["net_cm"] < config.SMALL_CM * 0.5:
        return "nowhere"
    share = measured["vertical_share"]
    if share > 0.7:
        return "up"
    if share < -0.7:
        return "down"
    if share > 0.35:
        return "up and out"
    if share < -0.35:
        return "down and out"
    return "sideways"


# ── Strokes ───────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Stroke:
    """One leg of a continuous movement: travel in a consistent direction."""

    distance_cm: float
    direction: str
    vertical_cm: float
    duration_s: float
    peak_speed_ms: float


def _smooth(v: np.ndarray, k: int = 7) -> np.ndarray:
    """Boxcar along each axis. Raw velocity jitters enough to fake a turn."""
    if len(v) < k:
        return v
    kernel = np.ones(k) / k
    pad = k // 2
    out = np.empty_like(v)
    for axis in range(v.shape[1]):
        padded = np.pad(v[:, axis], pad, mode="edge")
        out[:, axis] = np.convolve(padded, kernel, mode="valid")[:len(v)]
    return out


def strokes(path: Path) -> list[Stroke]:
    """Split the path where the direction of travel changes.

    Boundaries are heading changes rather than speed minima. A person reversing
    a movement does not necessarily pause at the turn — a fluent up-and-down has
    no still moment in it at all — so waiting for the speed to dip would miss the
    reversal that matters most.

    Strokes shorter than STROKE_MIN_CM are merged away: they are the wobble at a
    turn, not a leg of the journey, and reporting them would bury the shape of
    the movement in punctuation.
    """
    if path is None or len(path.velocity) < 8:
        return []

    v = _smooth(path.velocity)
    speed = np.linalg.norm(v, axis=1)
    moving = speed > max(config.STILL_SPEED_MS, 1e-6)
    if not moving.any():
        return []

    cos_limit = math.cos(math.radians(config.STROKE_TURN_DEG))

    bounds: list[list[int]] = []
    start = None
    ref = None
    for i in range(len(v)):
        if not moving[i]:
            continue
        heading = v[i] / speed[i]
        if ref is None:
            start, ref = i, heading
            continue
        if float(np.dot(heading, ref)) < cos_limit:
            bounds.append([start, i])
            start, ref = i, heading
        else:
            # drift the reference so a gradual arc stays one stroke, while a
            # genuine reversal still trips the test
            ref = ref * 0.9 + heading * 0.1
            ref = ref / max(float(np.linalg.norm(ref)), 1e-9)
    if start is not None:
        bounds.append([start, len(v) - 1])

    return _build(path, _merge(path, bounds))


def _merge(path: Path, bounds: list[list[int]]) -> list[list[int]]:
    """Fold negligible legs into the neighbour they most resemble."""
    if len(bounds) <= 1:
        return bounds

    def span(b) -> float:
        return float(np.linalg.norm(path.position[b[1]] - path.position[b[0]])) * 100

    out = bounds
    changed = True
    while changed and len(out) > 1:
        changed = False
        for i, b in enumerate(out):
            if span(b) >= config.STROKE_MIN_CM:
                continue
            if i == 0:
                out[1][0] = b[0]
            elif i == len(out) - 1:
                out[-2][1] = b[1]
            else:
                # give it to whichever neighbour is already longer
                if span(out[i - 1]) >= span(out[i + 1]):
                    out[i - 1][1] = b[1]
                else:
                    out[i + 1][0] = b[0]
            del out[i]
            changed = True
            break
    return out


def _build(path: Path, bounds: list[list[int]]) -> list[Stroke]:
    out = []
    for a, b in bounds:
        delta = path.position[b] - path.position[a]
        distance = float(np.linalg.norm(delta)) * 100
        out.append(Stroke(
            distance_cm=round(distance, 1),
            direction=_name_direction(delta, distance),
            vertical_cm=round(float(delta[2]) * 100, 1),
            duration_s=round(float(path.t[b] - path.t[a]), 2),
            peak_speed_ms=round(float(path.speed[a:b + 1].max()) if b > a else 0.0, 2),
        ))
    return out


def circling(path: Path) -> float:
    """How much this path goes *around* rather than back and forth. 0..1.

    Four attempts failed before this one: cumulative heading rotation, the
    rotation of the acceleration vector, the monotonicity of stroke headings,
    and an isoperimetric ratio. All were more complicated than the question,
    and one of them ranked throws as the most circular thing in the corpus.

    What actually separates them is embarrassingly simple. A circle is *flat* —
    it spreads over two dimensions where a shake or a sweep collapses onto one —
    and it travels much further than the box it stays inside, because it keeps
    going round. Measured on the same movement a person called "a circle in the
    air": flatness 0.88 against 0.16-0.39 for every shake in the corpus.

    Small movements are excluded on purpose. Integrated noise is isotropic, so
    anything tiny looks perfectly round and every 8 cm wobble claimed to be a
    circle.
    """
    pos = path.position
    if len(pos) < 20:
        return 0.0
    p = pos - pos.mean(axis=0)
    sv = np.linalg.svd(p, compute_uv=False)
    flat = float(sv[1] / max(sv[0], 1e-9))
    perimeter = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    extent = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))
    laps = perimeter / max(extent, 1e-9)
    net = float(np.linalg.norm(pos[-1] - pos[0])) / max(perimeter, 1e-9)

    if (extent * 100 < config.CIRCLE_MIN_CM or laps < config.CIRCLE_MIN_LAPS
            or net > config.CIRCLE_MAX_NET):
        return 0.0
    return flat


def shape(legs: list[Stroke]) -> str:
    """What the sequence of strokes did, as a whole."""
    if len(legs) < 2:
        return "single"
    verticals = [s.vertical_cm for s in legs]
    # a there-and-back has its vertical travel cancel out
    if len(legs) == 2 and verticals[0] * verticals[1] < 0:
        return "there and back"
    if len(legs) >= 3:
        sizes = [s.distance_cm for s in legs]
        if all(b < a for a, b in zip(sizes, sizes[1:])):
            return "shrinking"
        if all(b > a for a, b in zip(sizes, sizes[1:])):
            return "growing"
        return "back and forth"
    return "two legs"


def orientation_change_deg(frames) -> float:
    """Net angle between the first and last attitude, in degrees.

    Demoted from its old role as the main event but still worth having: turning
    a ball over in the hands is a real thing people do, and it is invisible to
    the trajectory.
    """
    if len(frames) < 2:
        return 0.0
    a = np.array([frames[0].qw, frames[0].qx, frames[0].qy, frames[0].qz])
    b = np.array([frames[-1].qw, frames[-1].qx, frames[-1].qy, frames[-1].qz])
    d = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return 2.0 * math.acos(d) * 180.0 / math.pi
