"""Segment -> one English line (PLAN 5.2).

The LLM sees only the string this produces, never the numbers behind it.
That boundary is the point: it lets what the device *notices* be retuned
independently of how it *talks*.
"""

from __future__ import annotations

import math

import numpy as np

from . import config
from .segmenter import Segment

DEG = 180.0 / math.pi


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v from the sensor frame into the world frame. q is (w,x,y,z)."""
    w, xyz = q[0], q[1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _pose(q: np.ndarray) -> str:
    """Where the limb is, from gravity alone.

    Game Rotation Vector's world frame has +Z up, so the elevation of the limb
    axis is absolute and needs no tare. Heading is *not* available (no
    magnetometer, by design), so PLAN 5.2's "across-body" is not derivable
    here — it is a yaw distinction and gravity cannot see yaw.
    """
    e = np.zeros(3)
    e[config.LIMB_AXIS] = config.LIMB_SIGN
    elevation = math.asin(float(np.clip(_rotate(q, e)[2], -1.0, 1.0))) * DEG
    if elevation > config.POSE_ELEVATION_DEG:
        return "overhead"
    if elevation < -config.POSE_ELEVATION_DEG:
        return "hanging"
    return "horizontal"


def _repetition(signal: np.ndarray, dt: float) -> tuple[int, float | None]:
    """Reversal count and repetition frequency (Hz) of the dominant-axis rate."""
    reversals = 0
    for i in range(1, len(signal)):
        if signal[i - 1] * signal[i] < 0 and max(
                abs(signal[i - 1]), abs(signal[i])) > config.REVERSAL_MIN_DPS:
            reversals += 1

    freq = None
    x = signal - signal.mean()
    if len(x) > 8 and np.any(x):
        ac = np.correlate(x, x, mode="full")[len(x) - 1:]
        if ac[0] > 0:
            ac = ac / ac[0]
            # first local maximum past the zero-lag lobe
            lag = None
            for i in range(2, len(ac) - 1):
                if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.3:
                    lag = i
                    break
            if lag:
                period = lag * dt
                if period > 0 and len(x) * dt / period >= config.REPEAT_MIN_PERIODS:
                    freq = 1.0 / period
    return reversals, freq


def extract(seg: Segment) -> dict:
    frames = seg.motion()
    if len(frames) < 2:
        return {}

    dt = 1.0 / config.IMU_HZ
    quats = np.array([[f.qw, f.qx, f.qy, f.qz] for f in frames])
    gyro = np.array([[f.gx, f.gy, f.gz] for f in frames]) * DEG
    mags = np.linalg.norm(gyro, axis=1)

    duration = seg.duration_ms / 1000.0

    # integrated path vs net rotation -> how direct the route was
    path = float(np.sum(mags) * dt)
    d = float(np.clip(abs(np.dot(quats[0], quats[-1])), 0.0, 1.0))
    net = 2.0 * math.acos(d) * DEG
    directness = (net / path) if path > 1e-6 else 0.0

    peak = float(mags.max())
    time_to_peak = float(mags.argmax()) / max(1, len(mags) - 1)

    # dominant rotation axis: first principal component of the rate vectors
    centred = gyro - gyro.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    projected = gyro @ axis
    reversals, freq = _repetition(projected, dt)

    return {
        "duration_s": round(duration, 2),
        "path_deg": round(path, 1),
        "net_deg": round(net, 1),
        "directness": round(directness, 3),
        "peak_dps": round(peak, 1),
        "time_to_peak": round(time_to_peak, 3),
        "dominant_axis": ["x", "y", "z"][int(np.argmax(np.abs(axis)))],
        "axis_vector": [round(float(v), 3) for v in axis],
        "reversals": reversals,
        "repeat_hz": round(freq, 2) if freq else None,
        "start_pose": _pose(quats[0]),
        "end_pose": _pose(quats[-1]),
        "n_frames": len(frames),
        "ended": seg.ended,
    }


def describe(f: dict) -> str:
    """The one line the LLM sees. Round figures only, no telemetry."""
    if not f:
        return "a movement too short to read"

    dur = f["duration_s"]
    peak = f["peak_dps"]
    reversals = f["reversals"]
    repeat = f["repeat_hz"]
    direct = f["directness"]
    # percussive means a strike, so it has to be brief as well as fast
    percussive = peak >= config.PERCUSSIVE_DPS and dur <= config.PERCUSSIVE_MAX_S

    if repeat and reversals >= 3:
        kind = "repeated shake"
    elif percussive and f["time_to_peak"] <= config.EARLY_PEAK_FRAC:
        kind = "sharp jab"
    elif direct >= config.DIRECT_RATIO:
        kind = "single arc"
    elif reversals >= 3:
        kind = "scribble"
    elif direct < config.RETURN_RATIO:
        kind = "out and back"
    else:
        kind = "loose sweep"

    parts = [f"{dur:.1f}s {kind}"]

    if kind == "repeated shake":
        parts.append(f"{reversals} reversals")
        parts.append(f"~{repeat:.1f} Hz")
    else:
        parts.append(f"{f['path_deg']:.0f}° path")
        if percussive:
            parts.append("percussive")
        elif direct >= config.DIRECT_RATIO:
            parts.append("very direct")
        elif direct < config.RETURN_RATIO and kind != "out and back":
            parts.append("wandering")

        if f["time_to_peak"] <= config.EARLY_PEAK_FRAC:
            parts.append("early acceleration")
        elif f["time_to_peak"] >= config.LATE_PEAK_FRAC:
            parts.append("late acceleration")

    parts.append(f"{f['start_pose']} → {f['end_pose']}")
    return ", ".join(parts)
