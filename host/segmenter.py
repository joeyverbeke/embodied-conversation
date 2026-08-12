"""Motion stream -> gesture segments (PLAN 5.1).

Dual-threshold hysteresis, structurally voice activity detection, and it fails
the same way if you collapse it to a single threshold: a gesture that dips
briefly below the line gets cut in two.

What changed for the ball: there are now *two* activity gates, on rotation and
on translation, and either one opens a gesture. Gating on rotation alone was
correct for a board strapped to a forearm, where nothing moves without turning.
A ball held in a hand can be lifted straight up, carried across the body, or
tossed gently with almost no rotation at all — |w| stays under the threshold and
the movement simply never happens as far as the machine is concerned. That was
not a tuning problem, it was a blind spot.

Free fall gets its own path. It is the one unambiguous thing an IMU can see,
and a throw must never be swallowed into a surrounding gesture.

This finds movements inside a stream that already has a person in it. Whether
anyone is there at all, and what the quiet between movements means, belongs to
host/lifecycle.py — which owns this class and only feeds it frames while the
ball is actually in a hand. Stillness used to be emitted from here and moved
there, because "not moving" means completely different things on a table and in
someone's grip.

Timing comes from the device's own t_ms, never from host arrival time, so
network jitter cannot move a boundary.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from . import config
from .protocol import MotionFrame

DEG = 180.0 / math.pi

IDLE = "IDLE"
ONSET = "ONSET"
ARMED = "ARMED"
GESTURING = "GESTURING"
SETTLING = "SETTLING"
REFRACTORY = "REFRACTORY"


def omega_dps(f: MotionFrame) -> float:
    """Rotation rate magnitude, degrees per second."""
    return math.sqrt(f.gx * f.gx + f.gy * f.gy + f.gz * f.gz) * DEG


def accel_ms2(f: MotionFrame) -> float:
    """Linear acceleration magnitude, gravity already removed."""
    return math.sqrt(f.lax * f.lax + f.lay * f.lay + f.laz * f.laz)


def total_g(f: MotionFrame) -> float:
    """Raw accelerometer magnitude, gravity included.

    Near zero means the ball is in free fall — it and the sensor are falling
    together, so there is nothing for the accelerometer to push against. This
    must come from the raw reading: while weightless the fusion's idea of which
    way gravity points is unanchored, so anything reconstructed from the
    quaternion is least trustworthy at the exact moment it matters most.
    """
    return math.sqrt(f.ax * f.ax + f.ay * f.ay + f.az * f.az)


@dataclass(slots=True)
class Segment:
    """frames includes up to PREBUFFER_MS of pre-roll before t_start.

    The pre-roll is what lets thresholds be lowered later and an earlier onset
    re-found in already-captured data. Timing metrics should use `motion()`,
    which starts at the threshold crossing.
    """
    frames: list[MotionFrame]
    t_start: int
    t_end: int
    ended: str                      # "settled", "force-cut" or "airborne"
    hesitations: int = 0            # sub-threshold starts that came to nothing
    airborne_ms: int = 0            # 0 unless the ball left the hand

    @property
    def duration_ms(self) -> int:
        return self.t_end - self.t_start

    def motion(self) -> list[MotionFrame]:
        """Frames from the threshold crossing onward, pre-roll excluded."""
        return [f for f in self.frames if f.t_ms >= self.t_start]




# Hard cap on the pre-buffer, in frames. The time-based trim below is the real
# policy; this is the guard rail that makes a bad timestamp survivable instead
# of unbounded. Generous enough never to bind in normal operation.
_BUF_MAX = int(config.PREBUFFER_MS / 1000.0 * config.IMU_HZ) + 20

# A gap larger than this, or any step backwards, is not a dropped frame — it is
# a different timeline. Opening the serial port resets the ESP32, so frames sent
# before the reset arrive carrying the old millis() and are followed by frames
# starting again from zero.
_DISCONTINUITY_MS = 2000


@dataclass(slots=True)
class Segmenter:
    state: str = IDLE
    _buf: deque = field(default_factory=lambda: deque(maxlen=_BUF_MAX))
    _seg: list = field(default_factory=list)     # frames since t_start
    _t_start: int = 0
    _t_mark: int = 0                             # onset / settle / refractory
    _hesitations: int = 0
    _fall_since: int | None = None               # start of the current free fall
    _airborne_ms: int = 0
    _last_t: int | None = None
    last_state_change: int = 0

    def reset(self) -> None:
        """Forget everything. The timeline this state described is gone."""
        self._buf.clear()
        self._seg = []
        self.state = IDLE
        self._t_start = self._t_mark = 0
        self._hesitations = 0
        self._fall_since = None
        self._airborne_ms = 0

    def push(self, f: MotionFrame) -> Optional[Segment]:
        """Feed one frame. Returns an event on the frame that completes one."""
        t = f.t_ms

        # Every duration here is a difference of device timestamps, so the
        # stream has to be one continuous timeline or none of the arithmetic
        # means anything.
        #
        # Opening the serial port resets the board. Frames already in flight
        # carry the pre-reset millis() and arrive just before the clock starts
        # again from zero — so one frame stamped 186000 lands at the head of the
        # pre-buffer, and the trim below stops at the first frame that is not
        # old enough to drop. That frame never is. The buffer then grows for the
        # whole session, every gesture's pre-roll becomes the entire recording,
        # and the evidence is invisible afterwards because _complete() filters
        # out anything past t_end. Logs went from kilobytes to 55 MB and the
        # features were computed over minutes of unrelated movement.
        if self._last_t is not None and not (
                0 <= t - self._last_t <= _DISCONTINUITY_MS):
            self.reset()
        self._last_t = t

        w = omega_dps(f)
        a = accel_ms2(f)

        # Either kind of activity counts. A ball can move without turning and
        # turn without moving, and both are things a person did.
        loud = w > config.ARM_ON_DPS or a > config.ACCEL_ON_MS2
        quiet = w < config.ARM_OFF_DPS and a < config.ACCEL_OFF_MS2

        self._buf.append(f)
        cutoff = t - config.PREBUFFER_MS
        while self._buf and self._buf[0].t_ms < cutoff:
            self._buf.popleft()

        if self.state in (ONSET, ARMED, GESTURING, SETTLING):
            self._seg.append(f)

        airborne = self._track_freefall(f, t)

        # A throw ends the gesture the moment the ball is back in a hand, and
        # cannot wait for the settle timer: whatever happens next is a separate
        # act, and merging the two would lose both.
        if airborne and self.state in (ARMED, GESTURING, SETTLING):
            return self._complete(t, "airborne")

        if self.state == IDLE:
            if loud:
                self._enter(ONSET, t)
                self._t_start = t
                # seed from the pre-buffer so the wind-up into the threshold
                # survives, not just everything after it
                self._seg = list(self._buf)

        elif self.state == ONSET:
            if not loud:
                self._enter(IDLE, t)
                self._seg = []
            elif t - self._t_start >= config.ONSET_HOLD_MS:
                self._enter(ARMED, t)

        elif self.state == ARMED:
            if quiet:
                # Started, then thought better of it. Worth counting: a stretch
                # full of false starts is a person being tentative, which is
                # not visible in any single gesture.
                self._hesitations += 1
                self._enter(IDLE, t)
                self._seg = []
            elif t - self._t_start > config.MIN_DURATION_MS:
                self._enter(GESTURING, t)

        elif self.state == GESTURING:
            if t - self._t_start > config.MAX_DURATION_MS:
                return self._complete(t, "force-cut")
            if quiet:
                self._enter(SETTLING, t)

        elif self.state == SETTLING:
            if loud:
                self._enter(GESTURING, t)
            elif t - self._t_mark >= config.SETTLE_HOLD_MS:
                # the gesture ended when it went quiet, not when we noticed
                return self._complete(self._t_mark, "settled")

        elif self.state == REFRACTORY:
            if t - self._t_mark >= config.REFRACTORY_MS:
                self._enter(IDLE, t)

        return None

    # ── free fall ─────────────────────────────────────────────────────────

    def _track_freefall(self, f: MotionFrame, t: int) -> bool:
        """True on the frame where a completed flight lands."""
        if total_g(f) < config.FREEFALL_MS2:
            if self._fall_since is None:
                self._fall_since = t
            return False

        if self._fall_since is None:
            return False

        flight = t - self._fall_since
        self._fall_since = None
        if flight >= config.FREEFALL_MIN_MS:
            self._airborne_ms = flight
            return True
        return False

    # ── transitions ───────────────────────────────────────────────────────

    def _enter(self, state: str, t: int) -> None:
        self.state = state
        self._t_mark = t
        self.last_state_change = t

    def _complete(self, t_end: int, how: str) -> Segment:
        frames = [f for f in self._seg if f.t_ms <= t_end]
        seg = Segment(frames=frames, t_start=self._t_start, t_end=t_end,
                      ended=how, hesitations=self._hesitations,
                      airborne_ms=self._airborne_ms)
        self._hesitations = 0
        self._airborne_ms = 0
        self._seg = []
        self._enter(REFRACTORY, t_end)
        return seg
