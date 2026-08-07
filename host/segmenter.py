"""Motion stream -> gesture segments (PLAN 5.1).

Dual-threshold hysteresis on gyro magnitude. Structurally this is voice
activity detection, and it fails the same way if you collapse it to a single
threshold: a gesture that dips briefly below the line gets cut in two.

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
    return math.sqrt(f.gx * f.gx + f.gy * f.gy + f.gz * f.gz) * DEG


@dataclass(slots=True)
class Segment:
    """frames includes up to PREBUFFER_MS of pre-roll before t_start.

    The pre-roll is what lets Phase 2 lower ARM_ON and re-find an earlier
    onset in already-captured data. Timing metrics should use `motion()`,
    which starts at the threshold crossing.
    """
    frames: list[MotionFrame]
    t_start: int
    t_end: int
    ended: str                      # "settled" or "force-cut"
    rejected_twitches: int = 0

    @property
    def duration_ms(self) -> int:
        return self.t_end - self.t_start

    def motion(self) -> list[MotionFrame]:
        """Frames from the threshold crossing onward, pre-roll excluded."""
        return [f for f in self.frames if f.t_ms >= self.t_start]


@dataclass(slots=True)
class Segmenter:
    state: str = IDLE
    _buf: deque = field(default_factory=deque)   # rolling pre-buffer
    _seg: list = field(default_factory=list)     # frames since t_start
    _t_start: int = 0
    _t_mark: int = 0                             # onset / settle / refractory
    _twitches: int = 0
    last_state_change: int = 0

    def push(self, f: MotionFrame) -> Optional[Segment]:
        """Feed one frame. Returns a Segment on the frame that completes one."""
        w = omega_dps(f)
        t = f.t_ms

        self._buf.append(f)
        cutoff = t - config.PREBUFFER_MS
        while self._buf and self._buf[0].t_ms < cutoff:
            self._buf.popleft()

        if self.state in (ONSET, ARMED, GESTURING, SETTLING):
            self._seg.append(f)

        if self.state == IDLE:
            if w > config.ARM_ON_DPS:
                self._enter(ONSET, t)
                self._t_start = t
                # seed from the pre-buffer so the wind-up into the threshold
                # survives, not just everything after it
                self._seg = list(self._buf)

        elif self.state == ONSET:
            if w < config.ARM_ON_DPS:
                self._enter(IDLE, t)
                self._seg = []
            elif t - self._t_start >= config.ONSET_HOLD_MS:
                self._enter(ARMED, t)

        elif self.state == ARMED:
            if w < config.ARM_OFF_DPS:
                self._twitches += 1
                self._enter(IDLE, t)
                self._seg = []
            elif t - self._t_start > config.MIN_DURATION_MS:
                self._enter(GESTURING, t)

        elif self.state == GESTURING:
            if t - self._t_start > config.MAX_DURATION_MS:
                return self._complete(t, "force-cut")
            if w < config.ARM_OFF_DPS:
                self._enter(SETTLING, t)

        elif self.state == SETTLING:
            if w > config.ARM_ON_DPS:
                self._enter(GESTURING, t)
            elif t - self._t_mark >= config.SETTLE_HOLD_MS:
                # the gesture ended when it went quiet, not when we noticed
                return self._complete(self._t_mark, "settled")

        elif self.state == REFRACTORY:
            if t - self._t_mark >= config.REFRACTORY_MS:
                self._enter(IDLE, t)

        return None

    def _enter(self, state: str, t: int) -> None:
        self.state = state
        self._t_mark = t
        self.last_state_change = t

    def _complete(self, t_end: int, how: str) -> Segment:
        frames = [f for f in self._seg if f.t_ms <= t_end]
        seg = Segment(frames=frames, t_start=self._t_start, t_end=t_end,
                      ended=how, rejected_twitches=self._twitches)
        self._twitches = 0
        self._seg = []
        self._enter(REFRACTORY, t_end)
        return seg
