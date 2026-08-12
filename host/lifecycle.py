"""Is anyone there, and what are they doing right now?

The ball sits on a table until someone picks it up. They might hold it a while
before doing anything, or move it immediately; they might move continuously, or
in bursts, with pauses of any length or none. Nothing about that is a sequence
of discrete gestures, and until now the machine assumed it was.

This layer answers the prior question — whether there is a person at all — and
turns the raw stream into events something can decide about. It does not decide
anything itself. That is host/policy.py.

The one signal it all rests on: **a hand cannot hold an object without
micro-rotation**. On a table the gyro reads a median of 0.05 deg/s; held still
in a hand it reads 0.78, with a p95 of 1.58 against the table's 0.25. Fifteen
times apart, essentially no overlap, and it needs no calibration because it is a
property of hands rather than of this room. Acceleration separates the two far
less cleanly and is deliberately not used.

Getting this right is what makes the piece showable. A machine that cannot tell
an empty table from a person holding it still will narrate to an empty room all
night, and no amount of good writing survives that.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Union

from . import config
from .protocol import MotionFrame
from .segmenter import Segment, Segmenter, omega_dps

DORMANT = "DORMANT"      # on the table, nobody here
HELD = "HELD"            # in a hand, not going anywhere
MOVING = "MOVING"        # in a hand, being moved


# ── Events ────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class PickedUp:
    """Someone just arrived. The start of a visit.

    The most valuable instant in the piece: the only moment the machine
    addresses a person who has not done anything yet, and so the only moment a
    suggestion lands before an intention has formed.
    """
    t_ms: int
    away_ms: int             # how long it sat untouched before this


@dataclass(slots=True)
class PutDown:
    """Back on the table. The visit is over."""
    t_ms: int
    visit_ms: int


@dataclass(slots=True)
class Holding:
    """Held, and not moving. A person deciding — not an empty room."""
    t_ms: int
    duration_ms: int
    first: bool


@dataclass(slots=True)
class Pause:
    """A lull between movements. The polite moment to speak into."""
    t_ms: int
    duration_ms: int


Event = Union[PickedUp, PutDown, Holding, Pause, Segment]


# ── The machine ───────────────────────────────────────────────────────────

@dataclass(slots=True)
class Lifecycle:
    state: str = DORMANT
    segmenter: Segmenter = field(default_factory=Segmenter)

    _win: deque = field(default_factory=deque)     # recent (t_ms, omega)
    _handled_since: int | None = None              # first frame that looked held
    _settled_since: int | None = None              # first frame that looked idle
    _quiet_since: int | None = None                # last movement ended
    _held_reported: int | None = None
    _pause_sent: bool = False
    _visit_start: int = 0
    _last_putdown: int = 0
    _last_t: int | None = None

    # ── inbound ───────────────────────────────────────────────────────────

    def push(self, f: MotionFrame) -> Optional[Event]:
        t = f.t_ms

        # The segmenter owns the discontinuity check for its own state; this
        # layer keeps its own because it must not carry a window across a device
        # reboot either.
        if self._last_t is not None and not (0 <= t - self._last_t <= 2000):
            self._reset(t)
        self._last_t = t

        self._win.append((t, omega_dps(f)))
        cutoff = t - config.LIFECYCLE_WINDOW_MS
        while len(self._win) > 1 and self._win[0][0] < cutoff:
            self._win.popleft()

        handled = self._looks_handled()

        if self.state == DORMANT:
            return self._dormant(t, handled)
        return self._present(f, t, handled)

    # ── classification ────────────────────────────────────────────────────

    def _looks_handled(self) -> bool:
        """Hysteresis on the median rotation across the window.

        The median rather than the mean: a single knock against the table would
        drag a mean over the line and invent a person. It takes a sustained hand
        to move the middle of a one-second window.
        """
        if len(self._win) < 5:
            return self.state != DORMANT
        rates = sorted(w for _, w in self._win)
        median = rates[len(rates) // 2]
        if self.state == DORMANT:
            return median > config.HELD_GYRO_DPS
        return median > config.TABLE_GYRO_DPS

    # ── states ────────────────────────────────────────────────────────────

    def _dormant(self, t: int, handled: bool) -> Optional[Event]:
        if not handled:
            self._handled_since = None
            return None
        if self._handled_since is None:
            self._handled_since = t
            return None
        if t - self._handled_since < config.PICKUP_HOLD_MS:
            return None

        # Date the pickup from when the handling started, not from when we were
        # convinced by it — the person reached for it PICKUP_HOLD_MS ago.
        t_pick = self._handled_since
        self._handled_since = None
        self._settled_since = None
        self._quiet_since = None
        self._held_reported = None
        self._pause_sent = False
        self._visit_start = t_pick
        self.state = HELD
        self.segmenter.reset()
        return PickedUp(t_ms=t_pick,
                        away_ms=t_pick - self._last_putdown if self._last_putdown else 0)

    def _present(self, f: MotionFrame, t: int, handled: bool) -> Optional[Event]:
        if not handled:
            if self._settled_since is None:
                self._settled_since = t
            elif t - self._settled_since >= config.PUTDOWN_HOLD_MS:
                return self._put_down(t)
        else:
            self._settled_since = None

        # The gesture machine runs only while someone is holding it, so table
        # noise can never become a movement.
        seg = self.segmenter.push(f)
        if seg is not None:
            self.state = MOVING
            self._quiet_since = None
            self._pause_sent = False
            self._held_reported = None
            return seg

        active = self.segmenter.state not in ("IDLE", "REFRACTORY")
        if active:
            self.state = MOVING
            self._quiet_since = None
            self._pause_sent = False
            return None

        # Quiet, and still in a hand.
        self.state = HELD
        if self._quiet_since is None:
            self._quiet_since = t
            return None

        quiet = t - self._quiet_since
        if not self._pause_sent and quiet >= config.PAUSE_MS:
            self._pause_sent = True
            return Pause(t_ms=t, duration_ms=quiet)

        if quiet >= config.HOLDING_MS:
            first = self._held_reported is None
            if first or t - self._held_reported >= config.HOLDING_REPEAT_MS:
                self._held_reported = t
                return Holding(t_ms=t, duration_ms=quiet, first=first)
        return None

    def _put_down(self, t: int) -> PutDown:
        # Date it from when it went still, not from when we accepted it.
        t_down = self._settled_since or t
        visit = t_down - self._visit_start
        self.state = DORMANT
        self._last_putdown = t_down
        self._settled_since = None
        self._quiet_since = None
        self._held_reported = None
        self._pause_sent = False
        self.segmenter.reset()
        return PutDown(t_ms=t_down, visit_ms=visit)

    def _reset(self, t: int) -> None:
        self._win.clear()
        self.state = DORMANT
        self._handled_since = None
        self._settled_since = None
        self._quiet_since = None
        self._held_reported = None
        self._pause_sent = False
        self.segmenter.reset()

    # ── for the view ──────────────────────────────────────────────────────

    @property
    def present(self) -> bool:
        return self.state != DORMANT
