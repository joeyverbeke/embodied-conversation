"""When to speak, and — mostly — when not to.

This replaces should_respond(), which was three booleans: is the device idle, is
nothing in flight, has the refractory elapsed. That predicate answered "may I
speak" and the machine therefore spoke every time it could. Sixty-four gestures
produced sixty-four attempts at a sentence.

The question here is different: *should* I. Silence is the common answer.

## The bar

One number does almost all the work. It sits at BAR_BASE, jumps by BAR_STEP
whenever the machine speaks, and decays back while it is quiet.

    speak when salience >= bar

That single mechanism produces three things at once:

  a variable rate       — talkative after silence, reticent after speaking
  differential attention — repetition scores low and stops clearing a raised
                           bar, novelty still clears it
  a gradient            — the machine visibly cares more about some movements
                          than others, and people go looking for the ones that
                          get a reaction

Nobody is told any of this, which is the point. A person discovers that certain
movements earn attention and drifts toward them, and can say afterwards only
that they were playing.

## Timing

Speech prefers a lull. A candidate that clears the bar mid-movement waits up to
PENDING_MS for a pause rather than talking over someone — unless it clears
BAR_INTERRUPT, in which case cutting in is the right call and waiting would land
the line after the moment it describes has passed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from . import config
from .lifecycle import DORMANT


@dataclass(slots=True)
class Decision:
    speak: bool
    reason: str
    score: float = 0.0
    bar: float = 0.0

    def __repr__(self) -> str:
        verb = "SPEAK" if self.speak else "hold "
        return f"{verb} {self.score:.2f} vs bar {self.bar:.2f} — {self.reason}"


@dataclass(slots=True)
class Pending:
    """A line that cleared the bar and is waiting for a polite moment."""
    payload: object
    score: float
    since: float

    def expired(self, now: float) -> bool:
        return (now - self.since) * 1000 >= config.PENDING_MS


@dataclass(slots=True)
class Policy:
    bar: float = config.BAR_BASE
    _last_move: float = 0.0         # monotonic, when the bar last changed
    spoke_at: float = 0.0
    said: int = 0
    withheld: int = 0
    pending: Pending | None = None

    # ── the bar ───────────────────────────────────────────────────────────

    def _decayed(self, now: float) -> float:
        """Exponential return to BAR_BASE.

        Exponential rather than linear so the machine loosens up quickly after
        one remark and only stays quiet after a run of them — which is the shape
        of someone becoming less impressed, not of a timer running out.
        """
        if self._last_move == 0.0:
            return self.bar
        elapsed = now - self._last_move
        k = math.exp(-elapsed / max(config.BAR_DECAY_S, 1e-6))
        return config.BAR_BASE + (self.bar - config.BAR_BASE) * k

    def height(self, now: float | None = None) -> float:
        return self._decayed(now if now is not None else time.monotonic())

    def _raise(self, now: float) -> None:
        self.bar = min(config.BAR_MAX, self._decayed(now) + config.BAR_STEP)
        self._last_move = now

    # ── the decision ──────────────────────────────────────────────────────

    def consider(self, score: float, *, state: str, moving: bool,
                 always: bool = False, busy: bool = False,
                 ending: bool = False, now: float | None = None) -> Decision:
        """Should this be said, and said now?"""
        now = time.monotonic() if now is None else now
        bar = self._decayed(now)

        # Nobody is here. No score gets through, ever. A piece that narrates to
        # an empty room all night is unshowable, and this is the one rule that
        # guarantees it cannot.
        #
        # `ending` is the one exemption: a putdown is *reported* from DORMANT
        # because setting the ball down is what caused the transition, but the
        # person who set it down is still standing there. Without this the last
        # word is unreachable by construction — it scored 0.45 against a bar of
        # 0.39 and was still silenced.
        if state == DORMANT and not ending:
            return Decision(False, "nobody here", score, bar)

        if busy:
            self.withheld += 1
            return Decision(False, "already speaking", score, bar)

        # Pickup bypasses the bar. It is the only moment the machine addresses
        # someone before they have done anything, and it is worth more than any
        # pacing rule.
        if always:
            self._raise(now)
            self.spoke_at = now
            self.said += 1
            return Decision(True, "greeting", score, bar)

        if score < bar:
            self.withheld += 1
            return Decision(False, "under the bar", score, bar)

        if moving and score < config.BAR_INTERRUPT:
            return Decision(False, "waiting for a lull", score, bar)

        self._raise(now)
        self.spoke_at = now
        self.said += 1
        reason = "cut in" if moving else "cleared the bar"
        return Decision(True, reason, score, bar)

    # ── waiting for a lull ────────────────────────────────────────────────

    def hold(self, payload, score: float, now: float | None = None) -> None:
        """Keep a line that cleared the bar until a pause arrives."""
        self.pending = Pending(payload, score, now or time.monotonic())

    def release(self, now: float | None = None):
        """The pause arrived. Hand back the waiting line, if it is still fresh.

        A held line has a shelf life. Said four seconds late it describes a
        movement the person has already stopped making, and arriving after the
        moment is worse than never arriving.
        """
        now = time.monotonic() if now is None else now
        pending, self.pending = self.pending, None
        if pending is None:
            return None
        if pending.expired(now):
            self.withheld += 1
            return None
        self._raise(now)
        self.spoke_at = now
        self.said += 1
        return pending.payload

    def drop_stale(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.pending is not None and self.pending.expired(now):
            self.pending = None
            self.withheld += 1

    def reset(self) -> None:
        """A new person. The bar is about the machine's mood, not theirs —
        it stays where it is, but nothing is left waiting from the last visit."""
        self.pending = None
