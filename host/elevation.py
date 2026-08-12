"""Where the ball is, as opposed to how far it moved.

Everything the device measures today is displacement within one gesture. There
is no persistent frame, so the machine can say "it went up 30 cm" and can never
say "it was down near the floor" — and the difference matters more than it
sounds. An absence phrased as displacement is not somewhere a person can decide
to go, which is why "nobody has done anything low all night" fired thirteen
times and moved nobody. Phrased as a place, it is a destination.

A barometer is on the way. It is the only option that suits a sealed sphere:
pressure does not care which way the ball is facing, where a rangefinder would
need to know which way is down on a tumbling object and would be blocked by the
hand holding it anyway. Drift is handled by the table — the ball returns to a
known height between visits, so every putdown re-zeros the reference and error
never accumulates past one visit.

This module exists so that arrives as a drop-in. Everything downstream asks the
same question now and gets None, and keeps using rise_cm, which is relative,
comparative and honest.

One thing absolute height will *not* buy, and it is worth being clear about
before the part arrives: anything body-relative. "Above your head" is 1.2 m on a
child and 1.8 m on a tall adult, and no sensor inside the ball knows whose hand
it is in.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Reading:
    """Height above the floor, in centimetres, or None if unknowable."""
    cm: float | None
    trusted: bool = False


class Source:
    """What a height provider has to do."""

    def update(self, frame) -> None:          # pragma: no cover - interface
        raise NotImplementedError

    def read(self) -> Reading:                # pragma: no cover - interface
        raise NotImplementedError

    def datum(self, cm: float) -> None:
        """Re-zero against a known height — the table, on putdown."""


class RelativeOnly(Source):
    """The honest answer until there is a barometer in the ball.

    Deliberately not a guess. Accumulating the per-gesture displacement across a
    visit would produce a number, and that number would be wrong in a way that
    grows without bound and cannot be checked — the machine would assert a place
    with the same confidence it asserts a measured distance. Saying nothing is
    the behaviour the whole design rests on.
    """

    def update(self, frame) -> None:
        return None

    def read(self) -> Reading:
        return Reading(cm=None, trusted=False)


def band(reading: Reading) -> str | None:
    """A place, coarse enough to be worth trusting when a barometer lands."""
    if reading.cm is None or not reading.trusted:
        return None
    if reading.cm < 40:
        return "near the floor"
    if reading.cm < 110:
        return "low"
    if reading.cm < 170:
        return "about chest height"
    return "up high"
