"""How much is this worth saying?

Everything the machine could say becomes a scored candidate, and the score is
what makes selectivity possible. Sixty-four logged gestures showed what happens
without it: the machine responded to every one with identical enthusiasm, its
comparative vocabulary collapsed into "unremarkable so far" thirteen times, and
there was no gradient for anyone to climb.

The repetition penalty is the load-bearing part. Doing the same thing again
scores lower each time, so the machine goes quiet on someone who has settled
into a rut and speaks up the moment they try something else. Nobody is told
this. They feel it, and they go looking for the thing that gets a reaction —
which is the entire mechanism the piece is built on.

Nothing here decides anything. It only scores. When and whether to speak is
host/policy.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from . import features as _features

# Wording the corpus uses for the facts worth the most. Kept as a check against
# corpus.observe() rather than recomputed, so the score and the sentence can
# never disagree about what was notable.
# Maxima only. "The smallest so far" is technically a record and scored +0.40,
# so doing almost nothing rated 0.80 and the machine enthusiastically rewarded
# people for barely moving — teaching exactly the opposite of what it is for.
_RECORD_WORDS = ("largest so far", "hardest so far", "longest so far",
                 "higher than anyone")
_NOVEL_WORDS = ("nobody has done",)
_ABSENCE_WORDS = ("all night",)


@dataclass(slots=True)
class Score:
    value: float
    reasons: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"{self.value:.2f} ({', '.join(self.reasons) or 'nothing'})"


def _add(value: float, reasons: list[str], amount: float, why: str) -> float:
    reasons.append(f"{why} {amount:+.2f}")
    return value + amount


def for_gesture(feat: dict, facts: list[str], relation: list[str],
                repeats: int, new_to_visit: bool = False,
                kind: str = "") -> Score:
    """Score one movement.

    `facts` and `relation` are the corpus and within-visit comparisons that will
    actually be spoken, so the score is derived from the same evidence the
    sentence is — not from a parallel judgement that could drift out of step.
    """
    reasons: list[str] = []
    value = config.SAL_BASE
    reasons.append(f"base {config.SAL_BASE:+.2f}")

    joined = " ".join(facts).lower()
    if any(w in joined for w in _RECORD_WORDS):
        value = _add(value, reasons, config.SAL_RECORD, "record")
    if any(w in joined for w in _NOVEL_WORDS):
        value = _add(value, reasons, config.SAL_NEW_FAMILY, "first of its kind")
    if any(w in joined for w in _ABSENCE_WORDS):
        value = _add(value, reasons, config.SAL_ABSENCE, "names an absence")

    # The first time *this person* does something, even if the night has seen it
    # a hundred times. Without this an ordinary opening movement scores 0.10 and
    # a visitor's very first act is met with silence, which reads as broken
    # rather than as selective.
    if new_to_visit:
        value = _add(value, reasons, config.SAL_FIRST_THIS_VISIT,
                     "first of its kind this visit")

    # Size, on its own terms. Needs no corpus and no precedent: a movement that
    # crosses most of a person's reach is a thing they will know they just did,
    # and a machine that ignores it is not watching.
    span = feat.get("span_cm", 0.0) if feat.get("trusted") else 0.0
    travel = sum(s["cm"] for s in (feat.get("strokes") or []))
    if span or travel:
        big = max(span / max(config.LARGE_CM, 1e-6),
                  travel / max(config.SAL_BIG_TRAVEL_CM, 1e-6))
        if big > 0.35:
            value = _add(value, reasons,
                         config.SAL_MAGNITUDE * min(1.0, big), "big")

    # Drama needs no corpus: a ball leaving someone's hand is the most
    # unambiguous thing this device can detect, and it is always worth a word.
    if feat.get("airborne_ms"):
        value = _add(value, reasons, config.SAL_DRAMA, "airborne")
    elif feat.get("peak_accel", 0.0) >= config.FORCEFUL_MS2:
        value = _add(value, reasons, config.SAL_DRAMA * 0.5, "forceful")

    if repeats and repeats % config.SAL_RUN_NOTICE == 0:
        # Not the repetition being forgiven — the run itself being the news.
        value = _add(value, reasons, config.SAL_RUN_BONUS,
                     f"a run of {repeats}")
    elif repeats >= 2:
        penalty = -config.SAL_REPEAT_PENALTY * (repeats - 1)
        value = _add(value, reasons, penalty, f"{repeats} in a row")

    # A movement the machine could not even name is not notable, whatever else
    # happens to be true tonight. Without this ceiling an unrelated standing
    # fact could carry a nothing-gesture over the bar, and "you barely moved it"
    # was said five times in one session.
    if kind and _features.family(kind) == "nothing much":
        value = min(value, config.SAL_NOTHING_MUCH_MAX)
        reasons.append(f"capped at {config.SAL_NOTHING_MUCH_MAX:.2f}, unnamed")

    return Score(max(0.0, min(1.0, value)), reasons)


def for_pickup(away_ms: int, visits: int) -> Score:
    """Always worth saying. The score is a formality.

    This is the one moment the machine addresses someone who has not done
    anything yet, so it is also the only moment a suggestion can land before an
    intention has formed. The policy bypasses the bar for it; the score exists
    so the trace reads consistently.
    """
    return Score(1.0, [f"pickup, visit {visits + 1}, away {away_ms // 1000}s"])


def for_holding(duration_ms: int, first: bool, moved_yet: bool) -> Score:
    """Someone holding it and not moving.

    Worth most from a person who has not moved at all yet — that is hesitation
    before a first act, and the moment a nudge is cheapest. Someone pausing
    mid-play has already shown what they do, so it is worth much less.
    """
    reasons = [f"holding {duration_ms // 1000}s"]
    if not moved_yet and first:
        return Score(0.55, reasons + ["has not moved yet"])
    if first:
        return Score(0.30, reasons)
    return Score(0.12, reasons + ["still holding"])


def for_putdown(visit_ms: int, gestures: int) -> Score:
    """A last word. Only worth it if they actually did something."""
    if gestures == 0:
        return Score(0.0, ["put down without doing anything"])
    return Score(0.45, [f"putdown after {gestures} movements"])
