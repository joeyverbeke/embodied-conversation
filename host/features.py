"""Segment -> one English line (PLAN 5.2).

The LLM sees only the string this produces, never the numbers behind it. That
boundary is the point: it lets what the device *notices* be retuned
independently of how it *talks*.

What changed for the ball. The old model measured rotation and inferred a limb
pose from gravity — correct for a board strapped to a forearm, meaningless for
a ball, which has no canonical orientation and no shoulder to hang from. It also
meant 81% of everything came out as "horizontal to horizontal", because pose was
sampled at two instants and a movement that went overhead and came back read as
flat.

The new model measures where the ball went, in centimetres, and describes it
with verbs a person would recognise. Three groups, ordered by how much they
trust integration:

  dynamics   — straight off the accelerometer. Always trustworthy.
  rotation   — straight off the gyro. Always trustworthy, but demoted: a ball
               has no front, so how far it spun is rarely a fact about the
               person.
  trajectory — double-integrated, so guarded by a confidence value. When that
               confidence is low the distances are simply not claimed, and the
               description falls back on the other two groups.

That last rule matters more than it looks. The persona is built on only ever
asserting what it was handed, and a made-up distance would break the one thing
holding the whole piece together.
"""

from __future__ import annotations

import math

import numpy as np

from . import config, trajectory
from .segmenter import Segment

DEG = 180.0 / math.pi


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 below lo, 1 above hi, linear between. Membership without a cliff."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _repetition(signal: np.ndarray, dt: float,
                min_swing: float) -> tuple[int, float | None]:
    """Reversal count and repetition frequency (Hz) of a one-dimensional rate.

    Both are measured on the mean-centred signal. Counting sign changes of the
    raw signal instead missed any oscillation riding on a constant offset, which
    is most of them — fidgeting with the ball scored zero reversals while
    visibly reversing throughout.
    """
    x = signal - signal.mean()

    reversals = 0
    for i in range(1, len(x)):
        if x[i - 1] * x[i] < 0 and max(abs(x[i - 1]), abs(x[i])) > min_swing:
            reversals += 1

    freq = None
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
                if (period > 0
                        and 1.0 / period <= config.REPEAT_MAX_HZ
                        and len(x) * dt / period >= config.REPEAT_MIN_PERIODS):
                    freq = 1.0 / period
    return reversals, freq


# ── extraction ────────────────────────────────────────────────────────────

def extract(seg: Segment) -> dict:
    frames = seg.motion()
    if len(frames) < 3:
        return {}

    dt = 1.0 / config.IMU_HZ
    duration = seg.duration_ms / 1000.0

    gyro = np.array([[f.gx, f.gy, f.gz] for f in frames]) * DEG
    rates = np.linalg.norm(gyro, axis=1)
    lin = np.array([[f.lax, f.lay, f.laz] for f in frames])
    lin_mag = np.linalg.norm(lin, axis=1)
    raw_mag = np.array([math.sqrt(f.ax ** 2 + f.ay ** 2 + f.az ** 2)
                        for f in frames])

    # ── dynamics: no integration, so these are always safe to assert ──────
    peak_accel = float(lin_mag.max())
    jerk = np.abs(np.diff(lin_mag)) / dt
    onset_window = max(2, int(0.1 / dt))
    onset_jerk = float(jerk[:onset_window].max()) if len(jerk) else 0.0
    time_to_peak = float(lin_mag.argmax()) / max(1, len(lin_mag) - 1)
    impact = float(raw_mag.max())

    # Oscillation, measured on acceleration rather than rotation. Shaking a ball
    # is a translation event; the old version looked for it in the gyro and
    # would miss a straight up-and-down shake entirely.
    centred = lin - lin.mean(axis=0)
    if np.any(centred):
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        shake_axis = vt[0]
    else:
        shake_axis = np.array([1.0, 0.0, 0.0])
    # In m/s^2, with a threshold of its own. This was briefly scaled by an
    # arbitrary factor so the gyro's degrees-per-second threshold could be
    # reused, which set the bar near 2 m/s^2 and made small repeated movements
    # score zero reversals while visibly reversing.
    projected = lin @ shake_axis
    reversals, repeat_hz = _repetition(projected, dt, config.REVERSAL_MIN_MS2)

    # ── rotation ──────────────────────────────────────────────────────────
    spin_path = float(np.sum(rates) * dt)
    spin_rate = float(np.median(rates))
    peak_dps = float(rates.max())
    axes = gyro / np.maximum(np.linalg.norm(gyro, axis=1, keepdims=True), 1e-9)
    mean_axis = axes.mean(axis=0)
    # 1.0 = every instant turned about the same axis (a clean spin);
    # near 0 = the axis wandered (tumbling, or being turned over in the hands)
    axis_stability = float(np.linalg.norm(mean_axis))
    attitude_deg = trajectory.orientation_change_deg(frames)

    f = {
        "duration_s": round(duration, 2),
        "n_frames": len(frames),
        "ended": seg.ended,
        "hesitations": seg.hesitations,
        "airborne_ms": seg.airborne_ms,

        "peak_accel": round(peak_accel, 2),
        "onset_jerk": round(onset_jerk, 1),
        "time_to_peak": round(time_to_peak, 3),
        "impact": round(impact, 1),
        "reversals": reversals,
        "repeat_hz": round(repeat_hz, 2) if repeat_hz else None,

        "spin_deg": round(spin_path, 1),
        "spin_rate": round(spin_rate, 1),
        "peak_dps": round(peak_dps, 1),
        "axis_stability": round(axis_stability, 3),
        "attitude_deg": round(attitude_deg, 1),
    }

    # ── trajectory: guarded by confidence ─────────────────────────────────
    path = trajectory.reconstruct(frames)
    if path is not None:
        measured = trajectory.measure(path)
        f.update(measured)
        f["direction"] = trajectory.direction(measured)
        legs = trajectory.strokes(path)
        f["strokes"] = [
            {"cm": s.distance_cm, "dir": s.direction, "up_cm": s.vertical_cm,
             "s": s.duration_s}
            for s in legs
        ]
        f["stroke_count"] = len(legs)
        f["shape"] = trajectory.shape(legs)
        # How much of the journey was one leg. A directed verb — lifted,
        # lowered, swept — is only honest when one leg dominates; otherwise the
        # net displacement is an artefact of where the person happened to stop,
        # and naming the gesture after it describes an accident.
        travelled = sum(s.distance_cm for s in legs)
        f["dominant_share"] = round(
            max((s.distance_cm for s in legs), default=0.0) / travelled, 3
        ) if travelled > 0 else 0.0
        # A throw breaks both assumptions the reconstruction rests on. In free
        # fall the linear-acceleration reading is not the ball's acceleration
        # relative to the room, and the gesture does not end at rest in the
        # sense the zero-velocity update needs — it ends in a hand, mid-flight.
        # The flight itself is measured directly and unambiguously; the path
        # through it is not, and must not be quoted.
        f["trusted"] = path.trusted and seg.airborne_ms == 0
    else:
        f["confidence"] = 0.0
        f["trusted"] = False
        f["direction"] = "nowhere"
        f["strokes"] = []
        f["stroke_count"] = 0
        f["shape"] = "single"

    return f


# ── vocabulary ────────────────────────────────────────────────────────────
#
# Physical verbs, scored rather than chosen. Keeping the scores instead of an
# argmax is what allows "barely a throw" and "a shake that gave up halfway" —
# a hard winner throws away exactly the information that makes a description
# specific rather than merely correct.
#
# Verbs that depend on the reconstructed path are marked, and are suppressed
# when confidence is low. What is left still says something true.

_NEEDS_PATH = {"lifted", "lowered", "swung", "swept", "carried"}

# Every verb is a past participle, so "barely shaken" and "the third shaken in
# a row" both work without special-casing. The corpus needs a noun for its own
# templates — "nobody has done a throw before" survives phrasing that
# "nobody has thrown it before" does not, once intransitive verbs are in the
# set. Keeping the map here means both agree by construction.
NOUNS = {
    "thrown": "throw", "caught": "catch", "dropped": "drop", "tapped": "tap",
    "shaken": "shake", "spun": "spin", "rolled": "roll", "flipped": "flip",
    "lifted": "lift", "lowered": "lowering", "swung": "swing",
    "swept": "sweep", "carried": "carry", "jiggled": "jiggle",
    "held": "hold", "moved": "movement",
}


# Coarse groupings for comparison. Sixteen verbs against sixty-four gestures made
# almost everything "the first of its kind" by construction — the machine sounded
# like it was noticing rarity when it was reporting a small sample against a fine
# vocabulary. Novelty is judged on these five; description still uses the verbs.
FAMILIES = {
    "lifted": "raising", "lowered": "raising",
    "swept": "travelling", "swung": "travelling", "carried": "travelling",
    "shaken": "agitating", "jiggled": "agitating",
    "spun": "turning", "rolled": "turning", "flipped": "turning",
    "thrown": "releasing", "caught": "releasing", "dropped": "releasing",
    "tapped": "releasing",
    "held": "nothing much", "moved": "nothing much",
}

# What to call a family out loud.
FAMILY_NOUNS = {
    "raising": "lift or a drop", "travelling": "movement across",
    "agitating": "shake", "turning": "turn", "releasing": "throw",
    "nothing much": "movement",
}


def family(kind: str) -> str:
    return FAMILIES.get(kind, "nothing much")


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def verbs(f: dict) -> list[tuple[str, float]]:
    """Every verb that fits, strongest first."""
    if not f:
        return []

    dur = f["duration_s"]
    trusted = f.get("trusted", False)
    span = f.get("span_cm", 0.0)
    net = f.get("net_cm", 0.0)
    rise = f.get("rise_cm", 0.0)
    drop = f.get("drop_cm", 0.0)
    direct = f.get("travel_directness", 0.0)
    speed = f.get("peak_speed_ms", 0.0)
    direction = f.get("direction", "nowhere")

    scores: dict[str, float] = {}

    # Airborne is not a judgement call — either the accelerometer went
    # weightless or it did not.
    if f["airborne_ms"] > 0:
        flight = _ramp(f["airborne_ms"], config.FREEFALL_MIN_MS, 500)
        scores["thrown"] = 0.6 + 0.4 * flight
        if f["impact"] >= config.IMPACT_MS2:
            scores["caught"] = 0.5 + 0.5 * _ramp(f["impact"],
                                                 config.IMPACT_MS2, 120.0)
        # No "dropped". Nothing here can tell a catch from a landing: the
        # segment ends at the instant of impact, so there are no frames after it
        # to say whether the ball carried on being handled or came to rest on
        # the floor. Impact strength does not separate them either — a soft
        # catch and a soft landing look identical. Saying "you did not catch it
        # cleanly" is asserting something unknown about somebody, which is the
        # one thing this whole design refuses to do. When it cannot tell, it
        # says "thrown" and stops there.

    elif f["impact"] >= config.IMPACT_MS2 and dur < 0.6:
        scores["tapped"] = _ramp(f["impact"], config.IMPACT_MS2, 120.0)

    if f["repeat_hz"] and f["reversals"] >= 3:
        scores["shaken"] = 0.5 + 0.5 * _ramp(f["reversals"], 3, 10)

    # A tossed ball tumbles. That rotation is gravity and release angle, not a
    # decision, and calling it "spun" credits the person with something they did
    # not do — eight hand-to-hand tosses all came back as "spun and thrown".
    if f["spin_rate"] >= config.SPIN_DPS and not f["airborne_ms"]:
        # Scored on rate alone. Weighting this by axis stability suppressed
        # every real spin: turning a ball over in two hands measures around
        # 0.43, not the ~1.0 an idealised single-axis spin would give.
        scores["spun"] = 0.4 + 0.6 * _ramp(f["spin_rate"], config.SPIN_DPS, 300.0)
        if f["axis_stability"] < config.TUMBLE_STABILITY:
            scores["rolled"] = 0.4 + 0.6 * (
                1.0 - f["axis_stability"] / config.TUMBLE_STABILITY)

    if f["attitude_deg"] >= 90.0 and f["spin_rate"] < config.SPIN_DPS:
        scores["flipped"] = _ramp(f["attitude_deg"], 90.0, 180.0)

    # Fidgeting and carrying describe the *character* of a movement rather than
    # its size, so neither needs the reconstruction. Both sat behind the trusted
    # gate at first and were therefore unreachable in practice: a slow carry is
    # precisely the movement whose distance cannot be measured, so requiring a
    # measured distance before it could be named meant it never was.
    # Measured separation at REVERSAL_MIN_MS2: a single deliberate move scores
    # 0-1 reversals, a smooth carry 1, fidgeting 3, a spin 15, a shake 18. The
    # spin exclusion matters — a spin reverses constantly and would otherwise
    # outscore itself as fidgeting.
    if (f["reversals"] >= 3 and not f["repeat_hz"]
            and f["peak_accel"] < config.FORCEFUL_MS2
            and f["spin_rate"] < config.SPIN_DPS
            and (not trusted or f.get("span_cm", 0.0) < config.SMALL_CM)):
        scores["jiggled"] = 0.4 + 0.6 * _ramp(f["reversals"], 3, 12)

    if (dur >= 2.5 and f["reversals"] < 3
            and config.GENTLE_MS2 * 0.4 < f["peak_accel"] <= config.GENTLE_MS2 * 1.6
            and f["spin_rate"] < config.SPIN_DPS):
        scores["carried"] = 0.5 + 0.5 * _ramp(dur, 2.5, 8.0)

    if trusted:
        size = _ramp(span, config.SMALL_CM, config.LARGE_CM)
        # One leg carried the movement, so it is fair to name the movement after
        # a direction. Below this it went several ways and "lifted" would be a
        # claim about the endpoint rather than the gesture.
        directed = f.get("stroke_count", 1) <= 1 or f.get("dominant_share", 1.0) >= 0.6

        if directed and direction in ("up", "up and out") and net >= config.SMALL_CM:
            scores["lifted"] = 0.4 + 0.6 * size
        if directed and direction in ("down", "down and out") and net >= config.SMALL_CM:
            scores["lowered"] = 0.4 + 0.6 * size

        if span >= config.SMALL_CM:
            # Every measured movement of real size gets named. The first version
            # only named the extremes — returned-to-start, or dead straight and
            # sideways — so anything in between fell through to "moved". A live
            # 39 cm sweep came out as "barely moved, 39 cm sideways", which is
            # the machine describing something it clearly saw as nothing.
            if direct < config.RETURN_RATIO or not directed:
                # went out and came back, or went several ways
                scores["swung"] = 0.4 + 0.6 * size * _ramp(speed, 0.5, 2.5)
            elif direction not in ("up", "down"):
                # went somewhere and stayed; purely vertical is a lift or a drop
                scores["swept"] = 0.4 + 0.6 * size

        if span < config.SMALL_CM and speed < config.STILL_SPEED_MS:
            scores["held"] = 0.6

    if not scores:
        # Nothing named it, but the gates fired, so something happened. Say the
        # least specific true thing rather than inventing a specific one.
        scores["moved"] = 0.3

    return sorted(scores.items(), key=lambda kv: -kv[1])


def classify(f: dict) -> str:
    """The primary verb. Shared with the corpus so both agree."""
    ranked = verbs(f)
    return ranked[0][0] if ranked else "moved"


# ── comparison to the last few ────────────────────────────────────────────

def compare(f: dict, recent: list[dict]) -> list[str]:
    """How this movement sits against the last few, in this stretch.

    The corpus knows what people have done all night. This knows what has
    happened in the last minute. The second is what makes the machine seem to
    be following someone, and it is the dimension the old system had none of —
    escalation, repetition and retreat were all invisible.
    """
    if not f or not recent:
        return []

    prev = recent[-1]
    out: list[str] = []

    def shift(key: str, bigger: str, smaller: str) -> str | None:
        a, b = f.get(key), prev.get(key)
        if a is None or b is None or b <= 0:
            return None
        ratio = (a - b) / b
        if ratio > config.SIMILAR_RATIO:
            return bigger
        if ratio < -config.SIMILAR_RATIO:
            return smaller
        return None

    if f.get("trusted") and prev.get("trusted"):
        for word in (shift("span_cm", "bigger", "smaller"),
                     shift("rise_cm", "higher", "lower")):
            if word:
                out.append(word)
    for word in (shift("peak_accel", "harder", "gentler"),
                 shift("duration_s", "slower", "quicker")):
        if word:
            out.append(word)

    kind = classify(f)
    fam = family(kind)
    same = 1
    for e in reversed(recent):
        if family(e.get("_kind", "")) == fam:
            same += 1
        else:
            break
    if same >= 3:
        out.insert(0, f"the {_ordinal(same)} of these in a row")
    elif not out:
        out.append("much the same as the last")

    return out[:2]


# ── the one line the LLM sees ─────────────────────────────────────────────

def _travel(f: dict) -> list[str]:
    """How far it went, leg by leg.

    One stroke reads as it always did. Two or more become a sequence, because
    that is what the person actually did — and because "up, then down about half
    as far" is a thing they can recognise, while an averaged endpoint is not.
    """
    legs = f.get("strokes") or []
    big = [s for s in legs if s["cm"] >= config.SMALL_CM]

    if not big:
        span = f.get("span_cm", 0.0)
        return [f"{span:.0f} cm"] if span >= config.SMALL_CM else []

    if len(big) == 1:
        s = big[0]
        return [f"{s['cm']:.0f} cm {s['dir']}" if s["dir"] != "nowhere"
                else f"{s['cm']:.0f} cm"]

    if len(big) == 2:
        a, b = big
        return [f"{a['cm']:.0f} cm {a['dir']}, then {b['cm']:.0f} cm {b['dir']}"]

    shape = f.get("shape", "back and forth")
    total = sum(s["cm"] for s in big)
    # Say metres once the number gets long. Asked to render "188 cm in all" the
    # model returned "ninety-five centimetres" — it does arithmetic on numbers
    # it is handed, and the fix is to hand it one it does not have to convert.
    said = f"{total / 100:.1f} metres" if total >= 100 else f"{total:.0f} cm"
    if shape == "shrinking":
        tail = "each one smaller"
    elif shape == "growing":
        tail = "each one bigger"
    else:
        tail = "back and forth"
    return [f"{len(big)} legs, {tail}", f"{said} in all"]


def describe(f: dict, facts: list[str] | None = None,
             relation: list[str] | None = None) -> str:
    """Round figures only, no telemetry.

    `facts` is what the corpus makes of this gesture across the whole night;
    `relation` is how it compares to the last few. Putting both here rather than
    leaving the model to infer comparison from context is the difference between
    a claim that happens to be true and one that is checked.
    """
    if not f:
        return "a movement too short to read"

    ranked = verbs(f)
    kind, strength = ranked[0]

    # A weak primary verb is said weakly. This is the graded membership earning
    # its keep: "barely a throw" is more accurate than "a throw", and more
    # interesting than dropping to a vaguer word.
    if strength < 0.45:
        head = f"barely {kind}"
    elif len(ranked) > 1 and ranked[1][1] > 0.5:
        # a throw and its catch are sequential; everything else is simultaneous
        joiner = ", then " if kind == "thrown" else " and "
        head = f"{kind}{joiner}{ranked[1][0]}"
    else:
        head = kind

    parts = [head]
    said_a_distance = False

    if f["airborne_ms"] > 0:
        parts.append(f"airborne {f['airborne_ms'] / 1000:.1f}s")
        # h = g*t^2/8 for a throw that returns to the height it left. Comes
        # straight from the flight time, so it holds even though the trajectory
        # is untrustworthy through free fall — the one real distance available
        # about a throw, where everything else has to say "unclear".
        apex = config.G * (f["airborne_ms"] / 1000.0) ** 2 / 8 * 100
        if apex >= 5.0:
            parts.append(f"about {apex:.0f} cm up")
            said_a_distance = True

    # Distances only when the reconstruction earned the right to be quoted, and
    # attached to the leg they belong to.
    #
    # Quoting one distance for the whole gesture is what produced "79 cm down"
    # for a movement that went up 79 and came back down 31: the number was the
    # rise and the word was the net. A continuous movement has no single
    # distance and no single direction, and pretending otherwise is not a
    # rounding error, it is a false statement about what someone did.
    if f.get("trusted"):
        parts.extend(_travel(f))
    elif kind not in _NEEDS_PATH and not said_a_distance:
        parts.append("distance unclear")

    # Autocorrelation finds a period in anything, including sensor noise on a
    # device sitting on a table — which reported a confident "~9.1 times a
    # second" about a board nobody was touching. The reversal count is what
    # distinguishes a rhythm from a fit to noise, so both must agree, exactly as
    # they must for the "shaken" verb.
    if f["repeat_hz"] and f["reversals"] >= 3:
        parts.append(f"~{f['repeat_hz']:.1f} times a second")

    # Effort, from peak acceleration. Exactly zero means there was no
    # acceleration data to look at rather than a gentle movement — replayed
    # legacy logs land there, and calling that "gentle" would be an assertion
    # made from an absence. A movement that never went anywhere has no force
    # worth characterising either.
    if kind != "held" and f["peak_accel"] > 0.0:
        if f["peak_accel"] >= config.FORCEFUL_MS2:
            parts.append("hard")
        elif f["peak_accel"] <= config.GENTLE_MS2:
            parts.append("barely any force in it")

    if f["duration_s"] >= 3.0:
        parts.append(f"{f['duration_s']:.0f}s, unhurried")

    if f["hesitations"] >= 2:
        parts.append(f"{f['hesitations']} false starts first")

    line = ", ".join(parts)
    if relation:
        line += "\n  against the last one: " + ", ".join(relation)
    if facts:
        line += "\n  against everyone tonight: " + "; ".join(facts)
    return line


# ── situations ────────────────────────────────────────────────────────────
#
# Not every thing worth saying is a movement. A person picking the ball up, or
# holding it and not yet deciding, or putting it down, are all situations the
# machine can now see — and the pickup in particular is the only moment it
# addresses somebody who has not done anything yet.


def describe_pickup(facts: list[str] | None = None) -> str:
    """Someone just lifted it off the table.

    Deliberately carries no movement at all. There has not been one. Anything
    the model is handed here it will try to use, so handing it kinematics would
    invite a comment on a gesture that never happened.
    """
    line = "someone has just picked it up, and has not done anything yet"
    if facts:
        line += "\n  what is true so far: " + "; ".join(facts)
    return line


def describe_holding(hold, moved_yet: bool,
                     facts: list[str] | None = None) -> str:
    """Held, not moving. A person deciding, not an empty room."""
    secs = hold.duration_ms / 1000.0
    if not moved_yet:
        line = f"holding it, {secs:.0f}s now, still has not moved it"
    elif hold.first:
        line = f"has stopped, holding it still for {secs:.0f}s"
    else:
        line = f"still not moving, {secs:.0f}s now"
    if facts:
        line += "\n  against everyone tonight: " + "; ".join(facts)
    return line


def describe_putdown(visit_ms: int, kinds: list[str],
                     facts: list[str] | None = None) -> str:
    """They have put it back. A last word, if they did anything at all."""
    secs = visit_ms / 1000.0
    fams = sorted({family(k) for k in kinds})
    line = (f"put it back down after {secs:.0f}s and {len(kinds)} movements, "
            f"{len(fams)} different kinds")
    if facts:
        line += "\n  against everyone tonight: " + "; ".join(facts)
    return line
