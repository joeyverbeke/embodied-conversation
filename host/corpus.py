"""What the machine has watched, across everyone (PLAN 6, Phase 3 — pulled
forward because the piece went public).

The device is handed from stranger to stranger and cannot tell them apart, so
this deliberately does not model individuals. It knows what *people* have done
tonight, not what *you* have done. That is both the honest reading of the data
and the better voice: being lumped in with strangers is stranger than being
remembered.

Persists to disk so an evening accumulates across restarts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from . import config
from .features import FAMILY_NOUNS, NOUNS, family

# Below this many gestures, "nobody has done that before" is a statement about
# the sample rather than about people. Sixteen verbs over sixty-four gestures
# made almost everything novel; the claim needs a corpus behind it to mean
# anything, and saying so plainly is better than a hollow superlative.
NOVELTY_MIN = 12

# Regions of the movement space we can notice nobody has visited. Naming an
# absence is the one way to move people without instructing them: it is a
# statement about the corpus, and they fill it in themselves.
#
# Rewritten for the ball. The old regions were about limb pose, which no longer
# exists; these are about what was done with an object, which is what a person
# would recognise as a thing they had or had not tried.
def _num(entry: dict, key: str, missing: float) -> float:
    """Read a numeric field, treating an unmeasured one as `missing`.

    Not dict.get(key, default): these keys are *present* and set to None on
    entries whose trajectory could not be trusted, so the default never applies
    and the comparison blows up on None. `missing` is chosen per region to mean
    "this does not count as an example", so an unmeasured gesture never
    contributes evidence that a region has been visited.
    """
    value = entry.get(key)
    return missing if value is None else value


REGIONS = {
    "high": lambda e: _num(e, "rise_cm", 0.0) >= 50.0,
    "low": lambda e: _num(e, "drop_cm", 0.0) <= -30.0,
    "thrown": lambda e: _num(e, "airborne_ms", 0) > 0,
    "repeated": lambda e: e.get("repeat_hz") is not None,
    "hard": lambda e: _num(e, "peak_accel", 0.0) >= 15.0,
    "slow": lambda e: _num(e, "duration_s", 0.0) >= 4.0,
    "small": lambda e: _num(e, "span_cm", 999.0) <= 10.0,
}


@dataclass
class Corpus:
    path: object = None
    entries: list = field(default_factory=list)
    visits: list = field(default_factory=list)

    def __post_init__(self):
        if self.path is None:
            self.path = config.CORPUS_PATH
        self.load()

    @property
    def visits_path(self):
        return self.path.with_name(self.path.stem + "-visits.jsonl")

    # ── persistence ───────────────────────────────────────────────────────

    def load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                self.entries = [json.loads(line) for line in fh if line.strip()]
        except FileNotFoundError:
            self.entries = []
        try:
            with open(self.visits_path, encoding="utf-8") as fh:
                self.visits = [json.loads(line) for line in fh if line.strip()]
        except FileNotFoundError:
            self.visits = []

    def add(self, kind: str, f: dict) -> None:
        entry = {
            "t": time.time(),
            "kind": kind,
            "duration_s": f["duration_s"],
            "peak_accel": f["peak_accel"],
            "onset_jerk": f["onset_jerk"],
            "reversals": f["reversals"],
            "repeat_hz": f.get("repeat_hz"),
            "airborne_ms": f["airborne_ms"],
            "spin_deg": f["spin_deg"],
            "attitude_deg": f["attitude_deg"],
            # Only recorded when the reconstruction was trustworthy. An entry
            # without these is not a small movement, it is an unmeasured one,
            # and ranking against it as though it were zero would quietly
            # manufacture records.
            "trusted": f.get("trusted", False),
            "span_cm": f.get("span_cm") if f.get("trusted") else None,
            "rise_cm": f.get("rise_cm") if f.get("trusted") else None,
            "drop_cm": f.get("drop_cm") if f.get("trusted") else None,
            "direction": f.get("direction") if f.get("trusted") else None,
        }
        self.entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")

    # ── what it knows ─────────────────────────────────────────────────────

    def _rank(self, key: str, value: float | None) -> float | None:
        """Fraction of previous gestures this one exceeds.

        None when there is nothing to say — either this gesture was not measured
        on that dimension, or nothing before it was. Returning 0.5 in that case
        would look like a middling result rather than an unknown one, and the
        caller would go on to assert it.
        """
        if value is None:
            return None
        prior = [e[key] for e in self.entries if e.get(key) is not None]
        if len(prior) < 3:
            return None
        return sum(1 for v in prior if v < value) / len(prior)

    def observe(self, kind: str, f: dict) -> list[str]:
        """The notable facts about this gesture, most striking first.

        Deliberately returns few: handing the model every statistic makes it
        pick badly, and a connoisseur who lists everything isn't one.
        """
        n = len(self.entries)
        facts: list[str] = []
        if n < 3:
            # Saying "the first one" invites the model to invent a comparison
            # against a corpus that does not exist yet. Say so plainly instead.
            return ["nothing to compare it against yet"]

        trusted = f.get("trusted", False)
        size = self._rank("span_cm", f.get("span_cm") if trusted else None)
        high = self._rank("rise_cm", f.get("rise_cm") if trusted else None)
        dur = self._rank("duration_s", f["duration_s"])
        force = self._rank("peak_accel", f["peak_accel"])
        same_kind = sum(1 for e in self.entries if e["kind"] == kind)

        # records first — they are the most worth remarking on
        if size is not None:
            if size >= 1.0:
                facts.append("the largest so far")
            elif size <= 0.0:
                facts.append("the smallest so far")
        if high is not None and high >= 1.0:
            facts.append("higher than anyone has taken it")
        if force is not None and force >= 1.0:
            facts.append("the hardest so far")
        if dur is not None and dur >= 1.0:
            facts.append("the longest so far")

        # Novelty, judged on the coarse family rather than the exact verb, and
        # only once there is a corpus worth comparing against.
        fam = family(kind)
        same_family = sum(1 for e in self.entries
                          if family(e.get("kind", "")) == fam)
        if n >= NOVELTY_MIN:
            noun = FAMILY_NOUNS.get(fam, NOUNS.get(kind, kind))
            if same_family == 0:
                facts.append(f"nobody has done a {noun} before")
            elif same_family == 1:
                facts.append(f"one other {noun} so far")

        if not facts:
            if size is not None:
                if size >= 0.85:
                    facts.append("larger than most")
                elif size <= 0.15:
                    facts.append("smaller than most")
            if force is not None:
                if force >= 0.9:
                    facts.append("harder than most")
                elif force <= 0.1:
                    facts.append("gentler than most")
            if same_kind >= max(4, n * 0.3):
                facts.append(f"{same_kind} of these tonight, the usual")
            elif not facts:
                facts.append("unremarkable so far")

        # No absence here. It is a standing fact about the evening rather than
        # anything about this movement, and attaching it to whatever gesture
        # happened to be current inflated nothing-movements over the bar —
        # "barely moved ... nobody has done anything high all night" scored 0.65.
        # It belongs to greet(), where it lands before an intention has formed.
        return facts[:2]

    def absence(self) -> str | None:
        """A region of the movement space nobody has visited tonight.

        The one lever that moves people without instructing them: it asserts
        nothing about anyone and leaves a gap they fill in themselves. Needs
        enough history behind it for "nobody" to be a claim rather than an
        accident of a quiet start.

        Also the only thing worth saying about stillness, which cannot be ranked
        against gestures — there is no size or speed to compare.
        """
        if len(self.entries) < 12:
            return None
        for name, test in REGIONS.items():
            if not any(test(e) for e in self.entries):
                return f"nobody has done anything {name} all night"
        return None

    # ── visits ────────────────────────────────────────────────────────────
    #
    # Pickup and putdown finally give the corpus *people*, not just gestures.
    # Until now "nobody has done that" meant nobody-among-the-movements, which
    # is a much weaker and stranger claim than nobody-among-the-people. It also
    # still refuses to identify anyone: a visit is an interval, not a person.

    def add_visit(self, *, duration_ms: int, kinds: list[str],
                  biggest_cm: float | None, threw: bool) -> None:
        entry = {
            "t": time.time(),
            "duration_s": round(duration_ms / 1000.0, 1),
            "families": sorted({family(k) for k in kinds}),
            "gestures": len(kinds),
            "biggest_cm": biggest_cm,
            "threw": threw,
        }
        self.visits.append(entry)
        self.visits_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.visits_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")

    def greet(self, away_ms: int) -> list[str]:
        """What is true about this person before they have done anything.

        The only moment the machine speaks to someone with no intention yet
        formed, so it is the only moment a suggestion lands ahead of a decision
        rather than as a comment on one. The absence belongs here — the same
        sentence fired thirteen times mid-play and moved nobody.
        """
        n = len(self.visits)
        facts: list[str] = []
        if n == 0:
            return ["the first person to pick it up tonight"]

        facts.append(f"the {n + 1}{_suffix(n + 1)} person to pick it up tonight")

        # The absence takes the second slot ahead of anything else true. It is
        # the only thing said here that can change what the person does next.
        never = self.absence()
        if never:
            facts.append(never)
        elif not any(v["threw"] for v in self.visits):
            facts.append("nobody has let go of it yet")
        elif away_ms >= 120_000:
            facts.append(f"nobody has touched it for {away_ms // 60_000} minutes")
        return facts[:2]

    def summary(self) -> str:
        return f"{len(self.entries)} movements, {len(self.visits)} visits tonight"


def _suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
