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

# Regions of the movement space we can notice nobody has visited. Naming an
# absence is the one way to move people without instructing them: it is a
# statement about the corpus, and they fill it in themselves.
REGIONS = {
    "overhead": lambda f: "overhead" in (f["start_pose"], f["end_pose"]),
    "low": lambda f: "hanging" in (f["start_pose"], f["end_pose"]),
    "repeated": lambda f: f.get("repeat_hz") is not None,
    "percussive": lambda f: f["peak_dps"] >= config.PERCUSSIVE_DPS,
    "slow": lambda f: f["duration_s"] >= 3.0,
    "tiny": lambda f: f["path_deg"] <= 30.0,
}


@dataclass
class Corpus:
    path: object = None
    entries: list = field(default_factory=list)

    def __post_init__(self):
        if self.path is None:
            self.path = config.CORPUS_PATH
        self.load()

    # ── persistence ───────────────────────────────────────────────────────

    def load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                self.entries = [json.loads(line) for line in fh if line.strip()]
        except FileNotFoundError:
            self.entries = []

    def add(self, kind: str, f: dict) -> None:
        entry = {
            "t": time.time(),
            "kind": kind,
            "duration_s": f["duration_s"],
            "path_deg": f["path_deg"],
            "peak_dps": f["peak_dps"],
            "directness": f["directness"],
            "reversals": f["reversals"],
            "repeat_hz": f.get("repeat_hz"),
            "start_pose": f["start_pose"],
            "end_pose": f["end_pose"],
        }
        self.entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")

    # ── what it knows ─────────────────────────────────────────────────────

    def _rank(self, key: str, value: float) -> float:
        """Fraction of previous gestures this one exceeds."""
        prior = [e[key] for e in self.entries]
        if not prior:
            return 0.5
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

        size = self._rank("path_deg", f["path_deg"])
        dur = self._rank("duration_s", f["duration_s"])
        peak = self._rank("peak_dps", f["peak_dps"])
        same_kind = sum(1 for e in self.entries if e["kind"] == kind)

        # records first — they are the most worth remarking on
        if size >= 1.0:
            facts.append("the largest so far")
        elif size <= 0.0:
            facts.append("the smallest so far")
        if peak >= 1.0:
            facts.append("the fastest so far")
        if dur >= 1.0:
            facts.append("the longest so far")

        # novelty: has anyone done this shape before?
        if same_kind == 0:
            facts.append(f"nobody has done a {kind} before")
        elif same_kind == 1:
            facts.append(f"one other {kind} so far")

        # Height is the dimension the model is most tempted to invent, so give
        # it the true version whenever there is one to give.
        high = sum(1 for e in self.entries
                   if "overhead" in (e["start_pose"], e["end_pose"]))
        low = sum(1 for e in self.entries
                  if "hanging" in (e["start_pose"], e["end_pose"]))
        if f["end_pose"] == "overhead" and high <= n * 0.15:
            facts.append("higher than almost anyone has gone")
        elif f["end_pose"] == "hanging" and low <= n * 0.15:
            facts.append("lower than almost anyone has gone")

        if not facts:
            if size >= 0.85:
                facts.append("larger than most")
            elif size <= 0.15:
                facts.append("smaller than most")
            if peak >= 0.9:
                facts.append("faster than most")
            elif peak <= 0.1:
                facts.append("slower than most")
            if same_kind >= max(4, n * 0.3):
                facts.append(f"{same_kind} of these tonight, the usual")
            elif not facts:
                facts.append("unremarkable so far")

        # an absence is worth one slot, but only when there is enough history
        # for "nobody" to mean something
        if n >= 12 and len(facts) < 2:
            for name, test in REGIONS.items():
                if not any(test(e) for e in self.entries):
                    facts.append(f"nobody has done anything {name} all night")
                    break

        return facts[:2]

    def summary(self) -> str:
        return f"{len(self.entries)} seen tonight"
