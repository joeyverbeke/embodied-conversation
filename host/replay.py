"""Re-run logged movement through the pipeline with no hardware attached.

This is the development loop. Every session record carries its raw frames
(session_log.py), so segmentation, trajectory and features can all be retuned
at a desk against real movement, and the effect of a threshold change is
visible in seconds rather than in a gallery.

    python -m host.replay logs/session-20260808-163514.jsonl
    python -m host.replay logs/*.jsonl --resegment
    python -m host.replay logs/latest.jsonl --verbose

By default each stored segment is re-featurised in place, which is the fastest
way to see how the description of a known movement changed. `--resegment` feeds
every frame back through a fresh Segmenter instead, which is the only way to
test the gates themselves — including whether movements that used to be missed
entirely are now caught.

Logs recorded before the accelerometer was enabled carry nine fields per frame
instead of fifteen. They still replay, and the rotation features still mean
what they always did, but nothing that depends on the trajectory can be
computed from them and it is reported as such rather than guessed at.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import config, features, trajectory
from .protocol import MOTION_FRAME_SIZE, MotionFrame
from .segmenter import Segmenter

LEGACY_FIELDS = 9                   # seq, t_ms, quat, gyro — no acceleration
FIELDS = 15


def _frame(row: list) -> MotionFrame:
    if len(row) == LEGACY_FIELDS:
        row = list(row) + [0.0] * (FIELDS - LEGACY_FIELDS)
    return MotionFrame(int(row[0]), int(row[1]), *(float(v) for v in row[2:]))


def _records(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


class Stored:
    """Just enough of a Segment for features.extract(), rebuilt from a log."""

    def __init__(self, rec: dict):
        seg = rec["segment"]
        self.frames = [_frame(r) for r in seg["frames"]]
        self.t_start = seg["t_start"]
        self.t_end = seg["t_end"]
        self.ended = seg["ended"]
        # renamed when hesitations stopped being discarded
        self.hesitations = seg.get("hesitations", seg.get("rejected_twitches", 0))
        self.airborne_ms = seg.get("airborne_ms", 0)

    @property
    def duration_ms(self) -> int:
        return self.t_end - self.t_start

    def motion(self) -> list[MotionFrame]:
        return [f for f in self.frames if f.t_ms >= self.t_start]


def _report(feat: dict, descriptor: str, verbose: bool) -> None:
    print(descriptor)
    if verbose and feat:
        ranked = features.verbs(feat)
        print("    verbs   " + ", ".join(f"{k} {v:.2f}" for k, v in ranked))
        print("    conf    %.2f   trusted=%s   ended=%s"
              % (feat.get("confidence", 0.0), feat.get("trusted"),
                 feat.get("ended")))
        print("    dyn     peak %.1f m/s2, onset jerk %.0f, impact %.0f"
              % (feat["peak_accel"], feat["onset_jerk"], feat["impact"]))
        print("    rot     %.0f deg, %.0f dps median, stability %.2f"
              % (feat["spin_deg"], feat["spin_rate"], feat["axis_stability"]))
    print()


def replay_stored(paths: list[Path], verbose: bool) -> None:
    n = legacy = 0
    for path in paths:
        print(f"── {path.name} " + "─" * max(0, 60 - len(path.name)))
        stretch: list[dict] = []
        for rec in _records(path):
            if not rec.get("segment"):
                continue
            seg = Stored(rec)
            if len(seg.frames[0:1]) and not any(
                    f.lax or f.lay or f.laz for f in seg.frames):
                legacy += 1
            feat = features.extract(seg)
            kind = features.classify(feat) if feat else None
            relation = features.compare(feat, stretch) if feat else []
            _report(feat, features.describe(feat, None, relation), verbose)
            if feat:
                entry = dict(feat)
                entry["_kind"] = kind
                stretch.append(entry)
                del stretch[:-config.SESSION_WINDOW]
            n += 1
    _footer(n, legacy)


def replay_resegment(paths: list[Path], verbose: bool) -> None:
    """Feed every frame back through a fresh Segmenter.

    Segment boundaries in the log came from the old gates, so this is the only
    way to see whether a movement that used to fall below them is now caught.
    Frames are replayed in log order; gaps between records are real gaps in the
    original capture, and the segmenter handles them the same way it handled
    them live.
    """
    seg_count = still_count = 0
    for path in paths:
        print(f"── {path.name} " + "─" * max(0, 60 - len(path.name)))
        segmenter = Segmenter()
        stretch: list[dict] = []
        for rec in _records(path):
            if not rec.get("segment"):
                continue
            for row in rec["segment"]["frames"]:
                event = segmenter.push(_frame(row))
                if event is None:
                    continue
                feat = features.extract(event)
                kind = features.classify(feat) if feat else None
                relation = features.compare(feat, stretch) if feat else []
                _report(feat, features.describe(feat, None, relation), verbose)
                if feat:
                    entry = dict(feat)
                    entry["_kind"] = kind
                    stretch.append(entry)
                    del stretch[:-config.SESSION_WINDOW]
                seg_count += 1
    print(f"{seg_count} segments, {still_count} stillnesses")


class Trimmed:
    """A frame list presented as a Segment, for features.extract().

    Used so the description and the measured distance in --check come from the
    same frames. They did not, briefly, and the table cheerfully printed "57 cm
    down" next to a measurement of 21 cm.
    """

    def __init__(self, frames: list[MotionFrame]):
        self.frames = frames
        self.t_start = frames[0].t_ms
        self.t_end = frames[-1].t_ms
        self.ended = "settled"
        self.hesitations = 0
        self.airborne_ms = 0

    @property
    def duration_ms(self) -> int:
        return self.t_end - self.t_start

    def motion(self) -> list[MotionFrame]:
        return self.frames


def _movement_within(frames: list[MotionFrame]) -> list[MotionFrame]:
    """Trim a labelled capture window down to the movement inside it.

    A prompt says "lift it and hold" and allows five seconds; the lift itself
    might take one. Integrating the whole window means integrating four seconds
    of someone standing still, and the drift accumulated over those four seconds
    swamps the movement. The live pipeline never sees a window — the segmenter
    hands it a gesture — so grading whole windows measures a situation that does
    not occur.
    """
    active = [i for i, f in enumerate(frames)
              if math.sqrt(f.lax ** 2 + f.lay ** 2 + f.laz ** 2)
              > config.ACCEL_OFF_MS2]
    if len(active) < 3:
        return []
    pad = int(0.15 * config.IMU_HZ)      # keep the wind-up and the settle
    return frames[max(0, active[0] - pad):min(len(frames), active[-1] + pad + 1)]


def _freefall_truth(frames: list[MotionFrame]) -> list[tuple[float, float]]:
    """Validate the integration against flight time. Returns (true, measured).

    This is the real gate, and it needs no human to estimate anything.

    A thrown ball is weightless for exactly as long as physics says, and the
    accelerometer sees that directly — no integration involved. Flight time
    fixes the release speed at v = g*t/2. Integrating the throw up to the
    moment of release gives a second, completely independent estimate of the
    same number. If the two agree, the integration is sound.

    The first version of this tool graded against remembered distances instead,
    and reported FAIL on a pipeline that was working: hand-travel estimates run
    roughly 2x high, consistently enough to look exactly like a scale bug.
    """
    g = np.array([math.sqrt(f.ax ** 2 + f.ay ** 2 + f.az ** 2) for f in frames])
    runs, start = [], None
    for i, weightless in enumerate(g < config.FREEFALL_MS2):
        if weightless and start is None:
            start = i
        elif not weightless and start is not None:
            runs.append((start, i - start))
            start = None

    out = []
    for s, n in runs:
        if n * 1000 // config.IMU_HZ < config.FREEFALL_MIN_MS:
            continue
        flight = n / config.IMU_HZ
        v_true = config.G * flight / 2.0        # symmetric up and down
        run_up = frames[max(0, s - 60):s]       # ~600 ms into the release
        if len(run_up) < 10:
            continue
        q = np.array([[f.qw, f.qx, f.qy, f.qz] for f in run_up])
        a = np.array([[f.lax, f.lay, f.laz] for f in run_up])
        # No zero-velocity update here: a throw ends at maximum speed, which is
        # the one thing the correction assumes never happens.
        v = np.cumsum(trajectory.rotate(q, a) / config.IMU_HZ, axis=0)
        out.append((v_true, float(v[-1][2])))
    return out


def check(paths: list[Path]) -> bool:
    """Grade a host.capture recording.

    Physics decides pass or fail. Remembered distances are printed because the
    *ratios* between them are informative, but they do not vote.
    """
    ok = True
    for path in paths:
        print(f"── {path.name} " + "─" * max(0, 60 - len(path.name)))

        # ── the gate: physics, no human estimate involved ─────────────────
        pairs: list[tuple[float, float]] = []
        for rec in _records(path):
            pairs += _freefall_truth([_frame(r) for r in rec["frames"]])

        print("  GROUND TRUTH — release speed, flight time vs integration")
        if not pairs:
            print("     no throws detected; toss it and catch it to enable "
                  "this check\n")
            ok = False
        else:
            for v_true, v_meas in pairs:
                err = (v_meas / v_true - 1) * 100 if v_true else 0.0
                mark = "ok  " if abs(err) <= 25 else "OFF "
                print(f"     {mark} flight says {v_true:5.2f} m/s, "
                      f"integration says {v_meas:5.2f} m/s   ({err:+.0f}%)")
            worst = max(abs(m / t - 1) for t, m in pairs if t)
            ok &= worst <= 0.25
            print(f"     -> integration accurate to {worst * 100:.0f}%"
                  f"  {'PASS' if worst <= 0.25 else 'FAIL'}\n")

        print("  REMEMBERED SIZES — informational. People judge their own hand's")
        print("  travel about 2x high, so the ratios matter and the absolutes do not.")
        print(f"  {'movement':12s} {'asked':>7s} {'measured':>9s} {'ratio':>7s}  "
              f"{'conf':>5s}  what it made of it")
        ratios: list[float] = []
        for rec in _records(path):
            label = rec["label"]
            truth = rec.get("true_cm")
            frames = _movement_within([_frame(r) for r in rec["frames"]])
            if len(frames) < 3:
                print(f"  {label:12s} {'-':>7s} {'no motion':>9s}")
                continue
            path_obj = trajectory.reconstruct(frames)
            if path_obj is None:
                print(f"  {label:12s} {'-':>7s} {'no data':>9s}")
                continue
            m = trajectory.measure(path_obj)
            got = m["net_cm"]

            trusted = path_obj.trusted
            shown = f"{got:8.1f}cm" if trusted else f"{'unclear':>9s}"

            if truth is None:
                verdict = ""
                cells = f"{'-':>7s} {shown} {'-':>7s}"
            elif truth == 0.0:
                # Stillness is the one case a human cannot misjudge, so it still
                # votes: the test is that the reconstruction did NOT invent
                # travel where there was none.
                good = not trusted or got <= 5.0
                ok &= good
                verdict = "PASS" if good else "FAIL"
                cells = f"{truth:6.0f}cm {shown} {got:+6.1f}"
            elif not trusted:
                # Declining to measure is not a failure. Asserting a wrong
                # distance is. The whole design rests on that distinction.
                verdict = "----"
                cells = f"{truth:6.0f}cm {shown} {'-':>6s}"
            else:
                ratios.append(got / truth)
                verdict = ""
                cells = f"{truth:6.0f}cm {shown} {got / truth:7.2f}"

            feat = features.extract(Trimmed(frames))
            desc = features.describe(feat).split("\n")[0] if feat else "-"
            print(f"  {label:12s} {cells} {m['confidence']:5.2f}  "
                  f"{verdict:4s} {desc}")

        if len(ratios) >= 3:
            mean = sum(ratios) / len(ratios)
            spread = max(ratios) - min(ratios)
            print(f"\n     mean ratio {mean:.2f}, spread {spread:.2f} — "
                  f"{'consistent' if spread <= 0.35 else 'scattered'}.")
            if spread <= 0.35:
                # A consistent offset is a property of the person estimating,
                # not of the sensor. A scattered one would mean neither the
                # estimates nor the measurements can be relied on.
                print(f"     A steady {1 / mean:.1f}x offset across every "
                      f"movement is one person's sense of scale,")
                print("     not a sensor fault — a scale error would not "
                      "leave the ratios intact.")
        print()

    print("TRAJECTORY TRUSTWORTHY" if ok else
          "FAILED — physics disagrees with the integration; fix that first")
    return ok


def trace_policy(paths: list[Path]) -> int:
    """Replay a session and show every judgement, including the silent ones.

    A policy whose main output is silence cannot be debugged from the outside:
    working correctly and completely broken look identical. This prints what was
    considered, what it scored, what bar it faced, and what happened.
    """
    for path in paths:
        print(f"── {path.name} " + "─" * max(0, 60 - len(path.name)))
        print(f"  {'kind':9s} {'score':>6s} {'bar':>6s}  {'':5s} line")
        said = held = 0
        first = last = None
        for rec in _records(path):
            t = rec.get("t")
            first = first if first is not None else t
            last = t
            timings = rec.get("timings_ms") or {}
            score = timings.get("salience")
            bar = timings.get("bar")
            spoke = rec.get("responded")
            said, held = said + bool(spoke), held + (not spoke)
            line = (rec.get("utterance")
                    or (rec.get("descriptor") or "").split("\n")[0])
            print("  %-9s %6s %6s  %-5s %s" % (
                rec.get("kind", "gesture"),
                f"{score:.2f}" if score is not None else "-",
                f"{bar:.2f}" if bar is not None else "-",
                "SAID" if spoke else "",
                line[:78]))
        minutes = ((last or 0) - (first or 0)) / 60.0
        rate = said / minutes if minutes > 0.02 else 0.0
        print(f"\n  said {said}, withheld {held}"
              + (f", {rate:.1f} utterances per minute" if rate else ""))
    return 0


def report_effect(paths: list[Path]) -> int:
    """Did people move differently after the machine spoke?

    The only question that matters and the one never asked. Accuracy was always
    the wrong measure for this piece: a perfectly accurate machine that changes
    nobody's behaviour has failed at the only thing it was built to do.

    Compares the gesture that follows an utterance against gestures that follow
    silence. A real effect shows up as a difference in how quickly people move
    next, and in how big that movement is.
    """
    after_speech, after_silence = [], []
    for path in paths:
        prev_spoke = False
        prev_t = None
        for rec in _records(path):
            feat = rec.get("features") or {}
            t = rec.get("t")
            if feat and prev_t is not None:
                gap = t - prev_t
                row = (gap, feat.get("span_cm") or 0.0, feat.get("peak_accel") or 0.0)
                (after_speech if prev_spoke else after_silence).append(row)
            if feat:
                prev_t, prev_spoke = t, bool(rec.get("responded"))

    def summarise(name, rows):
        if not rows:
            print(f"  {name:16s} (nothing yet)")
            return
        n = len(rows)
        print("  %-16s n=%-4d  next move after %5.1fs   %5.1f cm   %5.1f m/s2"
              % (name, n, sum(r[0] for r in rows) / n,
                 sum(r[1] for r in rows) / n, sum(r[2] for r in rows) / n))

    print("What people did next:")
    summarise("after it spoke", after_speech)
    summarise("after silence", after_silence)
    if after_speech and after_silence:
        a = sum(r[1] for r in after_speech) / len(after_speech)
        b = sum(r[1] for r in after_silence) / len(after_silence)
        if b > 0:
            print(f"\n  movements are {a / b:.2f}x the size after it speaks")
            print("  (one session proves nothing; watch this across an evening)")
    return 0


def _footer(n: int, legacy: int) -> None:
    print(f"{n} gestures")
    if legacy:
        print(f"{legacy} of them predate the accelerometer — no trajectory, "
              f"so distances are absent rather than estimated.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--resegment", action="store_true",
                    help="re-run the gates instead of using stored boundaries")
    ap.add_argument("--check", action="store_true",
                    help="grade a host.capture recording against its true sizes")
    ap.add_argument("--policy", action="store_true",
                    help="trace every candidate: score, bar, spoke or withheld")
    ap.add_argument("--effect", action="store_true",
                    help="did people move differently after being spoken to?")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="print the numbers behind each description")
    args = ap.parse_args()

    paths = [p for p in args.logs if p.is_file()]
    if not paths:
        print("no readable logs", file=sys.stderr)
        return 1

    print(f"frame size {MOTION_FRAME_SIZE} bytes; "
          f"gates {config.ARM_ON_DPS:.0f} dps / {config.ACCEL_ON_MS2:.1f} m/s2\n")
    if args.check:
        return 0 if check(paths) else 2
    if args.policy:
        return trace_policy(paths)
    if args.effect:
        return report_effect(paths)
    if args.resegment:
        replay_resegment(paths, args.verbose)
    else:
        replay_stored(paths, args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
