"""WebSocket lifecycle and orchestration (PLAN 5.5).

Single-slot policy: one utterance in flight, ever. No queue, no backlog. A
device narrating gestures from thirty seconds ago is broken in a way that is
very hard to un-see.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass, field

import websockets

from . import config, features, link, protocol, salience
from .lifecycle import (DORMANT, MOVING, Holding, Lifecycle, Pause, PickedUp,
                        PutDown)
from .policy import Policy
from .protocol import MotionFrame
from .segmenter import Segment
from .corpus import Corpus
from .session_log import SessionLog
from .tts import TTS
from .voice import Voice

log = logging.getLogger("gradi")


@dataclass
class State:
    """Whether the device is mid-utterance. Not a speaking policy —
    that is host/policy.py; this only tracks the hardware."""
    device: str = "idle"
    inflight: bool = False
    last_end: float = 0.0
    now: float = 0.0


def _busy(state: "State") -> bool:
    """The device is mid-utterance. Nothing to do with whether to speak."""
    return state.device != "idle" or state.inflight


@dataclass
class Visit:
    """One person, from pickup to putdown.

    Not an identity — an interval. The device is handed between strangers and
    cannot tell them apart, and this deliberately does not try. What it knows is
    that this stretch of movement began when somebody lifted it off the table
    and ends when it goes back down, which is enough to say "the fourth person
    tonight" without ever knowing who any of them were.

    Replaces Stretch, which reset on a 20-second stillness and so merged two
    strangers who swapped quickly into one set of comparisons.
    """
    started_at: int = 0                 # device ms
    recent: list = field(default_factory=list)
    kinds: list = field(default_factory=list)
    families: set = field(default_factory=set)
    biggest_cm: float = 0.0
    threw: bool = False

    @property
    def moved_yet(self) -> bool:
        return bool(self.kinds)

    def repeats_of(self, kind: str) -> int:
        """How many of this family in an unbroken run at the end.

        Consecutive, not total: someone alternating two movements is varying,
        someone doing the same one six times is in a rut, and only the second
        should go quiet.
        """
        fam = features.family(kind)
        n = 0
        for k in reversed(self.kinds):
            if features.family(k) != fam:
                break
            n += 1
        return n + 1

    def remember(self, feat: dict, kind: str) -> None:
        entry = dict(feat)
        entry["_kind"] = kind
        self.recent.append(entry)
        del self.recent[:-config.SESSION_WINDOW]
        self.kinds.append(kind)
        self.families.add(features.family(kind))
        if feat.get("trusted"):
            self.biggest_cm = max(self.biggest_cm, feat.get("span_cm", 0.0))
        if feat.get("airborne_ms"):
            self.threw = True


@dataclass
class Session:
    ws: object
    voice: Voice
    tts: TTS
    logbook: SessionLog
    corpus: Corpus
    view: object = None                 # host.view.View, or None
    speak: bool = True                  # False under --no-audio
    show_rate: bool = True              # False under --quiet
    lifecycle: Lifecycle = field(default_factory=Lifecycle)
    policy: Policy = field(default_factory=Policy)
    state: State = field(default_factory=State)
    visit: Visit = field(default_factory=Visit)
    utt_id: int = 0
    _drained: asyncio.Event = field(default_factory=asyncio.Event)
    _tasks: set = field(default_factory=set)
    frames_seen: int = 0
    last_seq: int | None = None
    dropped: int = 0
    _rate_t0: float = 0.0
    _rate_n: int = 0

    # ── inbound ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        async for message in self.ws:
            if isinstance(message, str):
                log.warning("text frame ignored: %r", message[:80])
                continue
            try:
                kind, payload = protocol.unpack(message)
            except protocol.ProtocolError as exc:
                log.error("bad frame: %s", exc)
                continue

            if kind == protocol.MOTION:
                self._on_motion(payload)
            elif kind == protocol.STATE:
                self._on_state(payload)
            elif kind == protocol.LOG:
                log.info("[device] %s", payload)

    def _on_motion(self, frames: list[MotionFrame]) -> None:
        now = time.monotonic()
        if self._rate_t0 == 0.0:
            self._rate_t0 = now
        elif now - self._rate_t0 >= 5.0:
            # Everything downstream assumes 100 Hz. If this drifts, the
            # segmenter's millisecond thresholds quietly stop meaning what
            # they say, so it is worth saying out loud.
            hz = self._rate_n / (now - self._rate_t0)
            # A rate drop is worth hearing about even under --quiet:
            # every threshold downstream is stated in milliseconds and
            # silently means something else if this is not 100 Hz.
            if hz < 90:
                log.warning("motion %.0f Hz (%d dropped)", hz, self.dropped)
            elif self.show_rate:
                log.info("motion %.0f Hz (%d dropped)", hz, self.dropped)
            self._rate_t0, self._rate_n = now, 0

        for f in frames:
            self.frames_seen += 1
            self._rate_n += 1
            if self.last_seq is not None and f.seq != self.last_seq + 1:
                gap = f.seq - self.last_seq - 1
                # A device reboot restarts the sequence, and the reset chatter
                # left in the port can parse as one garbage frame carrying an
                # arbitrary seq. Either way the "gap" is not dropped data, and
                # counting it reported 1.6 billion dropped frames on a link that
                # had lost nothing. Anything this large is a discontinuity, so
                # resynchronise instead of accusing the link.
                if 0 < gap < 1000:
                    self.dropped += gap
            self.last_seq = f.seq

            before = self.lifecycle.state
            event = self.lifecycle.push(f)
            if self.view and self.lifecycle.state != before:
                # State-only ping so the page's header tracks presence even
                # through long stretches where nothing is worth saying.
                self.view.publish(tick=True, state=self.lifecycle.state,
                                  bar=round(self.policy.height(), 2))
            if event is not None:
                # Held in a set for the same reason as the view task above:
                # an unreferenced task can be collected before it runs, which
                # here would silently lose an utterance.
                task = asyncio.create_task(self._on_event(event, time.monotonic()))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

    def _on_state(self, value: int) -> None:
        self.state.device = "playing" if value == protocol.PLAYING else "idle"
        log.debug("device -> %s", self.state.device)
        if self.state.device == "idle":
            self._drained.set()

    # ── the pipeline ──────────────────────────────────────────────────────

    async def _on_event(self, event, t_detected: float) -> None:
        """One door for everything the lifecycle notices.

        Each event becomes a *candidate*: a sentence that could be said and a
        score for how much it is worth saying. The policy decides. Silence is a
        normal, frequent, deliberate outcome and is recorded as one — a
        suppressed candidate is data about the machine's judgement, not a
        failure to have any.
        """
        if isinstance(event, PickedUp):
            return await self._picked_up(event, t_detected)
        if isinstance(event, PutDown):
            return await self._put_down(event, t_detected)
        if isinstance(event, Pause):
            return await self._pause(event, t_detected)
        if isinstance(event, Holding):
            return await self._holding(event, t_detected)
        if isinstance(event, Segment):
            return await self._gesture(event, t_detected)

    # ── the situations ────────────────────────────────────────────────────

    async def _picked_up(self, ev: PickedUp, t0: float) -> None:
        self.visit = Visit(started_at=ev.t_ms)
        self.policy.reset()

        facts = self.corpus.greet(ev.away_ms)
        descriptor = features.describe_pickup(facts)
        score = salience.for_pickup(ev.away_ms, len(self.corpus.visits))

        # Bypasses the bar. This is the only moment the machine speaks to
        # somebody who has not done anything yet, and so the only moment a
        # suggestion arrives before an intention rather than after one.
        await self._maybe(descriptor, score, ev, always=True, kind="pickup")

    async def _put_down(self, ev: PutDown, t0: float) -> None:
        # Too brief to have been anybody. Forget it happened rather than record
        # a phantom visitor and say goodbye to a table.
        if ev.visit_ms < config.MIN_VISIT_MS and not self.visit.moved_yet:
            log.debug("discarding a %dms visit", ev.visit_ms)
            self.visit = Visit()
            return
        if self.visit.moved_yet:
            self.corpus.add_visit(duration_ms=ev.visit_ms,
                                  kinds=list(self.visit.kinds),
                                  biggest_cm=self.visit.biggest_cm or None,
                                  threw=self.visit.threw)
        descriptor = features.describe_putdown(ev.visit_ms, self.visit.kinds)
        score = salience.for_putdown(ev.visit_ms, len(self.visit.kinds))
        await self._maybe(descriptor, score, ev, kind="putdown", ending=True)
        self.visit = Visit()

    async def _holding(self, ev: Holding, t0: float) -> None:
        absence = self.corpus.absence()
        descriptor = features.describe_holding(
            ev, self.visit.moved_yet, [absence] if absence else [])
        score = salience.for_holding(ev.duration_ms, ev.first,
                                     self.visit.moved_yet)
        await self._maybe(descriptor, score, ev, kind="holding")

    async def _pause(self, ev: Pause, t0: float) -> None:
        """A lull. If something was waiting for a polite moment, this is it."""
        waiting = self.policy.release()
        if waiting is None:
            return
        descriptor, score, feat, seg, kind = waiting
        if _busy(self.state):
            return
        await self._speak_it(descriptor, score, feat=feat, segment=seg, kind=kind)

    async def _gesture(self, seg: Segment, t0: float) -> None:
        feat = features.extract(seg)
        if not feat:
            return
        kind = features.classify(feat)

        # Read before writing, or every gesture is its own precedent.
        relation = features.compare(feat, self.visit.recent)
        facts = self.corpus.observe(kind, feat)
        repeats = self.visit.repeats_of(kind)
        new_to_visit = features.family(kind) not in self.visit.families

        self.corpus.add(kind, feat)
        self.visit.remember(feat, kind)

        descriptor = features.describe(feat, facts, relation)
        score = salience.for_gesture(feat, facts, relation, repeats,
                                     new_to_visit=new_to_visit, kind=kind)
        await self._maybe(descriptor, score, seg, feat=feat, kind="gesture")

    # ── the gate ──────────────────────────────────────────────────────────

    async def _maybe(self, descriptor: str, score, ev, *, feat: dict = None,
                     always: bool = False, ending: bool = False,
                     kind: str = "gesture") -> None:
        moving = self.lifecycle.state == MOVING
        decision = self.policy.consider(
            score.value, state=self.lifecycle.state, moving=moving,
            always=always, ending=ending, busy=_busy(self.state))

        segment = ev if isinstance(ev, Segment) else None

        if decision.speak:
            return await self._speak_it(descriptor, score, feat=feat or {},
                                        segment=segment, kind=kind)

        # Cleared the bar but arrived mid-movement: hold it for a lull rather
        # than talk over someone. It expires if the lull never comes, because a
        # line delivered after the moment it describes is worse than none.
        #
        # Deliberately not logged. A deferral is not an outcome — the outcome is
        # recorded when it resolves — and writing one anyway put the same
        # segment in the log twice, once as withheld and once as spoken.
        if decision.reason == "waiting for a lull":
            self.policy.hold((descriptor, score, feat or {}, segment, kind),
                             score.value)
            if self.view:
                self.view.publish(kind=kind, descriptor=descriptor,
                                  utterance=None, responded=False,
                                  reason="waiting for a lull",
                                  score=round(decision.score, 2),
                                  bar=round(decision.bar, 2),
                                  state=self.lifecycle.state)
            return

        log.info("[silent %.2f<%.2f] %s", decision.score, decision.bar,
                 descriptor.split("\n")[0])
        self.logbook.write(segment=segment, features=feat or {},
                           descriptor=descriptor, utterance=None,
                           timings={"salience": score.value, "bar": decision.bar},
                           responded=False, reason=decision.reason, kind=kind)
        if self.view:
            self.view.publish(kind=kind, descriptor=descriptor, utterance=None,
                              responded=False, reason=decision.reason,
                              score=round(decision.score, 2),
                              bar=round(decision.bar, 2),
                              why="; ".join(score.reasons),
                              state=self.lifecycle.state)

    async def _speak_it(self, descriptor: str, score, *, feat: dict,
                        segment, kind: str) -> None:
        timings: dict[str, float] = {"salience": score.value,
                                     "bar": round(self.policy.height(), 3)}
        t0 = time.monotonic()
        # How fast somebody moves after being spoken to is the only place the
        # piece's own effect on them is visible.
        if self.policy.spoke_at:
            timings["since_last_spoke"] = (t0 - self.policy.spoke_at) * 1000
        await self._respond(descriptor, timings, t0, segment=segment,
                            feat=feat, kind=kind, score=score)

    async def _respond(self, descriptor: str, timings: dict, t0: float, *,
                       segment, feat: dict, kind: str = "gesture",
                       score=None) -> None:
        self.state.inflight = True
        self._drained.clear()
        utterance = None
        try:
            t_llm = time.monotonic()
            utterance, retried = await self.voice.say(descriptor)
            timings["llm"] = (time.monotonic() - t_llm) * 1000
            timings["llm_retried"] = 1 if retried else 0
            log.info("%s  ->  %s", descriptor, utterance)

            if self.speak:
                await self._speak(utterance, timings, t0)
            else:
                # Nothing is sent to the device, so it never reports PLAYING and
                # the single-slot gate reopens immediately. That is the point:
                # reading the output makes the next gesture testable in a second
                # rather than after several seconds of speech.
                timings["audio_ms"] = 0
        except Exception:
            log.exception("pipeline failed; returning device to idle")
            try:
                await self.ws.send(protocol.pack_flush())
            except Exception:
                pass
        finally:
            self.state.inflight = False
            self.state.last_end = time.monotonic()
            timings["total_to_idle"] = (time.monotonic() - t0) * 1000
            self.logbook.write(segment=segment, features=feat,
                               descriptor=descriptor, utterance=utterance,
                               timings=timings, kind=kind,
                               responded=utterance is not None)
            if self.view:
                self.view.publish(kind=kind, descriptor=descriptor,
                                  utterance=utterance, responded=True,
                                  ms=round(timings.get("llm", 0)),
                                  score=round(score.value, 2) if score else None,
                                  bar=round(self.policy.height(), 2),
                                  why="; ".join(score.reasons) if score else "",
                                  state=self.lifecycle.state)

    async def _speak(self, text: str, timings: dict, t0: float) -> None:
        self.utt_id = (self.utt_id + 1) % 65536
        uid = self.utt_id

        await self.ws.send(protocol.pack_utt_begin(config.SAMPLE_RATE, uid))

        t_tts = time.monotonic()
        first = True
        total = 0
        async for chunk in self.tts.stream(text):
            if first:
                timings["tts_first_chunk"] = (time.monotonic() - t_tts) * 1000
                timings["gesture_to_first_audio"] = (time.monotonic() - t0) * 1000
                first = False
            total += len(chunk) // 2
            await self.ws.send(protocol.pack_pcm(uid, chunk))

        await self.ws.send(protocol.pack_utt_end(uid))
        timings["audio_ms"] = total / config.SAMPLE_RATE * 1000
        timings["synth_total"] = (time.monotonic() - t_tts) * 1000

        # the device is authoritative for when speech actually finished
        budget = timings["audio_ms"] / 1000 + 5.0
        try:
            await asyncio.wait_for(self._drained.wait(), timeout=budget)
        except asyncio.TimeoutError:
            # Never leave the device stuck at "playing" — _busy() would be
            # true forever and the piece would go quietly dead.
            log.warning("no STATE idle within %.1fs; flushing device", budget)
            timings["drain_timeout"] = 1
            try:
                await self.ws.send(protocol.pack_flush())
            except Exception:
                pass
            self.state.device = "idle"
        timings["playback_ms"] = (time.monotonic() - t_tts) * 1000


def _parse_args():
    ap = argparse.ArgumentParser(
        description="gradi-remark host. Reads movement, says something about it.")
    ap.add_argument("--no-audio", action="store_true",
                    help="do not speak: skip TTS and send nothing to the "
                         "device. Much faster to test against, because the "
                         "device never goes busy and no gesture is dropped "
                         "waiting for speech to finish.")
    ap.add_argument("--view", action="store_true",
                    help="serve a page showing what it would say")
    ap.add_argument("--view-port", type=int, default=8770)
    ap.add_argument("--quiet", action="store_true",
                    help="hide the periodic motion-rate lines")
    return ap.parse_args()


async def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx narrates every ollama call at INFO, which buries the one line worth
    # reading. Nothing is lost: failures still raise.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    voice = Voice()
    tts = TTS()

    log.info("warming models (first run downloads Kokoro, be patient)...")
    t0 = time.monotonic()
    await voice.warm()
    log.info("  ollama %s ready (%.1fs)", config.OLLAMA_MODEL, time.monotonic() - t0)
    if args.no_audio:
        log.info("  --no-audio: not loading Kokoro, nothing will be spoken")
    else:
        t0 = time.monotonic()
        await asyncio.get_running_loop().run_in_executor(None, tts.warm)
        log.info("  kokoro ready (%.1fs)", time.monotonic() - t0)

    view = None
    if args.view:
        from .view import View
        view = View(args.view_port)
        # The reference matters. asyncio holds only a *weak* reference to a
        # task, so a create_task() whose result is discarded can be garbage
        # collected mid-run and vanish with no exception and no log line. That
        # is exactly what happened: the page stopped loading while the server
        # carried on processing movement perfectly, and nothing anywhere said
        # why. Keeping it in a local that outlives main() is the whole fix.
        view_task = asyncio.create_task(view.serve())

        def _view_died(task: asyncio.Task) -> None:
            # Belt and braces. The reference above should keep it alive, but the
            # failure mode here is silence, and silence is exactly what makes a
            # bug like this cost an hour instead of a minute.
            if task.cancelled():
                return
            exc = task.exception()
            log.error("view server stopped: %s", exc or "no reason given")

        view_task.add_done_callback(_view_died)

    logbook = SessionLog()
    corpus = Corpus()
    log.info("corpus: %s", corpus.summary())
    log.info("logging to %s", logbook.path)

    async def handler(ws):
        peer = getattr(ws, "remote_address", ("?",))[0]
        log.info("device connected from %s", peer)
        session = Session(ws=ws, voice=voice, tts=tts, logbook=logbook,
                          corpus=corpus, view=view, speak=not args.no_audio,
                          show_rate=not args.quiet)
        try:
            await session.run()
        except (websockets.ConnectionClosed, link.LinkClosed):
            pass
        finally:
            log.info("device disconnected (%d frames, %d dropped); "
                     "said %d, withheld %d",
                     session.frames_seen, session.dropped,
                     session.policy.said, session.policy.withheld)

    await link.serve(handler)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
