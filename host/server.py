"""WebSocket lifecycle and orchestration (PLAN 5.5).

Single-slot policy: one utterance in flight, ever. No queue, no backlog. A
device narrating gestures from thirty seconds ago is broken in a way that is
very hard to un-see.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import websockets

from . import config, features, link, protocol
from .protocol import MotionFrame
from .segmenter import Segmenter
from .corpus import Corpus
from .session_log import SessionLog
from .tts import TTS
from .voice import Voice

log = logging.getLogger("gradi")


@dataclass
class State:
    """Everything should_respond() is allowed to look at."""
    device: str = "idle"
    inflight: bool = False
    last_end: float = 0.0
    now: float = 0.0


def should_respond(state: State) -> bool:
    """The one gate. Changing the policy is a change to this predicate."""
    return (state.device == "idle"
            and not state.inflight
            and state.now - state.last_end >= config.REFRACTORY_MS / 1000.0)


@dataclass
class Session:
    ws: object
    voice: Voice
    tts: TTS
    logbook: SessionLog
    corpus: Corpus
    segmenter: Segmenter = field(default_factory=Segmenter)
    state: State = field(default_factory=State)
    utt_id: int = 0
    _drained: asyncio.Event = field(default_factory=asyncio.Event)
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
            level = log.info if hz >= 90 else log.warning
            level("motion %.0f Hz (%d dropped)", hz, self.dropped)
            self._rate_t0, self._rate_n = now, 0

        for f in frames:
            self.frames_seen += 1
            self._rate_n += 1
            if self.last_seq is not None and f.seq != self.last_seq + 1:
                gap = f.seq - self.last_seq - 1
                if gap > 0:
                    self.dropped += gap
            self.last_seq = f.seq

            seg = self.segmenter.push(f)
            if seg is not None:
                asyncio.create_task(self._on_segment(seg, time.monotonic()))

    def _on_state(self, value: int) -> None:
        self.state.device = "playing" if value == protocol.PLAYING else "idle"
        log.debug("device -> %s", self.state.device)
        if self.state.device == "idle":
            self._drained.set()

    # ── the pipeline ──────────────────────────────────────────────────────

    async def _on_segment(self, seg, t_detected: float) -> None:
        timings: dict[str, float] = {}
        t0 = t_detected

        feat = features.extract(seg)
        kind = features.classify(feat) if feat else None
        # Read the corpus before adding to it, or every gesture is its own
        # precedent and nothing is ever new.
        facts = self.corpus.observe(kind, feat) if feat else []
        if feat:
            self.corpus.add(kind, feat)
        descriptor = features.describe(feat, facts)
        timings["features"] = (time.monotonic() - t0) * 1000

        self.state.now = time.monotonic()
        if not should_respond(self.state):
            reason = ("device_playing" if self.state.device != "idle"
                      else "inflight" if self.state.inflight else "refractory")
            log.info("dropped (%s): %s", reason, descriptor)
            self.logbook.write(segment=seg, features=feat, descriptor=descriptor,
                               utterance=None, timings=timings, responded=False,
                               reason=reason)
            return

        self.state.inflight = True
        self._drained.clear()
        utterance = None
        try:
            t_llm = time.monotonic()
            utterance, retried = await self.voice.say(descriptor)
            timings["llm"] = (time.monotonic() - t_llm) * 1000
            timings["llm_retried"] = 1 if retried else 0
            log.info("%s  ->  %s", descriptor, utterance)

            await self._speak(utterance, timings, t0)
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
            self.logbook.write(segment=seg, features=feat, descriptor=descriptor,
                               utterance=utterance, timings=timings,
                               responded=utterance is not None)

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
            # Never leave the device stuck at "playing" — should_respond would
            # be false forever and the piece would go quietly dead.
            log.warning("no STATE idle within %.1fs; flushing device", budget)
            timings["drain_timeout"] = 1
            try:
                await self.ws.send(protocol.pack_flush())
            except Exception:
                pass
            self.state.device = "idle"
        timings["playback_ms"] = (time.monotonic() - t_tts) * 1000


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    voice = Voice()
    tts = TTS()

    log.info("warming models (first run downloads Kokoro, be patient)...")
    t0 = time.monotonic()
    await voice.warm()
    log.info("  ollama %s ready (%.1fs)", config.OLLAMA_MODEL, time.monotonic() - t0)
    t0 = time.monotonic()
    await asyncio.get_running_loop().run_in_executor(None, tts.warm)
    log.info("  kokoro ready (%.1fs)", time.monotonic() - t0)

    logbook = SessionLog()
    corpus = Corpus()
    log.info("corpus: %s", corpus.summary())
    log.info("logging to %s", logbook.path)

    async def handler(ws):
        peer = getattr(ws, "remote_address", ("?",))[0]
        log.info("device connected from %s", peer)
        session = Session(ws=ws, voice=voice, tts=tts, logbook=logbook,
                          corpus=corpus)
        try:
            await session.run()
        except (websockets.ConnectionClosed, link.LinkClosed):
            pass
        finally:
            log.info("device disconnected (%d frames, %d dropped)",
                     session.frames_seen, session.dropped)

    await link.serve(handler)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
