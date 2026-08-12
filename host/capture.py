"""Guided capture of movements whose true size is known.

The trajectory reconstruction is the one part of the pipeline that can be
confidently wrong, and no amount of desk testing settles it — synthetic data
proves the arithmetic, not the sensor. This walks a person through a short
scripted sequence and records each movement against what it was *supposed* to
be, so `host.replay --check` can compare the two and say plainly whether the
reconstruction is trustworthy.

    python -m host.capture

Stop host.server first; it holds the port. Nothing is spoken, nothing reaches
the LLM — this only listens.

Prompts are timed rather than keypress-driven on purpose. Both hands are on the
device, and reaching for a keyboard mid-movement is exactly the contamination
this is trying to avoid.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime

from . import config, link, protocol

log = logging.getLogger("capture")

# (label, spoken instruction, seconds to do it in, true size in cm or None).
#
# BRISK is not a stylistic preference, it is the difference between a
# measurable movement and an unmeasurable one. Acceleration falls with the
# square of duration: a 30 cm lift taking one second peaks near 1.7 m/s^2 and is
# twenty times the tremor floor; the same lift taking four seconds peaks near
# 0.11 and is *below* it. The first capture was done slowly and every distance
# came back wrong — 14 cm for a 30 cm lift, and a downward move read as upward.
#
# So each prompt asks for one decisive movement, then stillness. Distance
# accuracy matters far less than speed: an honest "about 30, done briskly"
# measures well, while a painstaking 30.0 cm done slowly cannot be measured at
# all.
SEQUENCE = [
    ("warmup",      "Get comfortable holding it. Anything you like.",   5, None),
    ("still",       "Now hold it completely still",                     8, 0.0),
    ("lift30",      "ONE brisk lift, about 30 cm up. Then hold still",  5, 30.0),
    ("lift60",      "ONE brisk lift, about 60 cm up. Then hold still",  5, 60.0),
    ("lower30",     "ONE brisk drop, about 30 cm down. Then hold",      5, 30.0),
    ("sideways50",  "ONE brisk move, about 50 cm sideways. Then hold",  5, 50.0),
    ("toss",        "Toss it up and catch it. Twice.",                  8, None),
    ("shake",       "Shake it, about three times a second",             6, None),
    ("spin",        "Spin it in your hands, staying in one place",      6, None),
    ("carry",       "Carry it slowly across your body, take your time", 8, None),
    ("tiny",        "Fidget with it. Small movements only.",            6, None),
]


class Recorder:
    """Collects every frame with the label that was on screen when it arrived."""

    def __init__(self) -> None:
        self.frames: list[tuple[str, list]] = []
        self.label = "idle"
        self.done = asyncio.Event()
        self.seen = 0

    async def run(self, ws) -> None:
        asyncio.create_task(self._script())
        async for message in ws:
            if isinstance(message, str):
                continue
            try:
                kind, payload = protocol.unpack(message)
            except protocol.ProtocolError:
                continue
            if kind != protocol.MOTION:
                continue
            for f in payload:
                self.seen += 1
                self.frames.append((self.label, [
                    f.seq, f.t_ms, f.qw, f.qx, f.qy, f.qz, f.gx, f.gy, f.gz,
                    f.lax, f.lay, f.laz, f.ax, f.ay, f.az,
                ]))
            if self.done.is_set():
                break

    async def _script(self) -> None:
        # Let the stream settle before asking for anything. A capture that
        # begins mid-handshake starts with a gap the segmenter has to guess at.
        await asyncio.sleep(1.5)
        if not self.seen:
            print("\n  no frames arriving — is the board flashed with the new "
                  "firmware?\n")

        print("\n" + "=" * 62)
        print("  Follow the prompts. Each one counts down, then says GO.")
        print("=" * 62 + "\n")

        for label, instruction, secs, truth in SEQUENCE:
            print(f"  next:  {instruction}")
            for n in (3, 2, 1):
                print(f"         starting in {n}...", end="\r", flush=True)
                await asyncio.sleep(1)
            print(f"  GO ->  {instruction}" + " " * 12)
            self.label = label
            await asyncio.sleep(secs)
            self.label = "idle"
            print(f"         ...done ({label})\n")
            await asyncio.sleep(1.0)

        self.done.set()
        print("=" * 62)
        print(f"  captured {self.seen} frames")


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rec = Recorder()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = config.LOG_DIR / f"capture-{stamp}.jsonl"

    serving = asyncio.create_task(link.serve(rec.run))
    print("waiting for the device (stop host.server first — it holds the port)")

    while not rec.done.is_set():
        if serving.done():
            await serving              # re-raise whatever killed it
            return 1
        await asyncio.sleep(0.2)
    await asyncio.sleep(0.5)
    serving.cancel()

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    truths = {label: truth for label, _, _, truth in SEQUENCE}
    with open(path, "w", encoding="utf-8") as fh:
        for label, _, _, _ in SEQUENCE:
            rows = [r for lbl, r in rec.frames if lbl == label]
            if not rows:
                continue
            fh.write(json.dumps({
                "t": time.time(),
                "label": label,
                "true_cm": truths[label],
                "frames": rows,
            }, separators=(",", ":")) + "\n")

    print(f"  written to {path}")
    print(f"\n  now run:  uv run python -m host.replay {path} --check\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        pass
