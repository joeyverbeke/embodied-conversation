"""One JSONL record per gesture (PLAN 5.6).

This is not instrumentation, it is the development loop. Every record carries
enough to re-run the gesture through the pipeline with no hardware attached,
which is what makes Phase 2 possible at a desk.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

from . import config
from .segmenter import Segment


class SessionLog:
    def __init__(self, log_dir=config.LOG_DIR):
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = log_dir / f"session-{stamp}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")
        self.count = 0

    def write(self, *, segment: Segment, features: dict, descriptor: str,
              utterance: str | None, timings: dict, responded: bool,
              reason: str = "") -> None:
        record = {
            "t": time.time(),
            "index": self.count,
            "responded": responded,
            "reason": reason,
            "segment": {
                "t_start": segment.t_start,
                "t_end": segment.t_end,
                "duration_ms": segment.duration_ms,
                "ended": segment.ended,
                "rejected_twitches": segment.rejected_twitches,
                # raw frames, pre-roll included, so replay can re-segment
                "frames": [
                    [f.seq, f.t_ms, f.qw, f.qx, f.qy, f.qz, f.gx, f.gy, f.gz]
                    for f in segment.frames
                ],
            },
            "features": features,
            "descriptor": descriptor,
            "utterance": utterance,
            "timings_ms": timings,
        }
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        self._fh.close()
