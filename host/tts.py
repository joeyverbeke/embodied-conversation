"""Utterance -> PCM chunks (PLAN 5.4).

mlx-audio running Kokoro-82M on Metal. Kokoro's native output is 24 kHz, which
is exactly what the device plays, so there is no resampling step here and there
should never be one.

Synthesis is synchronous and Metal-bound, so it runs in a worker thread and
pushes chunks onto an asyncio queue as they appear.
"""

from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator

import numpy as np

from . import config


class TTS:
    def __init__(self):
        self._model = None
        self.sample_rate = config.SAMPLE_RATE

    def load(self) -> None:
        from mlx_audio.tts.utils import load_model
        self._model = load_model(config.TTS_MODEL)

    def warm(self) -> None:
        """First synthesis costs ~40 s (spaCy download, Metal warm-up).
        Pay it at startup, never in front of a participant."""
        if self._model is None:
            self.load()
        for _ in self._synth("Ready."):
            pass

    def _synth(self, text: str):
        """Yield int16 mono PCM bytes, sliced to PCM_CHUNK_MS."""
        chunk = int(self.sample_rate * config.PCM_CHUNK_MS / 1000)
        tail = np.empty(0, dtype=np.int16)

        for result in self._model.generate(
                text=text,
                voice=config.TTS_VOICE,
                speed=config.TTS_SPEED,
                lang_code=config.TTS_LANG,
        ):
            if result.sample_rate != self.sample_rate:
                raise RuntimeError(
                    f"Kokoro returned {result.sample_rate} Hz, expected "
                    f"{self.sample_rate}. Something would have to resample, "
                    "which PLAN §2 forbids — stop and look at this.")
            audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
            pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
            tail = np.concatenate([tail, pcm])
            while len(tail) >= chunk:
                yield tail[:chunk].tobytes()
                tail = tail[chunk:]

        if len(tail):
            yield tail.tobytes()

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Async view of _synth, so the event loop keeps serving motion."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        DONE = object()

        def worker():
            try:
                for buf in self._synth(text):
                    asyncio.run_coroutine_threadsafe(queue.put(buf), loop).result()
            except Exception as exc:                     # surfaced to the caller
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(DONE), loop).result()

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = await queue.get()
            if item is DONE:
                return
            if isinstance(item, Exception):
                raise item
            yield item
