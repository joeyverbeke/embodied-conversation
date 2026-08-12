"""Descriptor -> utterance (PLAN 5.3).

The persona file is hot-reloaded on every request. The human edits it
constantly and must never restart the server to do it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import ollama

from . import config

_WORD = re.compile(r"[a-z']+")


class Persona:
    """System prompt, re-read whenever the file changes on disk."""

    def __init__(self, path=config.PERSONA_PATH):
        self.path = path
        self._mtime = None
        self._text = ""
        self.reload()

    def reload(self) -> bool:
        """Returns True if the file changed since last read."""
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            if self._text:
                return False
            self._text = "Say one short, dry observation about how someone moved."
            return True
        if mtime != self._mtime:
            self._mtime = mtime
            self._text = self.path.read_text(encoding="utf-8").strip()
            return True
        return False

    @property
    def text(self) -> str:
        return self._text


# Short lines share function words by construction — "you", "that", "to" carry
# no signal about whether two observations are the same observation. Comparing
# raw tokens made almost every five-word line look like a repeat.
_STOP = frozenset("""
a an the and or but so of to in on at by for with from up down out back
is are was were be been am do does did done have has had you your yours
it its that this these those there here then than as if not no
i me my we us our he she they them one only just very much more most
""".split())


def _tokens(s: str) -> set[str]:
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP}


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _clean(text: str) -> str:
    text = text.strip().strip('"').strip()
    text = " ".join(text.split())
    # models like to add a second sentence; keep the first
    parts = re.split(r"(?<=[.!?])\s+", text)
    if parts:
        text = parts[0]
    words = text.split()
    if len(words) > config.MAX_WORDS:
        text = " ".join(words[:config.MAX_WORDS]).rstrip(",;:") + "."
    return text


@dataclass
class Voice:
    persona: Persona = field(default_factory=Persona)
    history: list[tuple[str, str]] = field(default_factory=list)
    _client: ollama.AsyncClient = field(default=None, repr=False)

    def __post_init__(self):
        self._client = ollama.AsyncClient(host=config.OLLAMA_HOST)

    async def warm(self) -> None:
        """Load the model into memory so the first gesture doesn't pay for it."""
        await self._client.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 1},
            keep_alive=-1,
        )

    def _messages(self, descriptor: str) -> list[dict]:
        msgs = [{"role": "system", "content": self.persona.text}]
        for desc, utt in self.history[-config.CONTEXT_PAIRS:]:
            msgs.append({"role": "user", "content": desc})
            msgs.append({"role": "assistant", "content": utt})
        msgs.append({"role": "user", "content": descriptor})
        return msgs

    async def _generate(self, descriptor: str) -> str:
        out = []
        async for part in await self._client.chat(
                model=config.OLLAMA_MODEL,
                messages=self._messages(descriptor),
                stream=True,
                keep_alive=-1,
                options={"temperature": config.TEMPERATURE, "num_predict": 60},
        ):
            out.append(part["message"]["content"])
        return _clean("".join(out))

    async def say(self, descriptor: str) -> tuple[str, bool]:
        """Returns (utterance, was_retried)."""
        self.persona.reload()
        text = await self._generate(descriptor)

        recent = [u for _, u in self.history[-config.CONTEXT_PAIRS:]]
        retried = False
        if any(_overlap(text, prev) >= config.OVERLAP_THRESHOLD for prev in recent):
            retried = True
            text = await self._generate(descriptor)

        self.history.append((descriptor, text))
        return text, retried
