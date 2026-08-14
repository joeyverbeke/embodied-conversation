"""One page: what it said, what you did, and what you make of it.

The review used to be a separate tool over past sessions, and that was the wrong
shape. Nobody remembers a movement from yesterday, so judging one is guesswork —
and the physical loop it forced (play, stop, open another page, scroll back) is
long enough that the memory is gone before you arrive.

The real constraint is that both hands are on the ball. Nothing can be typed or
clicked mid-movement; the most that fits in the pause between two gestures is a
single keypress. So the loop is two phases and this page serves both:

    playing   — glance, hit one key to rate what it just said
    stopped   — set the ball down, type corrections on the ones it got wrong

The second phase works because it happens two minutes later rather than two days
later, and because every movement is drawn. The drawing is what makes a gesture
recognisable after the fact; recency is what makes the judgement trustworthy.

Ratings travel back over the same socket and append to logs/review.jsonl.


Testing by ear is slow. Each utterance takes several seconds to speak, the
device is busy for all of them, and gestures made during that window are
dropped — so hearing the machine costs more time than making the movement does.
Reading it removes that entirely.

It is a page rather than terminal output because both hands are on the ball. You
cannot watch a scrolling terminal while moving an object around, but you can
glance at a second screen or a phone propped up nearby.

    python -m host.server --no-audio          # read instead of listen
    python -m host.server --no-audio --view   # ...on http://localhost:8770

One file, no build step, no dependencies beyond the websockets library already
carrying the device link.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from websockets.asyncio.server import serve

from . import config
from websockets.datastructures import Headers
from websockets.http11 import Response

log = logging.getLogger("gradi.view")

PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>gradi — what it would say</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 1.5rem;
    background: #0f0f11; color: #e8e8ea;
    font: 16px/1.5 ui-sans-serif, -apple-system, system-ui, sans-serif;
  }
  header {
    display: flex; align-items: baseline; gap: .75rem;
    padding-bottom: 1rem; border-bottom: 1px solid #2a2a30; margin-bottom: 1rem;
  }
  h1 { font-size: .95rem; font-weight: 600; margin: 0; letter-spacing: .02em; }
  #status { font-size: .8rem; color: #8a8a94; }
  #state {
    font: .72rem/1 ui-monospace, Menlo, monospace; letter-spacing: .08em;
    padding: .2rem .45rem; border-radius: .3rem;
    background: #23232b; color: #8a8a94;
  }
  #state.HELD   { background: #1f3350; color: #9dc4f5; }
  #state.MOVING { background: #1f4030; color: #86e0ac; }
  #bar { font: .72rem/1 ui-monospace, Menlo, monospace; color: #6a6a74; }
  #status.live::before {
    content: ""; display: inline-block; width: .5rem; height: .5rem;
    border-radius: 50%; background: #4ade80; margin-right: .4rem;
    vertical-align: middle;
  }
  #status.down::before {
    content: ""; display: inline-block; width: .5rem; height: .5rem;
    border-radius: 50%; background: #f87171; margin-right: .4rem;
    vertical-align: middle;
  }
  #clear {
    margin-left: auto; background: none; border: 1px solid #2a2a30;
    color: #8a8a94; border-radius: .35rem; padding: .25rem .6rem;
    font: inherit; font-size: .8rem; cursor: pointer;
  }
  #clear:hover { color: #e8e8ea; border-color: #4a4a54; }
  .row {
    padding: .85rem 0; border-bottom: 1px solid #1c1c22;
    animation: in .25s ease-out;
  }
  @keyframes in { from { opacity: 0; transform: translateY(-4px); } }
  .said { font-size: 1.35rem; line-height: 1.35; }
  .desc {
    font: .82rem/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
    color: #7c7c88; margin-top: .4rem; white-space: pre-wrap;
  }
  .meta { font-size: .72rem; color: #55555f; margin-top: .35rem; }
  .row.dropped { opacity: .62; }
  .row.dropped .said { color: #6a6a74; font-style: italic; font-size: 1rem; }
  .score {
    font: .72rem/1.4 ui-monospace, Menlo, monospace; margin-top: .35rem;
    color: #6a6a74;
  }
  .score b { color: #b8b8c2; font-weight: 600; }
  .score.win b { color: #86e0ac; }
  .row.stillness .said { color: #93b8e8; }
  .pic { margin: .5rem 0 .2rem; max-width: 620px; }
  .rate { display: flex; gap: .35rem; align-items: center; margin-top: .5rem;
          flex-wrap: wrap; }
  .rate button {
    background: #20202a; border: 1px solid #2e2e38; color: #9a9aa6;
    border-radius: .35rem; padding: .2rem .5rem; font: inherit;
    font-size: .74rem; cursor: pointer;
  }
  .rate button:hover { color: #fff; border-color: #4a4a58; }
  .rate button.on { background: #2b4a35; border-color: #3f7a51; color: #c9f5d8; }
  .rate button.bad.on { background: #4a2b2b; border-color: #7a3f3f; color: #f5c9c9; }
  .rate .lbl { font-size: .7rem; color: #4a4a54; margin-right: .15rem; }
  .fix {
    width: 100%; max-width: 620px; margin-top: .45rem; background: #101014;
    border: 1px solid #2e2e38; color: #e8e8ea; border-radius: .35rem;
    padding: .4rem .55rem; font: inherit; font-size: .85rem;
  }
  .row.latest { border-left: 2px solid #3f6ea8; padding-left: .7rem; }
  .judged { color: #86e0ac; font-size: .7rem; margin-left: .3rem; }
  .empty { color: #55555f; font-style: italic; padding-top: 2rem; }
</style>
<header>
  <h1>what it would say</h1>
  <span id="status" class="down">connecting</span>
  <span id="state" class="DORMANT">DORMANT</span>
  <span id="bar">bar —</span>
  <button id="clear">clear</button>
</header>
<div id="feed"><div class="empty">Move the ball.</div></div>
<script>
  const feed = document.getElementById('feed');
  const status = document.getElementById('status');
  const stateEl = document.getElementById('state');
  const barEl = document.getElementById('bar');
  document.getElementById('clear').onclick = () => feed.innerHTML = '';

  function add(e) {
    const empty = feed.querySelector('.empty');
    if (empty) empty.remove();
    const row = document.createElement('div');
    row.className = 'row' + (e.responded ? '' : ' dropped')
                  + (e.kind === 'stillness' ? ' stillness' : '');
    const said = document.createElement('div');
    said.className = 'said';
    said.textContent = e.utterance || ('(not spoken \\u2014 ' + (e.reason || 'skipped') + ')');
    const desc = document.createElement('div');
    desc.className = 'desc';
    desc.textContent = e.descriptor || '';
    if (e.picture) {
      const pic = document.createElement('div');
      pic.className = 'pic';
      pic.innerHTML = e.picture;
      row.append(pic);
    }
    row.append(said, desc);
    if (e.score != null) {
      // The score and the bar it faced. Once silence is the main behaviour a
      // broken policy and a working one look identical without this.
      const sc = document.createElement('div');
      sc.className = 'score' + (e.responded ? ' win' : '');
      sc.innerHTML = 'salience <b>' + e.score.toFixed(2) + '</b> vs bar <b>'
                   + (e.bar != null ? e.bar.toFixed(2) : '?') + '</b>'
                   + (e.why ? '  \u2014 ' + e.why : '');
      row.append(sc);
    }
    if (e.ms) {
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = e.ms + ' ms';
      row.append(meta);
    }
    // Rating lives on the row itself, so the thing being judged and the
    // judgement are never more than a glance apart.
    if (e.descriptor) {
      const rate = document.createElement('div');
      rate.className = 'rate';
      rate.innerHTML =
        '<span class="lbl">right?</span>'
      + '<button class="bad" data-g="accuracy" data-v="wrong">1 wrong</button>'
      + '<button data-g="accuracy" data-v="vague">2 vague</button>'
      + '<button data-g="accuracy" data-v="right">3 yes</button>'
      + '<span class="lbl" style="margin-left:.6rem">speak?</span>'
      + '<button data-g="timing" data-v="yes">4 yes</button>'
      + '<button class="bad" data-g="timing" data-v="no">5 no</button>';
      const fix = document.createElement('input');
      fix.className = 'fix';
      fix.placeholder = 'what should it have said?';
      fix.onkeydown = ev => {
        if (ev.key !== 'Enter') return;
        send(row, {should_have_said: fix.value.trim()});
        fix.blur();
      };
      rate.onclick = ev => {
        const b = ev.target.closest('button'); if (!b) return;
        [...rate.querySelectorAll('button')].forEach(o => {
          if (o.dataset.g === b.dataset.g) o.classList.toggle('on', o === b);
        });
        send(row, {[b.dataset.g]: b.dataset.v});
      };
      row.append(rate, fix);
      row._event = e;
    }
    [...feed.querySelectorAll('.latest')].forEach(r => r.classList.remove('latest'));
    row.classList.add('latest');
    feed.prepend(row);
    while (feed.children.length > 60) feed.lastChild.remove();
  }

  let sock = null;

  function send(row, patch) {
    // Merge, so rating accuracy then timing then typing a correction is three
    // keystrokes rather than three separate records to reconcile later.
    row._judgement = Object.assign({}, row._judgement, patch);
    const e = row._event || {};
    if (sock && sock.readyState === 1) {
      sock.send(JSON.stringify(Object.assign({
        judge: true, descriptor: e.descriptor, utterance: e.utterance,
        spoke: !!e.responded, salience: e.score, bar: e.bar, kind: e.kind,
      }, row._judgement)));
    }
    if (!row.querySelector('.judged')) {
      const tick = document.createElement('span');
      tick.className = 'judged';
      tick.textContent = 'saved';
      row.querySelector('.rate').append(tick);
    }
  }

  // Number keys rate the newest row, so a movement can be judged one-handed
  // without looking away from what you are holding.
  document.addEventListener('keydown', ev => {
    if (ev.target.tagName === 'INPUT') return;
    const map = {'1':['accuracy','wrong'],'2':['accuracy','vague'],
                 '3':['accuracy','right'],'4':['timing','yes'],'5':['timing','no']};
    const m = map[ev.key];
    if (!m) return;
    const row = feed.querySelector('.latest');
    if (!row || !row._event) return;
    const b = row.querySelector('[data-g="'+m[0]+'"][data-v="'+m[1]+'"]');
    if (b) b.click();
    ev.preventDefault();
  });

  function connect() {
    const ws = new WebSocket('ws://' + location.host + '/feed');
    sock = ws;
    ws.onopen = () => { status.className = 'live'; status.textContent = 'live'; };
    ws.onmessage = m => {
      const e = JSON.parse(m.data);
      if (e.state) { stateEl.textContent = e.state; stateEl.className = e.state; }
      if (e.bar != null) barEl.textContent = 'bar ' + e.bar.toFixed(2);
      if (e.tick) return;               // state-only ping, nothing to list
      add(e);
    };
    ws.onclose = () => {
      status.className = 'down'; status.textContent = 'reconnecting';
      setTimeout(connect, 1000);
    };
    ws.onerror = () => ws.close();
  }
  connect();
</script>
"""


class View:
    """Fan out one event to every open page. Never blocks the pipeline."""

    def __init__(self, port: int, judgements=None) -> None:
        self.port = port
        self.judgements = judgements or (config.LOG_DIR / "review.jsonl")
        self._clients: set = set()
        self._sends: set = set()        # strong refs; see publish()

    async def _handler(self, ws) -> None:
        self._clients.add(ws)
        try:
            async for message in ws:
                try:
                    payload = json.loads(message)
                except (json.JSONDecodeError, TypeError):
                    continue
                if payload.pop("judge", False):
                    self._record(payload)
        except Exception:
            pass
        finally:
            self._clients.discard(ws)

    def _record(self, judgement: dict) -> None:
        """Append one judgement. Never let a bad one take the server with it."""
        judgement["t"] = time.time()
        try:
            self.judgements.parent.mkdir(parents=True, exist_ok=True)
            with open(self.judgements, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(judgement, separators=(",", ":")) + "\n")
        except OSError as exc:
            log.warning("could not save judgement: %s", exc)

    def _http(self, connection, request):
        """Serve the page for anything that is not the feed socket."""
        if request.path.rstrip("/") in ("", "/feed".rstrip("/")) and \
                request.path.startswith("/feed"):
            return None                     # let the WebSocket handshake proceed
        body = PAGE.encode()
        return Response(200, "OK", Headers({
            "Content-Type": "text/html; charset=utf-8",
            "Content-Length": str(len(body)),
        }), body)

    async def serve(self) -> None:
        async with serve(self._handler, "127.0.0.1", self.port,
                         process_request=self._http):
            log.info("view on http://localhost:%d", self.port)
            await asyncio.Future()

    def publish(self, **event) -> None:
        """Fire and forget. A stalled browser must never stall a gesture."""
        if not self._clients:
            return
        event.setdefault("t", time.time())
        payload = json.dumps(event)
        for ws in list(self._clients):
            # Keep a reference until it finishes. An unreferenced task may be
            # garbage collected before it ever sends.
            task = asyncio.create_task(self._send(ws, payload))
            self._sends.add(task)
            task.add_done_callback(self._sends.discard)

    async def _send(self, ws, payload: str) -> None:
        try:
            await ws.send(payload)
        except Exception:
            self._clients.discard(ws)
