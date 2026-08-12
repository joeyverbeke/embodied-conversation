"""A page that shows what the device would have said.

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
    feed.prepend(row);
    while (feed.children.length > 200) feed.lastChild.remove();
  }

  function connect() {
    const ws = new WebSocket('ws://' + location.host + '/feed');
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

    def __init__(self, port: int) -> None:
        self.port = port
        self._clients: set = set()
        self._sends: set = set()        # strong refs; see publish()

    async def _handler(self, ws) -> None:
        self._clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            self._clients.discard(ws)

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
