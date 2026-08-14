"""Judge what the machine said, against a picture of what actually happened.

    python -m host.review logs/session-*.jsonl

Every problem so far has been found the same way: notice something wrong while
playing, describe it, fix that one case. Each fix was right and the set never
got smaller, because there was no way to check a change except by going and
playing again. Accuracy was being optimised toward "true", when the real
criterion is "recognisable" — and those come apart badly.

This closes the loop. It replays logged movements with a drawing of the
reconstructed path, so a movement from an hour ago can be judged as easily as
one from a second ago, and records two judgements that are genuinely separate:

    was the description right?      — the words
    should it have spoken at all?   — the timing

Conflating those is why tuning has been going in circles. A perfect sentence at
the wrong moment and a clumsy one at the right moment are different failures
with different fixes, and from the outside they feel the same.

Judgements append to logs/review.jsonl, which is what any future change gets
measured against.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, draw, trajectory
from .replay import Stored, _records

JUDGEMENTS = config.LOG_DIR / "review.jsonl"


def _load(paths: list[Path]) -> list[dict]:
    """Every gesture worth judging, with its picture already drawn."""
    items = []
    for path in paths:
        for rec in _records(path):
            if not rec.get("segment") or not rec.get("features"):
                continue
            seg = Stored(rec)
            frames = seg.motion()
            if len(frames) < 3:
                continue
            feat = rec["features"]

            # A throw is the one case with genuinely no path: the ball is not
            # accelerating under anyone's control through the flight, so the
            # reconstruction is meaningless rather than merely imprecise.
            picture, kind = "", "signals"
            if not feat.get("airborne_ms"):
                built = trajectory.reconstruct(frames)
                if built is not None:
                    picture = draw.path_svg(built.position,
                                            trusted=bool(feat.get("trusted")))
                    kind = "path" if feat.get("trusted") else "path (rough)"
            picture += draw.signal_svg(frames)

            items.append({
                "id": f"{path.name}#{rec['index']}",
                "session": path.name,
                "index": rec["index"],
                "picture": picture,
                "picture_kind": kind,
                "descriptor": rec.get("descriptor") or "",
                "utterance": rec.get("utterance"),
                "spoke": bool(rec.get("responded")),
                "salience": (rec.get("timings_ms") or {}).get("salience"),
                "bar": (rec.get("timings_ms") or {}).get("bar"),
                "facts": _numbers(feat),
            })
    return items


def _numbers(f: dict) -> str:
    """The handful of figures worth checking a description against."""
    bits = [f"{f.get('duration_s', 0):.1f}s"]
    if f.get("trusted"):
        bits.append(f"span {f.get('span_cm', 0):.0f}cm")
        bits.append(f"{f.get('stroke_count', 0)} legs")
        bits.append(f"conf {f.get('confidence', 0):.2f}")
    else:
        bits.append("path untrusted")
    if f.get("airborne_ms"):
        bits.append(f"airborne {f['airborne_ms']}ms")
    bits.append(f"peak {f.get('peak_accel', 0):.1f}m/s2")
    bits.append(f"spin {f.get('spin_rate', 0):.0f}dps")
    return " · ".join(bits)


PAGE = """<!doctype html>
<meta charset="utf-8"><title>review</title>
<style>
 :root{color-scheme:dark}
 *{box-sizing:border-box}
 body{margin:0;padding:1.2rem 1.5rem 4rem;background:#0f0f11;color:#e8e8ea;
      font:15px/1.5 ui-sans-serif,-apple-system,system-ui,sans-serif}
 header{display:flex;gap:.8rem;align-items:baseline;margin-bottom:1rem}
 h1{font-size:.9rem;margin:0;font-weight:600}
 #count{font:.78rem ui-monospace,Menlo,monospace;color:#8a8a94}
 #done{font:.78rem ui-monospace,Menlo,monospace;color:#86e0ac;margin-left:auto}
 .card{background:#17171c;border:1px solid #26262e;border-radius:10px;padding:1rem}
 .said{font-size:1.3rem;margin:.2rem 0 .5rem}
 .said.mute{color:#6a6a74;font-style:italic;font-size:1.05rem}
 .desc{font:.8rem/1.5 ui-monospace,Menlo,monospace;color:#7c7c88;
       white-space:pre-wrap;margin-bottom:.5rem}
 .nums{font:.72rem ui-monospace,Menlo,monospace;color:#55555f;margin-bottom:.8rem}
 .pic{margin:.4rem 0 .9rem}
 .q{font-size:.82rem;color:#8a8a94;margin:.7rem 0 .35rem}
 .row{display:flex;gap:.4rem;flex-wrap:wrap}
 button{background:#20202a;border:1px solid #2e2e38;color:#c8c8d2;
        border-radius:.4rem;padding:.45rem .8rem;font:inherit;font-size:.85rem;
        cursor:pointer}
 button:hover{border-color:#4a4a58;color:#fff}
 button.on{background:#2b4a35;border-color:#3f7a51;color:#c9f5d8}
 button.no.on{background:#4a2b2b;border-color:#7a3f3f;color:#f5c9c9}
 input[type=text]{width:100%;background:#101014;border:1px solid #2e2e38;
   color:#e8e8ea;border-radius:.4rem;padding:.5rem .6rem;font:inherit;
   font-size:.9rem;margin-top:.5rem}
 #next{margin-top:1rem;background:#2f4f7a;border-color:#3f6ea8;color:#fff;
       padding:.55rem 1.4rem}
 .hint{font-size:.72rem;color:#4a4a54;margin-top:.6rem}
 .empty{color:#55555f;padding-top:3rem;text-align:center}
</style>
<header>
  <h1>does this describe what happened?</h1>
  <span id="count"></span>
  <span id="done"></span>
</header>
<div id="app"></div>
<script>
const items = ITEMS;
let i = 0, judged = 0;
let cur = {accuracy:null, timing:null};

function render(){
  const app = document.getElementById('app');
  if(i >= items.length){
    app.innerHTML = '<div class="empty">Done — '+judged+' judged.<br>'
      + 'Saved to logs/review.jsonl</div>';
    document.getElementById('count').textContent = '';
    return;
  }
  const it = items[i];
  document.getElementById('count').textContent = (i+1)+' / '+items.length;
  document.getElementById('done').textContent = judged ? judged+' judged' : '';
  cur = {accuracy:null, timing:null};
  app.innerHTML = `
   <div class="card">
     <div class="pic">${it.picture}</div>
     <div class="${it.spoke ? 'said' : 'said mute'}">${
        it.utterance ? esc(it.utterance) : '(stayed silent)'}</div>
     <div class="desc">${esc(it.descriptor)}</div>
     <div class="nums">${esc(it.facts)}${
        it.salience!=null ? ' · salience '+it.salience.toFixed(2)
                          + (it.bar!=null?' vs bar '+it.bar.toFixed(2):'') : ''}</div>

     <div class="q">Does the description match the picture?</div>
     <div class="row" id="acc">
       <button data-v="wrong" class="no">1 &nbsp;wrong</button>
       <button data-v="vague">2 &nbsp;true but vague</button>
       <button data-v="right">3 &nbsp;that's it</button>
     </div>

     <div class="q">Should it have spoken here?</div>
     <div class="row" id="tim">
       <button data-v="yes">4 &nbsp;yes</button>
       <button data-v="no" class="no">5 &nbsp;no</button>
       <button data-v="either">6 &nbsp;either way</button>
     </div>

     <input type="text" id="fix" placeholder="What should it have said? (optional)">
     <button id="next">save &amp; next &nbsp;↵</button>
     <div class="hint">keys 1–6 to rate · Enter to save · S to skip</div>
   </div>`;

  for(const g of ['acc','tim']){
    document.getElementById(g).onclick = e => {
      const b = e.target.closest('button'); if(!b) return;
      pick(g, b.dataset.v);
    };
  }
  document.getElementById('next').onclick = save;
}
function esc(s){const d=document.createElement('div');d.textContent=s||'';return d.innerHTML;}
function pick(group, v){
  const box = document.getElementById(group);
  [...box.querySelectorAll('button')].forEach(b =>
    b.classList.toggle('on', b.dataset.v === v));
  if(group==='acc') cur.accuracy = v; else cur.timing = v;
}
function save(){
  const it = items[i];
  const fix = document.getElementById('fix').value.trim();
  if(!cur.accuracy && !cur.timing && !fix){ i++; render(); return; }
  fetch('/judge', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id:it.id, session:it.session, index:it.index,
      descriptor:it.descriptor, utterance:it.utterance, spoke:it.spoke,
      salience:it.salience, accuracy:cur.accuracy, timing:cur.timing,
      should_have_said:fix})});
  judged++; i++; render();
}
document.addEventListener('keydown', e => {
  if(e.target.tagName === 'INPUT'){ if(e.key==='Enter') save(); return; }
  const m = {'1':['acc','wrong'],'2':['acc','vague'],'3':['acc','right'],
             '4':['tim','yes'],'5':['tim','no'],'6':['tim','either']}[e.key];
  if(m){ pick(m[0], m[1]); e.preventDefault(); }
  else if(e.key === 'Enter') save();
  else if(e.key.toLowerCase() === 's'){ i++; render(); }
});
render();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    items: list = []

    def log_message(self, *a):            # keep the terminal readable
        return

    def do_GET(self):
        body = PAGE.replace("ITEMS", json.dumps(self.items)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            payload = {}
        JUDGEMENTS.parent.mkdir(parents=True, exist_ok=True)
        with open(JUDGEMENTS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.send_response(204)
        self.end_headers()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--spoken-only", action="store_true",
                    help="only judge what it actually said")
    ap.add_argument("--silent-only", action="store_true",
                    help="only judge what it withheld — the timing question")
    args = ap.parse_args()

    paths = [p for p in args.logs if p.is_file()]
    items = _load(paths)
    if args.spoken_only:
        items = [i for i in items if i["spoke"]]
    if args.silent_only:
        items = [i for i in items if not i["spoke"]]
    if not items:
        print("nothing to review", file=sys.stderr)
        return 1

    Handler.items = items
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"{len(items)} movements to review — {url}")
    print(f"judgements append to {JUDGEMENTS}")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
