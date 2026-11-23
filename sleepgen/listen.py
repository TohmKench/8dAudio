"""The A/B listening page for `sleepgen tune`: pick, never describe.

  python -m sleepgen listen runs/page_turns

Serves a page on this computer only (127.0.0.1) that plays two clips at a time, the best 3
generated versions and a real recording in every pairing, and asks "Which sounds more like a
real page turn?": A is better, B is better, or Can't tell. Each pick is saved straight away
in <run>/listen/picks.json, so closing the page loses nothing. At the end it shows whether
the real recording still stands out. `sleepgen tune ... --resume` then learns from the picks.
"""

from __future__ import annotations

import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .eventmatch import verdict

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Which is real?</title>
<style>
:root{--bg:#f6f5f2;--fg:#1d1c1a;--muted:#6b6862;--card:#fff;--line:#dedbd4;--accent:#2f5d8a;--accent-fg:#fff;--good:#2e6b3f}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#ecebe8;--muted:#9a978f;--card:#1f1f1e;--line:#33322f;--accent:#7fb0e0;--accent-fg:#0d1a26;--good:#8fd19e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:620px;margin:0 auto;padding:32px 16px}
h1{font-size:1.4rem;margin:0 0 4px}.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin-top:20px}
.row{display:flex;gap:12px}.row>*{flex:1}
button{font:inherit;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--fg);padding:14px 10px;cursor:pointer}
button:disabled{opacity:.4;cursor:default}
.play{font-size:1.1rem;padding:22px 10px}.play.on{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent) inset}
.play.heard::after{content:" \2713";color:var(--good)}
.pick{background:var(--accent);color:var(--accent-fg);border-color:var(--accent)}
.pick.same{background:transparent;color:var(--fg);border-color:var(--line)}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:12px}.bar>i{display:block;height:100%;background:var(--accent);width:0}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:4px;padding:0 4px}
table{width:100%;border-collapse:collapse;margin-top:8px}td{padding:6px 0;border-top:1px solid var(--line)}td:last-child{text-align:right}
</style></head><body><main>
<h1 id="q">Loading…</h1>
<p class="muted">Play both, then pick. Go with your first impression. Keys: <kbd>1</kbd> <kbd>2</kbd> play, <kbd>A</kbd> <kbd>B</kbd> <kbd>S</kbd> (can't tell) pick.</p>
<div class="card" id="trial">
  <div class="muted" id="count"></div>
  <div class="row" style="margin-top:12px"><button class="play" id="pa">Play A</button><button class="play" id="pb">Play B</button></div>
  <div class="row" style="margin-top:16px"><button class="pick" id="ca" disabled>A is better</button><button class="pick same" id="cs" disabled>Can't tell</button><button class="pick" id="cb" disabled>B is better</button></div>
  <div class="bar"><i id="prog"></i></div>
</div>
<div class="card" id="done" hidden></div>
</main>
<script>
let T=null,i=0,heard={a:false,b:false};const audio=new Audio();
const $=id=>document.getElementById(id);
function play(side){const t=T.trials[i];audio.pause();audio.src="clips/"+(side==="a"?t.a_file:t.b_file);audio.currentTime=0;audio.play();
 $("pa").classList.toggle("on",side==="a");$("pb").classList.toggle("on",side==="b");heard[side]=true;
 $(side==="a"?"pa":"pb").classList.add("heard");if(heard.a&&heard.b)["ca","cb","cs"].forEach(x=>$(x).disabled=false)}
audio.onended=()=>{$("pa").classList.remove("on");$("pb").classList.remove("on")};
function show(){if(i>=T.trials.length){finish();return}heard={a:false,b:false};
 ["pa","pb"].forEach(x=>$(x).classList.remove("heard","on"));["ca","cb","cs"].forEach(x=>$(x).disabled=true);
 $("count").textContent="Pair "+(i+1)+" of "+T.trials.length;$("prog").style.width=(100*i/T.trials.length)+"%"}
async function pick(choice){if($("ca").disabled)return;audio.pause();const t=T.trials[i];
 await fetch("pick",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({trial:t.id,choice})});i++;show()}
async function finish(){$("trial").hidden=true;const v=await (await fetch("verdict")).json();const d=$("done");d.hidden=false;
 const names={c1:"Version 1",c2:"Version 2",c3:"Version 3"};let rows="";
 for(const [k,r] of Object.entries(v.real_beats).sort()) rows+="<tr><td>Real recording vs "+names[k]+"</td><td>real preferred "+Math.round(100*r)+"%</td></tr>";
 d.innerHTML="<h2 style='margin-top:0'>"+(v.done?"Done: the real one no longer stands out":"Thanks: the real one still stands out")+"</h2>"+
 "<p class='muted'>"+(v.done?"The best version held its own against the real recording.":"The next round learns from your picks. You can close this page.")+"</p><table>"+rows+"</table>"}
$("pa").onclick=()=>play("a");$("pb").onclick=()=>play("b");
$("ca").onclick=()=>pick("a");$("cb").onclick=()=>pick("b");$("cs").onclick=()=>pick("same");
document.onkeydown=e=>{const k=e.key.toLowerCase();if(k==="1")play("a");else if(k==="2")play("b");else if(k==="a")pick("a");else if(k==="b")pick("b");else if(k==="s")pick("same")};
(async()=>{T=await (await fetch("trials.json")).json();const done=await (await fetch("picks.json")).json();
 $("q").textContent=T.question;const ids=new Set(done.map(p=>p.trial));while(i<T.trials.length&&ids.has(T.trials[i].id))i++;show()})();
</script></body></html>
"""


def _paths(run_dir):
    d = os.path.join(run_dir, "listen")
    return d, os.path.join(d, "trials.json"), os.path.join(d, "picks.json")


def load_picks(run_dir):
    _, _, pp = _paths(run_dir)
    if not os.path.isfile(pp):
        return []
    with open(pp, encoding="utf-8") as fh:
        return json.load(fh)


def record_pick(run_dir, trial_id, choice):
    """Save one pick (replacing an earlier one for the same pair). The page sends only the
    pair's number and the choice; which clip was which comes from trials.json."""
    if choice not in ("a", "b", "same"):
        raise ValueError("choice must be a, b or same")
    d, tp, pp = _paths(run_dir)
    with open(tp, encoding="utf-8") as fh:
        trials = {t["id"]: t for t in json.load(fh)["trials"]}
    t = trials[int(trial_id)]
    picks = [p for p in load_picks(run_dir) if p["trial"] != t["id"]]
    picks.append({"trial": t["id"], "a": t["a"], "b": t["b"], "choice": choice, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    tmp = pp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(picks, fh, indent=1)
    os.replace(tmp, pp)
    return picks


def make_server(run_dir, port=8765):
    d, tp, _ = _paths(run_dir)
    if not os.path.isfile(tp):
        raise SystemExit(f"No listening set in {d}: run `sleepgen tune` first.")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):  # noqa: N802 - http.server API
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/trials.json":
                with open(tp, "rb") as fh:
                    data = json.load(fh)
                # The page needs only the clips and the question, not which clip is which.
                public = {"question": data["question"],
                          "trials": [{"id": t["id"], "a_file": t["a_file"], "b_file": t["b_file"]}
                                     for t in data["trials"]]}
                return self._json(public)
            if path == "/picks.json":
                return self._json([{"trial": p["trial"]} for p in load_picks(run_dir)])
            if path == "/verdict":
                return self._json(verdict(load_picks(run_dir)))
            if path.startswith("/clips/"):
                name = os.path.basename(path[len("/clips/"):])
                fp = os.path.join(d, "clips", name)
                if name.endswith(".wav") and os.path.isfile(fp):
                    with open(fp, "rb") as fh:
                        return self._send(200, fh.read(), "audio/wav")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802 - http.server API
            if self.path.split("?")[0] != "/pick":
                return self._send(404, b"not found", "text/plain")
            try:
                n = min(int(self.headers.get("Content-Length", 0)), 10_000)
                body = json.loads(self.rfile.read(n) or b"{}")
                picks = record_pick(run_dir, body["trial"], body["choice"])
            except (ValueError, KeyError, TypeError) as e:
                return self._json({"error": str(e)}, 400)
            return self._json({"saved": len(picks)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(run_dir, port=8765, open_browser=True):
    srv = make_server(run_dir, port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"Listening page: {url}  (Ctrl+C to stop; picks are saved as you go)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return load_picks(run_dir)
