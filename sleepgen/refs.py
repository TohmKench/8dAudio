"""Reference recordings for `sleepgen tune`: real sounds to measure generated ones against.

  python -m sleepgen refs fetch page_turns --query "page turn" -n 10
  python -m sleepgen refs check references/page_turns

`fetch` searches Freesound for CC0 (public domain) recordings only, downloads their
high-quality previews, and writes `sources.json` next to them with each one's author and
link, so the provenance is kept even though CC0 asks for none. It needs a free Freesound API
key (freesound.org/apiv2/apply) in FREESOUND_API_KEY, in a `.env` file, or via --key. The
key is never written anywhere by this module.

Your own recordings work too: drop them in the folder (any format ffmpeg reads).
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

API = "https://freesound.org/apiv2/search/text/"
CC0 = "publicdomain/zero"  # Freesound's CC0 licence URL contains this
AGENT = "sleepgen-reference-fetch/1.0"


def api_key(explicit=None, env_file=".env"):
    if explicit:
        return explicit
    if os.environ.get("FREESOUND_API_KEY"):
        return os.environ["FREESOUND_API_KEY"]
    for path in (env_file, os.path.join(os.path.expanduser("~"), "sleepgen", ".env")):
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    k, _, v = line.strip().partition("=")
                    if k.strip() == "FREESOUND_API_KEY" and v.strip():
                        return v.strip().strip('"').strip("'")
    return None


def _get(url, opener=None):
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with (opener or urllib.request.urlopen)(req, timeout=60) as r:
        return r.read()


def search(query, key, n=10, min_s=0.3, max_s=30.0, opener=None):
    """CC0 results for `query`, best rated first, at most n."""
    params = {
        "query": query,
        "filter": f'license:"Creative Commons 0" duration:[{min_s} TO {max_s}]',
        "fields": "id,name,username,license,url,previews,duration,avg_rating,num_downloads",
        "sort": "rating_desc",
        "page_size": str(min(150, max(n * 3, 15))),
        "token": key,
    }
    data = json.loads(_get(API + "?" + urllib.parse.urlencode(params), opener))
    out = []
    for r in data.get("results", []):
        if CC0 not in str(r.get("license", "")):  # belt and braces: CC0 only, whatever the filter did
            continue
        if not (r.get("previews") or {}).get("preview-hq-mp3"):
            continue
        out.append(r)
        if len(out) >= n:
            break
    return out


def fetch(query, out_dir, n=10, key=None, opener=None, log=print):
    """Download up to n CC0 recordings for `query` into out_dir. Returns the sources list."""
    key = api_key(key)
    if not key:
        raise SystemExit("No Freesound API key: get a free one at https://freesound.org/apiv2/apply "
                         "and put FREESOUND_API_KEY=... in a .env file (or pass --key).")
    os.makedirs(out_dir, exist_ok=True)
    src_path = os.path.join(out_dir, "sources.json")
    sources = json.load(open(src_path, encoding="utf-8")) if os.path.isfile(src_path) else []
    have = {s.get("id") for s in sources}
    for r in search(query, key, n, opener=opener):
        if r["id"] in have:
            continue
        fn = f"freesound_{r['id']}.mp3"
        with open(os.path.join(out_dir, fn), "wb") as fh:
            fh.write(_get(r["previews"]["preview-hq-mp3"], opener))
        sources.append({"file": fn, "id": r["id"], "name": r.get("name"), "author": r.get("username"),
                        "url": r.get("url"), "license": r.get("license"), "duration": r.get("duration"),
                        "source": "freesound", "query": query})
        log(f"  {fn}: {r.get('name')!r} by {r.get('username')} ({r.get('duration', 0):.1f} s, CC0)")
    with open(src_path, "w", encoding="utf-8") as fh:
        json.dump(sources, fh, indent=1)
    return sources


def check(folder, write_cuts=True, log=print):
    """Cut every recording in the folder into events and report them; with write_cuts, save
    each event as a WAV in <folder>/_cut so you can hear what the loop measures against."""
    import numpy as np

    from .eventmatch import Reference, write_wav
    ref = Reference.from_folder(folder)
    log(f"{len(ref.events)} events from {len({n.split('#')[0] for n in ref.names})} recordings; "
        f"room noise {ref.snr_db:.0f} dB under the events")
    if write_cuts:
        cut = os.path.join(folder, "_cut")
        os.makedirs(cut, exist_ok=True)
        for name, ev in zip(ref.names, ref.clips):
            safe = name.replace("#", "_").rsplit(".", 1)[0] + "_" + name.split("#")[1]
            y = ev / (np.max(np.abs(ev)) + 1e-9) * 0.5
            write_wav(os.path.join(cut, safe + ".wav"), np.stack([y, y], axis=1))
    for name, ev in zip(ref.names, ref.events):
        log(f"  {name}: {len(ev) / 48000:.2f} s")
    return ref
