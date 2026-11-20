"""Turns a recipe into a long audio file, streamed through ffmpeg in chunks.

Memory use stays flat whatever the length: a 12 hour render holds only one
chunk (10 s) in memory at a time.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import layers as L
from .dsp import SR, Leveler, MatchEQ, Reverb, equal_power_pan, loudness_lufs, lowpass, smooth_lfo, soft_limit

DEFAULT_TARGET_LUFS = -26.0  # quiet and steady for sleep; YouTube never turns quiet audio up


@dataclass
class Voice:
    """A layer plus its level, slow drift and movement."""

    layer: L.Layer
    level_db: float = 0.0
    pan: dict | None = None
    drift: dict | None = None
    gate: dict | None = None
    send: float = 0.0  # how much goes into the shared reverb (0..1)
    dry: float = 1.0  # direct sound; lower = further away
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    pos: int = 0

    def __post_init__(self):
        self.gain = 10 ** (self.level_db / 20)
        d = self.drift or {}
        self.drift_db = float(d.get("depth_db", 0.0))
        base = float(d.get("period_s", 120.0))
        self.drift_periods = [base, base * 0.37, base * 2.3]
        self.drift_phases = self.rng.uniform(0, 2 * np.pi, 3)
        p = self.pan or {}
        self.pan_mode = p.get("mode", "none") if self.layer.pannable else "none"
        self.pan_period = float(p.get("period_s", 60.0))
        self.pan_width = float(p.get("width", 0.5))
        self.pan_phase = self.rng.uniform(0, 2 * np.pi)
        if self.pan_mode == "circle":
            self.behind_lp = lowpass(2200)
        self.gate_env = np.zeros(0)
        g = self.gate or {}
        self.until = int(SR * g["until_s"]) if g.get("until_s") else None
        self.until_fade = int(SR * g.get("until_fade_s", 300))
        if g.get("on_s"):
            self.gate_env = np.zeros(int(SR * g.get("start_s", 0)))

    def _gate(self, n: int) -> np.ndarray:
        """Accent timing: on for a while, off for a while, with soft edges."""
        g = self.gate
        while len(self.gate_env) < n:
            on = int(SR * self.rng.uniform(*g["on_s"]))
            off = int(SR * self.rng.uniform(*g["off_s"]))
            fade = min(int(SR * g.get("fade_s", 3.0)), on // 2)
            seg = np.ones(on)
            ramp = (1 - np.cos(np.linspace(0, np.pi, fade))) / 2
            seg[:fade] = ramp
            seg[on - fade:] = ramp[::-1]
            self.gate_env = np.concatenate([self.gate_env, seg, np.zeros(off)])
        e, self.gate_env = self.gate_env[:n], self.gate_env[n:]
        return e

    def __call__(self, n: int) -> np.ndarray:
        start = self.pos
        if self.until is not None and start >= self.until:
            self.pos += n
            return np.zeros((n, 2))  # finished: costs nothing for the rest of the night
        x = self.layer(n) * self.gain
        t = (self.pos + np.arange(n)) / SR
        self.pos += n
        if self.gate and self.gate.get("on_s"):
            x = x * self._gate(n)[:, None]
        if self.until is not None:
            left = (self.until - (start + np.arange(n))) / max(self.until_fade, 1)
            x = x * ((1 - np.cos(np.pi * np.clip(left, 0, 1))) / 2)[:, None]
        if self.drift_db:
            db = self.drift_db * smooth_lfo(t, self.drift_periods, self.drift_phases, [1.0, 0.5, 0.7])
            x = x * (10 ** (db / 20))[:, None]
        if self.pan_mode in ("sweep", "circle"):
            angle = 2 * np.pi * t / self.pan_period + self.pan_phase
            gl, gr = equal_power_pan(self.pan_width * np.sin(angle))
            if self.pan_mode == "circle":
                # Sound passing behind the head: a little duller and quieter.
                behind = np.clip(-np.cos(angle), 0, 1)[:, None] * 0.6
                x = (1 - behind) * x + behind * self.behind_lp(x)
                x = x * (1 - 0.2 * behind)
            x = x * np.stack([gl, gr], axis=1)
        return x


def build_voices(recipe: dict) -> list[Voice]:
    seq = np.random.SeedSequence(int(recipe.get("seed", 0)))
    children = seq.spawn(len(recipe["layers"]) * 2)
    voices = []
    for i, spec in enumerate(recipe["layers"]):
        kind = spec["type"]
        if kind not in L.LAYER_TYPES:
            raise ValueError(f"unknown layer type {kind!r}")
        layer = L.LAYER_TYPES[kind](np.random.default_rng(children[2 * i]), **spec.get("params", {}))
        voices.append(Voice(layer, spec.get("level_db", 0.0), spec.get("pan"), spec.get("drift"),
                            spec.get("gate"), float(spec.get("send", 0.0)), float(spec.get("dry", 1.0)),
                            np.random.default_rng(children[2 * i + 1])))
    return voices


class Mix:
    """Sums the voices and applies the master chain."""

    def __init__(self, recipe: dict):
        self.voices = build_voices(recipe)
        m = recipe.get("master", {})
        self.lp = lowpass(m["lowpass_hz"], order=2) if m.get("lowpass_hz") else None
        self.target = float(m.get("target_lufs", DEFAULT_TARGET_LUFS))
        self.fade_in = float(m.get("fade_in_s", 20.0))
        self.fade_out = float(m.get("fade_out_s", 45.0))
        rv = m.get("reverb")
        rv_rng = np.random.default_rng(int(recipe.get("seed", 0)) + 7919)
        self.reverb = Reverb(rv_rng, **{k: rv[k] for k in ("rt60", "predelay_ms", "damping") if k in rv}) \
            if rv else None
        self.wet = float(rv.get("level", 1.0)) if rv else 0.0
        self.leveler = Leveler(self.target) if m.get("leveler", True) else None
        self.eq = MatchEQ(m["eq"]) if m.get("eq") else None
        self.gain = 1.0

    def raw(self, n: int) -> np.ndarray:
        dry = np.zeros((n, 2))
        send = np.zeros((n, 2))
        for v in self.voices:
            x = v(n)
            dry += x * v.dry
            if v.send:
                send += x * v.send
        if self.reverb is not None:
            dry += self.reverb(send) * self.wet
        if self.eq is not None:
            dry = self.eq(dry)
        return self.lp(dry) if self.lp else dry

    def master(self, n: int) -> np.ndarray:
        """Final-level audio: raw mix x calibrated gain, eased by the leveler."""
        x = self.raw(n) * self.gain
        return self.leveler(x) if self.leveler else x

    def calibrate(self, seconds: float = 120.0) -> float:
        """Measure a stretch of the mix and set the gain that hits the target loudness."""
        block = np.concatenate([self.raw(SR * 10) for _ in range(int(seconds // 10))])
        measured = loudness_lufs(block)
        self.gain = 10 ** ((self.target - measured) / 20)
        return measured


def _ffmpeg_cmd(out_path: str, bitrate: str) -> list[str]:
    ext = os.path.splitext(out_path)[1].lower()
    codec = {
        ".m4a": ["-c:a", "aac", "-b:a", bitrate],
        ".aac": ["-c:a", "aac", "-b:a", bitrate],
        ".mp3": ["-c:a", "libmp3lame", "-b:a", bitrate],
        ".opus": ["-c:a", "libopus", "-b:a", bitrate],
        ".flac": ["-c:a", "flac", "-sample_fmt", "s16"],
        ".wav": ["-c:a", "pcm_s16le"],
    }.get(ext)
    if codec is None:
        raise ValueError(f"unsupported output type {ext!r} (use .m4a, .mp3, .opus, .flac or .wav)")
    return ["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "pipe:0",
            *codec, out_path]


class _Writer:
    """Writes blocks to a pipe from a background thread (a few blocks of buffering)."""

    def __init__(self, pipe, depth: int = 3):
        self.pipe = pipe
        self.q = queue.Queue(maxsize=depth)
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            block = self.q.get()
            if block is None:
                return
            if self.error is None:
                try:
                    self.pipe.write(block)
                except OSError as e:  # ffmpeg died; keep draining so put() never blocks
                    self.error = e

    def put(self, block: bytes):
        if self.error is not None:
            raise RuntimeError(f"ffmpeg stopped accepting audio: {self.error}")
        self.q.put(block)

    def close(self):
        """Finish writing everything queued."""
        self.q.put(None)
        self.thread.join()

    def stop(self):
        if self.thread.is_alive():
            self.q.put(None)
            self.thread.join()


def render(recipe: dict, out_path: str, seconds: float, chunk_s: float = 10.0,
           bitrate: str = "384k", progress: bool = True) -> dict:
    """Render `seconds` of audio for `recipe` to `out_path`. Returns a render report."""
    started = time.time()
    # Measure loudness on a throwaway copy so the real render starts from t=0
    # (otherwise the first thunder roll etc. would be spent on the measurement).
    measured = Mix(recipe).calibrate()
    calibrated = time.time()
    mix = Mix(recipe)
    mix.gain = 10 ** ((mix.target - measured) / 20)
    total = int(seconds * SR)
    chunk = int(chunk_s * SR)
    fi, fo = int(mix.fade_in * SR), int(mix.fade_out * SR)
    peak = 0.0
    proc = subprocess.Popen(_ffmpeg_cmd(out_path, bitrate), stdin=subprocess.PIPE)
    # Encoding runs in ffmpeg's own process; a writer thread keeps feeding it, so the next
    # chunk is synthesised while ffmpeg encodes this one (instead of taking turns).
    writer = _Writer(proc.stdin)
    try:
        done = 0
        last_report = 0.0
        while done < total:
            n = min(chunk, total - done)
            x = mix.master(n)
            idx = done + np.arange(n)
            fade = np.ones(n)
            if done < fi:
                fade = np.minimum(fade, (1 - np.cos(np.pi * np.clip(idx / fi, 0, 1))) / 2)
            if done + n > total - fo:
                fade = np.minimum(fade, (1 - np.cos(np.pi * np.clip((total - idx) / fo, 0, 1))) / 2)
            x = soft_limit(x * fade[:, None])
            peak = max(peak, float(np.max(np.abs(x))))
            writer.put(x.astype(np.float32).tobytes())
            done += n
            if progress and (time.time() - last_report > 5 or done == total):
                last_report = time.time()
                pct = 100 * done / total
                rate = done / SR / max(time.time() - started, 1e-6)
                print(f"\r  {pct:5.1f}%  {done / SR / 3600:5.2f} h rendered  ({rate:.0f}x realtime)",
                      end="", file=sys.stderr, flush=True)
        if progress:
            print(file=sys.stderr)
        writer.close()
    finally:
        writer.stop()
        try:
            proc.stdin.close()
        except OSError:
            pass
        code = proc.wait()
    if code != 0 or writer.error is not None:
        raise RuntimeError(f"ffmpeg exited with code {code}" + (f" ({writer.error})" if writer.error else ""))
    return {
        "output": out_path,
        "seconds": seconds,
        "measured_lufs_before_gain": round(float(measured), 2),
        "target_lufs": mix.target,
        "peak_dbfs": round(float(20 * np.log10(peak + 1e-12)), 2),
        "render_seconds": round(time.time() - started, 1),
        "loudness_check_seconds": round(calibrated - started, 1),
    }


def render_with_sidecar(recipe: dict, out_path: str, seconds: float, **kw) -> dict:
    """Render and write `<out>.json` holding the recipe and report (used later for titles/analytics)."""
    report = render(recipe, out_path, seconds, **kw)
    sidecar = os.path.splitext(out_path)[0] + ".json"
    with open(sidecar, "w") as f:
        json.dump({"recipe": recipe, "render": report}, f, indent=2)
    return report
