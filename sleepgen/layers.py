"""Sound layers. Each one is an endless stereo stream with roughly unit RMS.

A layer is called with a frame count and returns an array shaped (n, 2).
Levels, panning and slow drift are applied on top by the engine, so layers
only need to sound right on their own.
"""

from __future__ import annotations

import subprocess

import numpy as np

from scipy import fft as sp_fft
from scipy import signal

from .dsp import SOS, SR, Noise, bandpass, highpass, lowpass, rms


def child(rng: np.random.Generator) -> np.random.Generator:
    """An independent generator, so each noise source keeps its own stream."""
    return np.random.default_rng(int(rng.integers(2**63)))


class Layer:
    pannable = True  # binaural tones must keep their exact left/right split
    continuous = True  # False for layers that are mostly silence (thunder)

    def __call__(self, n: int) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def _unit(self, render, seconds: float = 4.0) -> float:
        """Gain that brings `render`'s output to unit RMS (advances its state)."""
        return 1.0 / rms(render(int(SR * seconds)))


class NoiseLayer(Layer):
    """Coloured noise with optional tone shaping (e.g. deep brown noise)."""

    def __init__(self, rng, color="brown", lowpass_hz=None, highpass_hz=None):
        self.noise = Noise(color, rng)
        self.filters = []
        if highpass_hz:
            self.filters.append(highpass(highpass_hz))
        if lowpass_hz:
            self.filters.append(lowpass(lowpass_hz, order=2))
        self.gain = 1.0
        self.gain = self._unit(self)

    def __call__(self, n):
        x = self.noise(n)
        for f in self.filters:
            x = f(x)
        return x * self.gain


class DropBank:
    """Places many short drop sounds at random times, per channel, quickly.

    Each drop is one of a bank of pre-made waveforms. Rather than looping over
    thousands of drops per chunk, every waveform gets its own impulse train,
    which is convolved once; the tail spills over into the next chunk.
    """

    def __init__(self, rng, kernels, channels=2):
        self.rng = rng
        self.kernels = kernels
        self.tail = np.zeros((max(len(k) for k in kernels), channels))
        self.channels = channels
        self._spectra = {}

    def __call__(self, n, rate):
        out = np.zeros((n + len(self.tail), self.channels))
        out[: len(self.tail)] += self.tail
        nk = len(self.kernels)
        k = self.rng.poisson(rate * n / SR * self.channels)
        if k:
            idx = self.rng.integers(0, n, k)
            ch = self.rng.integers(0, self.channels, k)
            which = self.rng.integers(0, nk, k)
            # Capped so no single drop pokes out of the bed and startles a sleeper.
            amp = np.minimum(self.rng.lognormal(0.0, 0.4, k), 2.0)
            slot = (which * self.channels + ch) * n + idx
            imp = np.bincount(slot, weights=amp, minlength=nk * self.channels * n)
            imp = imp.reshape(nk, self.channels, n)
            # One batched FFT for every waveform's impulse train, multiply by each
            # waveform's spectrum, sum, then a single inverse FFT.
            size = sp_fft.next_fast_len(n + len(self.tail))
            if size not in self._spectra:
                self._spectra = {size: np.stack([sp_fft.rfft(k, size) for k in self.kernels])}
            spec = sp_fft.rfft(imp, size, axis=-1, workers=-1)
            y = sp_fft.irfft(np.einsum("jcf,jf->cf", spec, self._spectra[size]), size, axis=-1, workers=-1)
            out += y[:, : len(out)].T
        self.tail = out[n:].copy()
        return out[:n]


def _bubble(rng, f_lo, f_hi, softness=1.0, rise=(0.005, 0.02)):
    """One water-drop 'plink': a decaying sine whose pitch rises (a Minnaert bubble)."""
    f0 = rng.uniform(f_lo, f_hi)
    tau = softness * 1.6 / (0.13 * f0 + 0.0072 * f0 ** 1.5)
    n = int(SR * tau * 6)
    t = np.arange(n) / SR
    sigma = rng.uniform(*rise) * f0  # a big rise sounds "bloopy", so keep it small
    phase = 2 * np.pi * f0 * (t + sigma * t ** 2 / 2)
    env = np.exp(-t / tau) * (1 - np.exp(-t / 0.0004))
    k = env * np.sin(phase)
    return k / np.sqrt(np.sum(k ** 2))


def _thud(rng, cutoff):
    """A soft, dull impact: a tiny burst of low-passed noise."""
    n = int(SR * rng.uniform(0.004, 0.012))
    t = np.arange(n) / SR
    k = rng.standard_normal(n) * np.exp(-t / (n / SR / 4))
    k = signal.sosfilt(signal.butter(2, cutoff, "lowpass", fs=SR, output="sos"), k)
    return k / np.sqrt(np.sum(k ** 2))


class Rain(Layer):
    """Synthetic rain heard from indoors: a soft dark wash, many tiny impacts,
    and water-drop 'plinks', rather than bright hiss.

    intensity  0..1  light drizzle to downpour
    closeness  0..1  how many big, close, lower drops there are
    gusts      0..1  depth of slow intensity swells
    plinks     0..0.3  faint pitched water drops; above ~0.15 it starts to sound bubbly
    """

    def __init__(self, rng, intensity=0.5, closeness=0.4, gusts=0.3, gust_period=75.0, plinks=0.06):
        self.rng = rng
        self.wash = Noise("pink", child(rng))
        self.wash_f = [highpass(120), lowpass(900 + 1400 * intensity, order=2)]
        self.patter = DropBank(child(rng), [_thud(rng, rng.uniform(1500, 4500)) for _ in range(16)])
        self.patter_rate = 500 + 3500 * intensity
        self.plinks = DropBank(child(rng), [_bubble(rng, 2500, 5000, 0.8) for _ in range(24)])
        self.plink_rate = 15 + 80 * intensity
        # Big close drops are dull taps, not bubbles.
        self.big = DropBank(child(rng), [_thud(rng, rng.uniform(400, 1100)) for _ in range(12)])
        self.big_rate = 3 + 40 * intensity * closeness
        self.tone = lowpass(6000, order=2)
        self.gusts = gusts
        self.gust_periods = [gust_period, gust_period * 0.43, gust_period * 2.7]
        self.gust_phases = rng.uniform(0, 2 * np.pi, 3)
        self.pos = 0

        # Mix (relative loudness of each part, after each is set to unit level).
        self.wash_mix = 0.45 + 0.35 * intensity
        self.patter_mix = 0.75
        self.plink_mix = float(plinks)
        self.big_mix = 0.25 + 0.3 * closeness

        self.wash_g = self._unit(self._wash)
        self.patter_g = self._unit(lambda k: self.patter(k, self.patter_rate))
        self.plink_g = self._unit(lambda k: self.plinks(k, self.plink_rate))
        self.big_g = self._unit(lambda k: self.big(k, self.big_rate), seconds=20)
        self.gain = 1.0
        self.gain = self._unit(self)

    def _wash(self, n):
        x = self.wash(n)
        for f in self.wash_f:
            x = f(x)
        return x

    def __call__(self, n):
        t = (self.pos + np.arange(n)) / SR
        self.pos += n
        swell = np.sin(2 * np.pi * t[:, None] / np.array(self.gust_periods)[None, :]
                       + self.gust_phases[None, :]).mean(axis=1)
        g = (1.0 + self.gusts * 0.6 * swell)[:, None]
        wash = self._wash(n) * self.wash_g * self.wash_mix
        patter = self.patter(n, self.patter_rate * float(g.mean())) * self.patter_g * self.patter_mix
        plinks = self.plinks(n, self.plink_rate) * self.plink_g * self.plink_mix
        big = self.big(n, self.big_rate) * self.big_g * self.big_mix
        return self.tone(wash * g + patter + plinks + big) * self.gain


class Thunder(Layer):
    """Rare, soft, distant rolling thunder. Silent between events."""

    continuous = False

    def __init__(self, rng, min_gap=120.0, max_gap=420.0, darkness=0.5):
        self.rng = rng
        self.min_gap, self.max_gap = min_gap, max_gap
        self.cutoff = 260 - 140 * darkness
        self.noise = Noise("brown", child(rng))
        self.pending = np.zeros((0, 2))
        self.until_next = int(SR * rng.uniform(20, min_gap))

    def _event(self) -> np.ndarray:
        length = self.rng.uniform(7, 16)
        n = int(SR * length)
        t = np.arange(n) / SR
        env = np.zeros(n)
        for _ in range(self.rng.integers(2, 5)):
            onset = self.rng.uniform(0, length * 0.45)
            attack = self.rng.uniform(0.4, 1.6)
            decay = self.rng.uniform(1.5, 4.5)
            amp = self.rng.uniform(0.4, 1.0)
            local = t - onset
            rise = np.clip(local / attack, 0, 1)
            fall = np.exp(-np.clip(local - attack, 0, None) / decay)
            env += amp * np.where(local > 0, (1 - np.cos(np.pi * rise)) / 2 * fall, 0)
        env *= (1 - np.cos(np.pi * np.clip((length - t) / 1.5, 0, 1))) / 2  # clean tail
        lp = lowpass(self.cutoff, order=4)
        body = lp(self.noise(n))
        body /= rms(body[env > 0.2 * env.max()]) or 1.0
        x = body * env[:, None] / max(env.max(), 1e-9)
        pan = self.rng.uniform(-0.5, 0.5)
        x[:, 0] *= np.sqrt(1 - pan)
        x[:, 1] *= np.sqrt(1 + pan)
        return x

    def __call__(self, n):
        out = np.zeros((n, 2))
        used = min(n, len(self.pending))
        out[:used] = self.pending[:used]
        self.pending = self.pending[used:]
        cursor = 0
        while cursor + self.until_next < n:
            cursor += self.until_next
            ev = self._event()
            take = min(len(ev), n - cursor)
            out[cursor:cursor + take] += ev[:take]
            rest = ev[take:]
            if len(rest):
                if len(rest) > len(self.pending):
                    self.pending = np.pad(self.pending, ((0, len(rest) - len(self.pending)), (0, 0)))
                self.pending[:len(rest)] += rest
            self.until_next = int(SR * self.rng.uniform(self.min_gap, self.max_gap))
        self.until_next -= n - cursor
        return out


class Ocean(Layer):
    """Waves rolling in and out. Each wave has its own length and size."""

    def __init__(self, rng, min_period=7.0, max_period=13.0, brightness=0.5, calm=0.3):
        self.rng = rng
        self.min_p, self.max_p = min_period, max_period
        self.calm = calm
        self.src = Noise("pink", child(rng))
        self.body = lowpass(380)
        self.wash = lowpass(1200 + 3500 * brightness)
        self.env = np.zeros(0)
        self.delay = int(SR * rng.uniform(0.15, 0.6))
        self.hist = np.zeros(self.delay)
        self.gain = 1.0
        self.gain = self._unit(self, seconds=60)

    def _wave(self):
        p = self.rng.uniform(self.min_p, self.max_p)
        n = int(SR * p)
        x = np.arange(n) / n
        r = self.rng.uniform(0.3, 0.45)
        rise = (1 - np.cos(np.pi * np.clip(x / r, 0, 1))) / 2
        fall = ((1 + np.cos(np.pi * np.clip((x - r) / (1 - r), 0, 1))) / 2) ** 1.6
        shape = np.where(x < r, rise, fall)
        return shape * self.rng.uniform(0.55, 1.0)

    def _envelope(self, n):
        while len(self.env) < n:
            self.env = np.concatenate([self.env, self._wave()])
        e, self.env = self.env[:n], self.env[n:]
        both = np.concatenate([self.hist, e])
        right = both[: n]
        self.hist = both[n:]
        return np.stack([e, right], axis=1)

    def __call__(self, n):
        env = self._envelope(n)
        x = self.src(n)
        base = self.calm + (1 - self.calm) * env
        out = self.body(x) * base * 1.4 + self.wash(x) * 0.65 * env ** 2
        return out * self.gain


class Fan(Layer):
    """A steady room fan: broadband air, faint motor hum, gentle blade flutter."""

    def __init__(self, rng, size=0.5, hum=0.04, flutter=0.03):
        self.air = Noise("brown", child(rng))
        self.hiss = Noise("pink", child(rng))
        self.hiss_lp = lowpass(700 + 1600 * (1 - size))
        self.hum_hz = rng.uniform(48, 110)
        self.hum = hum
        self.blade_hz = rng.uniform(6, 14)
        self.flutter = flutter
        self.pos = 0
        self.gain = 1.0
        self.gain = self._unit(self)

    def __call__(self, n):
        t = (self.pos + np.arange(n)) / SR
        self.pos += n
        air = self.air(n) * 0.8 + self.hiss_lp(self.hiss(n)) * 0.45
        air *= (1 + self.flutter * np.sin(2 * np.pi * self.blade_hz * t))[:, None]
        h = sum(a * np.sin(2 * np.pi * self.hum_hz * k * t) for k, a in ((1, 1.0), (2, 0.5), (3, 0.2)))
        return (air + self.hum * h[:, None]) * self.gain


class Binaural(Layer):
    """Two pure tones, one per ear, a few Hz apart. Needs headphones."""

    pannable = False

    def __init__(self, rng, carrier_hz=150.0, beat_hz=2.0):
        self.fl = carrier_hz - beat_hz / 2
        self.fr = carrier_hz + beat_hz / 2
        self.ph = rng.uniform(0, 2 * np.pi)
        self.pos = 0

    def __call__(self, n):
        t = (self.pos + np.arange(n)) / SR
        self.pos += n
        l = np.sin(2 * np.pi * self.fl * t + self.ph)
        r = np.sin(2 * np.pi * self.fr * t + self.ph)
        return np.stack([l, r], axis=1) * np.sqrt(2)


class Drone(Layer):
    """A soft sustained chord on a chosen frequency (e.g. 528 Hz), slowly breathing."""

    def __init__(self, rng, freq_hz=432.0, warmth=0.5):
        ratios = [0.5, 1.0, 1.5, 2.0]
        amps = [0.6 * (0.5 + warmth), 1.0, 0.45, 0.2 * (1.2 - warmth)]
        self.partials = []
        for r, a in zip(ratios, amps):
            detune = rng.uniform(0.05, 0.35)
            self.partials.append((freq_hz * r, detune, a, rng.uniform(18, 55), rng.uniform(0, 2 * np.pi)))
        self.pos = 0
        self.gain = 1.0
        self.gain = self._unit(self, seconds=60)

    def __call__(self, n):
        t = (self.pos + np.arange(n)) / SR
        self.pos += n
        out = np.zeros((n, 2))
        for f, d, a, lfo_p, ph in self.partials:
            breath = 0.65 + 0.35 * np.sin(2 * np.pi * t / lfo_p + ph)
            out[:, 0] += a * breath * np.sin(2 * np.pi * (f - d) * t + ph)
            out[:, 1] += a * breath * np.sin(2 * np.pi * (f + d) * t + 2 * ph)
        return out * self.gain


class Wind(Layer):
    """Soft wind: broadband air whose tone rises and falls with slow gusts.

    strength  0..1  how far the gusts swing
    whistle   0..1  a faint resonant whistle on the strongest gusts
    """

    def __init__(self, rng, strength=0.5, whistle=0.2, gust_period=22.0):
        self.src = Noise("pink", child(rng))
        self.bands = [lowpass(420), bandpass(300, 900), bandpass(700, 1900)]
        self.whistle_bp = SOS(signal.butter(1, [rng.uniform(550, 950), rng.uniform(1000, 1250)],
                                            "bandpass", fs=SR, output="sos"))
        self.tone = lowpass(3200)
        self.strength, self.whistle = strength, whistle
        self.periods = np.array([gust_period, gust_period * 0.43, gust_period * 2.3])
        self.phases = rng.uniform(0, 2 * np.pi, 3)
        self.pos = 0
        self.gain = 1.0
        self.gain = self._unit(self, seconds=60)

    def __call__(self, n):
        t = (self.pos + np.arange(n)) / SR
        self.pos += n
        lfo = np.sin(2 * np.pi * t[:, None] / self.periods[None, :] + self.phases[None, :]) @ [0.5, 0.2, 0.3]
        g = ((lfo + 1) / 2) ** 1.5  # 0 = lull, 1 = gust
        x = self.src(n)
        low, mid, high = (b(x) for b in self.bands)
        s = self.strength
        out = (low * (1 - 0.5 * s * g)[:, None]
               + mid * (0.3 + 0.9 * s * g)[:, None]
               + high * (0.08 + 0.6 * s * g ** 2)[:, None]
               + self.whistle_bp(x) * (self.whistle * 3 * g ** 3)[:, None])
        level = (1 - 0.6 * s) + 0.6 * s * g
        return self.tone(out * level[:, None]) * self.gain


class Crickets(Layer):
    """A few distant crickets, each with its own pitch, rhythm and position."""

    def __init__(self, rng, count=3, distance=0.5):
        self.crickets = []
        for _ in range(int(count)):
            self.crickets.append(dict(
                freq=rng.uniform(3800, 5200), pulse_rate=rng.uniform(24, 42),
                pulses=int(rng.integers(2, 6)), period=rng.uniform(0.45, 1.3),
                offset=rng.uniform(0, 2), pan=rng.uniform(-0.8, 0.8), amp=rng.uniform(0.5, 1.0),
                rest_period=rng.uniform(15, 50), rest_phase=rng.uniform(0, 2 * np.pi),
                ph=rng.uniform(0, 2 * np.pi)))
        self.distance = lowpass(6500 - 3500 * distance, order=2)
        self.pos = 0
        self.gain = 1.0
        self.gain = self._unit(self, seconds=30)

    def __call__(self, n):
        t = (self.pos + np.arange(n)) / SR
        self.pos += n
        out = np.zeros((n, 2))
        for c in self.crickets:
            phase = (t + c["offset"]) % c["period"]
            k = phase * c["pulse_rate"]
            active = k < c["pulses"]
            env = np.where(active, np.sin(np.pi * (k % 1.0)) ** 2, 0.0)
            # Each cricket pauses now and then, like real ones do.
            env *= np.clip(1.5 * np.sin(2 * np.pi * t / c["rest_period"] + c["rest_phase"]) + 0.8, 0, 1)
            tone = c["amp"] * env * np.sin(2 * np.pi * c["freq"] * t + c["ph"])
            out[:, 0] += tone * np.sqrt((1 - c["pan"]) / 2)
            out[:, 1] += tone * np.sqrt((1 + c["pan"]) / 2)
        return self.distance(out) * self.gain


def decode_audio(path: str, max_seconds: float = 900) -> np.ndarray:
    """Decode any audio file to 48 kHz stereo float with ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-t", str(max_seconds),
           "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).astype(np.float64)


class Sample(Layer):
    """A recording from a sound library, looped without an audible seam.

    Plays random stretches of the file and crossfades between them, so even a
    one minute clip never repeats in an obvious pattern.
    """

    def __init__(self, rng, path, crossfade=4.0, min_segment=25.0, max_segment=90.0):
        self.rng = rng
        self.data = decode_audio(path)
        if len(self.data) < SR * 3:
            raise ValueError(f"{path} is too short to loop (need at least 3 s)")
        self.xf = int(SR * min(crossfade, len(self.data) / SR / 3))
        self.min_seg = int(SR * min_segment)
        self.max_seg = int(SR * max_segment)
        self.buf = np.zeros((0, 2))
        self.started = False
        self.gain = 1.0 / rms(self.data)

    def _segment(self):
        total = len(self.data)
        length = int(self.rng.integers(min(self.min_seg, total), min(self.max_seg, total) + 1))
        length = max(length, 2 * self.xf + 1)
        start = int(self.rng.integers(0, total - length + 1))
        seg = self.data[start:start + length].copy()
        ramp = np.sin(np.linspace(0, np.pi / 2, self.xf))[:, None]  # equal-power
        if self.started:  # the very first segment starts at full level
            seg[: self.xf] *= ramp
        self.started = True
        seg[-self.xf:] *= ramp[::-1]
        return seg

    def __call__(self, n):
        while len(self.buf) < n + self.xf:
            seg = self._segment()
            if len(self.buf) >= self.xf:
                self.buf[-self.xf:] += seg[: self.xf]
                self.buf = np.concatenate([self.buf, seg[self.xf:]])
            else:
                self.buf = np.concatenate([self.buf, seg])
        out, self.buf = self.buf[:n], self.buf[n:]
        return out * self.gain


LAYER_TYPES = {
    "noise": NoiseLayer,
    "rain": Rain,
    "thunder": Thunder,
    "ocean": Ocean,
    "fan": Fan,
    "binaural": Binaural,
    "drone": Drone,
    "wind": Wind,
    "crickets": Crickets,
    "sample": Sample,
}
