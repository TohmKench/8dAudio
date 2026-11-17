"""Small streaming DSP helpers.

Everything here works on blocks shaped (frames, channels) and keeps its own
filter state between calls, so a 12 hour render can be produced in short
chunks with no clicks at the chunk boundaries.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sp_fft
from scipy import signal

SR = 48_000

# Paul Kellet's pink noise filter (-3 dB/octave), good from ~10 Hz to Nyquist.
_PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
_PINK_A = [1.0, -2.494956002, 2.017265875, -0.522189400]


class SOS:
    """Stateful second-order-sections filter."""

    def __init__(self, sos, channels: int = 2):
        self.sos = np.asarray(sos, dtype=np.float64)
        self.zi = np.zeros((self.sos.shape[0], 2, channels))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


class LFilter:
    """Stateful direct-form IIR filter."""

    def __init__(self, b, a, channels: int = 2):
        self.b = np.asarray(b, dtype=np.float64)
        self.a = np.asarray(a, dtype=np.float64)
        self.zi = np.zeros((max(len(self.a), len(self.b)) - 1, channels))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = signal.lfilter(self.b, self.a, x, axis=0, zi=self.zi)
        return y


def lowpass(hz: float, order: int = 2, channels: int = 2) -> SOS:
    return SOS(signal.butter(order, min(hz, SR * 0.45), "lowpass", fs=SR, output="sos"), channels)


def highpass(hz: float, order: int = 2, channels: int = 2) -> SOS:
    return SOS(signal.butter(order, hz, "highpass", fs=SR, output="sos"), channels)


def bandpass(lo: float, hi: float, order: int = 2, channels: int = 2) -> SOS:
    return SOS(signal.butter(order, [lo, min(hi, SR * 0.45)], "bandpass", fs=SR, output="sos"), channels)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-20))


class Noise:
    """Endless white, pink or brown noise with roughly unit RMS.

    Channels are independent, which gives a wide, natural stereo image.
    """

    COLORS = ("white", "pink", "brown")

    def __init__(self, color: str, rng: np.random.Generator, channels: int = 2):
        if color not in self.COLORS:
            raise ValueError(f"unknown noise colour {color!r}")
        self.color = color
        self.rng = rng
        self.channels = channels
        self._chain = self._make_chain()
        # Measure the chain's gain on throwaway noise, then reset its state.
        probe_rng = np.random.default_rng(0)
        probe = self._apply(self._make_chain(), probe_rng.standard_normal((SR * 4, channels)))
        self.gain = 1.0 / rms(probe[SR:])

    def _make_chain(self):
        if self.color == "white":
            return []
        if self.color == "pink":
            return [LFilter(_PINK_B, _PINK_A, self.channels)]
        # Brown: leaky integrator, then remove sub-audio drift.
        return [LFilter([1.0], [1.0, -0.998], self.channels), highpass(18, 2, self.channels)]

    @staticmethod
    def _apply(chain, x):
        for f in chain:
            x = f(x)
        return x

    def __call__(self, n: int) -> np.ndarray:
        x = self.rng.standard_normal((n, self.channels))
        return self._apply(self._chain, x) * self.gain


def smooth_lfo(t: np.ndarray, periods, phases, weights) -> np.ndarray:
    """Sum of slow sines, normalised to the range [-1, 1]."""
    out = np.zeros_like(t)
    for p, ph, w in zip(periods, phases, weights):
        out += w * np.sin(2 * np.pi * t / p + ph)
    return out / max(sum(abs(w) for w in weights), 1e-9)


def equal_power_pan(position: np.ndarray):
    """position in [-1, 1] -> (left_gain, right_gain), unity at centre."""
    theta = (np.clip(position, -1, 1) + 1) * np.pi / 4
    return np.sqrt(2) * np.cos(theta), np.sqrt(2) * np.sin(theta)


# ITU-R BS.1770 K-weighting at 48 kHz.
_K1_B = [1.53512485958697, -2.69169618940638, 1.19839281085285]
_K1_A = [1.0, -1.69065929318241, 0.73248077421585]
_K2_B = [1.0, -2.0, 1.0]
_K2_A = [1.0, -1.99004745483398, 0.99007225036621]


def loudness_lufs(x: np.ndarray) -> float:
    """Ungated BS.1770 loudness of a stereo block.

    Ungated is fine here: our soundscapes are continuous beds with no silence.
    """
    y = signal.lfilter(_K1_B, _K1_A, x, axis=0)
    y = signal.lfilter(_K2_B, _K2_A, y, axis=0)
    power = float(np.sum(np.mean(np.square(y), axis=0)))
    return -0.691 + 10 * np.log10(power + 1e-20)


def soft_limit(x: np.ndarray, threshold: float = 0.6, ceiling: float = 0.89) -> np.ndarray:
    """Transparent below threshold, smoothly saturates towards ceiling (-1 dBFS)."""
    a = np.abs(x)
    over = a > threshold
    if not np.any(over):
        return x
    span = ceiling - threshold
    y = x.copy()
    y[over] = np.sign(x[over]) * (threshold + span * np.tanh((a[over] - threshold) / span))
    return y


class Reverb:
    """Streaming stereo convolution reverb with a synthetic, natural-sounding tail.

    The impulse response is decorrelated noise per ear with an exponential decay that
    is faster for high frequencies than low ones (like air and soft surfaces), plus a
    few sparse early reflections. It places every layer in one shared space, which is
    what makes separate sounds feel like one environment rather than a collage.
    """

    def __init__(self, rng, rt60=2.2, predelay_ms=25.0, damping=0.5, channels=2):
        n = int(SR * rt60 * 1.1)
        t = np.arange(n) / SR
        ir = np.zeros((n, channels))
        bands = [(None, 400), (400, 1500), (1500, 5000), (5000, None)]
        # Higher bands die away sooner; damping 0..1 controls how much sooner.
        rt_scale = [1.0, 0.9, 0.75 - 0.3 * damping, 0.5 - 0.35 * damping]
        for (lo, hi), scale in zip(bands, rt_scale):
            noise = rng.standard_normal((n, channels))
            if lo is None:
                sos = signal.butter(4, hi, "lowpass", fs=SR, output="sos")
            elif hi is None:
                sos = signal.butter(4, lo, "highpass", fs=SR, output="sos")
            else:
                sos = signal.butter(4, [lo, hi], "bandpass", fs=SR, output="sos")
            decay = np.exp(-6.91 * t / (rt60 * scale))  # -60 dB at rt60
            ir += signal.sosfilt(sos, noise, axis=0) * decay[:, None]
        # Soft onset so the tail swells in rather than clicking.
        ir *= (1 - np.exp(-t / 0.012))[:, None]
        for _ in range(6):  # early reflections
            d = int(SR * rng.uniform(0.008, 0.07))
            ir[d, rng.integers(0, channels)] += rng.uniform(0.3, 0.8) * rng.choice([-1, 1]) * 8
        pre = int(SR * predelay_ms / 1000)
        ir = np.concatenate([np.zeros((pre, channels)), ir])
        ir /= np.sqrt(np.sum(ir ** 2) / channels)
        self.ir = ir
        self.tail = np.zeros((len(ir) - 1, channels))
        self._spec = {}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        size = sp_fft.next_fast_len(n + len(self.ir) - 1)
        if size not in self._spec:
            self._spec = {size: sp_fft.rfft(self.ir, size, axis=0)}
        y = sp_fft.irfft(sp_fft.rfft(x, size, axis=0, workers=-1) * self._spec[size], size, axis=0,
                         workers=-1)[: n + len(self.tail)]
        y[: len(self.tail)] += self.tail
        self.tail = y[n:].copy()
        return y[:n]


class Leveler:
    """Slow, gentle level rider: turns down anything that rises well above the target.

    Measures short-term loudness (K-weighted, 400 ms) and applies a soft 3:1 reduction
    above `target + headroom`, with a 0.5 s attack and 4 s release, so a big wave or a
    thunder roll is eased down rather than clipped. It never turns anything up.
    """

    def __init__(self, target_lufs, headroom_db=3.0, ratio=3.0, channels=2):
        self.threshold = target_lufs + headroom_db
        self.ratio = ratio
        self.k1 = LFilter(_K1_B, _K1_A, channels)
        self.k2 = LFilter(_K2_B, _K2_A, channels)
        self.block = SR // 10
        self.hist = np.zeros(4)  # last 4 blocks of power = 400 ms
        self.gain_db = 0.0
        self.a_att = np.exp(-0.1 / 0.5)
        self.a_rel = np.exp(-0.1 / 4.0)
        self.last_gain = 1.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        y = self.k2(self.k1(x))
        blocks = int(np.ceil(n / self.block))
        gains = np.empty(blocks)
        for b in range(blocks):
            seg = y[b * self.block:(b + 1) * self.block]
            self.hist = np.roll(self.hist, -1)
            self.hist[-1] = float(np.sum(np.mean(seg ** 2, axis=0)))
            lufs = -0.691 + 10 * np.log10(np.mean(self.hist) + 1e-20)
            over = max(0.0, lufs - self.threshold)
            want = -over * (1 - 1 / self.ratio)
            a = self.a_att if want < self.gain_db else self.a_rel
            self.gain_db = a * self.gain_db + (1 - a) * want
            gains[b] = 10 ** (self.gain_db / 20)
        # Smooth per-sample gain: ramp from the previous block's gain to each new one.
        points = np.concatenate([[self.last_gain], gains])
        g = np.interp(np.arange(n) / self.block, np.arange(blocks + 1), points)
        self.last_gain = gains[-1]
        return x * g[:, None]
