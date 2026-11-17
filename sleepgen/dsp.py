"""Small streaming DSP helpers.

Everything here works on blocks shaped (frames, channels) and keeps its own
filter state between calls, so a 12 hour render can be produced in short
chunks with no clicks at the chunk boundaries.
"""

from __future__ import annotations

import numpy as np
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
