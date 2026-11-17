import json
import os
import subprocess

import numpy as np
import pytest

from sleepgen import engine, layers
from sleepgen.dsp import SR, loudness_lufs
from sleepgen.recipes import BASES, available_bases, random_recipe


def _render_raw(recipe, seconds, chunk_s):
    mix = engine.Mix(recipe)
    n, chunk = int(seconds * SR), int(chunk_s * SR)
    out = []
    while n > 0:
        k = min(chunk, n)
        out.append(mix.raw(k))
        n -= k
    return np.concatenate(out)


@pytest.mark.parametrize("spec", [
    {"type": "noise", "params": {"color": "brown", "lowpass_hz": 500}, "pan": {"mode": "circle"}},
    {"type": "noise", "params": {"color": "pink"}, "drift": {"depth_db": 2}},
    {"type": "fan", "params": {}},
    {"type": "binaural", "params": {"carrier_hz": 150, "beat_hz": 2}},
    {"type": "drone", "params": {"freq_hz": 528}, "pan": {"mode": "sweep"}},
])
def test_chunk_size_does_not_change_output(spec):
    """Same audio whether rendered in 1 s or 3.7 s chunks: no seams at chunk edges."""
    recipe = {"seed": 5, "layers": [spec]}
    a = _render_raw(recipe, 8, 1.0)
    b = _render_raw(recipe, 8, 3.7)
    np.testing.assert_allclose(a, b, atol=1e-9)


@pytest.mark.parametrize("theme", available_bases())
def test_every_theme_renders_to_target(theme, tmp_path):
    recipe = random_recipe(11, theme)
    json.dumps(recipe)  # recipes must be plain JSON
    recipe["master"].update(fade_in_s=3, fade_out_s=3)
    out = tmp_path / f"{theme}.wav"
    report = engine.render(recipe, str(out), 40, progress=False)
    assert report["peak_dbfs"] <= -0.9
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-ss", "10", "-t", "25",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).astype(np.float64)
    if theme != "ocean":  # waves swing a lot over 25 s
        assert abs(loudness_lufs(x) - engine.DEFAULT_TARGET_LUFS) < 2.0


def test_same_seed_same_recipe():
    assert random_recipe(42) == random_recipe(42)
    assert random_recipe(42) != random_recipe(43)


def test_binaural_keeps_ears_separate():
    recipe = {"seed": 1, "layers": [{"type": "binaural", "params": {"carrier_hz": 200, "beat_hz": 4},
                                     "pan": {"mode": "circle", "width": 1}}]}
    x = _render_raw(recipe, 2, 1)
    spec_l = np.abs(np.fft.rfft(x[:, 0]))
    spec_r = np.abs(np.fft.rfft(x[:, 1]))
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    assert freqs[spec_l.argmax()] == pytest.approx(198, abs=1)
    assert freqs[spec_r.argmax()] == pytest.approx(202, abs=1)


def test_thunder_is_mostly_silent_with_events():
    th = layers.Thunder(np.random.default_rng(3), min_gap=20, max_gap=40)
    x = np.concatenate([th(SR * 10) for _ in range(12)])
    active = np.abs(x).max(axis=1) > 0.01
    assert 0.05 < active.mean() < 0.8


def test_sample_layer_loops_without_gaps(tmp_path):
    src = tmp_path / "rain" / "clip.wav"
    src.parent.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anoisesrc=d=8:c=pink:a=0.3",
                    "-ac", "2", str(src)], check=True)
    layer = layers.Sample(np.random.default_rng(0), str(src), min_segment=3, max_segment=5, crossfade=1)
    x = np.concatenate([layer(SR) for _ in range(30)])
    win = SR // 10
    level = np.sqrt(np.mean(x[: len(x) // win * win].reshape(-1, win, 2) ** 2, axis=(1, 2)))
    assert level.min() > 0.4 * np.median(level)  # no dropouts at the joins

    picked = [random_recipe(seed, "rain", library=str(tmp_path)) for seed in range(10)]
    assert any(l["type"] == "sample" for r in picked for l in r["layers"])


def test_sidecar_written(tmp_path):
    out = tmp_path / "v.m4a"
    engine.render_with_sidecar(random_recipe(2, "fan"), str(out), 5, progress=False)
    side = json.loads((tmp_path / "v.json").read_text())
    assert side["recipe"]["meta"]["theme"] == "fan"
    assert os.path.getsize(out) > 10_000


def _lib(tmp_path, *cats):
    for c in cats:
        d = tmp_path / c
        d.mkdir()
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anoisesrc=d=6:c=brown:a=0.2",
                        "-ac", "2", str(d / "a.wav")], check=True)
    return str(tmp_path)


def test_library_only_parts_need_the_library(tmp_path):
    assert "stream" not in available_bases()
    with pytest.raises(ValueError):
        random_recipe(1, "stream")
    lib = _lib(tmp_path, "stream", "owl")
    assert "stream" in available_bases(lib)
    r = random_recipe(1, "stream", library=lib, accents=["owl"])
    assert r["meta"]["headline"].endswith("Owl")
    assert r["meta"]["sources"]


def test_tiers_and_headline():
    r = random_recipe(4, "rain", secondaries=["noise_bed", "wind"], accents=["crickets"])
    m = r["meta"]
    assert m["secondaries"] == ["noise_bed", "wind"] and m["accents"] == ["crickets"]
    assert m["headline"].endswith("with Distant Wind and Crickets")
    assert m["combo"] == "rain+noise_bed+wind+crickets"
    levels = [l["level_db"] for l in r["layers"]]
    assert levels[0] == 0 and all(-16 <= v <= -6 for v in levels[1:])


def test_bases_only_offer_known_parts():
    from sleepgen.recipes import ACCENTS, SECONDARIES
    for spec in BASES.values():
        assert set(spec["secondaries"]) <= set(SECONDARIES)
        assert set(spec["accents"]) <= set(ACCENTS)


def test_accents_come_and_go_then_stop():
    spec = {"type": "crickets", "params": {}, "gate": {"on_s": [5, 5], "off_s": [10, 10], "start_s": 2,
                                                       "until_s": 60, "until_fade_s": 10}}
    x = _render_raw({"seed": 1, "layers": [spec]}, 80, 1)
    level = np.sqrt(np.mean(x.reshape(80, SR, 2) ** 2, axis=(1, 2)))
    assert level[:2].max() == 0                 # silent until start_s
    assert level[2:7].min() > 0                 # first burst
    assert level[8:16].max() == 0               # gap
    assert level[60:].max() == 0                # stopped for the rest of the night


def test_wind_and_crickets_are_unit_level():
    for kind in (layers.Wind, layers.Crickets):
        x = kind(np.random.default_rng(2))(SR * 20)
        assert 0.5 < np.sqrt(np.mean(x ** 2)) < 2
