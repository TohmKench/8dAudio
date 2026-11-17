# 8dAudio / sleepgen

`sleepgen` makes long (6 to 12 hour), calm soundscapes for a sleep channel: rain, thunder, ocean,
fan, wind, crickets, brown/pink noise, binaural beats and soft drones, with slow 8D-style movement.
Every video gets a randomly generated recipe, so no two uploads are the same mix.

## How a mix is built: three tiers

| Tier | How many | Level | Examples |
|---|---|---|---|
| **Base** | 1, always on | 0 dB | rain, ocean, brown/pink noise, fan, solfeggio drone, stream*, fireplace* |
| **Secondary** | 0 to 2, always on | 6 to 14 dB under | wind, noise bed, binaural beats, drone, soft rain, fire*, stream* |
| **Accent** | 0 to 1, comes and goes | 8 to 14 dB under | crickets, distant thunder, owl*, frogs*, wind chimes* |

\* needs recordings in the sound library.

Each base lists the secondaries and accents that suit it (`python -m sleepgen themes`). Accents play
only in the first 45 to 60 minutes and then fade out, so nothing can startle someone who is already
asleep. Accents are night sounds only: no daytime birdsong. The title headline is built from the
tiers, e.g. "Gentle Rain with Distant Wind and Crickets".

Needs Python 3.10+, `ffmpeg` on the PATH, and `pip install -r requirements.txt`.

## Quick start

```bash
python -m sleepgen themes                                  # list themes
python -m sleepgen previews -d previews --seconds 90       # a short clip of every theme
python -m sleepgen random -o out/video1.m4a --hours 10     # random theme, 10 hours
python -m sleepgen random --theme rain --secondary wind --accent crickets -o out/v.m4a --hours 8
python -m sleepgen recipe --theme ocean -o ocean.json      # save a recipe, edit it by hand...
python -m sleepgen render ocean.json -o out/ocean.m4a --hours 12   # ...then render it
```

Each render writes `<name>.json` beside the audio with the full recipe and a render report. The
recipe's `meta` block (theme, headline, features) feeds video titles and the weekly analytics report.

## How it keeps things right for sleep

- **Steady, quiet volume.** Every mix is measured and set to -26 LUFS, a slow leveler eases down
  anything that swells more than 3 dB above that (big waves, thunder), and a soft limiter sits at
  -1 dBFS. YouTube only ever turns loud videos down, never quiet ones up.
- **One shared space.** All layers feed a convolution reverb with a natural, darkening tail. The
  base is mostly direct, secondaries sit further back, and accents are mostly reverb, so a cricket
  sounds like it's out in the garden, not in your ear.
- **No surprises.** Rain drops are capped so none pokes out of the bed, thunder is rare, low and
  distant, and slow level drift keeps it alive without anything sudden.
- **No audible loop.** Everything is generated live, and library recordings are played as random
  stretches with crossfades between them.
- **8D movement** (`pan.mode`: `sweep` or `circle`) is gentle by default (width 0.3 to 0.75) so it
  stays comfortable for someone lying on one ear. Binaural beats are never panned.
- **Quality over speed.** Rain is built from thousands of individually modelled drops per second
  (impacts plus pitched water "plinks"), with dozens of drop shapes for variety. A 10 hour layered
  mix takes roughly 1 to 3 hours to render depending on the CPU. Audio streams to ffmpeg in
  10 second chunks, so memory stays flat, and output is 384 kbps AAC (YouTube's recommended rate).

## Sound library (optional)

Point `--library` at a folder with one subfolder per category and sleepgen will blend real
recordings into matching themes:

```
sounds/
  rain/  rain_roof/  rain_window/  ocean/  stream/  fire/  wind/      (bases and secondaries)
  crickets/  owl/  frogs/  chimes/                                   (accents)
```

Only use recordings you may use commercially: CC0 / public domain (e.g. Freesound filtered to
"Creative Commons 0") or Pixabay's licence. Keep a note of each file's source. A recording should be
at least 30 seconds long and free of voices, cars, birds and other sharp sounds.

## Recipe format

```json
{
  "seed": 7,
  "master": {"target_lufs": -26, "lowpass_hz": 9000, "fade_in_s": 20, "fade_out_s": 45,
             "reverb": {"rt60": 2.2, "predelay_ms": 25, "damping": 0.6}},
  "layers": [
    {"type": "rain", "level_db": 0, "params": {"intensity": 0.6, "closeness": 0.4},
     "pan": {"mode": "circle", "width": 0.6, "period_s": 50},
     "drift": {"depth_db": 1.5, "period_s": 120}},
    {"type": "wind", "level_db": -10, "params": {"strength": 0.5}, "send": 0.35},
    {"type": "crickets", "level_db": -18, "params": {"count": 3}, "send": 0.75, "dry": 0.4,
     "gate": {"on_s": [20, 60], "off_s": [30, 120], "start_s": 40, "until_s": 3000, "until_fade_s": 300}}
  ],
  "meta": {"base": "rain", "secondaries": ["wind"], "accents": ["crickets"],
           "headline": "Rain with Distant Wind and Crickets"}
}
```

Layer types: `noise`, `rain`, `thunder`, `ocean`, `fan`, `wind`, `crickets`, `binaural`, `drone`, `sample`.
`send` (0 to 1) feeds the shared reverb and `dry` (0 to 1) lowers the direct sound to push a layer
further away. `gate` makes any layer come and go (`on_s` / `off_s` ranges) and stop for good at `until_s`.
See `sleepgen/layers.py` for each one's parameters.

## Tests

```bash
pip install pytest && python -m pytest -q
```

## Older 8D web app

The earlier React + Node/XAMPP 8D audio app is not in this repository.
