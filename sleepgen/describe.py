"""YouTube title, description and tags for a rendered video, built from its recipe.

The background/thumbnail image's owner is ALWAYS credited at the end of the description
(Thomas's rule, 2026-11-20), even where the licence doesn't require it. The credit lives in
the recipe's `image` block, filled in when the image is picked:

    "image": {"source": "unsplash", "photographer": "Jane Doe",
              "photographer_url": "https://unsplash.com/@janedoe",
              "photo_url": "https://unsplash.com/photos/abc123", "fetched": "2026-11-20"}

`video_metadata` refuses (MissingCredit) a recipe without a complete credit, so the upload
step can never publish a video whose image owner isn't named.
"""

from __future__ import annotations

from urllib.parse import urlparse

IMAGE_SOURCES = {"unsplash": "Unsplash", "pexels": "Pexels"}
CREDIT_FIELDS = ("source", "photographer", "photographer_url", "photo_url")
TITLE_MAX = 100        # YouTube's limit
DESCRIPTION_MAX = 5000


class MissingCredit(ValueError):
    """The recipe's image has no complete owner credit, so the video must not be published."""


def _is_url(s) -> bool:
    try:
        u = urlparse(str(s))
    except ValueError:
        return False
    return u.scheme in ("http", "https") and bool(u.netloc)


def credit_line(image: dict | None) -> str:
    """'Background photo by Jane Doe on Unsplash' plus links. Raises MissingCredit."""
    if not image:
        raise MissingCredit("the recipe has no `image` block: pick and credit an image first")
    missing = [k for k in CREDIT_FIELDS if not str(image.get(k) or "").strip()]
    if missing:
        raise MissingCredit(f"the image credit is missing: {', '.join(missing)}")
    source = IMAGE_SOURCES.get(str(image["source"]).lower())
    if source is None:
        raise MissingCredit(f"unknown image source {image['source']!r}; use one of {', '.join(IMAGE_SOURCES)}")
    for k in ("photographer_url", "photo_url"):
        if not _is_url(image[k]):
            raise MissingCredit(f"image {k} is not a web link: {image[k]!r}")
    name = " ".join(str(image["photographer"]).split())
    return (f"Background photo by {name} on {source}: {image['photo_url']}\n"
            f"More from {name}: {image['photographer_url']}")


def _hours(hours: float) -> str:
    return f"{hours:g} Hour" + ("" if hours == 1 else "s")


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def title(recipe: dict, hours: float) -> str:
    m = recipe["meta"]
    purpose = "for Studying" if m.get("video_type") == "study" else "for Sleep"
    if purpose.split()[-1][:5].lower() in m["headline"].lower():  # "Studying by a Rainy Window"
        purpose = ""
    full = f"{m['headline']} · {_hours(hours)} {purpose}".rstrip()
    if len(full) > TITLE_MAX:
        full = f"{m['headline']} · {_hours(hours)}"
    return full[:TITLE_MAX].rstrip()


def description(recipe: dict, hours: float) -> str:
    m = recipe["meta"]
    credit = credit_line(recipe.get("image"))  # first, so nothing is written without it
    study = m.get("video_type") == "study"
    layers = [m.get("base_title", m.get("headline", ""))] + m.get("secondary_labels", []) \
        + m.get("tertiary_labels", [])
    sounds = _join([l.lower() if i else l for i, l in enumerate(layers)])
    lead = f"{m['headline']}: " if m.get("scene") else ""
    lines = [f"{lead}{sounds}, {_hours(hours).lower()} {'to study and focus to' if study else 'to fall asleep to'}."]
    lines.append("")
    if study:
        lines.append("A steady, quiet background for reading, writing and focus. No music and no talking.")
    else:
        lines.append("Made for sleep: one steady, quiet level all night, with nothing sudden. "
                     "Any night sounds fade out within the first hour, so nothing wakes you.")
    if "3D Audio" in m.get("features", []):
        lines.append("The sound is placed in 3D around you: best with headphones, fine on speakers.")
    if m.get("binaural"):
        lines.append("Includes binaural beats: use headphones for the effect.")
    lines += ["", "The screen is kept dark on purpose, so you can leave it playing.", "", credit]
    text = "\n".join(lines)
    if len(text) > DESCRIPTION_MAX:  # never cut the credit
        head = "\n".join(lines[:-1])[: DESCRIPTION_MAX - len(credit) - 2].rstrip()
        text = head + "\n\n" + credit
    return text


def tags(recipe: dict) -> list[str]:
    m = recipe["meta"]
    study = m.get("video_type") == "study"
    words = [m.get("base_label", ""), m.get("base_title", ""), *m.get("secondary_labels", []),
             *m.get("tertiary_labels", [])]
    words += ["study ambience", "focus sounds", "study with me"] if study else \
        ["sleep sounds", "sleep music", "relaxing sounds", "white noise"]
    out, seen = [], set()
    for w in words:
        w = w.strip().lower()
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    while sum(len(t) + 2 for t in out) > 480:  # YouTube allows about 500 characters of tags
        out.pop()
    return out


def video_metadata(recipe: dict, hours: float) -> dict:
    """Everything the upload step needs. Raises MissingCredit if the image owner isn't credited."""
    desc = description(recipe, hours)
    return {"title": title(recipe, hours), "description": desc, "tags": tags(recipe),
            "image": dict(recipe["image"])}
