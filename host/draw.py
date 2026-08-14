"""Draw a movement, so it can be recognised after the fact.

Until now the only way to judge a description was to have just made the movement
and still remember it. That limits review to the few seconds after each gesture,
which is exactly why every problem has been found one anecdote at a time.

A picture of the path fixes that: a hand-to-hand toss, a lift, a shake and a
spin look nothing like each other on paper, so a movement from an hour ago can
be judged as easily as one from a second ago.

Two projections, because one is ambiguous. Seen from the side a lift and a
sideways sweep can look identical; seen from above a lift is a dot. Both
together are unmistakable. Time runs from pale to bright, so direction is
visible without an arrow.

Untrusted reconstructions are still drawn, clearly labelled. That inverts the
rule everywhere else in this codebase, and only because the job here is
recognition rather than assertion: a bad scale does not destroy the *shape*, and
a rough shape is far easier to recognise than three signal traces. Nothing is
claimed from these drawings — they exist so a person can say "yes, that is what
I did".

A throw is the exception. Through free fall nobody is controlling the ball, so
the path is meaningless rather than imprecise, and only the raw signals are
shown — where a flight reads unmistakably as a flat weightless notch.
"""

from __future__ import annotations

import math

import numpy as np

BG = "#141418"
INK = "#e8e8ea"
DIM = "#55555f"
GRID = "#26262e"
HOT = "#86e0ac"
COLD = "#3d6ea8"


def _lerp_colour(t: float) -> str:
    """Cold at the start of the movement, hot at the end."""
    a = (0x3d, 0x6e, 0xa8)
    b = (0x86, 0xe0, 0xac)
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _panel(w: int, h: int, title: str, sub: str = "") -> list[str]:
    return [
        f'<rect width="{w}" height="{h}" fill="{BG}" rx="6"/>',
        f'<text x="10" y="18" fill="{DIM}" font-size="11" '
        f'font-family="ui-monospace,Menlo,monospace">{title}</text>',
        f'<text x="{w - 10}" y="18" fill="{DIM}" font-size="11" '
        f'text-anchor="end" font-family="ui-monospace,Menlo,monospace">{sub}</text>',
    ]


def _project(pts: np.ndarray, ax: int, ay: int, w: int, h: int,
             pad: int = 26) -> tuple[np.ndarray, float]:
    """Fit a 2-D projection into the panel, preserving aspect ratio.

    Both panels share one scale so the two views stay comparable — a movement
    that is wide and flat should look wide and flat, not be stretched to fill.
    """
    x, y = pts[:, ax], pts[:, ay]
    cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    span = max(x.max() - x.min(), y.max() - y.min(), 0.05)
    scale = (min(w, h) - 2 * pad) / span
    sx = w / 2 + (x - cx) * scale
    sy = h / 2 - (y - cy) * scale          # screen y grows downward
    return np.column_stack([sx, sy]), scale


def _trace(pts: np.ndarray, w: int, h: int, ax: int, ay: int,
           label_a: str, label_b: str) -> list[str]:
    xy, scale = _project(pts, ax, ay, w, h)
    out = [f'<line x1="0" y1="{h/2}" x2="{w}" y2="{h/2}" stroke="{GRID}"/>',
           f'<line x1="{w/2}" y1="0" x2="{w/2}" y2="{h}" stroke="{GRID}"/>']

    n = len(xy)
    step = max(1, n // 90)                  # cap the segment count
    for i in range(0, n - step, step):
        t = i / max(1, n - 1)
        out.append(
            f'<line x1="{xy[i,0]:.0f}" y1="{xy[i,1]:.0f}" '
            f'x2="{xy[i+step,0]:.0f}" y2="{xy[i+step,1]:.0f}" '
            f'stroke="{_lerp_colour(t)}" stroke-width="2.5" '
            f'stroke-linecap="round"/>')

    out.append(f'<circle cx="{xy[0,0]:.1f}" cy="{xy[0,1]:.1f}" r="4.5" '
               f'fill="none" stroke="{COLD}" stroke-width="2"/>')
    out.append(f'<circle cx="{xy[-1,0]:.1f}" cy="{xy[-1,1]:.1f}" r="4.5" '
               f'fill="{HOT}"/>')
    # a scale bar, so "how big was that" is answerable at a glance
    bar_cm = 10.0
    px = bar_cm / 100.0 * scale
    if 12 < px < w * 0.7:
        out.append(f'<line x1="12" y1="{h-14}" x2="{12+px:.0f}" y2="{h-14}" '
                   f'stroke="{DIM}" stroke-width="2"/>')
        out.append(f'<text x="{12+px+6:.0f}" y="{h-10}" fill="{DIM}" '
                   f'font-size="10" font-family="ui-monospace,Menlo,monospace">'
                   f'10 cm</text>')
    out.append(f'<text x="{w-10}" y="{h-10}" fill="{DIM}" font-size="10" '
               f'text-anchor="end" font-family="ui-monospace,Menlo,monospace">'
               f'{label_a} / {label_b}</text>')
    return out


def path_svg(position: np.ndarray, w: int = 300, h: int = 210,
             trusted: bool = True) -> str:
    """Side elevation and plan view, side by side.

    Drawn even when the reconstruction is untrustworthy, which is the opposite
    of what the rest of the pipeline does — and right here, because the job is
    recognition rather than assertion. The *shape* of a movement survives a bad
    scale, and a rough shape is far easier to recognise than three signal
    traces. It is labelled, so nobody reads a distance off it.
    """
    if position is None or len(position) < 3:
        return ""
    # Rotate so the dominant horizontal direction runs left-right in the side
    # view. Without this a movement along Y appears as a dot from the side and
    # looks like the person did nothing.
    flat = position[:, :2] - position[:, :2].mean(axis=0)
    if np.any(flat):
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
        main = vt[0]
        along = position[:, :2] @ main
        across = position[:, :2] @ np.array([-main[1], main[0]])
    else:
        along, across = position[:, 0], position[:, 1]
    side = np.column_stack([along, position[:, 2]])
    plan = np.column_stack([along, across])

    note = "" if trusted else "shape only — scale not trustworthy"
    parts = [f'<svg viewBox="0 0 {w*2+10} {h}" width="100%" '
             f'xmlns="http://www.w3.org/2000/svg">']
    parts += ['<g>'] + _panel(w, h, "from the side", note) + \
        _trace(np.column_stack([side, np.zeros(len(side))]), w, h, 0, 1,
               "across", "height") + ['</g>']
    parts += [f'<g transform="translate({w+10},0)">'] + \
        _panel(w, h, "from above") + \
        _trace(np.column_stack([plan, np.zeros(len(plan))]), w, h, 0, 1,
               "across", "depth") + ['</g>']
    parts.append('</svg>')
    return "".join(parts)


def signal_svg(frames, w: int = 610, h: int = 120) -> str:
    """Raw traces, for movements whose path cannot be trusted.

    A shake, a spin and a throw are all obvious here even though none of them
    has a usable trajectory — oscillation, sustained rotation, and a flat
    weightless notch respectively.
    """
    if not frames:
        return ""
    acc = np.array([math.sqrt(f.lax ** 2 + f.lay ** 2 + f.laz ** 2) for f in frames])
    gyr = np.array([math.sqrt(f.gx ** 2 + f.gy ** 2 + f.gz ** 2) for f in frames])
    raw = np.array([math.sqrt(f.ax ** 2 + f.ay ** 2 + f.az ** 2) for f in frames])

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" '
             f'xmlns="http://www.w3.org/2000/svg">']
    parts += _panel(w, h, "signals", "accel · rotation · weightlessness")

    def line(series, colour, top, height, cap=None):
        v = np.clip(series, 0, cap) if cap else series
        hi = max(float(v.max()), 1e-6)
        n = len(v)
        step = max(1, n // 300)
        pts = " ".join(
            f"{26 + i / max(1, n - 1) * (w - 40):.1f},"
            f"{top + height - v[i] / hi * height:.1f}"
            for i in range(0, n, step))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                     f'stroke-width="1.6" opacity="0.9"/>')

    line(acc, HOT, 30, 34)
    line(gyr, COLD, 60, 34)
    # free fall shows as this dipping toward zero
    line(raw, "#c98fd6", 88, 26, cap=20.0)
    parts.append('</svg>')
    return "".join(parts)
