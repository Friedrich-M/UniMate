"""Shared helpers for the dataset-statistics distribution figures.

Used by the ``data_process/tools/vis_*.py`` distribution tools
(``vis_clip_frames.py`` — frames per clip, ``vis_joint_count.py`` — joints
per skeleton). Provides input resolution for the stage-1 summary JSONs
(``{key: int}`` maps next to ``summary.json``), the paper-figure style
shared with the joint-count / category distribution figures — one muted
indigo accent with a soft kernel-density fill, mean / median guides,
proportion axes, PNG + PDF at 300 dpi with embedded TrueType fonts — and
the distribution math behind the panels. Pure numpy + matplotlib: no scipy
(the Gaussian KDE is implemented here) and no bpy.
"""

import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

from data_process.utils.plotting import SANS_STACK  # noqa: E402


# ---------------------------------------------------------------------------
# Palette & style
# ---------------------------------------------------------------------------

# One muted indigo accent (bars, curves) with a soft fill (KDE / area), and
# the category palette's leading hues when several datasets are overlaid so
# a dataset keeps one color across every figure.
ACCENT = '#3D5A80'
ACCENT_DK = '#2A3F5F'
ACCENT_FL = '#98C1D9'
INK = '#1F2937'
MUTED = '#6B7280'
GRID = '#E5E7EB'
EDGE = '#9CA3AF'
SERIES = ('#3B5BA5', '#E25E54', '#43A47F', '#3FA9D6', '#EFA94A', '#A06CD5', '#8C95A8')

KDE_BW = 0.30          # bandwidth factor (× std), as in the joint-count figure
KDE_POINTS = 700
HEADROOM = 1.18        # y-limit over the tallest bar / curve: room for guide labels

# Two-panel layout; the tick-thinning estimate of a panel's width reads it.
FIG_W, FIG_H = 13.0, 4.8
LEFT, RIGHT, TOP, BOTTOM, WSPACE = 0.065, 0.985, 0.80, 0.14, 0.24
TICK_PT = 10.5


def configure_style():
    """Shared rcParams of the dataset-statistics figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        # Single source of truth, shared with the clip previews' captions.
        'font.sans-serif': SANS_STACK,
        'mathtext.fontset': 'dejavusans',
        'axes.titlesize': 14,
        'axes.titleweight': 'semibold',
        'axes.titlepad': 14,
        'axes.labelsize': 12,
        'axes.labelweight': 'medium',
        'axes.labelcolor': INK,
        'axes.edgecolor': EDGE,
        'axes.linewidth': 0.9,
        'xtick.labelsize': TICK_PT,
        'ytick.labelsize': TICK_PT,
        'xtick.color': '#374151',
        'ytick.color': '#374151',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': False,
        'legend.fontsize': 10.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.dpi': 300,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


# ---------------------------------------------------------------------------
# Inputs: the stage-1 summary JSONs
# ---------------------------------------------------------------------------

def resolve_inputs(paths, filename, default_glob):
    """Expand export dirs / JSON paths (or *default_glob*) to JSON files.

    A directory argument resolves to ``<dir>/<filename>``; a file argument
    is taken as-is (so worker shards like ``<stem>_worker3.json`` work).
    """
    if not paths:
        paths = sorted(glob.glob(default_glob))
        if not paths:
            raise FileNotFoundError(
                f'No files match {default_glob}; pass export directories or '
                f'{filename} paths explicitly.')
    resolved = []
    for p in paths:
        path = os.path.join(p, filename) if os.path.isdir(p) else p
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{filename} not found: {path}')
        resolved.append(path)
    return resolved


def label_for(path, filename):
    """Export directory name for the canonical file, else the file stem."""
    base = os.path.basename(path)
    if base == filename:
        return os.path.basename(os.path.dirname(os.path.abspath(path)))
    return os.path.splitext(base)[0]


def unique_labels(paths, filename):
    """Per-path labels, suffixed when two inputs would otherwise collide."""
    seen = {}
    out = []
    for lab in (label_for(p, filename) for p in paths):
        n = seen.get(lab, 0)
        seen[lab] = n + 1
        out.append(lab if n == 0 else f'{lab}_{n}')
    return out


def display_name(label):
    """Dataset directory names are lowercase; figures show them capitalized."""
    return label[:1].upper() + label[1:]


def load_int_map(path):
    """Load a ``{key: int}`` JSON and return the values as an int array."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f'{path}: expected a {{name: count}} object, '
                         f'got {type(data).__name__}')
    values = np.array([int(v) for v in data.values()], dtype=np.int64)
    if values.size == 0:
        raise ValueError(f'{path}: no entries')
    return values


def sibling_json_field(path, filename, key, default):
    """``key`` from ``<dir of path>/<filename>``, else *default*."""
    sibling = os.path.join(os.path.dirname(path), filename)
    try:
        with open(sibling) as f:
            value = json.load(f).get(key, default)
        return type(default)(value) if default is not None else value
    except (OSError, ValueError, TypeError):
        return default


def parse_int_list(spec, flag):
    """``"60,90,120"`` → ``[60, 90, 120]`` (sorted, deduped); empty → []."""
    spec = (spec or '').strip()
    if not spec:
        return []
    out = sorted({int(tok) for tok in spec.split(',') if tok.strip()})
    if any(t <= 0 for t in out):
        raise ValueError(f'{flag} must be positive integers, got {spec!r}')
    return out


def percentile_stats(values):
    """The percentile block shared by every summary."""
    def q(p):
        return float(np.percentile(values, p))
    return {
        'min': int(values.min()),
        'p10': q(10), 'p25': q(25), 'median': q(50), 'mean': float(values.mean()),
        'p75': q(75), 'p90': q(90), 'p99': q(99),
        'max': int(values.max()),
    }


# ---------------------------------------------------------------------------
# Distribution math
# ---------------------------------------------------------------------------

def to_axis(x, log_x):
    x = np.asarray(x, dtype=float)
    return np.log10(x) if log_x else x


def from_axis(u, log_x):
    return 10.0 ** u if log_x else u


def value_range(values, log_x):
    """Padded axis range: ×1.15 either side on log axes, 3 % on linear."""
    lo, hi = float(values.min()), float(values.max())
    if log_x:
        return max(1.0, lo / 1.15), hi * 1.15
    pad = max(1.0, (hi - lo) * 0.03)
    return max(0.0, lo - pad), hi + pad


def even_bins(lo, hi, n, log_x):
    """*n* bins of equal width in axis units (decades on a log axis)."""
    return from_axis(np.linspace(*to_axis([lo, hi], log_x), n + 1), log_x)


def integer_bins(values, width=None):
    """Integer-aligned bins for count data (bars centered on the integers).

    Width defaults to 1 / 2 / 3 for a value span of ≤30 / ≤60 / more, as in
    the joint-count figure. Returns ``(edges, width)``.
    """
    lo, hi = int(values.min()), int(values.max())
    if width is None:
        span = hi - lo
        width = 1 if span <= 30 else (2 if span <= 60 else 3)
    edges = np.arange(lo, hi + width + 1, width) - 0.5
    return edges, width


def kde_grid(lo, hi, log_x, n=KDE_POINTS):
    return from_axis(np.linspace(*to_axis([lo, hi], log_x), n), log_x)


def kde_proportion(values, grid, bin_width, log_x, bw=KDE_BW):
    """Gaussian KDE on *grid*, scaled to the expected proportion per bin.

    Evaluated in log10 space for log axes so the curve matches log-spaced
    bins (``bin_width`` is then in decades). Bandwidth is ``bw * std``, the
    scalar-factor rule scipy's ``gaussian_kde`` applies for
    ``bw_method=bw``. Returns None for a degenerate sample (fewer than two
    distinct values).
    """
    u = to_axis(values, log_x)
    if u.size < 2:
        return None
    std = u.std(ddof=1)
    if not std > 0:
        return None
    h = bw * std
    g = to_axis(grid, log_x)
    dens = np.zeros(g.size)
    for start in range(0, u.size, 4096):          # bound the (grid × samples) temp
        z = (g[:, None] - u[None, start:start + 4096]) / h
        dens += np.exp(-0.5 * z * z).sum(axis=1)
    dens /= u.size * h * np.sqrt(2.0 * np.pi)
    return dens * bin_width


def survival_curve(values, span):
    """Share of entries with ``>= x``, as ``(xs, ys, where)`` for ``ax.step``.

    ``P(X >= x)`` holds its value on ``(u[i-1], u[i]]`` and drops just past
    each value, so the step mode is ``'pre'``. The curve is extended across
    *span* ``(lo, hi)``: 1 below the minimum, 0 above the maximum — which
    also makes a single-valued sample visible as one full-height step.
    """
    lo, hi = span
    uniq, counts = np.unique(values, return_counts=True)
    below = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ys = 1.0 - below / values.size
    xs = np.concatenate([[lo], uniq, [hi]])
    ys = np.concatenate([[1.0], ys, [0.0]])
    return xs, ys, 'pre'


def cdf_curve(values, span):
    """Share of entries with ``<= x``, as ``(xs, ys, where)`` for ``ax.step``.

    ``P(X <= x)`` holds its value on ``[u[i], u[i+1])``, so the step mode is
    ``'post'``. Extended across *span*: 0 below the minimum, 1 above the
    maximum.
    """
    lo, hi = span
    uniq, counts = np.unique(values, return_counts=True)
    ys = np.cumsum(counts) / values.size
    xs = np.concatenate([[lo], uniq, [hi]])
    ys = np.concatenate([[0.0], ys, [1.0]])
    return xs, ys, 'post'


# ---------------------------------------------------------------------------
# Axes & annotations
# ---------------------------------------------------------------------------

def two_panel_figure():
    return plt.subplots(1, 2, figsize=(FIG_W, FIG_H))


def style_axes(ax):
    ax.grid(True, axis='y', color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=4)
    ax.tick_params(axis='x', length=4)


def pct_axis(ax):
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))


def panel_width_in(ax):
    """Approximate width of one panel in inches under the shared layout."""
    return ax.figure.get_figwidth() * (RIGHT - LEFT) / (2 + WSPACE)


def _log_tick_candidates(lo, hi):
    """1-2-5 ticks per decade covering [lo, hi]."""
    e_lo, e_hi = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    return [m * 10 ** e for e in range(e_lo, e_hi + 1) for m in (1, 2, 5)
            if lo <= m * 10 ** e <= hi]


def thin_log_ticks(ticks, lo, hi, panel_in, fmt, font_pt=TICK_PT, gap_pt=8.0):
    """Drop ticks whose labels would overlap the previous kept label.

    Label width is estimated from character count (~0.6 em per glyph);
    positions are compared in points along the log axis.
    """
    pt_per_decade = panel_in * 72.0 / max(np.log10(hi / lo), 1e-9)
    kept, last_center, last_half = [], None, 0.0
    for t in ticks:
        center = np.log10(t) * pt_per_decade
        half = 0.6 * font_pt * len(fmt(t)) / 2.0
        if last_center is None or center - last_center >= last_half + half + gap_pt:
            kept.append(t)
            last_center, last_half = center, half
    return kept


def value_axis(ax, lo, hi, log_x, xlabel):
    """Integer-formatted x axis: thinned 1-2-5 ticks on log, MaxNLocator on
    linear. Sets the limits and label."""
    fmt = lambda v: f'{int(round(v)):,}'  # noqa: E731
    if log_x:
        ax.set_xscale('log')
        ticks = thin_log_ticks(_log_tick_candidates(lo, hi), lo, hi,
                               panel_width_in(ax), fmt)
        if len(ticks) < 2:
            ticks = [int(round(lo)), int(round(hi))]
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
    else:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt(v)))
    ax.set_xlim(lo, hi)
    ax.set_xlabel(xlabel)


def x_frac(ax, x):
    """Position of data *x* as a fraction of the axis width (log-aware)."""
    lo, hi = ax.get_xlim()
    if ax.get_xscale() == 'log':
        return (np.log10(x) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    return (x - lo) / (hi - lo)


def guide(ax, x, text, color, dash, side, y_frac=0.985, fontsize=10):
    """Vertical guide with an inline label hung from the top of the panel."""
    ax.axvline(x, color=color, linewidth=1.0, linestyle=dash, alpha=0.85, zorder=4)
    ax.text(x, y_frac, f' {text} ', transform=ax.get_xaxis_transform(),
            ha=side, va='top', fontsize=fontsize, color=color, zorder=5,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.85))


def mean_median_guides(ax, mean_v, median_v, edge_frac=0.15):
    """Mean (dashed, ink) and median (dotted, muted) guides.

    The smaller value is labeled on its left and the larger on its right so
    the labels never cross. When the smaller one sits within *edge_frac* of
    the left edge — where a left-hung label would run into the y tick labels
    — it is labeled on the right instead, one step lower than the other.
    """
    style = {
        'mean':   (lambda v: f'mean {v:.1f}', INK, (0, (5, 3))),
        'median': (lambda v: f'median {v:g}', MUTED, (0, (1.5, 2.5))),
    }
    (lo_v, lo_kind), (hi_v, hi_kind) = sorted([(mean_v, 'mean'), (median_v, 'median')])
    room_on_left = x_frac(ax, lo_v) >= edge_frac
    for value, kind, side, y_frac in (
        (lo_v, lo_kind, 'right' if room_on_left else 'left', 0.985 if room_on_left else 0.90),
        (hi_v, hi_kind, 'left', 0.985),
    ):
        text, color, dash = style[kind]
        guide(ax, value, text(value), color, dash, side, y_frac=y_frac)


def legend_loc_clear_of(ax, guide_xs, label_frac=0.2, legend_frac=0.3):
    """Upper corner for the legend that the top-hung guide labels don't reach.

    Guide labels extend ~*label_frac* of the axis from their line (the
    smaller value leftward, the larger rightward); the legend occupies
    ~*legend_frac* from its corner. Prefer upper right; fall back to upper
    left when the rightmost guide label would reach into it.
    """
    fr = [x_frac(ax, x) for x in guide_xs]
    if not fr or max(fr) + label_frac <= 1.0 - legend_frac:
        return 'upper right'
    if min(fr) - label_frac >= legend_frac:
        return 'upper left'
    return 'upper right'


def threshold_guides(ax, thresholds, fmt, at_bottom=()):
    """Dashed guide per threshold, labeled along the line with ``fmt(t)``.

    Labels hang from the top; thresholds in *at_bottom* (where a curve still
    runs near the top) are labeled from the baseline instead.
    """
    for t in thresholds:
        ax.axvline(t, color=EDGE, linewidth=1.0, linestyle=(0, (5, 3)), zorder=3)
        bottom = t in at_bottom
        ax.annotate(fmt(t), xy=(t, 0.03 if bottom else 0.985),
                    xycoords=ax.get_xaxis_transform(),
                    xytext=(4, 0), textcoords='offset points',
                    rotation=90, ha='left', va='bottom' if bottom else 'top',
                    fontsize=9, color=MUTED, zorder=4)


def crowded_top(value_at, thresholds, limit=0.8):
    """Thresholds where the curve value ``value_at(t)`` is high enough to
    collide with a top-hung label."""
    return {t for t in thresholds if value_at(t) > limit}


def mark_points(ax, points, fmt=lambda y: f'{y:.1%}', color=ACCENT_DK,
                prefer='above', edge=0.1):
    """Filled markers with a white ring and an offset value label.

    *prefer* puts the label above-right (free space beside a falling
    survival curve) or below-right (free space beside a rising cumulative
    curve); within *edge* of the y-range's top or bottom the side flips so
    the label stays inside the panel.
    """
    y_lo, y_hi = ax.get_ylim()
    span = y_hi - y_lo
    for x, y in points:
        ax.plot([x], [y], marker='o', markersize=6.5, color=color,
                markeredgecolor='white', markeredgewidth=1.2, zorder=6)
        above = prefer == 'above'
        if above and y > y_hi - edge * span:
            above = False
        elif not above and y < y_lo + edge * span:
            above = True
        ax.annotate(fmt(y), (x, y), xytext=(6, 5 if above else -6),
                    textcoords='offset points', va='bottom' if above else 'top',
                    fontsize=10, color=INK, zorder=6)


def finish_figure(fig, title, subtitle, out_base, pdf):
    """Title + subtitle, shared layout, save PNG (+ PDF), close. Returns the
    written paths."""
    fig.suptitle(title, y=0.985, fontsize=14, fontweight='semibold', color=INK)
    fig.text(0.5, 0.895, subtitle, ha='center', va='top', fontsize=10.5, color=MUTED)
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM, wspace=WSPACE)
    os.makedirs(os.path.dirname(os.path.abspath(out_base)), exist_ok=True)
    written = [out_base + '.png']
    fig.savefig(written[0])
    if pdf:
        written.append(out_base + '.pdf')
        fig.savefig(written[1])
    plt.close(fig)
    return written


def per_dataset_out_base(path, label, per_dataset_dir, suffix):
    """Where a per-dataset figure goes and whether a PDF is allowed.

    Default: beside the source JSON as ``<stem>_<suffix>`` (PNG only, so the
    export directory carries its own figure). With *per_dataset_dir*:
    ``<dir>/<label>_<stem>_<suffix>`` (PDF allowed).
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if per_dataset_dir:
        return os.path.join(per_dataset_dir, f'{label}_{stem}_{suffix}'), True
    return os.path.join(os.path.dirname(os.path.abspath(path)), f'{stem}_{suffix}'), False
