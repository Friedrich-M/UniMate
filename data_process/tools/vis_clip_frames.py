"""Plot the per-clip frame-count distribution of ``clip_frames.json`` files.

Stage 1 writes ``<export_dir>/clip_frames.json`` — a flat ``{clip_name:
nframes}`` map (see :func:`data_process.utils.blender_export.write_clip_frames_json`).
This tool turns one or more of those files into paper-style distribution
figures plus a summary table, so clip-length settings — stage 4's
``--max_clip_len`` and the training ``max_motion_length`` of 60 / 90 / 120
frames — can be checked against the data instead of guessed.

Read-only: the JSON inputs are never modified. Styling follows the other
dataset-statistics figures (see :mod:`data_process.utils.dist_plot`).

Outputs:

  * ``<export_dir>/clip_frames_distribution.png`` — per file, written next
    to its JSON (PNG only) so every export directory carries its own
    clip-length figure; the name is the JSON stem plus ``_distribution``
    (``clip_frames_worker3.json`` → ``clip_frames_worker3_distribution.png``).
    Two panels:
      - histogram of frames per clip as a proportion of the file's clips
        (log-x by default — frame counts are heavy-tailed), with a kernel
        density estimate and mean / median guides;
      - the survival curve "share of clips with at least N frames", with
        the threshold frame counts marked and the share labeled at each.
    ``--per_dataset_dir DIR`` collects them in one directory instead, as
    ``DIR/<label>_clip_frames_distribution.{png,pdf}``.
  * ``<output_dir>/clip_frames_distribution_comparison.{png,pdf}`` — when
    several files are given: the kernel densities and survival curves of
    every dataset overlaid, each normalized to its own clip count so a
    10k-clip set doesn't swamp a 1k-clip one.
  * ``<output_dir>/clip_frames_summary.json`` — the numbers behind the plots
    (percentiles, totals, per-threshold counts and shares), also printed as
    a table.

``<label>`` is the export directory name for a canonical ``clip_frames.json``
(``dataset/export/truebones/clip_frames.json`` → ``truebones``) and the file
stem otherwise (``clip_frames_worker3.json`` → ``clip_frames_worker3``). The
frame rate used to express thresholds in seconds is read from the sibling
``summary.json`` when present, else ``--fps``.

Usage:
    python -m data_process.tools.vis_clip_frames                       # every dataset/export/*/clip_frames.json
    python -m data_process.tools.vis_clip_frames dataset/export/truebones
    python -m data_process.tools.vis_clip_frames a/clip_frames.json b/clip_frames.json \\
        --thresholds 60,120 --per_dataset_dir outputs/clip_frames_vis --no-pdf
"""

import argparse
import json
import os

import numpy as np

from data_process.utils import dist_plot as dp


DEFAULT_GLOB = 'dataset/export/*/clip_frames.json'
DEFAULT_OUTPUT_DIR = 'outputs/clip_frames_vis'
FILENAME = 'clip_frames.json'
SUFFIX = 'distribution'
DEFAULT_THRESHOLDS = '60,90,120,200'   # training max_motion_length options + stage-4 max_clip_len
DEFAULT_FPS = 30


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def summarize(frames, fps, thresholds):
    total = int(frames.sum())
    stats = {'n_clips': int(frames.size)}
    stats.update(dp.percentile_stats(frames))
    stats.update({
        'total_frames': total,
        'fps': fps,
        'duration_min': total / fps / 60.0,
        # Share of clips at least N frames long: the fraction usable at a
        # given max_motion_length without padding.
        'at_least': {
            str(t): {'count': int((frames >= t).sum()),
                     'share': float((frames >= t).mean())}
            for t in thresholds
        },
    })
    return stats


def print_table(stats_by_label, thresholds):
    cols = ['clips', 'min', 'p10', 'median', 'mean', 'p90', 'max', 'minutes']
    cols += [f'>={t}f' for t in thresholds]
    width = max(len(lab) for lab in stats_by_label)
    print(f'{"dataset":<{width}}  ' + '  '.join(f'{c:>8}' for c in cols))
    for lab, s in stats_by_label.items():
        row = [f'{s["n_clips"]:,}', f'{s["min"]}', f'{s["p10"]:.0f}',
               f'{s["median"]:.0f}', f'{s["mean"]:.1f}', f'{s["p90"]:.0f}',
               f'{s["max"]:,}', f'{s["duration_min"]:.1f}']
        row += [f'{s["at_least"][str(t)]["share"]:.1%}' for t in thresholds]
        print(f'{lab:<{width}}  ' + '  '.join(f'{c:>8}' for c in row))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _xlabel(fps, log_x):
    return f'Frames per clip ({fps} fps' + (', log scale)' if log_x else ')')


def _threshold_label(fps):
    return lambda t: f'{t} f  ·  {t / fps:g} s'


def plot_dataset(label, frames, fps, stats, thresholds, bins, log_x, out_base, pdf):
    """Histogram + KDE + mean/median guides, and the survival curve with
    threshold guides, for one clip_frames.json."""
    fig, (ax_h, ax_s) = dp.two_panel_figure()
    lo, hi = dp.value_range(frames, log_x)
    edges = dp.even_bins(lo, hi, bins, log_x)
    shown = [t for t in thresholds if lo <= t <= hi]

    # ---- Histogram (proportion of clips) + kernel density ----
    counts, _ = np.histogram(frames, bins=edges)
    props = counts / frames.size
    widths = np.diff(edges)
    ax_h.bar(edges[:-1] + 0.04 * widths, props, width=0.92 * widths, align='edge',
             color=dp.ACCENT, edgecolor='white', linewidth=0.6, alpha=0.92,
             label='Histogram', zorder=2)
    grid = dp.kde_grid(lo, hi, log_x)
    bin_width = float(np.diff(dp.to_axis(edges[:2], log_x))[0])
    kde = dp.kde_proportion(frames, grid, bin_width, log_x)
    y_top = props.max()
    if kde is not None:
        ax_h.fill_between(grid, kde, color=dp.ACCENT_FL, alpha=0.30, linewidth=0, zorder=1)
        ax_h.plot(grid, kde, color=dp.ACCENT_DK, linewidth=1.8, solid_capstyle='round',
                  label='Kernel density', zorder=3)
        y_top = max(y_top, float(kde.max()))
    ax_h.set_ylim(0, y_top * dp.HEADROOM)
    dp.style_axes(ax_h)
    dp.value_axis(ax_h, lo, hi, log_x, _xlabel(fps, log_x))
    dp.pct_axis(ax_h)
    ax_h.set_ylabel('Proportion of clips')
    dp.mean_median_guides(ax_h, stats['mean'], stats['median'])
    ax_h.legend(loc=dp.legend_loc_clear_of(ax_h, (stats['mean'], stats['median'])),
                borderaxespad=0.6, handlelength=1.8)

    # ---- Survival: share of clips with at least N frames ----
    xs, ys, where = dp.survival_curve(frames, span=(lo, hi))
    ax_s.fill_between(xs, ys, step=where, color=dp.ACCENT_FL, alpha=0.30,
                      linewidth=0, zorder=1)
    ax_s.step(xs, ys, where=where, color=dp.ACCENT_DK, linewidth=1.8,
              solid_capstyle='round', solid_joinstyle='round', zorder=3)
    ax_s.set_ylim(0, 1.06)
    dp.style_axes(ax_s)
    dp.value_axis(ax_s, lo, hi, log_x, _xlabel(fps, log_x))
    dp.pct_axis(ax_s)
    ax_s.set_ylabel('Share of clips with ≥ N frames')
    shares = {t: stats['at_least'][str(t)]['share'] for t in shown}
    dp.threshold_guides(ax_s, shown, _threshold_label(fps),
                        at_bottom=dp.crowded_top(shares.get, shown))
    dp.mark_points(ax_s, [(t, shares[t]) for t in shown])

    subtitle = (f'{stats["n_clips"]:,} clips  ·  median {stats["median"]:.0f} frames'
                f'  ·  range {stats["min"]}–{stats["max"]:,}'
                f'  ·  {stats["duration_min"]:.1f} min at {fps} fps')
    return dp.finish_figure(fig, f'Distribution of Clip Lengths — {dp.display_name(label)}',
                            subtitle, out_base, pdf)


def plot_comparison(datasets, thresholds, bins, log_x, out_base, pdf):
    """Overlay several datasets: kernel densities (proportion per bin) and
    survival curves, each normalized to its own clip count.

    ``datasets`` is ``[(label, frames, fps), ...]``; one categorical hue per
    dataset, in input order.
    """
    fig, (ax_k, ax_s) = dp.two_panel_figure()
    all_frames = np.concatenate([f for _, f, _ in datasets])
    lo, hi = dp.value_range(all_frames, log_x)
    edges = dp.even_bins(lo, hi, bins, log_x)
    shown = [t for t in thresholds if lo <= t <= hi]
    fps_set = sorted({fps for _, _, fps in datasets})
    fps_label = fps_set[0] if len(fps_set) == 1 else '/'.join(map(str, fps_set))

    grid = dp.kde_grid(lo, hi, log_x)
    bin_width = float(np.diff(dp.to_axis(edges[:2], log_x))[0])
    y_top = 0.0
    for i, (label, frames, _) in enumerate(datasets):
        color = dp.SERIES[i]
        name = dp.display_name(label)
        kde = dp.kde_proportion(frames, grid, bin_width, log_x)
        if kde is not None:
            ax_k.fill_between(grid, kde, color=color, alpha=0.12, linewidth=0, zorder=1)
            ax_k.plot(grid, kde, color=color, linewidth=1.8, solid_capstyle='round',
                      label=name, zorder=3)
            y_top = max(y_top, float(kde.max()))
        xs, ys, where = dp.survival_curve(frames, span=(lo, hi))
        ax_s.step(xs, ys, where=where, color=color, linewidth=1.8, label=name,
                  solid_capstyle='round', solid_joinstyle='round', zorder=3)

    ax_k.set_ylim(0, (y_top or 1.0) * dp.HEADROOM)
    dp.style_axes(ax_k)
    dp.value_axis(ax_k, lo, hi, log_x, _xlabel(fps_label, log_x))
    dp.pct_axis(ax_k)
    ax_k.set_ylabel(f'Proportion of clips per bin ({bins} bins)')

    ax_s.set_ylim(0, 1.06)
    dp.style_axes(ax_s)
    dp.value_axis(ax_s, lo, hi, log_x, _xlabel(fps_label, log_x))
    dp.pct_axis(ax_s)
    ax_s.set_ylabel('Share of clips with ≥ N frames')
    crowded = dp.crowded_top(
        lambda t: max(float((f >= t).mean()) for _, f, _ in datasets), shown)
    fmt = _threshold_label(fps_set[0]) if len(fps_set) == 1 else (lambda t: f'{t} f')
    dp.threshold_guides(ax_s, shown, fmt, at_bottom=crowded)
    # Every dataset has a survival step (a density needs >= 2 distinct
    # values), so the legend lives on this panel.
    ax_s.legend(loc='upper right', borderaxespad=0.6, handlelength=1.8)

    subtitle = (f'{len(datasets)} datasets  ·  {all_frames.size:,} clips  ·  '
                + '  ·  '.join(f'{dp.display_name(lab)} {f.size:,}' for lab, f, _ in datasets))
    return dp.finish_figure(fig, 'Distribution of Clip Lengths across Datasets', subtitle,
                            out_base, pdf)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot the per-clip frame-count distribution of export-stage '
                    'clip_frames.json files.')
    parser.add_argument('inputs', nargs='*',
                        help='Export directories (containing clip_frames.json) or '
                             f'clip_frames*.json paths. Default: {DEFAULT_GLOB}')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help='Where the comparison figure and the summary JSON go '
                             f'(default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--per_dataset_dir', default=None,
                        help='Collect the per-dataset figures in this directory as '
                             '<label>_clip_frames_distribution.{png,pdf}. Default: write '
                             'each one as clip_frames_distribution.png next to its own '
                             'clip_frames.json.')
    parser.add_argument('--thresholds', default=DEFAULT_THRESHOLDS,
                        help='Comma-separated frame counts to mark (default: '
                             f'{DEFAULT_THRESHOLDS} — the training max_motion_length '
                             'options and stage-4 max_clip_len). Empty string disables.')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS,
                        help='Frame rate for the seconds labels when no sibling '
                             f'summary.json records one (default: {DEFAULT_FPS})')
    parser.add_argument('--bins', type=int, default=40,
                        help='Histogram bin count (default: 40)')
    parser.add_argument('--linear', action='store_true',
                        help='Linear frame axis instead of log scale')
    parser.add_argument('--no_comparison', action='store_true',
                        help='Skip the overlaid comparison figure')
    parser.add_argument('--pdf', action=argparse.BooleanOptionalAction, default=True,
                        help='Also write a PDF next to the comparison figure, and next '
                             'to the per-dataset figures when --per_dataset_dir is set '
                             '(default: on; figures beside the JSONs are always PNG only)')
    return parser.parse_args()


def main():
    args = parse_args()
    dp.configure_style()
    thresholds = dp.parse_int_list(args.thresholds, '--thresholds')
    paths = dp.resolve_inputs(args.inputs, FILENAME, DEFAULT_GLOB)
    labels = dp.unique_labels(paths, FILENAME)
    log_x = not args.linear
    os.makedirs(args.output_dir, exist_ok=True)

    datasets = []
    summary = {}
    for path, label in zip(paths, labels):
        frames = dp.load_int_map(path)
        fps = dp.sibling_json_field(path, 'summary.json', 'fps', args.fps)
        stats = summarize(frames, fps, thresholds)
        out_base, pdf_ok = dp.per_dataset_out_base(path, label, args.per_dataset_dir, SUFFIX)
        written = plot_dataset(label, frames, fps, stats, thresholds, args.bins,
                               log_x, out_base, pdf_ok and args.pdf)
        print(f'[{label}] {frames.size:,} clips from {path} -> {", ".join(written)}')
        datasets.append((label, frames, fps))
        summary[label] = dict(stats, source=os.path.abspath(path))

    if len(datasets) > 1 and not args.no_comparison:
        if len(datasets) > len(dp.SERIES):
            print(f'Skipping the comparison figure: {len(datasets)} inputs exceed the '
                  f'{len(dp.SERIES)} distinguishable series; compare in smaller groups.')
        else:
            written = plot_comparison(
                datasets, thresholds, args.bins, log_x,
                os.path.join(args.output_dir, 'clip_frames_distribution_comparison'), args.pdf)
            print(f'[comparison] {len(datasets)} datasets -> {", ".join(written)}')

    summary_path = os.path.join(args.output_dir, 'clip_frames_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary -> {summary_path}\n')
    print_table(summary, thresholds)


if __name__ == '__main__':
    main()
